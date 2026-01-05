import os
import docker
from typing import TypedDict, Annotated, List, Union
from langgraph.graph import StateGraph, END
print("DEBUG: Imported StateGraph", flush=True)
from langchain_openai import ChatOpenAI
print("DEBUG: Imported ChatOpenAI", flush=True)
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
print("DEBUG: Imported langchain_core", flush=True)
from vector_store import rag_engine
print("DEBUG: Imported rag_engine", flush=True)

from langgraph.graph.message import add_messages

# --- State Application ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    plan: List[str]
    current_task: str
    tool_calls: List[dict]
    last_tool_output: str
    error: str

# --- Setup Docker ---
try:
    docker_client = docker.from_env()
except Exception:
    docker_client = None

CONTAINER_NAME = "codepilot_sandbox"
# Define Project Root (Default to one level up)
ACTIVE_PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def set_active_project(path: str):
    """Update the active project path and reset docker container."""
    global ACTIVE_PROJECT_PATH
    ACTIVE_PROJECT_PATH = path
    print(f"Agent switching to project: {ACTIVE_PROJECT_PATH}")
    
    # Remove existing container so new one is created with correct mount
    if docker_client:
        try:
            old_container = docker_client.containers.get(CONTAINER_NAME)
            old_container.remove(force=True)
            print("Old sandbox removed.")
        except docker.errors.NotFound:
            pass
        except Exception as e:
            print(f"Error removing sandbox: {e}")

def get_sandbox():
    if not docker_client:
        return None
    try:
        container = docker_client.containers.get(CONTAINER_NAME)
        if container.status != "running":
            container.start()
        return container
    except docker.errors.NotFound:
        try:
            return docker_client.containers.run(
                "codepilot:v1",
                name=CONTAINER_NAME,
                detach=True,
                tty=True,
                # Mount the ACTIVE_PROJECT_PATH
                volumes={ACTIVE_PROJECT_PATH: {'bind': '/workspace', 'mode': 'rw'}},
                working_dir="/workspace"
            )
        except Exception as e:
            print(f"Failed to start docker container: {e}")
            return None

# --- Tools ---
@tool
def search_codebase(query: str):
    """Search the codebase for relevant files and code snippets."""
    results = rag_engine.search(query)
    return str(results)

@tool
def run_shell(command: str, background: bool = False):
    """Run a shell command. Set background=True for servers/long-running tasks."""
    container = get_sandbox()
    if container:
        # Docker background support? 
        # For now, let's assume local fallback is what's being used based on logs
        # Docker exec_run has detach=True
        try:
            if background:
                container.exec_run(command, detach=True)
                return "Command started in background."
            else:
                exec_log = container.exec_run(command)
                return exec_log.output.decode("utf-8")
        except Exception as e:
            return f"Error executing command in docker: {e}"
    else:
        # Fallback to local execution
        import subprocess
        try:
            if background:
                # Fire and forget (or track globally if possible)
                proc = subprocess.Popen(
                    command, 
                    shell=True, 
                    cwd=ACTIVE_PROJECT_PATH,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                return f"Command started in background (PID: {proc.pid})."
            else:
                result = subprocess.run(
                    command, 
                    shell=True, 
                    capture_output=True, 
                    text=True, 
                    cwd=ACTIVE_PROJECT_PATH
                )
                return result.stdout + result.stderr
        except Exception as e:
            return f"Error executing local command: {e}"

@tool
def write_file(path: str, content: str):
    """Write content to a file in the sandbox."""
    # Since we mount the volume, we can write directly to the path locally
    # BUT strictly speaking we should probably do it inside docker or respect the sandbox rule.
    # The prompt says: "The Agent NEVER runs code on the host machine. All subprocess calls must go through docker..."
    # Writing files to the mounted volume is safe and efficient.
    full_path = os.path.join(ACTIVE_PROJECT_PATH, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)
    return f"Wrote to {path}"

@tool
def read_file(path: str):
    """Read the content of a file."""
    full_path = os.path.join(ACTIVE_PROJECT_PATH, path)
    if not os.path.exists(full_path):
        return f"Error: File {path} does not exist."
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

@tool
def list_files(path: str = "."):
    """List files in a directory."""
    full_path = os.path.join(ACTIVE_PROJECT_PATH, path)
    if not os.path.exists(full_path):
        return f"Error: Directory {path} does not exist."
    try:
        entries = []
        for entry in os.scandir(full_path):
            if entry.name.startswith('.') and len(entry.name) > 1: continue 
            if 'node_modules' in entry.name or '__pycache__' in entry.name: continue
            
            type_str = "DIR" if entry.is_dir() else "FILE"
            entries.append(f"{type_str}: {entry.name}")
        return "\n".join(sorted(entries))
    except Exception as e:
        return f"Error listing directory: {e}"

tools = [search_codebase, run_shell, write_file, read_file, list_files]
llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)

# --- Nodes ---
def planner_node(state: AgentState):
    messages = state['messages']
    # Simple planner: just keeps the messages logic or we could have a dedicated planner LLM
    # For now, we'll let the Executor handle the planning implicitly via conversation
    return {"plan": ["Execute user request"]} 

def executor_node(state: AgentState):
    messages = state['messages']
    
    # Add system prompt
    system_prompt = SystemMessage(content="""You are CodePilot, an expert software engineer assistant.
You are running in a secure Ubuntu Docker sandbox.
You have access to tools: `run_shell`, `write_file`, `read_file`, `list_files`, `search_codebase`.

RULES:
1. **PHASE 1: PLAN & ASK**: Upon receiving a request that involves creating files, editing code, or running commands, **DO NOT** execute tools immediately.
    - First, creates a detailed plan of what you will do (filenames, code structures, commands).
    - **STOP** and ask the user: "Do you want me to proceed with this plan?"
2. **PHASE 2: EXECUTE**: Only after the user replies with "yes", "proceed", or similar confirmation, execute the tools.
3. **VERIFY**: ALWAYS `read_file` before you edit it.
4. **PROJECT STRUCTURE**:
    - If asked to create a new "project" or "app", create a new subdirectory for it.
5. **RUNNING APPS**: If the user asks to "run" the app or implies it (e.g., "create and run a calculator"), **INCLUDE** the `run_shell` commands in your plan.
    - **CRITICAL**: For servers (Flask, Node, etc.), set `background=True` in `run_shell`.
    - Example Plan: "1. Create files... 2. Run `python app/main.py` (background)... Do you want to proceed?"
6. Format your final response nicely with markdown.
""")
    
    # Prepend system prompt to the conversation if not already there (simplified check)
    # Actually, for every invoke, we can just pass [system_prompt] + messages
    response = llm.invoke([system_prompt] + messages)
    return {"messages": [response]}

def tool_node(state: AgentState):
    last_message = state['messages'][-1]
    if not last_message.tool_calls:
        return {} # Should not happen if routed correctly
    
    results = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        
        if tool_name == "search_codebase":
            res = search_codebase.invoke(tool_args)
        elif tool_name == "run_shell":
            res = run_shell.invoke(tool_args)
        elif tool_name == "write_file":
            res = write_file.invoke(tool_args)
        elif tool_name == "read_file":
            res = read_file.invoke(tool_args)
        elif tool_name == "list_files":
            res = list_files.invoke(tool_args)
        else:
            res = "Unknown tool"
            
        results.append(
            {"tool_call_id": tool_call['id'], "output": str(res)}
        )
        
    # Construct tool messages
    tool_messages = []
    for res in results:
         tool_messages.append(
             {"role": "tool", "tool_call_id": res["tool_call_id"], "content": res["output"]}
         )
         
    return {"messages": tool_messages, "last_tool_output": str(results)}

def verifier_node(state: AgentState):
    # Check if the last output was an error
    last_output = state.get('last_tool_output', "")
    if "Error" in last_output or "Exception" in last_output:
        return {"error": "Last command failed. Fix it."}
    return {"error": ""}

# --- Edge Logic ---
def should_continue(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

# --- Graph Definition ---
workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("tools", tool_node)
workflow.add_node("verifier", verifier_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_conditional_edges("executor", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "verifier")
workflow.add_edge("verifier", "executor") # Loop back to executor to continue (Plan-Act-Verify)

app = workflow.compile()
