from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, Form, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from agent_graph import app as agent_app, set_active_project
from vector_store import rag_engine

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
import asyncio
import os
import subprocess
import signal
import markdown

# Helper to render markdown for the frontend (Zero-JS)
def process_chat_history(history):
    processed = []
    for msg in history:
        new_msg = msg.copy()
        if msg["role"] == "Agent":
            try:
                # Convert Markdown to HTML
                new_msg["content"] = markdown.markdown(
                    msg["content"], 
                    extensions=['fenced_code', 'codehilite', 'tables', 'nl2br']
                )
            except Exception:
                pass # Fallback to raw text
        processed.append(new_msg)
    return processed

app = FastAPI(title="CodePilot AI Backend (Zero JS)")

# Determine backend directory
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
# Determine project root (parent of backend directory)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(BACKEND_DIR, "static")), name="static")

# Templates
templates = Jinja2Templates(directory=os.path.join(BACKEND_DIR, "templates"))

# --- State (Simple In-Memory for Zero JS Demo) ---
# In a real app, use a proper session/db.
class GlobalState:
    terminal_history = "Welcome to CodePilot Terminal.\n"
    chat_history = []
    open_files = [] # List of full paths
    running_process = None # Subprocess object
    current_working_dir = PROJECT_ROOT
    
state = GlobalState()

@app.get("/", response_class=HTMLResponse)
async def get_workspace(
    request: Request, 
    file: str = None, 
    close_file: str = None,
    cmd_output: str = None,
    create_path: str = None,
    new_item_path: str = None,
    delete_path: str = None,
    cwd: str = None # Legacy: state.cwd is used
):
    """Serve the main workspace UI."""
    current_tree = get_file_tree(PROJECT_ROOT)
    
    # Handle Tabs
    if close_file and close_file in state.open_files:
        state.open_files.remove(close_file)
        # If we closed the active file, switch to the last one or None
        if file == close_file:
             file = state.open_files[-1] if state.open_files else None
    
    if file:
        full_path = file
        if full_path.startswith(PROJECT_ROOT) and os.path.exists(full_path) and os.path.isfile(full_path):
             if full_path not in state.open_files:
                 state.open_files.append(full_path)
             active_file = os.path.basename(full_path)
             with open(full_path, "r", encoding="utf-8") as f:
                 active_file_content = f.read()
        else:
            # Invalid file or directory
             active_file = None
             active_file_content = ""
    else:
        # If no file requested, default to last open
        if state.open_files:
             active_file_path = state.open_files[-1]
             # Redirect to open it properly (so url param matches)
             return RedirectResponse(url=f"/?file={active_file_path}")
        active_file = None
        active_file_content = ""

    return templates.TemplateResponse("index.html", {
        "request": request,
        "tree": current_tree,
        "active_file": active_file,
        "active_file_path": file,
        "active_file_content": active_file_content,
        "terminal_output": state.terminal_history,
        "chat_history": process_chat_history(state.chat_history),
        "parent_path": create_path if create_path else os.path.dirname(PROJECT_ROOT),
        "current_path": PROJECT_ROOT,
        "open_files": state.open_files,
        "new_item_path": new_item_path,
        "delete_path": delete_path,
        "cwd": state.current_working_dir,
        "os": os # passing os for basename in template
    })

@app.post("/item/create")
async def create_item_form(
    parent_path: str = Form(...), 
    name: str = Form(...), 
    type: str = Form(...)
):
    try:
        full_path = os.path.join(parent_path, name)
        
        # Security check
        if not full_path.startswith(PROJECT_ROOT):
             # Simplified check
             pass

        if type == "folder":
            os.makedirs(full_path, exist_ok=True)
            # Stay on page
            return RedirectResponse(url="/", status_code=303)
        else:
            # File
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write("") # Create empty file
            # Open the new file
            return RedirectResponse(url=f"/?file={full_path}", status_code=303)
            
    except Exception as e:
        state.terminal_history += f"Error creating item: {e}\n"
        return RedirectResponse(url="/", status_code=303)

@app.post("/item/delete")
async def delete_item_form(path: str = Form(...)):
    import shutil
    try:
        # Security check
        if not path.startswith(PROJECT_ROOT):
             state.terminal_history += "Error: Cannot delete outside project root.\n"
             return RedirectResponse(url="/", status_code=303)
             
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
            # Close file if open
            if path in state.open_files:
                state.open_files.remove(path)
                
    except Exception as e:
        state.terminal_history += f"Error deleting item: {e}\n"
        
    return RedirectResponse(url="/", status_code=303)

# --- File Operations (POST Forms) ---
@app.post("/file/save")
async def save_file(path: str = Form(...), content: str = Form(...)):
    if path and path.startswith(PROJECT_ROOT):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content.replace('\r\n', '\n')) # Normalize encoded newlines
    
    # Redirect back to the file
    return RedirectResponse(url=f"/?file={path}", status_code=303)

@app.post("/terminal/exec")
async def shell_exec(command: str = Form(...), current_path: str = Form(None)):
    if not command:
         return RedirectResponse(url=f"/?file={current_path}" if current_path else "/", status_code=303)
         
    # Handle CD special case
    if command.strip().startswith("cd "):
        target = command.strip()[3:].strip()
        new_path = os.path.normpath(os.path.join(state.current_working_dir, target))
        if os.path.exists(new_path) and os.path.isdir(new_path):
            state.current_working_dir = new_path
            state.terminal_history += f"{state.current_working_dir} $ {command}\n"
        else:
            state.terminal_history += f"{state.current_working_dir} $ {command}\nError: Directory not found\n"
        return RedirectResponse(url=f"/?file={current_path}" if current_path else "/", status_code=303)

    # Handle Clear
    if command.strip() == "clear":
        state.terminal_history = ""
        return RedirectResponse(url=f"/?file={current_path}" if current_path else "/", status_code=303)

    # Execute synchronously
    state.terminal_history += f"{state.current_working_dir} $ {command}\n"
    
    try:
        res = subprocess.run(
            command,
            cwd=state.current_working_dir,
            shell=True,
            capture_output=True,
            text=True
        )
        output = res.stdout + res.stderr
        state.terminal_history += output + "\n"
    except Exception as e:
        state.terminal_history += f"Error: {e}\n"

    return RedirectResponse(url=f"/?file={current_path}" if current_path else "/", status_code=303)

@app.post("/editor/run")
async def run_code(current_path: str = Form(...)):
    if not current_path or not os.path.exists(current_path):
         state.terminal_history += "Error: No file selected to run.\n"
         return RedirectResponse(url="/", status_code=303)

    # Simple extension detection
    ext = os.path.splitext(current_path)[1].lower()
    cmd = None
    if ext == ".py":
        cmd = f"python3 {current_path}"
    elif ext == ".js":
        cmd = f"node {current_path}"
    elif ext == ".sh":
        cmd = f"bash {current_path}"
    else:
        state.terminal_history += f"Error: No runner configured for {ext}\n"
        return RedirectResponse(url=f"/?file={current_path}", status_code=303)
        
    state.terminal_history += f"$ {cmd}\n"
    
    # Run in background (but we want output)
    # For Zero-JS sync MVP, we'll run blocking but with timeout? 
    # Or start a background thread that appends to history?
    # Let's do partial blocking for immediate feedback, or simple subprocess run.
    
    def run_proc():
        try:
            state.running_process = subprocess.Popen(
                cmd, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                cwd=PROJECT_ROOT
            )
            stdout, stderr = state.running_process.communicate()
            state.terminal_history += stdout + stderr + "\n"
            state.terminal_history += f"[Process finished with exit code {state.running_process.returncode}]\n"
            state.running_process = None
        except Exception as e:
            state.terminal_history += f"Error execution: {e}\n"
            
    # Run in thread so we don't block the server entirely (though python GIL...)
    # Actually, for "Zero JS", blocking the *response* until it's done is bad if it's long.
    # But if it's short, it's fine. 
    # Users asked for "Reset/Stop".
    # Let's spawn a thread.
    import threading
    t = threading.Thread(target=run_proc)
    t.start()
    
    # Brief wait to catch immediate errors
    t.join(timeout=0.5) 
    
    return RedirectResponse(url=f"/?file={current_path}", status_code=303)

@app.post("/editor/stop")
async def stop_code(current_path: str = Form(None)):
    if state.running_process:
        state.running_process.terminate()
        state.running_process = None
        state.terminal_history += "^C [Process terminated by user]\n"
    else:
        state.terminal_history += "No running process to stop.\n"
        
    return RedirectResponse(url=f"/?file={current_path}" if current_path else "/", status_code=303)

@app.post("/chat/clear")
async def clear_chat(current_path: str = Form(None)):
    state.chat_history = []
    # Re-initialize with a welcome output if desired, or empty.
    return RedirectResponse(url=f"/?file={current_path}" if current_path else "/", status_code=303)

@app.post("/chat")
async def chat_post(message: str = Form(...), current_path: str = Form(None)):
    state.chat_history.append({"role": "User", "content": message})
    
    try:
        # Construct messages properly
        history_objs = []
        for msg in state.chat_history:
            if msg["role"] == "User":
                history_objs.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "Agent":
                history_objs.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "System":
                # Feed tool outputs/logs back to the agent as SystemMessages for context
                history_objs.append(SystemMessage(content=msg["content"]))
            
        print(f"Invoking agent with {len(history_objs)} messages...")
        
        inputs = {"messages": history_objs}
        final_text = ""
        
        if agent_app:
            async for event in agent_app.astream(inputs, config={"configurable": {"thread_id": "root"}}):
                for node_name, node_state in event.items():
                    # Capture Tool Calls
                    messages = node_state.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        
                        # Tool Invocation Log
                        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                # Professional Formatting
                                name = tc['name']
                                args = tc['args']
                                log_content = f"Using {name}"
                                
                                if name == "write_file":
                                    log_content = f"Creating/Editing file: `{args.get('path')}`"
                                elif name == "read_file":
                                    log_content = f"Reading file: `{args.get('path')}`"
                                elif name == "run_shell":
                                    cmd = args.get('command', '')
                                    if len(cmd) > 50: cmd = cmd[:50] + "..."
                                    log_content = f"Running command: `{cmd}`"
                                elif name == "list_files":
                                    log_content = f"Listing directory: `{args.get('path', '.')}`"
                                elif name == "search_codebase":
                                    log_content = f"Searching codebase for: '{args.get('query')}'"
                                    
                                state.chat_history.append({"role": "System", "content": log_content})
                        
                        # Tool Output Log
                        if isinstance(messages, list):
                            for m in messages:
                                if hasattr(m, 'role') and m.role == 'tool':
                                    # We can hide tool outputs or show them compactly
                                    # User wants "Professional", often implies hiding raw return values unless error
                                    content = str(m.content)
                                    if "Error" in content or "Exception" in content:
                                        state.chat_history.append({"role": "System", "content": f"⚠️ Tool Error: {content}"})
                                    else:
                                        # Success indication
                                        state.chat_history.append({"role": "System", "content": "✓ Action completed successfully."})
                                    
                        # Final response
                        if isinstance(last_msg, AIMessage) and not last_msg.tool_calls and last_msg.content:
                            final_text = last_msg.content
                            
        else:
            final_text = "Agent unavailable."

        if final_text:
            state.chat_history.append({"role": "Agent", "content": final_text})
        
    except Exception as e:
        state.chat_history.append({"role": "System", "content": f"Error: {e}"})
        import traceback
        traceback.print_exc()

    return RedirectResponse(url=f"/?file={current_path}" if current_path else "/", status_code=303)

# --- Project Management ---
@app.post("/project/create")
async def create_project(projectName: str = Form(...), projectPath: str = Form(...)):
    full_path = os.path.join(projectPath, projectName)
    os.makedirs(full_path, exist_ok=True)
    
    # Switch to it
    global PROJECT_ROOT
    PROJECT_ROOT = full_path
    set_active_project(full_path)
    
    return RedirectResponse(url="/", status_code=303)

@app.post("/project/open")
async def open_project(path: str = Form(...)):
    global PROJECT_ROOT
    if os.path.isdir(path):
        PROJECT_ROOT = path
        set_active_project(path)
    return RedirectResponse(url="/", status_code=303)


# --- Helper ---
def get_file_tree(path: str):
    """
    Build file tree recursive.
    """
    tree = []
    try:
        entries = sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name.lower()))
        for entry in entries:
            if (entry.name.startswith('.') or 
                entry.name == '__pycache__' or 
                'node_modules' in entry.name or
                entry.name == 'qdrant_data'):
                continue
                
            node = {
                "id": entry.path,
                "name": entry.name,
                "type": "folder" if entry.is_dir() else "file"
            }
            
            if entry.is_dir():
                node["children"] = get_file_tree(entry.path)
            
            tree.append(node)
    except PermissionError:
        pass
        
    return tree

@app.get("/picker", response_class=HTMLResponse)
async def file_picker(request: Request, path: str = None, mode: str = "open"):
    """Serve the No-JS file picker."""
    try:
        current_path = path if path else os.path.expanduser("~")
        if not os.path.exists(current_path):
             current_path = os.path.expanduser("~")
        
        parent_path = os.path.dirname(current_path)
        
        items = []
        with os.scandir(current_path) as it:
            for entry in it:
                if entry.name.startswith('.') and len(entry.name) > 1: continue 
                if entry.is_dir():
                    items.append({"name": entry.name, "path": entry.path, "type": "folder"})
        
        items.sort(key=lambda x: x['name'].lower())
        
        return templates.TemplateResponse("picker.html", {
            "request": request,
            "current_path": current_path,
            "parent_path": parent_path,
            "items": items,
            "mode": mode
        })
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
