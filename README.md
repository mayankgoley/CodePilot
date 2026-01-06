# CodePilot AI Assistant

CodePilot is an advanced AI-powered coding assistant designed to help developers with tasks ranging from code explanation to complex refactoring. It leverages a modern tech stack to provide an autonomous agentic experience, featuring vector-based semantic search (RAG) and a responsive web interface.

## Features

- **AI Agentic Workflow**: Built with **LangGraph** to handle complex multi-step reasoning and task execution.
- **Codebase RAG**: Uses **Qdrant** vector database to ingest and understand your entire codebase.
- **Smart Terminal**: Integrated terminal support for executing commands directly from the UI.
- **Modern UI**: A lightweight, server-side rendered interface using **Jinja2 Templates**.
- **Real-time Interaction**: seamless communication between the frontend and the AI agent.

## Tech Stack

### Backend
- **Language**: Python 3.12+
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) - High-performance web framework.
- **AI/LLM**: [LangChain](https://www.langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/) - For building stateful multi-agent applications.
- **Vector DB**: [Qdrant](https://qdrant.tech/) - For efficient vector similarity search.
- **WebSocket**: For real-time streaming of agent thoughts and logs.

### Frontend
- **Templating**: [Jinja2](https://jinja.palletsprojects.com/) - Server-side rendering.
- **Styling**: Custom CSS (or Tailwind via CDN)
- **Architecture**: Zero-JS / Low-JS approach for maximum performance and simplicity.
- **Terminal**: Integrated terminal emulator.

### Infrastructure
- **Docker**: Containerization for the vector database.
- **Docker Compose**: Orchestration for local development.

## Project Structure

CodePilot/
├── backend/            # Python FastAPI Backend
│   ├── agent_graph.py  # LangGraph agent definitions
│   ├── main.py         # Application entry point
│   ├── vector_store.py # RAG implementation with Qdrant
│   ├── templates/      # Jinja2 Templates (Frontend)
│   ├── static/         # Static assets (CSS, Images)
│   └── requirements.txt
└── docker-compose.yml  # Qdrant service configuration
```

## Getting Started

### Prerequisites
- **Python 3.10+**

- **Docker & Docker Compose**

### 1. Start the Vector Database
The vector database is required for the RAG functionality.
```bash
docker-compose up -d
```
This starts a Qdrant instance on `localhost:6333`.

### 2. Setup the Backend
Navigate to the backend directory and set up the environment.

```bash
cd backend
# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Configuration**:
Copy `.env.example` to `.env` and add your OpenAI API Key.
```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

**Run the Server**:
```bash
# Start the FastAPI server
uvicorn main:app --reload --port 8001
```

### 3. Access the Application
The application is served by the FastAPI backend.
Open your browser to [http://localhost:8001](http://localhost:8001).

## License
[MIT](LICENSE)
