# LocalDevAgent — Python MCP & FastAPI Backend

A complete, production-ready backend implementation for LocalDevAgent combining FastAPI, LangChain, Ollama (Qwen 2.5 Coder 32B), and Model Context Protocol (MCP) integrations across local scripts, HTTP endpoints, and containerized Elasticsearch tools.

## Features

- Asynchronous streaming via Server-Sent Events (SSE) using FastAPI
- Multi-user session management with automated raw history management and rolling summary digests
- Model Context Protocol (MCP) tool integrations via stdio, streamable HTTP, and containerized transport
- Ephemeral Docker-based code execution sandbox for Python and Node.js
- Robust fallback parsing for tool invocations and structured system prompts

## Architecture Overview

┌─────────────────────────────────┐       HTTP / SSE       ┌──────────────────────────────┐
│  React Frontend (IndexedDB)     │ ────────────────────►  │ FastAPI Backend (main.py)    │
└─────────────────────────────────┘                        └──────────────┬───────────────┘
│
Session-Scoped Agent (LocalDevAgent)
│
┌─────────────────────────────────────────────────┼─────────────────────────────────────────────────┐
│                                                 │                                                 │
▼                                                 ▼                                                 ▼
┌───────────────────────────────┐                 ┌───────────────────────────────┐                 ┌───────────────────────────────┐
│ Local Memory MCP (data.py)    │                 │ Coinfuty MCP (HTTP)           │                 │ Elasticsearch MCP (Docker)    │
└───────────────────────────────┘                 └───────────────────────────────┘                 └───────────────────────────────┘

## Getting Started

### Installation

Clone the repository and install the required dependencies inside a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### Running the Backend

Start the FastAPI server with hot-reload enabled:

python main.py

Your server will be running at http://127.0.0.1:8000.

## API Endpoints

### POST /ask

Sends a user prompt along with optional session history to trigger streaming responses via Server-Sent Events.

Request Body (JSON):

{
"user_id": "user_123",
"prompt": "What is the current weather on Zlatibor?",
"history": []
}

Response:

A text/event-stream stream containing JSON chunks:

data: {"content": "> **Analysis:** ..."}

data: {"content": "..."}

---

Built with ❤️ using FastAPI, LangChain, and Qwen Coder.