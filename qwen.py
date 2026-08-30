from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from typing import Dict, Any, List, Optional

from services.local_dev_agent import LocalDevAgent
from data import mcp  # Import your mcp instance from data1.py

agent = LocalDevAgent("data.py")

# In-memory session store mapping user_id -> session state dictionary
USER_SESSIONS: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    await agent.initialize()
    yield
    await agent.close()

app = FastAPI(title="Qwen Coder Local API with MCP", lifespan=lifespan)

# Mount the FastMCP ASGI application directly onto your FastAPI router
# This exposes the MCP endpoints (like /mcp or SSE transport) under the /mcp prefix
app.mount("/mcp", mcp.sse_app())

# Add CORS middleware to accept preflight OPTIONS requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageItem(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    user_id: str
    prompt: str
    history: Optional[List[MessageItem]] = None

@app.post("/ask")
async def ask_qwen(body: QueryRequest):
    if not body.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    if body.user_id not in USER_SESSIONS:
        USER_SESSIONS[body.user_id] = {
            "raw_history": [],
            "summary_digest": ""
        }

        if body.history:
            for msg in body.history:
                if msg.role == "user" and msg.content == body.prompt:
                    continue
                USER_SESSIONS[body.user_id]["raw_history"].append({
                    "role": msg.role,
                    "content": msg.content
                })

    user_session = USER_SESSIONS[body.user_id]

    async def event_generator():
        try:
            async for chunk in agent.process_message_stream(body.prompt, user_session):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # Bind to 0.0.0.0 so Kubernetes Ingress and internal services can reach it across the container network
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)