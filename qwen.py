# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from typing import Dict, Any, List, Optional

from services.local_dev_agent import LocalDevAgent

agent = LocalDevAgent("data.py")

# In-memory session store mapping user_id -> session state dictionary
USER_SESSIONS: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    await agent.initialize()
    yield
    await agent.close()

app = FastAPI(title="Qwen Coder Local API with MCP", lifespan=lifespan)

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

    # Check if session exists on the server (handles server restarts gracefully)
    if body.user_id not in USER_SESSIONS:
        USER_SESSIONS[body.user_id] = {
            "raw_history": [],
            "summary_digest": ""
        }

        # If client passed stored history from IndexedDB, populate the server session
        if body.history:
            for msg in body.history:
                # Avoid duplicating the current prompt if it was accidentally included in history
                if msg.role == "user" and msg.content == body.prompt:
                    continue
                USER_SESSIONS[body.user_id]["raw_history"].append({
                    "role": msg.role,
                    "content": msg.content
                })

    user_session = USER_SESSIONS[body.user_id]

    async def event_generator():
        try:
            # Pass the isolated user session into the agent method
            async for chunk in agent.process_message_stream(body.prompt, user_session):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)