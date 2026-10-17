from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from typing import Dict, Any, List, Optional

from services.local_dev_agent import LocalDevAgent

agent = LocalDevAgent("data.py")
USER_SESSIONS: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await agent.initialize()
    yield
    await agent.close()


app = FastAPI(title="Qwen Coder Local API with MCP", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://mcp.intovoid.dev/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MessageItem(BaseModel):
    role: str
    content: str
    tags: Optional[List[str]] = None


class QueryRequest(BaseModel):
    user_id: str
    prompt: str
    deepThink: bool
    tags: Optional[List[str]] = None
    history: Optional[List[MessageItem]] = None


@app.post("/ask")
async def ask_qwen(body: QueryRequest):
    if not body.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    if body.user_id not in USER_SESSIONS:
        USER_SESSIONS[body.user_id] = {"raw_history": [], "summary_digest": ""}

    user_session = USER_SESSIONS[body.user_id]

    if body.history:
        existing = {(m["role"], m["content"]) for m in user_session["raw_history"]}
        for msg in body.history:
            if msg.role == "user" and msg.content == body.prompt:
                continue
            key = (msg.role, msg.content)
            if key in existing:
                continue
            existing.add(key)
            user_session["raw_history"].append({
                "role": msg.role,
                "content": msg.content,
                "tags": msg.tags or ["general"],
            })

    active_tags = set(body.tags or ["general"])
    filtered_history = [
        {"role": rec["role"], "content": rec["content"]}
        for rec in user_session["raw_history"]
        if set(rec.get("tags") or ["general"]) & active_tags
    ]

    user_session_for_request = {
        "raw_history": filtered_history,
        "summary_digest": user_session["summary_digest"],
    }

    async def event_generator():
        try:
            async for chunk in agent.process_message_stream(
                    body.prompt, user_session_for_request, deep_think=body.deepThink
            ):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)