import os
import json
import traceback
from contextlib import AsyncExitStack
from typing import Dict, Any
from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage


class LocalDevAgent:
    def __init__(self, server_script_path: str = __file__):
        self.exit_stack = AsyncExitStack()
        self.tool_map = {}
        self.actionable_tools = []
        self.model = None
        self.profile_data = {}
        self.max_raw_turns = 10
        self.max_tool_rounds = 15

        if "KUBERNETES_SERVICE_HOST" in os.environ:
            self.custom_mcp_url = "https://mcp-service.intovoid.dev/sse"
        else:
            self.custom_mcp_url = "http://localhost:8001/sse"

    def _append_memory(self, session: Dict[str, Any], role: str, content: str):
        session["raw_history"].append({"role": role, "content": content})
        if len(session["raw_history"]) > (self.max_raw_turns * 2):
            overflow = session["raw_history"][:4]
            session["raw_history"] = session["raw_history"][4:]
            chunk = "\n".join([f"{m['role']}: {m['content']}" for m in overflow])
            session["summary_digest"] += f"\n[Past Context Summary]: {chunk}"

    def _get_context_payload(self, session: Dict[str, Any]) -> str:
        recent = json.dumps(session["raw_history"], indent=2)
        if session["summary_digest"]:
            return f"Background Context:\n{session['summary_digest']}\n\nRecent History:\n{recent}"
        return recent

    async def initialize(self):
        print("Initializing persistent MCP sessions...")

        try:
            custom_read, custom_write = await self.exit_stack.enter_async_context(
                sse_client(self.custom_mcp_url)
            )
            custom_session = await self.exit_stack.enter_async_context(ClientSession(custom_read, custom_write))
            await custom_session.initialize()
            custom_tools = await load_mcp_tools(custom_session)
            self.actionable_tools.extend(custom_tools)
            print(f"Successfully connected to Custom SSE MCP at {self.custom_mcp_url}.")
        except Exception as e:
            print(f"Warning: Could not connect to Custom SSE MCP server ({self.custom_mcp_url}): {e}")

        self.tool_map = {t.name: t for t in self.actionable_tools}

        if "about_author" in self.tool_map:
            try:
                self.profile_data = await self.tool_map["about_author"].ainvoke({})
            except Exception:
                pass

        self.model = ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            temperature=0.0,
            reasoning=True,

        ).bind_tools(self.actionable_tools)

    async def process_message_stream(self, user_input: str, session: Dict[str, Any]):
        try:
            system_msg = SystemMessage(
                content=(
                    f"Conversation history for this session:\n"
                    f"{self._get_context_payload(session)}\n\n"
                    "CRITICAL FORMATTING RULE: Whenever you write code blocks, you MUST explicitly specify the language identifier after the opening triple backticks. Never use bare triple backticks.\n\n"
                )
            )

            messages = [system_msg, HumanMessage(content=user_input)]
            final_answer = ""

            for _ in range(self.max_tool_rounds):
                content_text = ""
                tool_calls = []
                thinking_started = False
                async for chunk in self.model.astream(messages):
                    ak = getattr(chunk, "additional_kwargs", None) or {}
                    thinking = ak.get("reasoning_content")
                    if thinking:
                        if not thinking_started:
                            yield "\n\n> **[thinking]**\n> "
                            thinking_started = True
                        # re-prefix every newline so each line stays inside the blockquote
                        yield thinking.replace("\n", "\n> ")
                    elif thinking_started:
                        yield "\n\n"
                        thinking_started = False

                    if chunk.content:
                        content_text += chunk.content
                        yield chunk.content

                    if getattr(chunk, "tool_calls", None):
                        tool_calls.extend(chunk.tool_calls)

                if not tool_calls:
                    final_answer = content_text
                    break

                messages.append(AIMessage(content=content_text, tool_calls=tool_calls))
                yield "\n\n> *Executing tool calls...*\n\n"

                for tool_call in tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call.get("args", {})
                    tool_call_id = tool_call.get("id", "call_1")

                    if tool_name in self.tool_map:
                        try:
                            if isinstance(tool_args, str):
                                tool_args = json.loads(tool_args)
                            result = await self.tool_map[tool_name].ainvoke(tool_args)
                            result_str = json.dumps(result, ensure_ascii=False, indent=2)
                        except Exception as e:
                            result_str = f"Error: {str(e)}"
                    else:
                        result_str = f"Unknown tool: {tool_name}"

                    messages.append(ToolMessage(content=result_str, tool_call_id=tool_call_id))
                    yield f"\n\n> **Tool Result:**\n> {result_str}\n\n"

            self._append_memory(session, "human", user_input)
            self._append_memory(session, "ai", final_answer)

        except Exception as e:
            yield f"\n\n> **Error:** {type(e).__name__} – {str(e)}"
            traceback.print_exc()

    async def close(self):
        await self.exit_stack.aclose()
        print("Persistent MCP client sessions closed.")