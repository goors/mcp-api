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

        # 1. Connect to Custom SSE MCP server safely
        # Connect to Custom SSE MCP server safely with individual try/except
        try:
            custom_read, custom_write = await self.exit_stack.enter_async_context(
                # Remove client=http_client here
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


        # Bind tools once to the persistent model instance
        self.model = ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"),
            temperature=0.0
        ).bind_tools(self.actionable_tools)

    async def process_message_stream(self, user_input: str, session: Dict[str, Any]):
        try:
            system_msg = SystemMessage(
                content=(
                    f"You are a helpful local assistant built for the user.\n\n"
                    f"CRITICAL INSTRUCTION: The following context contains your conversation history and background digests for this specific session. "
                    f"You MUST read it, extract details from it, and use it to answer the user's questions:\n"
                    f"{self._get_context_payload(session)}\n\n"
                    "You have access to external data tools and your custom MCP server tools.\n\n"
                    "When given a task, always begin your response with a markdown blockquote analysis:\n"
                    "> **Analysis:** Analyze what the user is asking here.\n\n"
                    "CRITICAL FORMATTING RULE: Whenever you write code blocks, you MUST explicitly specify the language identifier after the opening triple backticks. Never use bare triple backticks.\n\n"
                    "After the blockquote analysis, provide your actual response or trigger the required tool call."
                )
            )

            messages = [system_msg, HumanMessage(content=user_input)]
            content_text = ""
            tool_calls_to_process = []

            # Use the pre-initialized persistent model instance
            async for chunk in self.model.astream(messages):
                if chunk.content:
                    content_text += chunk.content
                    yield chunk.content
                if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    tool_calls_to_process = chunk.tool_calls

            if not tool_calls_to_process and content_text and '{"name":' in content_text:
                try:
                    json_start = content_text.find('{"name":')
                    json_substr = content_text[json_start:]
                    brace_count = 0
                    json_end = 0
                    for i, char in enumerate(json_substr):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    if json_end > 0:
                        parsed_tool = json.loads(json_substr[:json_end])
                        if "name" in parsed_tool:
                            tool_calls_to_process = [{
                                "name": parsed_tool["name"],
                                "args": parsed_tool.get("arguments", {}),
                                "id": "fallback_call_1"
                            }]
                except Exception:
                    pass

            assistant_msg = AIMessage(content=content_text, tool_calls=tool_calls_to_process)
            messages.append(assistant_msg)
            final_answer = content_text

            if tool_calls_to_process:
                yield "\n\n> *Executing tool calls...*\n\n"
                for tool_call in tool_calls_to_process:
                    tool_name = tool_call["name"]
                    tool_args = tool_call.get("args", {})
                    tool_call_id = tool_call.get("id", "call_1")

                    if tool_name in self.tool_map:
                        try:
                            result = await self.tool_map[tool_name].ainvoke(tool_args)
                            result_str = json.dumps(result, ensure_ascii=False, indent=2)
                        except Exception as e:
                            result_str = f"Error: {str(e)}"
                    else:
                        result_str = f"Unknown tool: {tool_name}"

                    messages.append(ToolMessage(content=result_str, tool_call_id=tool_call_id))

                final_answer = ""
                async for chunk in self.model.astream(messages):
                    if chunk.content:
                        final_answer += chunk.content
                        yield chunk.content

            self._append_memory(session, "human", user_input)
            self._append_memory(session, "ai", final_answer)

        except Exception as e:
            yield f"\n\n> **Error:** {type(e).__name__} – {str(e)}"
            traceback.print_exc()

    async def close(self):
        await self.exit_stack.aclose()
        print("Persistent MCP client sessions closed.")