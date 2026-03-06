'''Мост между LLM tool calls и NLP tools'''

import asyncio
import json
import logging
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path

from mcp import ClientSession, StdioServerParameters, types
from mcp.server.stdio import stdio_server
from mcp.client.stdio import stdio_client

from llm_client.ollama_client import OllamaClient, LLMResponse, ToolCall

logger = logging.getLogger(__name__)

@dataclass
class ToolExecutionResult:
    '''Результат выполнения инструмента'''
    tool_name: str
    arguments: str
    result: str
    execution_time: float
    success: bool = True
    error: str | None = None

class McpBridge:
    '''Мост между LLM и NLP-tools'''
    
    def __init__(self, server_script: str | None = None,
                 python_command: str = 'python') -> None:
        self._server_script = server_script or str(Path(__file__).parent.parent/ "mcp_server" / "server.py")
        self._python_command = python_command
        self._session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()
    
    async def __aenter__(self) -> "McpBridge":
        await self.connect()
        return self
    
    async def __aexit__(self, *args) -> None:
        await self.disconnect() 
    
    async def connect(self) -> None:
        '''Подключиться к MCP-серверу через stdio'''

        server_params = StdioServerParameters(
            command = self._python_command,
            args = ["-m", "mcp_server.server"]
        )
        logger.info("Запуск MCP-сервера: %s -m mcp_server.server",
                    self._python_command)
        
        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read_stream, write_stream = stdio_transport

        self._session = await self._exit_stack.enter_async_context(ClientSession(read_stream, write_stream))

        await self._session.initialize()
        logger.info("MCP-соединение установлено")

    async def disconnect(self) -> None:
        '''Отключиться от MCP-сервера'''
        await self._exit_stack.aclose()
        self._session = None
        logger.info('MCP-соединение закрыто')

    async def list_tools(self) -> list[dict]:
        '''Получить список доступных инструментов от MCP-сервера'''
        if self._session is None:
            raise RuntimeError("MCP-сессия не установлена. Вызови connect()")
        result = await self._session.list_tools()

        tools = []
        for tool in result.tools:
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            })

        logger.info(
            "MCP-сервер предоставляет инструменты: %s",
            [t["name"] for t in tools],
        )
        return tools
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: dict,
    ) -> ToolExecutionResult:
        """Вызвать инструмент на MCP-сервере через MCP-протокол."""
        if self._session is None:
            raise RuntimeError("MCP-сессия не установлена. Вызовите connect().")

        start = time.perf_counter()

        try:
            result = await self._session.call_tool(tool_name, arguments)
            elapsed = time.perf_counter() - start

            text_parts = []
            for content in result.content:
                if isinstance(content, types.TextContent):
                    text_parts.append(content.text)

            result_text = "\n".join(text_parts)

            logger.info(
                "MCP tool %s выполнен за %.3f сек", tool_name, elapsed
            )

            return ToolExecutionResult(
                tool_name=tool_name,
                arguments=arguments,
                result=result_text,
                execution_time=elapsed,
                success=not result.isError,
            )

        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error("Ошибка MCP tool %s: %s", tool_name, e)

            return ToolExecutionResult(
                tool_name=tool_name,
                arguments=arguments,
                result="",
                execution_time=elapsed,
                success=False,
                error=str(e),
            )

    async def execute_tool_call(
        self, tool_call: ToolCall
    ) -> ToolExecutionResult:
        """Выполнить ToolCall от LLM через MCP."""
        return await self.call_tool(tool_call.name, tool_call.arguments)

    async def execute_and_respond(
        self,
        client: OllamaClient,
        initial_response: LLMResponse,
        system_prompt: str | None = None,
        user_message: str = "",
    ) -> tuple[LLMResponse, list[ToolExecutionResult]]:
        """Выполнить tool calls через MCP и получить финальный ответ."""
        if not initial_response.tool_calls:
            return initial_response, []

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if user_message:
            messages.append({"role": "user", "content": user_message})

        messages.append(initial_response.raw.get("message", {}))

        tool_results = []
        for tool_call in initial_response.tool_calls:
            exec_result = await self.execute_tool_call(tool_call)
            tool_results.append(exec_result)

            if exec_result.success:
                messages.append({
                    "role": "tool",
                    "content": exec_result.result,
                })
            else:
                messages.append({
                    "role": "tool",
                    "content": json.dumps(
                        {"error": exec_result.error},
                        ensure_ascii=False,
                    ),
                })

        final_response = client.chat_with_tool_results(messages)
        return final_response, tool_results
    
