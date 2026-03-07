'''Клиент для взаимодействия с Ollama LLM'''

import json
import logging
from dataclasses import dataclass, field

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class ToolCall:
    '''Вызов инструмента, запрошенный моделью'''
    name: str
    arguments: dict

@dataclass
class LLMResponse:
    '''Ответ от LLM'''
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

NLP_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "lemmatize",
            "description": (
                "Лемматизировать русский текст. Возвращает JSON-список "
                "с полями: token, lemma, pos, score для каждого слова."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Текст на русском языке для лемматизации.",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_ner",
            "description": (
                "Извлечь именованные сущности (PER, LOC, ORG) из русского "
                "текста. Возвращает JSON-список с полями: text, label, "
                "start, stop, normal_form."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Текст на русском языке.",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pos_tag",
            "description": (
                "Определить части речи для каждого токена в русском тексте. "
                "Возвращает JSON-список с полями: token, pos, feats, lemma."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Текст на русском языке.",
                    }
                },
                "required": ["text"],
            },
        },
    },
]

class OllamaClient:
    '''Клиент для Ollama API'''
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
    ) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.temperature = temperature if temperature is not None else settings.ollama_temperature
        self.timeout = timeout or settings.ollama_timeout
        self._client = httpx.Client(timeout=self.timeout)

    def chat(
        self,
        user_message: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Отправить запрос модели БЕЗ инструментов (режим LLM-only)"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        return self._send_request(payload)
    
    def chat_with_tools(
        self,
        user_message: str,
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Отправить запрос модели С инструментами (режим LLM+MCP)."""
        if tools is None:
            tools = NLP_TOOLS_SCHEMA

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        return self._send_request(payload)
    
    def chat_with_tool_results(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Продолжить диалог после выполнения инструмента"""
        if tools is None:
            tools = NLP_TOOLS_SCHEMA

        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        return self._send_request(payload)
    

    def _send_request(self, payload: dict) -> LLMResponse:
        """Отправить запрос к Ollama API и разобрать ответ"""
        url = f"{self.base_url}/api/chat"

        logger.debug("Запрос к %s: model=%s", url, self.model)

        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP ошибка от Ollama: %s", e)
            raise
        except httpx.ConnectError as e:
            logger.error(
                "Не удалось подключиться к Ollama по адресу %s. "
                "Убедитесь, что Ollama запущена.",
                self.base_url,
            )
            raise

        data = response.json()
        return self._parse_response(data)

    @staticmethod
    def _parse_response(data: dict) -> LLMResponse:
        """Разобрать сырой ответ Ollama в LLMResponse."""
        message = data.get("message", {})
        content = message.get("content", "")

        tool_calls = []
        raw_tool_calls = message.get("tool_calls", [])
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            tool_calls.append(
                ToolCall(
                    name=func.get("name", ""),
                    arguments=func.get("arguments", {}),
                )
            )

        if not tool_calls and content.strip():
            parsed = OllamaClient._try_parse_tool_calls_from_text(content)
            if parsed:
                tool_calls = parsed
                content = ""  

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            raw=data,
        )

    @staticmethod
    def _try_parse_tool_calls_from_text(text: str) -> list[ToolCall]:
        """Попытаться извлечь tool calls из текстового ответа"""
        import json

        text = text.strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and parsed and "name" in parsed[0]:
                return [
                    ToolCall(name=item["name"], arguments=item.get("arguments", {}))
                    for item in parsed
                    if isinstance(item, dict) and "name" in item
                ]
            if isinstance(parsed, dict) and "name" in parsed:
                return [ToolCall(name=parsed["name"], arguments=parsed.get("arguments", {}))]
        except json.JSONDecodeError:
            pass

        start = text.find("[")
        while start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "[":
                    depth += 1
                elif text[i] == "]":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            parsed = json.loads(candidate)
                            if (
                                isinstance(parsed, list)
                                and parsed
                                and isinstance(parsed[0], dict)
                                and "name" in parsed[0]
                            ):
                                return [
                                    ToolCall(
                                        name=item["name"],
                                        arguments=item.get("arguments", {}),
                                    )
                                    for item in parsed
                                    if isinstance(item, dict) and "name" in item
                                ]
                        except json.JSONDecodeError:
                            pass
                        break
            start = text.find("[", start + 1)

        return []
    
    def close(self) -> None:
        """Закрыть HTTP-клиент."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()