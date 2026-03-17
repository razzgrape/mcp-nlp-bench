'''Пайплайн экспериментов'''

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from config.settings import settings
from data.loader import DataLoader
from llm_client.ollama_client import OllamaClient
from llm_client.mcp_bridge import McpBridge
from llm_client.prompts import get_system_prompt, make_user_message

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

@dataclass
class ExperimentResult:
    '''Результат одного прогона(один текст, один режим)'''
    task: str
    mode: str
    text: str
    expected: list[dict]
    predicted: list[dict] = field(default_factory=list)
    raw_response: str = ""
    llm_time: float = 0.0
    tool_time: float = 0.0
    success: bool = True
    error: str | None = None

def parse_llm_json(text: str) -> list[dict]:
    '''Попытаться распарсить JSON из ответа LLM'''
    text = text.strip()

    if not text:
        return []

    if "```" in text:
        lines = text.split("\n")
        cleaned = []
        inside_block = False
        for line in lines:
            if line.strip().startswith("```"):
                inside_block = not inside_block
                continue
            if inside_block:
                cleaned.append(line)
        text = "\n".join(cleaned).strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start : end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return []

class ExperimentRunner:
    '''Оркестратор экспериментов'''
    def __init__(
        self,
        model: str | None = None,
        python_command: str = "python3",
    ) -> None:
        self._model = model
        self._python_command = python_command
        self._loader = DataLoader()

    async def run(
        self,
        task: str,
        n: int | None = None,
        modes: list[str] | None = None,
    ) -> list[ExperimentResult]:
        """Запустить эксперимент.

        Args:
            task: Задача — "pos" или "lemma".
            n: Количество примеров.
            modes: Режимы — ["llm_only", "tool_assisted"] по умолчанию.
        """
        n = n or settings.max_samples
        modes = modes or ["llm_only", "tool_assisted"]

        if task == "pos":
            samples = self._loader.load_pos_samples(n=n)
        elif task == "lemma":
            samples = self._loader.load_lemma_samples(n=n)
        else:
            raise ValueError(f"Неизвестная задача: {task}")

        logger.info(
            "Запуск эксперимента: task=%s, n=%d, modes=%s",
            task, len(samples), modes,
        )

        results = []

        if "llm_only" in modes:
            logger.info("--- Режим: LLM-only ---")
            llm_only_results = self._run_llm_only(task, samples)
            results.extend(llm_only_results)

        if "tool_assisted" in modes:
            logger.info("--- Режим: LLM+MCP (tool_assisted) ---")
            tool_results = await self._run_tool_assisted(task, samples)
            results.extend(tool_results)

        self._save_results(results, task)

        return results

    def _run_llm_only(
        self,
        task: str,
        samples: list,
    ) -> list[ExperimentResult]:
        """Прогнать все примеры в режиме LLM-only"""
        client_kwargs = {}
        if self._model:
            client_kwargs["model"] = self._model

        client = OllamaClient(**client_kwargs)
        system_prompt = get_system_prompt("llm_only", task)

        results = []

        for i, sample in enumerate(samples):
            user_msg = make_user_message(task, sample.text)

            logger.info(
                "[LLM-only] %d/%d: %s...",
                i + 1, len(samples), sample.text[:50],
            )

            start = time.perf_counter()
            try:
                response = client.chat(user_msg, system_prompt)
                llm_time = time.perf_counter() - start

                predicted = parse_llm_json(response.content)

                results.append(ExperimentResult(
                    task=task,
                    mode="llm_only",
                    text=sample.text,
                    expected=sample.tokens,
                    predicted=predicted,
                    raw_response=response.content,
                    llm_time=llm_time,
                    success=len(predicted) > 0,
                    error=None if predicted else "Не удалось распарсить JSON",
                ))

            except Exception as e:
                llm_time = time.perf_counter() - start
                logger.error("[LLM-only] Ошибка: %s", e)

                results.append(ExperimentResult(
                    task=task,
                    mode="llm_only",
                    text=sample.text,
                    expected=sample.tokens,
                    llm_time=llm_time,
                    success=False,
                    error=str(e),
                ))

        client.close()
        return results

    async def _run_tool_assisted(
        self,
        task: str,
        samples: list,
    ) -> list[ExperimentResult]:
        """Прогнать все примеры в режиме LLM+MCP"""
        client_kwargs = {}
        if self._model:
            client_kwargs["model"] = self._model

        client = OllamaClient(**client_kwargs)
        system_prompt = get_system_prompt("tool_assisted", task)

        results = []

        async with McpBridge(python_command=self._python_command) as bridge:
            tools = await bridge.list_tools()
            logger.info("MCP инструменты: %s", [t["name"] for t in tools])

            for i, sample in enumerate(samples):
                user_msg = make_user_message(task, sample.text)

                logger.info(
                    "[LLM+MCP] %d/%d: %s...",
                    i + 1, len(samples), sample.text[:50],
                )

                start = time.perf_counter()
                try:
                    response = client.chat_with_tools(user_msg, system_prompt)
                    llm_time = time.perf_counter() - start

                    if response.tool_calls:
                        final, tool_results = await bridge.execute_and_respond(
                            client, response, system_prompt, user_msg,
                        )

                        tool_time = sum(
                            tr.execution_time for tr in tool_results
                        )

                        tool_output = (
                            tool_results[0].result if tool_results else ""
                        )
                        predicted = parse_llm_json(tool_output)

                        results.append(ExperimentResult(
                            task=task,
                            mode="tool_assisted",
                            text=sample.text,
                            expected=sample.tokens,
                            predicted=predicted,
                            raw_response=final.content,
                            llm_time=llm_time,
                            tool_time=tool_time,
                            success=len(predicted) > 0,
                        ))
                    else:
                        predicted = parse_llm_json(response.content)

                        results.append(ExperimentResult(
                            task=task,
                            mode="tool_assisted",
                            text=sample.text,
                            expected=sample.tokens,
                            predicted=predicted,
                            raw_response=response.content,
                            llm_time=llm_time,
                            success=False,
                            error="Модель не вызвала инструмент",
                        ))

                except Exception as e:
                    llm_time = time.perf_counter() - start
                    logger.error("[LLM+MCP] Ошибка: %s", e)

                    results.append(ExperimentResult(
                        task=task,
                        mode="tool_assisted",
                        text=sample.text,
                        expected=sample.tokens,
                        llm_time=llm_time,
                        success=False,
                        error=str(e),
                    ))

        client.close()
        return results

    def _save_results(
        self,
        results: list[ExperimentResult],
        task: str,
    ) -> None:
        """Сохранить сырые результаты в JSON"""
        settings.ensure_dirs()

        output_path = settings.raw_results_dir / f"{task}_results.json"

        data = [asdict(r) for r in results]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("Результаты сохранены в %s", output_path)



async def main():
    parser = argparse.ArgumentParser(
        description="Запуск экспериментов LLM-only vs LLM+MCP"
    )
    parser.add_argument(
        "--task",
        choices=["pos", "lemma"],
        required=True,
        help="Задача: pos или lemma",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Количество примеров (по умолчанию 10)",
    )
    parser.add_argument(
        "--mode",
        choices=["llm_only", "tool_assisted", "both"],
        default="both",
        help="Режим: llm_only, tool_assisted или both",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Модель Ollama (по умолчанию из конфига)",
    )

    args = parser.parse_args()

    modes = (
        ["llm_only", "tool_assisted"]
        if args.mode == "both"
        else [args.mode]
    )

    runner = ExperimentRunner(model=args.model)
    results = await runner.run(task=args.task, n=args.n, modes=modes)

    for mode in modes:
        mode_results = [r for r in results if r.mode == mode]
        success = sum(1 for r in mode_results if r.success)
        total = len(mode_results)
        avg_llm = (
            sum(r.llm_time for r in mode_results) / total
            if total
            else 0
        )

        print(f"\n{'='*50}")
        print(f"Режим: {mode}")
        print(f"Успешных: {success}/{total}")
        print(f"Среднее время LLM: {avg_llm:.2f} сек")

        if mode == "tool_assisted":
            avg_tool = (
                sum(r.tool_time for r in mode_results) / total
                if total
                else 0
            )
            print(f"Среднее время MCP: {avg_tool:.3f} сек")


if __name__ == "__main__":
    asyncio.run(main())

