"""Инструмент лемматизации для MCP-сервера."""

from dataclasses import dataclass

import pymorphy3
from razdel import tokenize as razdel_tokenize

from mcp_server.utils.text_preprocessing import clean_text


@dataclass
class LemmaResult:
    """Результат лемматизации одного токена.

    Attributes:
        token: Исходное слово.
        lemma: Лемма (начальная форма).
        pos: Часть речи по pymorphy3.
        score: Уверенность морфологического разбора.
    """

    token: str
    lemma: str
    pos: str
    score: float


class Lemmatizer:
    """Лемматизатор на основе pymorphy3."""

    def __init__(self) -> None:
        self._morph = pymorphy3.MorphAnalyzer()

    def lemmatize(self, text: str) -> list[LemmaResult]:
        """Лемматизировать все токены в тексте."""

        text = clean_text(text)
        tokens = list(razdel_tokenize(text))
        results = []

        for token in tokens:
            word = token.text
            parsed = self._morph.parse(word)[0]

            result = LemmaResult(
                token=word,
                lemma=parsed.normal_form,
                pos=str(parsed.tag.POS) if parsed.tag.POS else "UNKNOWN",
                score=parsed.score,
            )
            results.append(result)

        return results

    def lemmatize_to_list(self, text: str) -> list[str]:
        """Получить только список лемм."""
        return [result.lemma for result in self.lemmatize(text)]