"""Инструмент POS-теггинга для MCP-сервера."""

from dataclasses import dataclass

from natasha import (
    Doc,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Segmenter,
)

from mcp_server.utils.text_preprocessing import clean_text


@dataclass
class PosTagResult:
    """Результат POS-теггинга для одного токена.

    Attributes:
        token: Исходное слово.
        pos: Часть речи (Universal Dependencies tagset).
        feats: Грамматические признаки.
        lemma: Лемма слова.
    """

    token: str
    pos: str
    feats: dict[str, str]
    lemma: str


class PosTagger:
    """POS-теггер."""

    def __init__(self) -> None:
        self._segmenter = Segmenter()
        self._morph_vocab = MorphVocab()
        self._emb = NewsEmbedding()
        self._morph_tagger = NewsMorphTagger(self._emb)

    def tag(self, text: str) -> list[PosTagResult]:
        """Определить части речи для всех токенов в тексте.

        Args:
            text: Входной текст на русском языке.

        Returns:
            Список результатов POS-теггинга.
        """
        text = clean_text(text)
        doc = Doc(text)

        doc.segment(self._segmenter)
        doc.tag_morph(self._morph_tagger)

        for token in doc.tokens:
            token.lemmatize(self._morph_vocab)

        results = []
        for token in doc.tokens:
            feats = self._parse_feats(token.feats)
            result = PosTagResult(
                token=token.text,
                pos=token.pos,
                feats=feats,
                lemma=token.lemma,
            )
            results.append(result)

        return results

    def tag_as_dicts(self, text: str) -> list[dict]:
        """Определить части речи и вернуть как список словарей.

        Args:
            text: Входной текст на русском языке.

        Returns:
            Список словарей с полями token, pos, feats, lemma.
        """
        return [
            {
                "token": result.token,
                "pos": result.pos,
                "feats": result.feats,
                "lemma": result.lemma,
            }
            for result in self.tag(text)
        ]

    @staticmethod
    def _parse_feats(feats_str: str | None) -> dict[str, str]:
        """Разобрать строку грамматических признаков.

        Args:
            feats_str: Строка вида "Case=Nom|Number=Sing".

        Returns:
            Словарь признаков.
        """
        if not feats_str:
            return {}

        result = {}
        for pair in feats_str.split("|"):
            if "=" in pair:
                key, value = pair.split("=", maxsplit=1)
                result[key] = value

        return result