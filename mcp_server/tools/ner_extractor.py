"""Инструмент извлечения именованных сущностей для MCP-сервера."""

from dataclasses import dataclass

from natasha import (
    Doc,
    MorphVocab,
    NamesExtractor,
    NewsEmbedding,
    NewsMorphTagger,
    NewsNERTagger,
    NewsSyntaxParser,
    Segmenter,
)

from mcp_server.utils.text_preprocessing import clean_text

NATASHA_LABEL_MAP = {
    "PER": "PER",
    "LOC": "LOC",
    "ORG": "ORG",
}


@dataclass
class NamedEntity:
    """Именованная сущность, извлечённая из текста.

    Attributes:
        text: Текст сущности как он встречается в тексте.
        label: Тип сущности (PER, LOC, ORG).
        start: Позиция начала сущности в тексте.
        stop: Позиция конца сущности в тексте.
        normal_form: Нормализованная форма сущности.
    """

    text: str
    label: str
    start: int
    stop: int
    normal_form: str | None = None


class NerExtractor:
    """Извлечение именованных сущностей через Natasha.

    Example:
        >>> extractor = NerExtractor()
        >>> entities = extractor.extract(
        ...     "Владимир Путин посетил Москву"
        ... )
        >>> [(e.text, e.label) for e in entities]
        [('Владимир Путин', 'PER'), ('Москву', 'LOC')]
    """

    def __init__(self) -> None:
        """Инициализировать все компоненты пайплайна Natasha."""
        self._segmenter = Segmenter()
        self._morph_vocab = MorphVocab()
        self._emb = NewsEmbedding()
        self._morph_tagger = NewsMorphTagger(self._emb)
        self._syntax_parser = NewsSyntaxParser(self._emb)
        self._ner_tagger = NewsNERTagger(self._emb)
        self._names_extractor = NamesExtractor(self._morph_vocab)

    def extract(self, text: str) -> list[NamedEntity]:
        """Извлечь именованные сущности из текста.

        Выполняет полный пайплайн: сегментация -> морфология ->
        синтаксис -> NER -> нормализация.

        Args:
            text: Входной текст на русском языке.

        Returns:
            Список найденных именованных сущностей.
        """
        text = clean_text(text)
        doc = Doc(text)

        doc.segment(self._segmenter)
        doc.tag_morph(self._morph_tagger)
        doc.parse_syntax(self._syntax_parser)
        doc.tag_ner(self._ner_tagger)

        for span in doc.spans:
            span.normalize(self._morph_vocab)

        entities = []
        for span in doc.spans:
            label = NATASHA_LABEL_MAP.get(span.type, span.type)

            entity = NamedEntity(
                text=span.text,
                label=label,
                start=span.start,
                stop=span.stop,
                normal_form=span.normal,
            )
            entities.append(entity)

        return entities

    def extract_as_dicts(self, text: str) -> list[dict]:
        """Извлечь сущности и вернуть как список словарей.

        Args:
            text: Входной текст на русском языке.

        Returns:
            Список словарей с полями text, label, start, stop.
        """
        return [
            {
                "text": entity.text,
                "label": entity.label,
                "start": entity.start,
                "stop": entity.stop,
                "normal_form": entity.normal_form,
            }
            for entity in self.extract(text)
        ]