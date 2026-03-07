"""Загрузчик датасета SynTagRus"""

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

from conllu import parse

from config.settings import settings

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = settings.project_root / "data" / "syntagrus"


@dataclass
class PosSample:
    """Один пример для задачи POS-теггинга."""

    text: str
    tokens: list[dict] = field(default_factory=list)


@dataclass
class LemmaSample:
    """Один пример для задачи лемматизации."""
    text: str
    tokens: list[dict] = field(default_factory=list)


class DataLoader:
    """
    Загрузчик датасета SynTagRus.
    Формирует выборки для POS и лемматизации
    """

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self._data_dir = Path(data_dir or DEFAULT_DATA_DIR)
        self._sentences = None

    def _load_conllu(self, split: str = "test") -> list:
        """Загрузить и распарсить CoNLL-U файл.

        Args:
            split: Какой файл загрузить — "test", "dev" или "train".
        """
        if self._sentences is not None:
            return self._sentences

        filepath = self._data_dir / f"{split}.conllu"

        if not filepath.exists():
            raise FileNotFoundError(
                f"Файл {filepath} не найден.")

        logger.info("Загрузка SynTagRus из %s ...", filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()

        self._sentences = parse(raw)
        logger.info("Загружено %d предложений", len(self._sentences))
        return self._sentences

    def load_pos_samples(
        self,
        n: int | None = None,
        seed: int | None = None,
        split: str = "test",
        min_tokens: int = 3,
    ) -> list[PosSample]:
        """
        Загрузить примеры для задачи POS-теггинга.
        """
        n = n or settings.max_samples
        seed = seed if seed is not None else settings.random_seed

        sentences = self._load_conllu(split)

        candidates = []
        for sent in sentences:
            tokens = self._extract_pos(sent)
            text = self._get_text(sent)
            if len(tokens) >= min_tokens and text:
                candidates.append(PosSample(text=text, tokens=tokens))

        logger.info("Подходящих предложений для POS: %d", len(candidates))

        random.seed(seed)
        if len(candidates) > n:
            candidates = random.sample(candidates, n)

        logger.info("POS выборка: %d примеров", len(candidates))
        return candidates

    def load_lemma_samples(
        self,
        n: int | None = None,
        seed: int | None = None,
        split: str = "test",
        min_tokens: int = 3,
    ) -> list[LemmaSample]:
        """
        Загрузить примеры для задачи лемматизации.
        """
        n = n or settings.max_samples
        seed = seed if seed is not None else settings.random_seed

        sentences = self._load_conllu(split)

        candidates = []
        for sent in sentences:
            tokens = self._extract_lemmas(sent)
            text = self._get_text(sent)
            if len(tokens) >= min_tokens and text:
                candidates.append(LemmaSample(text=text, tokens=tokens))

        logger.info("Подходящих предложений для Lemma: %d", len(candidates))

        random.seed(seed)
        if len(candidates) > n:
            candidates = random.sample(candidates, n)

        logger.info("Lemma выборка: %d примеров", len(candidates))
        return candidates

    @staticmethod
    def _get_text(sent) -> str:
        """Получить текст предложения из метаданных conllu."""
        if hasattr(sent, "metadata") and "text" in sent.metadata:
            return sent.metadata["text"]

        return " ".join(
            token["form"]
            for token in sent
            if isinstance(token["id"], int)
        )

    @staticmethod
    def _extract_pos(sent) -> list[dict]:
        """Извлечь POS-разметку из предложения conllu"""
        tokens = []
        for token in sent:
            if not isinstance(token["id"], int):
                continue

            form = token["form"]
            pos = token["upos"]

            if form and pos:
                tokens.append({"token": form, "pos": pos})

        return tokens

    @staticmethod
    def _extract_lemmas(sent) -> list[dict]:
        """Извлечь леммы из предложения conllu"""
        tokens = []
        for token in sent:
            if not isinstance(token["id"], int):
                continue

            form = token["form"]
            lemma = token["lemma"]

            if form and lemma:
                tokens.append({"token": form, "lemma": lemma})

        return tokens