'''MCP-сервер для NLP-инструментов'''

import json
import logging

from mcp.server.fastmcp import FastMCP

from mcp_server.tools.lemmatizer import Lemmatizer
from mcp_server.tools.ner_extractor import NerExtractor
from mcp_server.tools.pos_tagger import PosTagger

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("nlp-tools")

_lemmatizer = Lemmatizer()
_ner_extractor = NerExtractor()
_pos_tagger = PosTagger()

@mcp.tool()
def lemmatize(text: str) -> str:
    '''Лемматизировать русский текст'''
    logger.info("lemmatize вызван, длина текста: %d", len(text))
    results = _lemmatizer.lemmatize(text)
    return json.dumps(
        [
            {
                "token": r.token,
                "lemma": r.lemma,
                "pos": r.pos,
                "score": r.score,
            }
            for r in results
        ],
        ensure_ascii=False,
        indent=2,
    )

@mcp.tool()
def extract_ner(text: str) -> str:
    """Извлечь именованные сущности из русского текста."""
    logger.info("extract_ner вызван, длина текста: %d", len(text))
    entities = _ner_extractor.extract_as_dicts(text)
    return json.dumps(entities, ensure_ascii=False, indent=2)

@mcp.tool()
def pos_tag(text: str) -> str:
    """Определить части речи для каждого токена в русском тексте"""
    logger.info("pos_tag вызван, длина текста: %d", len(text))
    tags = _pos_tagger.tag_as_dicts(text)
    return json.dumps(tags, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    logger.info("Запуск MCP-сервера nlp-tools...")
    mcp.run()
