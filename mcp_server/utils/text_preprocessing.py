'''Утилиты для предобработки текста'''

import re
import unicodedata

def normalize_whitespace(text: str) -> str:
    '''
    Заменяет множественные проблемы, табуляции и переносы строк
    на одинарный пробел. Убирает пробелы по краям
    '''
    return re.sub(r"\s+", " ", text).strip()

def normalize_unicode(text: str) -> str:
    '''Привести Unicode к нормальной форме NFC.'''
    return unicodedata.normalize("NFC", text)

def clean_text(text: str) -> str:
    '''Полная очистка текста перед обработкой.'''
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    return text