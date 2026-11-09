# preprocess.py
import re
from dataclasses import dataclass
from typing import List

from utils import normalize_whitespace

# --- мягкий импорт razdel ---
try:
    from razdel import sentenize as _razdel_sentenize
    _HAS_RAZDEL = True
except Exception:                # ModuleNotFoundError, ImportError и пр.
    _razdel_sentenize = None
    _HAS_RAZDEL = False


# ВАЖНО: класс определён ДО любых вызовов, которые могут упасть
@dataclass
class Sentence:
    idx: int
    text: str
    paragraph_id: int


def _fallback_sentenize(text: str) -> List[str]:
    """Простое разбиение по . ! ? — если razdel недоступен."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _sentenize(text: str) -> List[str]:
    """Единая точка: razdel либо fallback."""
    if _HAS_RAZDEL:
        try:
            return [s.text for s in _razdel_sentenize(text)]
        except Exception:
            pass
    return _fallback_sentenize(text)


def split_paragraphs(text: str) -> List[str]:
    raw = re.split(r"\n{2,}", text)
    return [normalize_whitespace(p) for p in raw if p.strip()]


def split_sentences(text: str) -> List[Sentence]:
    sentences: List[Sentence] = []
    idx = 0
    for p_id, para in enumerate(split_paragraphs(text)):
        for s_text in _sentenize(para):
            s_text = normalize_whitespace(s_text)
            if len(s_text.split()) < 4:
                continue
            sentences.append(Sentence(idx=idx, text=s_text, paragraph_id=p_id))
            idx += 1
    return sentences


def filter_formula_sentences(sentences: List[Sentence]) -> List[Sentence]:
    bad = re.compile(r"[=<>≤≥∑∫√±×÷]|\b(рис|табл|формул)\w*\.?\s*\d")
    return [s for s in sentences if not bad.search(s.text)]
