# keywords.py
from collections import Counter
from dataclasses import dataclass
from typing import List
import math
import re

# --- sklearn ---
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _HAS_SKLEARN = True
except Exception:
    TfidfVectorizer = None
    _HAS_SKLEARN = False

# --- pymorphy3 ---
try:
    import pymorphy3
    _morph = pymorphy3.MorphAnalyzer()
    _HAS_MORPH = True
except Exception:
    _morph = None
    _HAS_MORPH = False

# --- razdel.tokenize ---
try:
    from razdel import tokenize as _razdel_tokenize
    _HAS_TOKENIZE = True
except Exception:
    _razdel_tokenize = None
    _HAS_TOKENIZE = False


def _tokenize(text: str):
    """Универсальная токенизация: razdel либо regex."""
    if _HAS_TOKENIZE:
        try:
            return [t.text for t in _razdel_tokenize(text)]
        except Exception:
            pass
    return re.findall(r"[А-Яа-яЁёA-Za-z\-]+", text)


@dataclass
class Keyword:
    lemma: str
    surface: str
    tfidf: float
    freq: int


def lemmatize(token: str) -> str:
    if not _HAS_MORPH:
        return token
    try:
        return _morph.parse(token)[0].normal_form
    except Exception:
        return token


def extract_keywords(sentences: List[str], top_n: int = 40) -> List[Keyword]:
    if not sentences:
        return []

    # 1. TF-IDF (если sklearn есть)
    tfidf_by_term = {}
    if _HAS_SKLEARN:
        try:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                token_pattern=r"(?u)\b[А-Яа-яЁё\-]{4,}\b",
                stop_words=list(STOPWORDS),
                max_features=2000,
            )
            matrix = vectorizer.fit_transform(sentences)
            scores = matrix.sum(axis=0).A1
            terms = vectorizer.get_feature_names_out()
            tfidf_by_term = dict(zip(terms, scores))
        except ValueError:
            pass

    # 2. Частоты + лемматизация
    freq = Counter()
    surface_by_lemma = {}
    for sent in sentences:
        for w in _tokenize(sent):
            w = w.lower()
            if w in STOPWORDS or len(w) < 4 or not w.replace("-", "").isalpha():
                continue
            lemma = lemmatize(w)
            freq[lemma] += 1
            surface_by_lemma.setdefault(lemma, w)

    # 3. Объединение
    result: List[Keyword] = []
    for lemma, f in freq.most_common(200):
        if f < 2:
            continue
        score = tfidf_by_term.get(surface_by_lemma.get(lemma, lemma), 0.0)
        result.append(Keyword(
            lemma=lemma,
            surface=surface_by_lemma[lemma],
            tfidf=float(score),
            freq=f,
        ))

    result.sort(key=lambda k: k.tfidf * math.log(k.freq + 1), reverse=True)
    return result[:top_n]
