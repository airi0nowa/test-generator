# keywords.py
from collections import Counter
from dataclasses import dataclass
from typing import List

import pymorphy3
from razdel import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from config import STOPWORDS

_morph = pymorphy3.MorphAnalyzer()


@dataclass
class Keyword:
    lemma: str          # нормальная форма
    surface: str        # как встречается в тексте
    tfidf: float
    freq: int


def lemmatize(token: str) -> str:
    return _morph.parse(token)[0].normal_form


def extract_keywords(sentences: List[str], top_n: int = 40) -> List[Keyword]:
    """TF-IDF по «документам» = предложениям + частотный фильтр."""
    if not sentences:
        return []

    # 1. TF-IDF
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[А-Яа-яЁё\-]{4,}\b",
        stop_words=list(STOPWORDS),
        max_features=2000,
    )
    try:
        matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return []
    scores = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()

    # 2. Частоты + лемматизация
    freq = Counter()
    surface_by_lemma = {}
    for sent in sentences:
        for tok in tokenize(sent):
            w = tok.text.lower()
            if w in STOPWORDS or len(w) < 4 or not w.replace("-", "").isalpha():
                continue
            lemma = lemmatize(w)
            freq[lemma] += 1
            surface_by_lemma.setdefault(lemma, w)

    # 3. Объединение
    tfidf_by_term = {t: s for t, s in zip(terms, scores)}
    result: List[Keyword] = []
    for lemma, f in freq.most_common(200):
        if f < 2:
            continue
        # tfidf может не найти лемму — берём по surface
        score = tfidf_by_term.get(surface_by_lemma.get(lemma, lemma), 0.0)
        result.append(Keyword(lemma=lemma, surface=surface_by_lemma[lemma],
                              tfidf=float(score), freq=f))

    # финальный ранжир: tfidf * log(freq+1)
    import math
    result.sort(key=lambda k: k.tfidf * math.log(k.freq + 1), reverse=True)
    return result[:top_n]