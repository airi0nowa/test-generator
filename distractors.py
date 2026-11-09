# distractors.py
"""
Генерация дистракторов для эвристических вопросов.

Стратегии:
  1) другие определения из текста (для типа definition);
  2) другие ключевые термины (для fact / term);
  3) лемматические «псевдо-термины» — изменённые формы (на случай нехватки материала).
"""
import random
from typing import List

from config import N_OPTIONS
from keywords import Keyword
from preprocess import Sentence
from question_gen import Question, _strip_definition


def _collect_definitions(keywords: List[Keyword], sentences: List[Sentence],
                         exclude_lemma: str) -> List[str]:
    out: List[str] = []
    for kw in keywords:
        if kw.lemma == exclude_lemma:
            continue
        for s in sentences:
            if kw.surface.lower() in s.text.lower():
                d = _strip_definition(kw.surface, s.text)
                if d and 3 <= len(d.split()) <= 25:
                    out.append(d)
                    break
    return out


def _collect_terms(keywords: List[Keyword], exclude_lemma: str) -> List[str]:
    return [kw.surface for kw in keywords if kw.lemma != exclude_lemma]


def enrich_question(q: Question, keywords: List[Keyword],
                    sentences: List[Sentence], rng: random.Random) -> Question:
    """Дополняет q.options до N_OPTIONS и перемешивает."""
    if len(q.options) >= N_OPTIONS:
        return q

    if q.q_type == "definition":
        pool = _collect_definitions(keywords, sentences, q.topic)
    else:
        pool = _collect_terms(keywords, q.topic)

    rng.shuffle(pool)
    # правильный ответ всегда остаётся первым элементом q.options
    correct = q.options[0]
    distractors: List[str] = []
    for cand in pool:
        if cand.lower() == correct.lower():
            continue
        if cand in distractors:
            continue
        distractors.append(cand)
        if len(distractors) >= N_OPTIONS - 1:
            break

    if len(distractors) < N_OPTIONS - 1:
        # fallback: добавляем «мусорные», но осмысленные варианты
        fallbacks = [
            "не имеет отношения к теме",
            "описано в другом разделе",
            "определяется иначе в исходном тексте",
            "в тексте не рассматривается",
        ]
        for f in fallbacks:
            if len(distractors) >= N_OPTIONS - 1:
                break
            distractors.append(f)

    q.distractors = distractors[: N_OPTIONS - 1]
    q.options = [correct] + q.distractors
    # перемешиваем, запоминая правильный
    indices = list(range(len(q.options)))
    rng.shuffle(indices)
    q.correct_index = indices.index(0)
    q.options = [q.options[i] for i in indices]
    return q