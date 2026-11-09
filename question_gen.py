# question_gen.py
"""
Генерация вопросов.

Два режима:
  * heuristic  — без ИИ, на шаблонах (вопрос-определение, вопрос-причина, вопрос-факт)
  * llm        — через OpenAI-совместимый API (опционально)
"""
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import List, Optional

from config import (LLM_API_KEY_ENV, LLM_BASE_URL, LLM_MODEL, N_OPTIONS,
                    STOPWORDS, USE_LLM)
from keywords import Keyword
from preprocess import Sentence


@dataclass
class Question:
    stem: str                       # формулировка вопроса
    options: List[str]              # 5 вариантов
    correct_index: int              # индекс правильного (0..4)
    source_sentence: str            # фрагмент-источник
    q_type: str                     # тип: definition / cause / fact / term
    topic: str = ""                 # ключевое слово-тема
    distractors: List[str] = field(default_factory=list)


# ---------------------- вспомогательные ----------------------

def _pick_sentence_with(keyword: Keyword, sentences: List[Sentence]) -> Optional[Sentence]:
    """Находим предложение, где встречается ключевое слово (по surface-форме)."""
    pattern = re.compile(rf"\b{re.escape(keyword.surface)}", re.IGNORECASE)
    candidates = [s for s in sentences if pattern.search(s.text)]
    if not candidates:
        return None
    # предпочитаем предложения средней длины — они информативнее
    candidates.sort(key=lambda s: abs(len(s.text.split()) - 18))
    return candidates[0]


def _strip_definition(lemma: str, sentence: str) -> str:
    """
    Пытаемся вытащить «определение» термина из предложения.
    Ищем паттерны: «X — это ...», «X называется ...», «X — ...»
    """
    patterns = [
        rf"{lemma}\s+[—-]\s+это\s+(.+)",
        rf"{lemma}\s+[—-]\s+(.+)",
        rf"{lemma}\s+называется\s+(.+)",
        rf"{lemma}\s+представляет\s+собой\s+(.+)",
    ]
    for p in patterns:
        m = re.search(p, sentence, re.IGNORECASE)
        if m:
            return m.group(1).strip(" .,;:")
    return ""


# ---------------------- эвристический генератор ----------------------

class HeuristicGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate(self, keywords: List[Keyword], sentences: List[Sentence],
                 n: int = 10) -> List[Question]:
        questions: List[Question] = []
        used_keywords = set()

        for kw in keywords:
            if len(questions) >= n:
                break
            if kw.lemma in used_keywords:
                continue
            sent = _pick_sentence_with(kw, sentences)
            if not sent:
                continue

            q = self._make_definition_question(kw, sent)
            if q is None:
                q = self._make_fact_question(kw, sent)
            if q is None:
                continue

            used_keywords.add(kw.lemma)
            questions.append(q)

        return questions

    def _make_definition_question(self, kw: Keyword, sent: Sentence) -> Optional[Question]:
        definition = _strip_definition(kw.surface, sent.text)
        if not definition or len(definition.split()) < 3:
            return None
        stem = f"Что такое «{kw.surface}» согласно тексту?"
        # правильный ответ — найденное определение
        correct = definition
        # дистракторы соберём позже в distractors.py
        return Question(
            stem=stem,
            options=[correct],          # заполним позже
            correct_index=0,
            source_sentence=sent.text,
            q_type="definition",
            topic=kw.lemma,
        )

    def _make_fact_question(self, kw: Keyword, sent: Sentence) -> Optional[Question]:
        """Вопрос «верно ли утверждение» -> превращаем в выбор из вариантов."""
        # Берём предложение, заменяем ключевое слово на пропуск
        if kw.surface.lower() not in sent.text.lower():
            return None
        blanked = re.sub(rf"\b{re.escape(kw.surface)}\b", "_____", sent.text,
                         count=1, flags=re.IGNORECASE)
        if "_____" not in blanked:
            return None
        stem = f"Заполните пропуск: {blanked}"
        return Question(
            stem=stem,
            options=[kw.surface],
            correct_index=0,
            source_sentence=sent.text,
            q_type="fact",
            topic=kw.lemma,
        )


# ---------------------- LLM-генератор (опционально) ----------------------

LLM_PROMPT = """Ты — методист, составляющий тестовые задания по учебному тексту.
Сгенерируй {n} вопросов с {k} вариантами ответа (1 правильный, остальные — правдоподобные дистракторы).
Язык — русский, стиль академический. Каждый вопрос должен опираться на конкретное предложение текста.

Верни строго JSON-массив объектов:
[
  {{
    "stem": "формулировка вопроса",
    "options": ["вар1", "вар2", ...],
    "correct_index": 0,
    "source_sentence": "точная цитата из текста",
    "q_type": "definition|fact|cause",
    "topic": "ключевое слово"
  }}
]

ТЕКСТ:
{text}
"""


class LLMGenerator:
    def __init__(self):
        if not USE_LLM:
            raise RuntimeError("USE_LLM=False, LLM-генератор недоступен")
        from openai import OpenAI
        api_key = os.getenv(LLM_API_KEY_ENV, "sk-no-key")
        kwargs = {"api_key": api_key}
        if LLM_BASE_URL:
            kwargs["base_url"] = LLM_BASE_URL
        self.client = OpenAI(**kwargs)

    def generate(self, text: str, n: int = 10) -> List[Question]:
        prompt = LLM_PROMPT.format(n=n, k=N_OPTIONS, text=text[:12000])
        resp = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        if isinstance(data, dict):
            data = data.get("questions", [])
        questions: List[Question] = []
        for item in data[:n]:
            opts = item["options"]
            if len(opts) < N_OPTIONS:
                continue
            questions.append(Question(
                stem=item["stem"],
                options=opts,
                correct_index=int(item.get("correct_index", 0)),
                source_sentence=item.get("source_sentence", ""),
                q_type=item.get("q_type", "fact"),
                topic=item.get("topic", ""),
            ))
        return questions