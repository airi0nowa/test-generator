# exporter.py
import csv
from collections import Counter
from pathlib import Path
from typing import List

from docx import Document
from docx.shared import Pt

from config import N_QUESTIONS
from question_gen import Question


def save_csv(questions: List[Question], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "№", "Вопрос", "Вариант 1", "Вариант 2", "Вариант 3",
            "Вариант 4", "Вариант 5", "Правильный вариант (№)",
            "Тип вопроса", "Тема", "Фрагмент-источник",
        ])
        for i, q in enumerate(questions, 1):
            opts = q.options + [""] * (5 - len(q.options))
            writer.writerow([
                i, q.stem, *opts[:5], q.correct_index + 1,
                q.q_type, q.topic, q.source_sentence,
            ])


def save_docx(questions: List[Question], path: Path) -> None:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)

    doc.add_heading("Тестовые задания", level=1)
    for i, q in enumerate(questions, 1):
        doc.add_paragraph(f"{i}. {q.stem}")
        for j, opt in enumerate(q.options, 1):
            marker = "✓" if (j - 1) == q.correct_index else " "
            doc.add_paragraph(f"   [{marker}] {j}) {opt}")
        doc.add_paragraph(f"   Тип: {q.q_type}; тема: {q.topic}")
        doc.add_paragraph(f"   Источник: «{q.source_sentence}»")
        doc.add_paragraph("")

    doc.add_heading("Статистика", level=1)
    for line in build_statistics(questions):
        doc.add_paragraph(line)

    doc.save(str(path))


def build_statistics(questions: List[Question]) -> List[str]:
    total = len(questions)
    by_type = Counter(q.q_type for q in questions)
    by_topic = Counter(q.topic for q in questions if q.topic)
    lines = [
        f"Всего вопросов: {total} (план: {N_QUESTIONS})",
        f"Распределение по типам: " + ", ".join(f"{k}={v}" for k, v in by_type.items()),
        f"Распределение по темам: " + ", ".join(f"{k}={v}" for k, v in by_topic.most_common()),
        f"Уникальных тем: {len(by_topic)}",
    ]
    return lines


def save_statistics(questions: List[Question], path: Path) -> None:
    path.write_text("\n".join(build_statistics(questions)), encoding="utf-8")