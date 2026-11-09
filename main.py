# main.py
"""
Точка входа.

Примеры:
    python main.py --input lecture.txt --format csv
    python main.py --input lecture.docx --format docx --outdir ./result
    python main.py --input lecture.pdf --mode llm
"""
import argparse
import random
import sys
import time
from pathlib import Path

from config import N_QUESTIONS, OUTPUT_DIR, USE_LLM
from distractors import enrich_question
from exporter import save_csv, save_docx, save_statistics
from keywords import extract_keywords
from loaders import load_text
from preprocess import filter_formula_sentences, split_sentences
from question_gen import HeuristicGenerator, LLMGenerator
from utils import count_words


def parse_args():
    p = argparse.ArgumentParser(description="Генератор тестовых вопросов по тексту")
    p.add_argument("--input", required=True, help="Путь к TXT/DOCX/PDF")
    p.add_argument("--format", choices=["csv", "docx"], default="csv")
    p.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic")
    p.add_argument("--n", type=int, default=N_QUESTIONS)
    p.add_argument("--outdir", default=str(OUTPUT_DIR))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    # 1. Загрузка
    print(f"[1/5] Загрузка: {args.input}")
    text = load_text(args.input)
    words = count_words(text)
    print(f"      Слов: {words}")
    if words < 1500:
        print("      ⚠ Текст короче 1500 слов — качество может быть низким")

    # 2. Предобработка
    print("[2/5] Предобработка…")
    sentences = filter_formula_sentences(split_sentences(text))
    sent_texts = [s.text for s in sentences]
    print(f"      Предложений-кандидатов: {len(sentences)}")

    # 3. Ключевые слова
    print("[3/5] Извлечение ключевых слов…")
    keywords = extract_keywords(sent_texts, top_n=60)
    print(f"      Ключевых слов: {len(keywords)}")
    if not keywords:
        print("      ✗ Не удалось извлечь ключевые слова — завершение")
        sys.exit(1)

    # 4. Генерация вопросов
    print(f"[4/5] Генерация вопросов (режим: {args.mode})…")
    if args.mode == "llm":
        if not USE_LLM:
            print("      ✗ USE_LLM=False в config.py — переключитесь на heuristic")
            sys.exit(1)
        gen = LLMGenerator()
        questions = gen.generate(text, n=args.n)
    else:
        gen = HeuristicGenerator(seed=args.seed)
        questions = gen.generate(keywords, sentences, n=args.n)

    # дополняем дистракторами (для эвристики; LLM уже вернул полный набор)
    rng = random.Random(args.seed)
    questions = [enrich_question(q, keywords, sentences, rng) for q in questions]

    # 5. Сохранение
    print("[5/5] Сохранение результатов…")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem

    if args.format == "csv":
        out_file = outdir / f"{stem}_questions.csv"
        save_csv(questions, out_file)
    else:
        out_file = outdir / f"{stem}_questions.docx"
        save_docx(questions, out_file)

    stats_file = outdir / f"{stem}_stats.txt"
    save_statistics(questions, stats_file)

    print(f"      Вопросов сгенерировано: {len(questions)}")
    print(f"      Файл: {out_file}")
    print(f"      Статистика: {stats_file}")
    print(f"      Время: {time.time() - t0:.2f} с")


if __name__ == "__main__":
    main()