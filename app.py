# app.py
"""
Streamlit-интерфейс для генератора тестовых вопросов.

Запуск:
    streamlit run app.py
"""
import io
import random
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from config import N_OPTIONS, N_QUESTIONS, USE_LLM
from distractors import enrich_question
from exporter import build_statistics, save_csv, save_docx, save_statistics
from keywords import extract_keywords
from loaders import LOADERS, load_text
from preprocess import filter_formula_sentences, split_sentences
from question_gen import HeuristicGenerator, LLMGenerator
from utils import count_words

# ---------------------- страница ----------------------
st.set_page_config(
    page_title="Генератор тестовых вопросов",
    page_icon="📝",
    layout="wide",
)

st.title("📝 Генератор тестовых вопросов")
st.caption("Загрузите конспект лекции или фрагмент учебника (TXT / DOCX / PDF, ≥1500 слов) "
           "и получите 10 вопросов с вариантами ответов и ссылками на источник.")

# ---------------------- сайдбар: настройки ----------------------
with st.sidebar:
    st.header("⚙️ Настройки")
    n_questions = st.slider("Количество вопросов", 5, 20, N_QUESTIONS)
    mode = st.radio(
        "Режим генерации",
        options=["heuristic", "llm"],
        index=0,
        help="heuristic — офлайн, на шаблонах; llm — через OpenAI-совместимый API",
        disabled=not USE_LLM if False else False,
    )
    if mode == "llm" and not USE_LLM:
        st.warning("`USE_LLM=False` в config.py — LLM-режим недоступен. "
                   "Переключитесь на heuristic или включите флаг.")
    seed = st.number_input("Seed (воспроизводимость)", value=42, step=1)

    st.divider()
    st.markdown("**Форматы вывода**")
    want_csv = st.checkbox("CSV", value=True)
    want_docx = st.checkbox("DOCX", value=True)

# ---------------------- загрузка файла ----------------------
uploaded = st.file_uploader(
    "Загрузите файл",
    type=["txt", "docx", "pdf"],
    accept_multiple_files=False,
)

if uploaded is None:
    st.info("⬆️ Загрузите файл, чтобы начать.")
    st.stop()

# показать метаданные файла
st.success(f"Файл загружен: **{uploaded.name}** ({uploaded.size / 1024:.1f} КБ)")

# сохраняем во временный файл, т.к. loaders работают с путями
suffix = Path(uploaded.name).suffix.lower()
with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.getbuffer())
    tmp_path = Path(tmp.name)

# ---------------------- кнопка генерации ----------------------
col_btn, col_meta = st.columns([1, 3])
with col_btn:
    generate = st.button("🚀 Сгенерировать", type="primary", use_container_width=True)

if not generate:
    # до нажатия показываем предпросмотр текста
    with st.expander("👀 Предпросмотр исходного текста"):
        try:
            preview = load_text(tmp_path)
            st.text(preview[:1500] + ("…" if len(preview) > 1500 else ""))
        except Exception as e:
            st.error(f"Не удалось прочитать файл: {e}")
    st.stop()

# ---------------------- пайплайн ----------------------
progress = st.progress(0, text="Старт…")
log = st.empty()
t0 = time.time()

try:
    # 1. Загрузка
    progress.progress(10, text="[1/5] Загрузка…")
    text = load_text(tmp_path)
    words = count_words(text)
    log.info(f"Слов в тексте: {words}")
    if words < 1500:
        st.warning(f"⚠️ Текст короче 1500 слов ({words}). Качество может быть низким.")

    # 2. Предобработка
    progress.progress(30, text="[2/5] Предобработка…")
    sentences = filter_formula_sentences(split_sentences(text))
    sent_texts = [s.text for s in sentences]
    if not sentences:
        st.error("Не удалось разбить текст на предложения.")
        st.stop()

    # 3. Ключевые слова
    progress.progress(50, text="[3/5] Извлечение ключевых слов…")
    keywords = extract_keywords(sent_texts, top_n=60)
    if not keywords:
        st.error("Не удалось извлечь ключевые слова — текст слишком короткий или специфичный.")
        st.stop()

    # 4. Генерация
    progress.progress(70, text="[4/5] Генерация вопросов…")
    if mode == "llm":
        gen = LLMGenerator()
        questions = gen.generate(text, n=n_questions)
    else:
        gen = HeuristicGenerator(seed=int(seed))
        questions = gen.generate(keywords, sentences, n=n_questions)

    rng = random.Random(int(seed))
    questions = [enrich_question(q, keywords, sentences, rng) for q in questions]

    if not questions:
        st.error("Вопросы не сгенерированы — попробуйте другой текст или режим.")
        st.stop()

    # 5. Сохранение (в память, для скачивания)
    progress.progress(90, text="[5/5] Подготовка файлов…")
    out_stem = Path(uploaded.name).stem

    csv_bytes = docx_bytes = None
    if want_csv:
        buf = io.BytesIO()
        tmp_csv = Path(tmp_path.parent) / f"{out_stem}_questions.csv"
        save_csv(questions, tmp_csv)
        csv_bytes = tmp_csv.read_bytes()
        tmp_csv.unlink(missing_ok=True)

    if want_docx:
        buf = io.BytesIO()
        tmp_docx = Path(tmp_path.parent) / f"{out_stem}_questions.docx"
        save_docx(questions, tmp_docx)
        docx_bytes = tmp_docx.read_bytes()
        tmp_docx.unlink(missing_ok=True)

    progress.progress(100, text="Готово!")
    elapsed = time.time() - t0
    log.success(f"✅ Сгенерировано вопросов: {len(questions)} за {elapsed:.2f} с")

except Exception as e:
    progress.empty()
    st.exception(e)
    st.stop()

# ---------------------- результаты ----------------------
st.divider()
st.subheader(f"📋 Сгенерированные вопросы ({len(questions)})")

for i, q in enumerate(questions, 1):
    with st.expander(f"Вопрос {i}. {q.stem}", expanded=(i == 1)):
        for j, opt in enumerate(q.options):
            if j == q.correct_index:
                st.markdown(f"✅ **{j + 1}) {opt}**  ← правильный")
            else:
                st.markdown(f"▫️ {j + 1}) {opt}")
        st.caption(f"Тип: `{q.q_type}` · Тема: `{q.topic}`")
        st.markdown(f"> **Источник:** {q.source_sentence}")

# ---------------------- статистика ----------------------
st.divider()
st.subheader("📊 Статистика")
for line in build_statistics(questions):
    st.markdown(f"- {line}")

# ---------------------- скачивание ----------------------
st.divider()
st.subheader("⬇️ Скачать результаты")
col1, col2 = st.columns(2)
with col1:
    if csv_bytes is not None:
        st.download_button(
            "📄 CSV",
            data=csv_bytes,
            file_name=f"{out_stem}_questions.csv",
            mime="text/csv",
            use_container_width=True,
        )
with col2:
    if docx_bytes is not None:
        st.download_button(
            "📝 DOCX",
            data=docx_bytes,
            file_name=f"{out_stem}_questions.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )

# ---------------------- исходный текст (на всякий случай) ----------------------
with st.expander("📖 Полный исходный текст"):
    st.text(text)

# очистка временного файла
try:
    tmp_path.unlink(missing_ok=True)
except Exception:
    pass