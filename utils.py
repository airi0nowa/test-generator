# utils.py
import re

def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def clean_token(tok: str) -> str:
    return re.sub(r"^[^\w\-]+|[^\w\-]+$", "", tok, flags=re.UNICODE)

def count_words(text: str) -> int:
    return len(re.findall(r"[А-Яа-яЁёA-Za-z\-]+", text))

def shorten(text: str, n: int = 200) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"