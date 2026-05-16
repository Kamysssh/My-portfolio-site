"""
RAG-чатбот для FAQ, адаптация из репо pcef9 под Flask.

Функции:
- ensure_index()  — лениво строит/загружает FAISS-индекс
- generate_answer(message, top_k=3) — возвращает ответ и использованный контекст
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

import faiss  # type: ignore
import numpy as np
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FAQS_PATH = DATA_DIR / "faqs.json"
INDEX_PATH = DATA_DIR / "faiss_index.bin"
META_PATH = DATA_DIR / "faqs_metadata.npy"

_index = None
_faqs: List[Dict[str, str]] = []
_dim: int | None = None
_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Создаёт и кеширует OpenAI-клиент."""
    global _client
    if _client is None:
        # Ключ должен быть в переменной окружения OPENAI_API_KEY
        _client = OpenAI()
    return _client


def load_faqs() -> List[Dict[str, str]]:
    """Загружает FAQ из JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not FAQS_PATH.exists():
        # Минимальный набор FAQ по умолчанию
        default_faqs = [
            {
                "question": "Чем вы занимаетесь?",
                "answer": "Делаю AI-ассистентов и Telegram-ботов для бизнеса на основе документов компании, "
                "чтобы снять нагрузку с поддержки и руководителей.",
            },
            {
                "question": "Как с вами связаться?",
                "answer": "Оставьте заявку в форме на странице «Контакты».",
            },
            {
                "question": "Сколько занимает запуск?",
                "answer": "Типичный срок MVP — 2–4 недели, точнее после короткого обсуждения задачи.",
            },
            {
                "question": "Нужно ли заказчику разбираться в ИИ?",
                "answer": "Нет, важен понятный результат и экономия времени; технические детали на стороне исполнителя.",
            },
        ]
        with FAQS_PATH.open("w", encoding="utf-8") as f:
            json.dump(default_faqs, f, ensure_ascii=False, indent=2)

    with FAQS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_embeddings(texts: List[str]) -> np.ndarray:
    """Строит эмбеддинги через OpenAI `text-embedding-3-small`."""
    client = get_client()
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    vectors = np.array([d.embedding for d in resp.data], dtype="float32")
    return vectors


def build_index() -> Tuple[faiss.IndexFlatIP, List[Dict[str, str]]]:
    """Строит FAISS-индекс по FAQ и сохраняет его на диск."""
    global _dim

    faqs = load_faqs()
    texts = [f"Вопрос: {f['question']}\nОтвет: {f['answer']}" for f in faqs]

    vectors = _build_embeddings(texts)
    _dim = vectors.shape[1]

    # Используем cosine similarity через нормализацию и inner product
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(_dim)
    index.add(vectors)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    np.save(META_PATH, np.array(faqs, dtype=object))

    return index, faqs


def load_index() -> Tuple[faiss.IndexFlatIP, List[Dict[str, str]]]:
    """Загружает индекс, если он уже построен."""
    global _dim

    if not INDEX_PATH.exists() or not META_PATH.exists():
        return build_index()

    index = faiss.read_index(str(INDEX_PATH))
    _dim = index.d
    faqs_arr = np.load(META_PATH, allow_pickle=True)
    faqs = list(faqs_arr.tolist())
    return index, faqs


def ensure_index() -> None:
    """Ленивая инициализация индекса и FAQ."""
    global _index, _faqs
    if _index is None or not _faqs:
        _index, _faqs = load_index()


def retrieve_similar(message: str, top_k: int = 3) -> List[Dict[str, str]]:
    """Ищет наиболее похожие FAQ по запросу пользователя."""
    ensure_index()
    assert _index is not None

    client = get_client()
    # Эмбеддинг запроса
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[message],
    )
    query_vec = np.array([resp.data[0].embedding], dtype="float32")
    faiss.normalize_L2(query_vec)

    scores, indices = _index.search(query_vec, top_k)
    idxs = indices[0]

    results: List[Dict[str, str]] = []
    for i, score in zip(idxs, scores[0]):
        if i < 0:
            continue
        faq = _faqs[i]
        results.append(
            {
                "question": faq.get("question", ""),
                "answer": faq.get("answer", ""),
                "score": float(score),
            }
        )
    return results


def generate_answer(message: str, top_k: int = 3) -> Dict:
    """
    Основная функция для Flask-роута /chat.
    Возвращает словарь: {answer: str, context: List[FAQ]}.
    """
    ensure_index()
    related = retrieve_similar(message, top_k=top_k)

    context_blocks = [
        f"Вопрос: {item['question']}\nОтвет: {item['answer']}"
        for item in related
    ]
    context_text = "\n\n".join(context_blocks) if context_blocks else "Нет контекста."

    system_prompt = (
        "Ты FAQ-ассистент сайта фрилансера в нише AI-ассистентов и Telegram-ботов для бизнеса "
        "(ответы на основе документов заказчика, меньше рутины для поддержки и сотрудников). "
        "Пиши кратко, по-русски, без технического жаргона. Если в контексте нет ответа — честно скажи "
        "и предложи оставить заявку в форме «Контакты»."
    )

    user_prompt = (
        f"Контекст (FAQ):\n{context_text}\n\n"
        f"Вопрос пользователя:\n{message}\n\n"
        "Сформируй понятный ответ на основе контекста. Если информации недостаточно, "
        "скажи об этом и предложи оставить заявку через форму на сайте."
    )

    client = get_client()
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=400,
    )
    answer = completion.choices[0].message.content.strip()

    return {
        "answer": answer,
        "context": related,
    }


