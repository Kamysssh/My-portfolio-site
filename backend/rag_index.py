"""
RAG-чатбот для FAQ: FAISS + LLM через ProxyAPI, интеграция с Flask.

- Загрузка FAQ из JSON и текстовых документов (.txt)
- Поиск релевантных фрагментов и генерация ответа
- Память диалога (история последних сообщений)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss  # type: ignore
import numpy as np
from dotenv import load_dotenv

from backend.llm_client import get_client

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FAQS_PATH = DATA_DIR / "faqs.json"
INDEX_PATH = DATA_DIR / "faiss_index.bin"
META_PATH = DATA_DIR / "faqs_metadata.npy"

EMBEDDING_MODEL = (
    os.getenv("AI_EMBEDDING_MODEL")
    or os.getenv("OPENAI_EMBEDDING_MODEL")
    or "text-embedding-3-small"
)
CHAT_MODEL = (
    os.getenv("AI_CHAT_MODEL")
    or os.getenv("OPENAI_CHAT_MODEL")
    or "gpt-4o-mini"
)
MAX_HISTORY_MESSAGES = int(os.getenv("CHAT_MAX_HISTORY", "10"))

_index: Optional[faiss.Index] = None
_items: List[Dict[str, str]] = []


def save_faiss_index(index: faiss.Index, path: Path) -> None:
    """
    Сохраняет индекс через Python.
    faiss.write_index на Windows не открывает пути с кириллицей и пробелами.
    serialize_index возвращает numpy-массив uint8 — его и пишем на диск.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ascontiguousarray(faiss.serialize_index(index), dtype=np.uint8)
    path.write_bytes(data.tobytes())


def load_faiss_index(path: Path) -> faiss.Index:
    """Загружает индекс с диска (совместимо с Unicode-путями на Windows)."""
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    return faiss.deserialize_index(data)


def load_faq_data(path: Path | str = FAQS_PATH) -> List[Dict[str, str]]:
    """Загружает FAQ из JSON (поля question и answer обязательны)."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        {
            "question": item["question"],
            "answer": item["answer"],
            "source": "faqs.json",
        }
        for item in data
    ]


def load_txt_documents(directory: Path | str = DATA_DIR) -> List[Dict[str, str]]:
    """
    Загружает .txt-файлы и приводит к формату FAQ.
    question — первая непустая строка (заголовок), answer — остальной текст.
    """
    directory = Path(directory)
    docs: List[Dict[str, str]] = []
    if not directory.is_dir():
        return docs

    for path in sorted(directory.glob("*.txt")):
        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not content:
            continue

        lines = content.splitlines()
        title = next((line.strip() for line in lines if line.strip()), path.name)
        body_lines = lines[1:] if len(lines) > 1 else []
        body = "\n".join(body_lines).strip() or content

        docs.append(
            {
                "question": title,
                "answer": body,
                "source": path.name,
            }
        )
    return docs


def collect_knowledge_items() -> List[Dict[str, str]]:
    """Собирает все элементы базы знаний из JSON и TXT."""
    items: List[Dict[str, str]] = []
    items.extend(load_faq_data())
    items.extend(load_txt_documents())
    return items


def _source_paths() -> List[Path]:
    """Пути к исходным файлам для проверки актуальности индекса."""
    paths: List[Path] = []
    if FAQS_PATH.exists():
        paths.append(FAQS_PATH)
    paths.extend(sorted(DATA_DIR.glob("*.txt")))
    return paths


def _needs_rebuild() -> bool:
    """True, если индекс отсутствует или исходники новее индекса."""
    if not INDEX_PATH.exists() or not META_PATH.exists():
        return True
    index_mtime = max(INDEX_PATH.stat().st_mtime, META_PATH.stat().st_mtime)
    for path in _source_paths():
        if path.stat().st_mtime > index_mtime:
            return True
    return False


def embed_texts(texts: List[str]) -> np.ndarray:
    """Строит эмбеддинги одним запросом (батч)."""
    if not texts:
        raise ValueError("Список текстов для эмбеддинга пуст")
    client = get_client()
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors = np.array([d.embedding for d in resp.data], dtype="float32")
    faiss.normalize_L2(vectors)
    return vectors


def build_index(force: bool = False) -> Tuple[faiss.Index, List[Dict[str, str]]]:
    """Строит FAISS-индекс и сохраняет метаданные на диск."""
    global _index, _items

    if not force and not _needs_rebuild() and _index is not None and _items:
        return _index, _items

    items = collect_knowledge_items()
    if not items:
        raise RuntimeError(
            "Нет данных для индекса: добавьте data/faqs.json или .txt-файлы в data/"
        )

    texts = [f"{item['question']}\n{item['answer']}" for item in items]
    vectors = embed_texts(texts)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    save_faiss_index(index, INDEX_PATH)
    np.save(META_PATH, np.array(items, dtype=object))

    _index = index
    _items = items
    return index, items


def load_index() -> Tuple[faiss.Index, List[Dict[str, str]]]:
    """Загружает индекс с диска или пересобирает при необходимости."""
    global _index, _items

    if _index is not None and _items and not _needs_rebuild():
        return _index, _items

    if _needs_rebuild():
        return build_index(force=True)

    index = load_faiss_index(INDEX_PATH)
    items_arr = np.load(META_PATH, allow_pickle=True)
    _index = index
    _items = list(items_arr.tolist())
    return _index, _items


def ensure_index() -> None:
    """Ленивая инициализация индекса в памяти."""
    load_index()


def search_similar(
    query: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Ищет наиболее похожие фрагменты базы знаний."""
    ensure_index()
    assert _index is not None

    client = get_client()
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    query_vec = np.array([resp.data[0].embedding], dtype="float32")
    faiss.normalize_L2(query_vec)

    k = min(top_k, len(_items))
    scores, indices = _index.search(query_vec, k)

    results: List[Dict[str, Any]] = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0:
            continue
        item = _items[idx]
        results.append(
            {
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "source": item.get("source", ""),
                "score": float(score),
            }
        )
    return results


def _trim_history(history: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """Оставляет только последние N сообщений с валидными ролями."""
    if not history:
        return []
    cleaned: List[Dict[str, str]] = []
    for msg in history:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            cleaned.append({"role": role, "content": content})
    return cleaned[-MAX_HISTORY_MESSAGES:]


def generate_answer(
    message: str,
    top_k: int = 3,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Основная функция для Flask-роута /chat.
    Возвращает: answer, context, history (обновлённая).
    """
    message = message.strip()
    if not message:
        raise ValueError("Пустое сообщение")

    related = search_similar(message, top_k=top_k)
    context_blocks = [
        f"Вопрос: {item['question']}\nОтвет: {item['answer']}"
        for item in related
    ]
    context_text = "\n\n".join(context_blocks) if context_blocks else "Нет контекста."

    system_prompt = (
        "Ты FAQ-ассистент сайта фрилансера в нише AI-ассистентов и Telegram-ботов для бизнеса. "
        "Отвечай кратко и по-русски, только на основе переданного контекста. "
        "Не выдумывай факты, цены и обещания. Если в контексте нет ответа — честно скажи "
        "и предложи оставить заявку в форме «Контакты» на сайте."
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(_trim_history(history))
    messages.append(
        {
            "role": "user",
            "content": (
                f"Контекст из базы знаний:\n{context_text}\n\n"
                f"Вопрос пользователя:\n{message}"
            ),
        }
    )

    client = get_client()
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=500,
    )
    answer = (completion.choices[0].message.content or "").strip()

    updated_history = _trim_history(history)
    updated_history.append({"role": "user", "content": message})
    updated_history.append({"role": "assistant", "content": answer})

    return {
        "answer": answer,
        "context": related,
        "history": updated_history,
    }
