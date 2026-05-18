"""
Скрипт сборки векторного индекса FAISS по data/faqs.json и data/*.txt.

Запуск из корня проекта:
    python -m backend.build_index
"""

from dotenv import load_dotenv

from backend.rag_index import build_index, collect_knowledge_items

load_dotenv()


def main() -> None:
    items = collect_knowledge_items()
    print(f"Найдено элементов базы знаний: {len(items)}")
    for item in items:
        source = item.get("source", "?")
        q = item.get("question", "")[:60]
        print(f"  - [{source}] {q}...")

    index, built_items = build_index(force=True)
    print(f"Индекс построен: {index.ntotal} векторов, размерность {index.d}")
    print("Файлы сохранены в data/faiss_index.bin и data/faqs_metadata.npy")


if __name__ == "__main__":
    main()
