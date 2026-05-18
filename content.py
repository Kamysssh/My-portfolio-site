"""
Тексты сайта: кейсы и подписи тем заявок.
Ниша: AI-ассистенты и Telegram-боты для бизнеса на основе внутренних документов.
"""

# Подписи значений поля «тема» для админки и отчётов
SUBJECT_LABELS = {
    "telegram_bot": "Telegram-бот с ответами из документов компании",
    "online_store": "Умная поддержка покупателей в интернет-магазине",
    "crm_automation": "Автоматизация CRM и черновики ответов для менеджеров",
    "corporate_site": "Корпоративный сайт и помощник для посетителей",
    "ai_assistant": "Внутренний AI-ассистент для сотрудников",
    "faq_assistant": "Универсальный ассистент для клиентов",
    "other": "Другое",
}

# Варианты для формы обратной связи (value, label)
SUBJECT_CHOICES = [
    ("", "Выберите тему"),
    ("faq_assistant", SUBJECT_LABELS["faq_assistant"]),
    ("telegram_bot", SUBJECT_LABELS["telegram_bot"]),
    ("corporate_site", SUBJECT_LABELS["corporate_site"]),
    ("ai_assistant", SUBJECT_LABELS["ai_assistant"]),
    ("online_store", SUBJECT_LABELS["online_store"]),
    ("crm_automation", SUBJECT_LABELS["crm_automation"]),
    ("other", SUBJECT_LABELS["other"]),
]

# Порядок карточек на сайте (только published=True попадут в витрину)
CASES_ORDER = [
    "corporate_site",
    "faq_assistant",
    "telegram_bot",
    "ai_assistant",
    "online_store",
    "crm_automation",
]

CASES_DATA = {
    "telegram_bot": {
        "id": "telegram_bot",
        "published": True,
        "icon": "chat-dots",
        "badge": "GitHub",
        "title": "Корпоративный ассистент в Telegram",
        "short_description": "Клиенты и сотрудники получают ответы по документам компании в мессенджере.",
        "description": """
        <p>Telegram-бот с RAG: ответы строятся по актуальным регламентам компании, а не «из головы» модели.</p>
        <ul>
            <li>Поиск по базе знаний (Google Docs и другие источники)</li>
            <li>Разные сценарии для ролей: HR, продажи, постпродажная поддержка</li>
            <li>Кеш типовых вопросов и логи обращений для аналитики</li>
        </ul>
        <p><a href="https://github.com/Kamysssh/Corporate_RAG_Assistant" target="_blank" rel="noopener noreferrer">Репозиторий на GitHub</a></p>
        """,
        "image": "telegram-bot.jpg",
    },
    "online_store": {
        "id": "online_store",
        "published": False,
        "icon": "cart",
        "badge": None,
        "title": "Первичная линия поддержки для интернет-магазина",
        "short_description": "Ответы о заказе и доставке без очереди к оператору.",
        "description": """
        <p>Покупатель в чате сайта или мессенджере сразу узнаёт статус заказа, условия доставки и возврата — по правилам магазина.</p>
        """,
        "image": "online-store.jpg",
    },
    "crm_automation": {
        "id": "crm_automation",
        "published": False,
        "icon": "gear",
        "badge": None,
        "title": "CRM: меньше рутины у менеджеров",
        "short_description": "Черновики ответов и напоминания по сделкам.",
        "description": """
        <p>Менеджер получает подсказки и черновики писем клиентам в стиле компании.</p>
        """,
        "image": "crm-automation.jpg",
    },
    "corporate_site": {
        "id": "corporate_site",
        "published": True,
        "icon": "globe",
        "badge": "Демо на сайте",
        "title": "Личный сайт с чат-ассистентом",
        "short_description": "Витрина портфолио и живой RAG-виджет — можно протестировать прямо сейчас.",
        "description": """
        <p>Этот сайт — рабочий кейс: Flask, форма заявок, админ-панель и чат-виджет в углу экрана.</p>
        <ul>
            <li>База знаний: FAQ (JSON) и документы в <code>data/</code>, поиск через FAISS</li>
            <li>Ответы только по найденному контексту, память диалога</li>
            <li>Готовность к деплою на VPS — ассистент доступен посетителям 24/7</li>
        </ul>
        <p>Откройте чат в правом нижнем углу и задайте вопрос об услугах или сроках — это и есть демонстрация продукта.</p>
        """,
        "image": "corporate-site.jpg",
    },
    "faq_assistant": {
        "id": "faq_assistant",
        "published": True,
        "icon": "headset",
        "badge": "Кейс курса",
        "title": "Универсальный ассистент для клиентов",
        "short_description": "Отвечает клиентам по FAQ, услугам и правилам компании — на сайте или в мессенджере.",
        "description": """
        <p>Решение для бизнеса, где много однотипных вопросов: оплата, доставка, сроки, возврат, контакты.
        Ассистент общается с <strong>клиентами</strong> на понятном языке и опирается только на материалы заказчика.</p>
        <ul>
            <li><strong>RAG:</strong> векторный поиск по FAQ и документам, без выдуманных фактов</li>
            <li><strong>Память диалога:</strong> учитывает предыдущие сообщения в разговоре</li>
            <li><strong>Каналы:</strong> виджет на сайте, Telegram или другой мессенджер — по задаче</li>
            <li><strong>Гибкость:</strong> базу знаний можно дополнять без переписывания всего проекта</li>
        </ul>
        <p>Технологический каркас: Python, FAISS, LLM через ProxyAPI (доступ из РФ, оплата в рублях).
        Сборка индекса: <code>python -m backend.build_index</code>.</p>
        <p>Для заказчика это снижение нагрузки на поддержку: до 70–80% типовых обращений закрываются автоматически,
        сложные случаи передаются человеку.</p>
        """,
        "image": "faq-assistant.jpg",
    },
    "ai_assistant": {
        "id": "ai_assistant",
        "published": True,
        "icon": "people",
        "badge": "Направление",
        "title": "Внутренний ассистент для сотрудников",
        "short_description": "Регламенты и инструкции — одним вопросом, без поиска по папкам.",
        "description": """
        <p>Формат решения для HR, операционки и обучения: сотрудник спрашивает «как правильно» —
        ассистент отвечает по внутренней базе компании.</p>
        <ul>
            <li>Ответы по актуальным документам и регламентам</li>
            <li>Разграничение доступа к чувствительным разделам</li>
            <li>Меньше шума в рабочих чатах, быстрее адаптация новичков</li>
        </ul>
        <p>Архитектура та же, что у клиентского ассистента, но с другими данными и правами доступа.</p>
        """,
        "image": "ai-assistant.jpg",
    },
}


def get_published_cases() -> dict:
    """Кейсы для отображения на сайте (в заданном порядке)."""
    result = {}
    for case_id in CASES_ORDER:
        case = CASES_DATA.get(case_id)
        if case and case.get("published", True):
            result[case_id] = case
    return result
