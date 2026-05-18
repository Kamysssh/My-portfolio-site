"""
Клиент LLM через ProxyAPI (https://proxyapi.ru).

OpenAI SDK + base_url ProxyAPI — запросы идут через серверы в РФ, без VPN.
Ключ: https://console.proxyapi.ru/keys
"""

from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI

# Нативный OpenAI-эндпоинт ProxyAPI (см. документацию proxyapi.ru)
DEFAULT_PROXYAPI_BASE_URL = "https://api.proxyapi.ru/openai/v1"

_client: Optional[OpenAI] = None


def get_api_key() -> str:
    """Ключ ProxyAPI (или legacy OPENAI_API_KEY для совместимости)."""
    key = os.getenv("PROXYAPI_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "Не задан PROXYAPI_KEY. Создайте ключ в личном кабинете: "
            "https://console.proxyapi.ru/keys и добавьте в .env"
        )
    return key


def is_llm_configured() -> bool:
    """True, если ключ для чат-бота задан."""
    return bool(os.getenv("PROXYAPI_KEY") or os.getenv("OPENAI_API_KEY"))


def get_base_url() -> str:
    return os.getenv("PROXYAPI_BASE_URL", DEFAULT_PROXYAPI_BASE_URL).rstrip("/")


def get_client() -> OpenAI:
    """Создаёт и кеширует OpenAI-клиент, направленный на ProxyAPI."""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=get_api_key(),
            base_url=get_base_url(),
        )
    return _client
