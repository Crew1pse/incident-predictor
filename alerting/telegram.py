from __future__ import annotations

import os
from typing import Optional

import httpx


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, api_base: str = "https://api.telegram.org", timeout: float = 10.0):
        self.url, self.chat_id, self.timeout = f"{api_base}/bot{token}/sendMessage", chat_id, timeout

    def send(self, text: str) -> None:
        httpx.post(self.url, json={"chat_id": self.chat_id, "text": text}, timeout=self.timeout).raise_for_status()

    @classmethod
    def from_env(cls) -> Optional["TelegramNotifier"]:
        token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
        return cls(token, chat) if token and chat else None
