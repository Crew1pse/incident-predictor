"""Решает, когда слать оповещение: порог + несколько замеров подряд + пауза между сообщениями + сообщение о выздоровлении."""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional, Protocol

log = logging.getLogger("alerting")


class Notifier(Protocol):
    def send(self, text: str) -> None: ...


class AlertManager:
    def __init__(self, notifier: Notifier, threshold: float, consecutive: int = 2, cooldown_s: float = 1800,
                 clock: Callable[[], float] = time.time):
        self.notifier, self.threshold, self.consecutive, self.cooldown_s, self.clock = \
            notifier, threshold, consecutive, cooldown_s, clock
        self.high_streak = 0
        self.low_streak = 0
        self.active = False
        self.last_sent: Optional[float] = None

    def _send(self, text: str) -> None:
        try:
            self.notifier.send(text)
        except Exception:                                  # сбой канала не должен ронять сервис
            log.exception("не удалось отправить оповещение")

    def handle(self, probability: float) -> Optional[str]:
        """Возвращает "alert" / "recovered", если было отправлено сообщение."""
        if probability >= self.threshold:
            self.high_streak, self.low_streak = self.high_streak + 1, 0
            now = self.clock()
            cooled = self.last_sent is None or now - self.last_sent >= self.cooldown_s
            if self.high_streak >= self.consecutive and not self.active and cooled:
                self.active, self.last_sent = True, now
                self._send(f"⚠️ Высокий риск инцидента в ближайшие минуты: {probability:.0%} (порог {self.threshold:.0%})")
                return "alert"
        else:
            self.low_streak, self.high_streak = self.low_streak + 1, 0
            if self.active and self.low_streak >= self.consecutive:
                self.active = False
                self._send(f"✅ Риск снизился: {probability:.0%}")
                return "recovered"
        return None
