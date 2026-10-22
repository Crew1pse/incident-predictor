"""Фоновый цикл: раз в минуту берёт замер из Prometheus, считает риск, при необходимости шлёт оповещение."""
from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Deque, Dict, Optional

from alerting.alert_manager import AlertManager
from collector.metrics_collector import PrometheusCollector
from inference.predictor import Predictor

log = logging.getLogger("monitor")


class LiveMonitor(threading.Thread):
    def __init__(self, predictor: Predictor, collector: PrometheusCollector, alerts: Optional[AlertManager],
                 interval_s: float = 60.0, on_prediction=None):
        super().__init__(daemon=True, name="live-monitor")
        self.predictor, self.collector, self.alerts, self.interval_s = predictor, collector, alerts, interval_s
        self.buffer: Deque[Dict[str, float]] = deque(maxlen=predictor.window)
        self.last: Optional[Dict] = None
        self.on_prediction = on_prediction
        self._stop_event = threading.Event()

    def step(self) -> Optional[Dict]:
        sample = self.collector.fetch()
        if sample is None:
            return None
        self.buffer.append(sample)
        if len(self.buffer) < self.predictor.window:
            return None                                    # накапливаем окно
        self.last = self.predictor.predict(list(self.buffer))
        if self.on_prediction:
            self.on_prediction(self.last)
        if self.alerts:
            self.alerts.handle(self.last["incident_probability"])
        return self.last

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.step()
            except Exception:
                log.exception("ошибка шага мониторинга")
            self._stop_event.wait(self.interval_s)

    def stop(self) -> None:
        self._stop_event.set()
