"""Забирает текущие значения 4 метрик из Prometheus (HTTP API). Запросы настраиваются переменными окружения.
Единицы ДОЛЖНЫ совпадать с обучением: cpu, memory, error_rate - %, latency - миллисекунды."""
from __future__ import annotations

import logging
import math
import os
from typing import Dict, Optional

import httpx

log = logging.getLogger("collector")

DEFAULT_QUERIES = {
    "cpu": '100 - avg(rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100',
    "memory": "(1 - sum(node_memory_MemAvailable_bytes) / sum(node_memory_MemTotal_bytes)) * 100",
    "error_rate": 'sum(rate(http_requests_total{status=~"5.."}[1m])) / sum(rate(http_requests_total[1m])) * 100',
    "latency": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[1m])) by (le)) * 1000",
}


class PrometheusCollector:
    def __init__(self, base_url: str, queries: Optional[Dict[str, str]] = None, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.queries = {k: os.environ.get(f"PROM_QUERY_{k.upper()}", v) for k, v in (queries or DEFAULT_QUERIES).items()}
        self.timeout = timeout

    def _query(self, client: httpx.Client, promql: str) -> float:
        r = client.get(f"{self.base_url}/api/v1/query", params={"query": promql})
        r.raise_for_status()
        body = r.json()
        if body.get("status") != "success":
            raise ValueError(f"Prometheus вернул ошибку: {body.get('error', body)}")
        result = body["data"]["result"]
        if not result:
            raise ValueError("пустой результат (метрики нет)")
        value = float(result[0]["value"][1])
        if math.isnan(value) or math.isinf(value):
            raise ValueError("значение NaN/Inf")
        return value

    def fetch(self) -> Optional[Dict[str, float]]:
        """Одна выборка. Если хоть одна метрика недоступна - None (лучше пропустить замер, чем подставить выдуманное)."""
        sample: Dict[str, float] = {}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                for name, promql in self.queries.items():
                    try:
                        sample[name] = self._query(client, promql)
                    except ValueError as exc:
                        log.warning("метрика %s недоступна: %s", name, exc)
                        return None
        except httpx.HTTPError as exc:
            log.warning("Prometheus недоступен: %s", exc)
            return None
        return sample
