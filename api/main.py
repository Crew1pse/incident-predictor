import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from alerting.alert_manager import AlertManager
from alerting.telegram import TelegramNotifier
from api.monitor import LiveMonitor
from api.routes import router
from collector.metrics_collector import PrometheusCollector
from inference.predictor import Predictor

logging.basicConfig(level=logging.INFO)


class AppMetrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter("predict_requests_total", "Запросов /predict", registry=self.registry)
        self.duration = Histogram("predict_duration_seconds", "Время прогноза", registry=self.registry)
        self.probability = Gauge("incident_probability", "Последняя оценка вероятности инцидента", registry=self.registry)

    def observe(self, prob: float, seconds: float) -> None:
        self.requests.inc()
        self.duration.observe(seconds)
        self.probability.set(prob)


def create_app(model_dir: Optional[str] = None, collector: Optional[PrometheusCollector] = None,
               notifier=None, start_monitor: Optional[bool] = None, interval_s: float = 60.0) -> FastAPI:
    predictor = Predictor(model_dir or os.environ.get("MODEL_DIR", "model"))
    metrics = AppMetrics()
    if collector is None and os.environ.get("PROMETHEUS_URL"):
        collector = PrometheusCollector(os.environ["PROMETHEUS_URL"])
    notifier = notifier or TelegramNotifier.from_env()
    alerts = AlertManager(notifier, predictor.threshold) if notifier else None
    monitor = LiveMonitor(predictor, collector, alerts, interval_s,
                          lambda r: metrics.probability.set(r["incident_probability"])) if collector else None
    should_start = (monitor is not None) if start_monitor is None else start_monitor

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if monitor and should_start:
            monitor.start()
        yield
        if monitor:
            monitor.stop()

    app = FastAPI(title="ML Incident Predictor", lifespan=lifespan)
    app.state.predictor, app.state.metrics, app.state.monitor = predictor, metrics, monitor
    app.include_router(router)
    return app


def app_factory() -> FastAPI:
    return create_app()
