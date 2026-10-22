import json

import pytest
from fastapi.testclient import TestClient

from alerting.alert_manager import AlertManager
from alerting.telegram import TelegramNotifier
from api.main import create_app
from collector.metrics_collector import PrometheusCollector

OK_SAMPLE = dict(cpu=50, memory=60, error_rate=0.2, latency=150)


class Box:
    def __init__(self): self.sent = []
    def send(self, text): self.sent.append(text)


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


# ---- AlertManager ----
def test_alert_needs_consecutive_and_cooldown_and_recovery():
    box, clock = Box(), Clock()
    am = AlertManager(box, threshold=0.7, consecutive=2, cooldown_s=600, clock=clock)
    assert am.handle(0.9) is None and not box.sent                     # одиночный всплеск игнорируем
    assert am.handle(0.9) == "alert" and len(box.sent) == 1
    assert am.handle(0.95) is None and len(box.sent) == 1               # не спамим, пока активно
    assert am.handle(0.1) is None
    assert am.handle(0.1) == "recovered" and len(box.sent) == 2
    clock.t = 100                                                       # новый всплеск внутри паузы - молчим
    am.handle(0.9); assert am.handle(0.9) is None
    clock.t = 700
    assert am.handle(0.9) == "alert" and len(box.sent) == 3


def test_single_dip_resets_streak():
    box = Box()
    am = AlertManager(box, 0.7, consecutive=3)
    for p in (0.9, 0.9, 0.1, 0.9, 0.9):
        am.handle(p)
    assert not box.sent


def test_broken_notifier_does_not_crash():
    class Bad:
        def send(self, t): raise RuntimeError("telegram down")
    am = AlertManager(Bad(), 0.5, consecutive=1)
    assert am.handle(0.9) == "alert"


def test_telegram_request_shape(fake_server):
    srv = fake_server(lambda m, p, b: (200, {"ok": True}))
    TelegramNotifier("TOKEN", "42", api_base=srv.url).send("привет")
    method, path, body = srv.requests[0]
    assert method == "POST" and path == "/botTOKEN/sendMessage"
    assert json.loads(body) == {"chat_id": "42", "text": "привет"}


def test_telegram_http_error_raises(fake_server):
    srv = fake_server(lambda m, p, b: (401, {"ok": False}))
    with pytest.raises(Exception):
        TelegramNotifier("BAD", "1", api_base=srv.url).send("x")


# ---- collector ----
def prom_ok(m, path, body):
    return 200, {"status": "success", "data": {"result": [{"metric": {}, "value": [1, "42.5"]}]}}


def test_collector_reads_all_metrics(fake_server):
    srv = fake_server(prom_ok)
    assert PrometheusCollector(srv.url).fetch() == {"cpu": 42.5, "memory": 42.5, "error_rate": 42.5, "latency": 42.5}
    assert len(srv.requests) == 4


def test_collector_skips_sample_on_missing_metric(fake_server):
    srv = fake_server(lambda m, p, b: (200, {"status": "success", "data": {"result": []}}))
    assert PrometheusCollector(srv.url).fetch() is None


def test_collector_skips_nan_and_prom_down(fake_server):
    srv = fake_server(lambda m, p, b: (200, {"status": "success", "data": {"result": [{"value": [1, "NaN"]}]}}))
    assert PrometheusCollector(srv.url).fetch() is None
    assert PrometheusCollector("http://127.0.0.1:1").fetch() is None


# ---- API ----
@pytest.fixture
def client(model_dir):
    return TestClient(create_app(model_dir, start_monitor=False))


def test_predict_and_validation(client):
    r = client.post("/predict", json={"samples": [OK_SAMPLE] * 10})
    assert r.status_code == 200 and 0 <= r.json()["incident_probability"] <= 1
    assert client.post("/predict", json={"samples": [OK_SAMPLE] * 3}).status_code == 422       # короткое окно
    assert client.post("/predict", json={"samples": [{**OK_SAMPLE, "cpu": 500}] * 10}).status_code == 422
    assert client.post("/predict", json={"samples": []}).status_code == 422


def test_metrics_and_model_endpoints(client):
    client.post("/predict", json={"samples": [OK_SAMPLE] * 10})
    text = client.get("/metrics").text
    assert "predict_requests_total 1.0" in text and "incident_probability" in text
    assert client.get("/model").json()["data"] == "synthetic"
    assert client.get("/health").json()["live_monitoring"] is False


def test_app_import_does_not_block():
    import importlib
    import api.main
    importlib.reload(api.main)                                           # раньше while True на уровне модуля вешал импорт


def test_live_monitor_end_to_end(model_dir, fake_server):
    """Prometheus (фейк) -> окно -> прогноз -> оповещение: спокойно - тишина, нарастание метрик - сообщение."""
    state = {"i": -1}                                                   # -1 = спокойный режим

    def prom(method, path, body):
        q, i = path, state["i"]
        calm = dict(cpu=45, memory=60, error_rate=0.2, latency=150)
        ramp = dict(cpu=45 + 3 * i, memory=60 + i, error_rate=0.2 + 0.2 * i, latency=150 + 15 * i) if i >= 0 else calm
        key = "cpu" if "node_cpu" in q else "memory" if "MemAvailable" in q else "error_rate" if "5.." in q else "latency"
        return 200, {"status": "success", "data": {"result": [{"value": [1, str(ramp[key])]}]}}

    prom_srv = fake_server(prom)
    box = Box()
    app = create_app(model_dir, collector=PrometheusCollector(prom_srv.url), notifier=box, start_monitor=False)
    mon = app.state.monitor
    assert mon.step() is None                                           # окно ещё не набрано
    for _ in range(14):
        r = mon.step()
    assert r["risk_level"] == "low" and not box.sent                    # спокойно -> тишина
    for i in range(10):                                                 # метрики начинают расти
        state["i"] = i
        mon.step()
    assert any("Высокий риск" in s for s in box.sent), (mon.last, box.sent)
