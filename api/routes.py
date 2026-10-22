import time

from fastapi import APIRouter, HTTPException, Request, Response
from prometheus_client import generate_latest

from api.schemas import PredictRequest, PredictResponse

router = APIRouter()


@router.get("/")
def root():
    return {"status": "ML Incident Predictor running"}


@router.get("/health")
def health(request: Request):
    monitor = request.app.state.monitor
    return {"status": "ok", "live_monitoring": monitor is not None,
            "last_prediction": monitor.last if monitor else None}


@router.get("/model")
def model_info(request: Request):
    meta = request.app.state.predictor.meta
    return {k: meta[k] for k in ("window", "horizon_minutes", "threshold", "data", "test_report")}


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request):
    state = request.app.state
    if len(body.samples) < state.predictor.window:
        raise HTTPException(422, f"Нужно минимум {state.predictor.window} замеров, получено {len(body.samples)}")
    started = time.perf_counter()
    result = state.predictor.predict([s.model_dump() for s in body.samples])
    state.metrics.observe(result["incident_probability"], time.perf_counter() - started)
    return result


@router.get("/metrics")
def metrics(request: Request):
    return Response(generate_latest(request.app.state.metrics.registry), media_type="text/plain; version=0.0.4")
