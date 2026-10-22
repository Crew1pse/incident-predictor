from typing import List

from pydantic import BaseModel, Field


class Sample(BaseModel):
    cpu: float = Field(ge=0, le=100, description="загрузка CPU, %")
    memory: float = Field(ge=0, le=100, description="занятая память, %")
    error_rate: float = Field(ge=0, le=100, description="доля 5xx ответов, %")
    latency: float = Field(ge=0, description="p95 задержка, мс")


class PredictRequest(BaseModel):
    samples: List[Sample] = Field(min_length=1, description="замеры по одному в минуту, от старых к новым; нужно не меньше окна модели")


class PredictResponse(BaseModel):
    incident_probability: float
    risk_level: str
    threshold: float
    horizon_minutes: int
