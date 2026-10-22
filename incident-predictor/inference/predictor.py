from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
import sklearn

from features.feature_builder import FEATURE_NAMES, METRICS, features_for_window

log = logging.getLogger("predictor")


class Predictor:
    def __init__(self, model_dir: str = "model"):
        d = Path(model_dir)
        self.meta: Dict = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        if self.meta["features"] != FEATURE_NAMES:
            raise RuntimeError("Признаки модели не совпадают с текущим кодом: переобучите модель (python -m training.train)")
        if self.meta.get("sklearn_version") != sklearn.__version__:
            warnings.warn(f"Модель обучена на scikit-learn {self.meta.get('sklearn_version')}, "
                          f"сейчас {sklearn.__version__}: лучше переобучить")
        # joblib = pickle: загружайте только свои собственные файлы модели
        self.model = joblib.load(d / "incident_model.joblib")
        self.threshold: float = float(self.meta["threshold"])
        self.window: int = int(self.meta["window"])

    def predict(self, samples: List[Dict[str, float]]) -> Dict:
        frame = pd.DataFrame(samples, columns=list(METRICS))
        prob = float(self.model.predict_proba(features_for_window(frame, self.window)[FEATURE_NAMES])[0][1])
        return {"incident_probability": round(prob, 4), "risk_level": "high" if prob >= self.threshold else "low",
                "threshold": round(self.threshold, 4), "horizon_minutes": self.meta["horizon_minutes"]}
