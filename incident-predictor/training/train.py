"""Обучение и честная оценка. Запуск:  python -m training.train"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve

from features.feature_builder import FEATURE_NAMES, METRICS, WINDOW, feature_frame
from training.dataset import HORIZON, generate


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    return float(thr[int(np.argmax(f1))])


def evaluate(y: np.ndarray, p: np.ndarray, thr: float, onsets_rel: np.ndarray, in_inc: np.ndarray,
             minutes: int, horizon: int = HORIZON) -> Dict[str, float]:
    pred = p >= thr
    tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    # событийные метрики: что важно человеку на дежурстве
    caught = sum(bool(pred[max(0, o - horizon):o].any()) for o in onsets_rel)
    alert_rows = pred & ~in_inc
    starts = np.flatnonzero(alert_rows & ~np.r_[False, alert_rows[:-1]])
    false_alarms = 0
    for s in starts:
        end = s
        while end + 1 < len(alert_rows) and alert_rows[end + 1]:
            end += 1
        if not any(s < o <= end + horizon for o in onsets_rel):
            false_alarms += 1
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "pr_auc": round(float(average_precision_score(y, p)), 3),
            "incidents_in_test": int(len(onsets_rel)), "incidents_caught_in_advance": int(caught),
            "false_alarms_per_day": round(false_alarms / (minutes / 1440), 2)}


def train(minutes: int = 200_000, seed: int = 42, out_dir: str = "model") -> Dict:
    df, onsets = generate(minutes, seed)
    feats = feature_frame(df)
    ok = feats.notna().all(axis=1) & ~df["in_incident"]           # первые окна неполные, идущие сбои не прогнозируем
    n1, n2 = int(minutes * 0.70), int(minutes * 0.85)             # ПО ВРЕМЕНИ: обучение -> подбор порога -> тест
    idx = np.arange(minutes)
    parts = {"train": ok & (idx < n1), "val": ok & (idx >= n1) & (idx < n2), "test": ok & (idx >= n2)}
    X = {k: feats[m][FEATURE_NAMES] for k, m in parts.items()}
    y = {k: df.loc[m, "label"].to_numpy() for k, m in parts.items()}

    model = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5, class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=seed)
    model.fit(X["train"], y["train"])
    thr = best_f1_threshold(y["val"], model.predict_proba(X["val"])[:, 1])

    base_thr = best_f1_threshold(y["val"], X["val"]["error_rate_last"].to_numpy())   # наивное правило «ошибки выросли»
    test_rows = np.flatnonzero(parts["test"].to_numpy())
    onsets_test = np.searchsorted(test_rows, onsets[onsets >= n2])   # начала сбоев в координатах тестового массива
    in_inc_test = df.loc[parts["test"], "in_incident"].to_numpy()
    p_model = model.predict_proba(X["test"])[:, 1]
    p_base = X["test"]["error_rate_last"].to_numpy()
    n_test_min = int(parts["test"].sum())
    report = {"model": evaluate(y["test"], p_model, thr, onsets_test, in_inc_test, n_test_min),
              "baseline_error_rate_rule": evaluate(y["test"], p_base, base_thr, onsets_test, in_inc_test, n_test_min)}

    meta = {"features": FEATURE_NAMES, "metrics": list(METRICS), "window": WINDOW, "horizon_minutes": HORIZON,
            "threshold": thr, "sklearn_version": sklearn.__version__, "data": "synthetic", "test_report": report}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "incident_model.joblib")
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    meta = train()
    print(json.dumps({"threshold": meta["threshold"], **meta["test_report"]}, ensure_ascii=False, indent=2))
    print("Данные СИНТЕТИЧЕСКИЕ: цифры показывают, что пайплайн работает, а не качество на вашем сервисе.", file=sys.stderr)
