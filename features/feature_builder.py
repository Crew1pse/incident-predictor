"""Признаки из окна последних W замеров. ОДИН код и для обучения, и для боевого прогноза - чтобы не было перекоса train/serve."""
from __future__ import annotations

import pandas as pd

METRICS = ("cpu", "memory", "error_rate", "latency")   # cpu, memory, error_rate - в процентах; latency - в мс
WINDOW = 10                                              # длина окна в замерах (1 замер = 1 минута)


def feature_frame(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Для каждой строки: статистика по последним `window` замерам. Первые window-1 строк содержат NaN."""
    out = {}
    for m in METRICS:
        roll = df[m].rolling(window, min_periods=window)
        out[f"{m}_last"] = df[m]
        out[f"{m}_mean"] = roll.mean()
        out[f"{m}_std"] = roll.std()
        out[f"{m}_max"] = roll.max()
        out[f"{m}_delta"] = df[m] - df[m].shift(window - 1)      # рост за окно: «растёт ли метрика»
    return pd.DataFrame(out, index=df.index)


FEATURE_NAMES = list(feature_frame(pd.DataFrame({m: [0.0] * WINDOW for m in METRICS})).columns)


def features_for_window(samples: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Признаки для боевого прогноза: берём последние `window` замеров, возвращаем одну строку."""
    if len(samples) < window:
        raise ValueError(f"Нужно минимум {window} замеров, получено {len(samples)}")
    tail = samples[list(METRICS)].tail(window).reset_index(drop=True)
    return feature_frame(tail, window).iloc[[-1]]
