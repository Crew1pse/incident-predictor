"""Синтетический временной ряд метрик с инцидентами.

ВАЖНО: это симуляция. Метка «инцидент в ближайшие H минут» - про БУДУЩЕЕ, а признаки считаются по ПРОШЛОМУ окну,
поэтому модель решает настоящую задачу прогноза (а не восстанавливает формулу из той же строки, как в старой версии).
Но «предвестники» (рост метрик перед сбоем) я заложил сам - на реальных данных качество будет другим.
Чтобы задача не была тривиальной: 30% инцидентов происходят внезапно (без предвестников), а «ложные всплески»
CPU/latency случаются и без инцидента.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

HORIZON = 10          # прогнозируем инцидент в ближайшие 10 минут


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    x = np.zeros(n)
    eps = rng.normal(0, sigma, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def generate(minutes: int = 200_000, seed: int = 42, horizon: int = HORIZON) -> Tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(minutes)
    daily = np.sin(2 * np.pi * t / 1440)

    cpu = 45 + 10 * daily + _ar1(rng, minutes, 0.9, 2.0)
    memory = 55 + 15 * ((t % 4000) / 4000) + _ar1(rng, minutes, 0.95, 0.5)       # «пила»: утечка и перезапуск
    error_rate = rng.exponential(0.2, minutes)
    latency = 150 + 1.2 * (cpu - 45) + rng.normal(0, 15, minutes)

    # ложные всплески: растут CPU и latency, ошибок нет, инцидента не будет
    for start in rng.choice(minutes - 20, size=minutes // 600, replace=False):
        dur = int(rng.integers(3, 9))
        cpu[start:start + dur] += 25
        latency[start:start + dur] += 80

    # инциденты
    onsets, pos = [], int(rng.integers(700, 1500))
    while pos < minutes - 40:
        onsets.append(pos)
        pos += int(rng.integers(700, 2500))
    onsets = np.array(onsets)

    in_incident = np.zeros(minutes, dtype=bool)
    for onset in onsets:
        dur = int(rng.integers(10, 25))
        if rng.random() < 0.7:                                  # у 70% инцидентов есть нарастание перед сбоем
            ramp = int(rng.integers(6, 14))
            frac = (np.arange(ramp) + 1) / ramp
            sl = slice(onset - ramp, onset)
            cpu[sl] += 30 * frac
            latency[sl] += 150 * frac
            error_rate[sl] += 1.5 * frac
            memory[sl] += 8 * frac
        sl = slice(onset, onset + dur)
        cpu[sl] += 35
        latency[sl] += 250
        error_rate[sl] += 3 + rng.exponential(1.0, dur)
        in_incident[sl] = True

    label = np.zeros(minutes, dtype=int)
    for onset in onsets:
        label[max(0, onset - horizon):onset] = 1               # за horizon минут до начала сбоя
    label[in_incident] = 0                                      # когда сбой уже идёт - не прогнозируем

    df = pd.DataFrame({"cpu": np.clip(cpu, 0, 100), "memory": np.clip(memory, 0, 100),
                       "error_rate": np.clip(error_rate, 0, 100), "latency": np.clip(latency, 1, None),
                       "in_incident": in_incident, "label": label})
    return df, onsets
