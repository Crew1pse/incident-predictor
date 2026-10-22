import numpy as np
import pandas as pd
import pytest

from features.feature_builder import FEATURE_NAMES, METRICS, WINDOW, feature_frame, features_for_window
from inference.predictor import Predictor
from training.dataset import HORIZON, generate


def test_generator_is_deterministic_and_labels_are_about_the_future():
    a, onsets = generate(20_000, seed=1)
    b, _ = generate(20_000, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert len(onsets) >= 5
    o = onsets[2]
    assert a["label"].iloc[o - HORIZON:o].sum() == HORIZON         # за horizon минут до сбоя метка = 1
    assert a["label"].iloc[o:o + 5].sum() == 0                     # когда сбой уже идёт - 0
    assert a["in_incident"].iloc[o]
    assert 0 < a["label"].mean() < 0.2


def test_training_features_equal_serving_features():
    """Главная защита от train/serve skew: боевой расчёт на окне == строка обучающей таблицы."""
    df, _ = generate(3000, seed=3)
    table = feature_frame(df)
    for end in (500, 1500, 2999):
        window = df[list(METRICS)].iloc[end - WINDOW + 1:end + 1]
        live = features_for_window(window)
        np.testing.assert_allclose(live.iloc[0].to_numpy(), table.iloc[end].to_numpy(), rtol=1e-9)


def test_features_need_full_window():
    with pytest.raises(ValueError):
        features_for_window(pd.DataFrame({m: [1.0] * (WINDOW - 1) for m in METRICS}))


def test_feature_use_only_past():
    """Изменение БУДУЩИХ замеров не должно менять признаки текущей минуты."""
    df, _ = generate(2000, seed=4)
    t = 1000
    before = feature_frame(df).iloc[t].copy()
    df2 = df.copy()
    df2.loc[t + 1:, "cpu"] = 99.0
    pd.testing.assert_series_equal(before, feature_frame(df2).iloc[t])


def test_trained_model_beats_chance_and_flags_ramp(model_dir):
    p = Predictor(model_dir)
    calm = [dict(cpu=45, memory=60, error_rate=0.2, latency=150)] * WINDOW
    ramp = [dict(cpu=45 + 3 * i, memory=60 + i, error_rate=0.2 + 0.2 * i, latency=150 + 15 * i) for i in range(WINDOW)]
    assert p.predict(calm)["incident_probability"] < p.predict(ramp)["incident_probability"]
    assert p.predict(calm)["risk_level"] == "low"
    assert set(p.meta["features"]) == set(FEATURE_NAMES)


def test_incompatible_model_is_rejected(model_dir, tmp_path):
    import json, shutil
    shutil.copytree(model_dir, tmp_path / "m")
    meta = json.loads((tmp_path / "m" / "metadata.json").read_text())
    meta["features"] = meta["features"][:-1]
    (tmp_path / "m" / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(RuntimeError):
        Predictor(str(tmp_path / "m"))
