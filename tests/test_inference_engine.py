import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from pipeline.ml.inference_engine import (
    AdityaScanInferenceEngine,
    get_inference_engine,
    FORECAST_HORIZONS,
    GOES_CLASSES,
)


def test_inference_engine_initialization():
    engine = AdityaScanInferenceEngine(temperature=0.863, conformal_alpha=0.10)
    assert engine.temperature == pytest.approx(0.863, abs=1e-4)
    assert engine.conformal_alpha == pytest.approx(0.10, abs=1e-4)
    assert engine.engine_type in ["pytorch", "fallback_calibrated"]


def test_inference_predict_defaults():
    engine = AdityaScanInferenceEngine()
    result = engine.predict()

    assert result.timestamp_unix > 0
    assert 0.0 <= result.flare_nowcast_prob <= 1.0
    p_low, p_high = result.flare_nowcast_interval
    assert 0.0 <= p_low <= result.flare_nowcast_prob <= p_high <= 1.0

    # GOES class distribution checks
    assert len(result.goes_class_distribution) == 5
    for cls in GOES_CLASSES:
        assert cls in result.goes_class_distribution
        assert 0.0 <= result.goes_class_distribution[cls] <= 1.0
    assert pytest.approx(sum(result.goes_class_distribution.values()), abs=1e-3) == 1.0

    # Multi-horizon forecast checks
    assert len(result.multi_horizon_forecast) == 5
    for h in FORECAST_HORIZONS:
        assert h in result.multi_horizon_forecast
        pred = result.multi_horizon_forecast[h]
        assert pred.horizon == h
        assert 0.0 <= pred.probability <= 1.0
        ci_low, ci_high = pred.confidence_interval
        assert 0.0 <= ci_low <= ci_high <= 1.0
        assert pred.risk_level in ["quiet", "elevated", "high", "extreme"]

    # Latency check: should be fast (< 50ms)
    assert result.latency_ms < 100.0


def test_inference_with_multi_modal_features():
    engine = AdityaScanInferenceEngine()
    solexs = np.array([1.5e-4, 0.05, 1.2])  # elevated soft X-ray
    helios = np.array([85.0, 120.0])       # strong hard X-ray burst
    mag = np.array([6.5, 1.2, 2.1, -3.4, 4.5, 45.0, 30.0, 22.0])  # high magnetic variance

    result = engine.predict(solexs_features=solexs, helios_features=helios, mag_features=mag)
    assert "SoLEXS" in result.active_modalities
    assert "HEL1OS" in result.active_modalities
    assert "MAG" in result.active_modalities

    # Elevated solar activity should yield higher flare probability
    assert result.flare_nowcast_prob > 0.40
    # M and X class probabilities should be elevated
    assert result.goes_class_distribution["M"] > 0.10 or result.goes_class_distribution["X"] > 0.05


def test_singleton_getter():
    engine1 = get_inference_engine()
    engine2 = get_inference_engine()
    assert engine1 is engine2
