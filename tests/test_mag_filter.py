import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from pipeline.physics.mag_variational_filter import MAGVariationalFilter, MAGFilteredSample


def test_mag_filter_packet_rejection():
    filt = MAGVariationalFilter()
    # Bad quality flag should be rejected
    assert filt.process_sample(100.0, 5.0, 2.0, -1.0, quality_flag=0) is None
    # NaN values should be rejected
    assert filt.process_sample(100.0, float('nan'), 2.0, -1.0, quality_flag=1) is None


def test_mag_filter_nominal_sliding_window():
    filt = MAGVariationalFilter(window_size=5)
    samples = []
    for i in range(10):
        s = filt.process_sample(100.0 + i * 10, 3.0, 2.0, -1.0, quality_flag=1)
        assert s is not None
        samples.append(s)

    last = samples[-1]
    assert pytest.approx(last.B_total, abs=0.01) == 3.74
    # In steady field, variance should be near zero
    assert last.variance_10s < 0.05
    assert not last.is_discontinuity


def test_mag_filter_discontinuity_detection():
    filt = MAGVariationalFilter(window_size=6, rotation_threshold_deg=35.0)
    # Steady background
    for i in range(5):
        filt.process_sample(1000.0 + i * 10, 3.0, 1.0, 5.0, quality_flag=1)

    # Abrupt rotational discontinuity: IMF rotation from northward Bz to strong southward Bz
    shock = filt.process_sample(1050.0, 8.0, -6.0, -9.0, quality_flag=1)
    assert shock is not None
    assert shock.is_discontinuity
    assert shock.shock_index > 0.5
    assert shock.variance_10s > 2.0
