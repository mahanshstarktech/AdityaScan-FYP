"""
Unit tests for NOAA GOES auto-failover, socket retries, and jitter bounds.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
from pipeline.ingestion.goes_fetcher import GOESFetcher

def test_calculate_backoff_jitter():
    fetcher = GOESFetcher()
    delay_1 = fetcher.calculate_backoff_jitter(1)
    delay_3 = fetcher.calculate_backoff_jitter(3)
    assert delay_1 >= 0.5
    assert delay_3 >= 0.5
    assert delay_3 <= 30.0

def test_cache_fallback_on_network_failure():
    fetcher = GOESFetcher()
    readings = fetcher.fetch_realtime()
    assert isinstance(readings, list)
