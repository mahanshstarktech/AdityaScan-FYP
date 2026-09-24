"""
Unit tests for NOAA GOES auto-failover and jitter retry behavior.
"""
from pipeline.ingestion.goes_fetcher import GOESFetcher

def test_calculate_backoff_jitter():
    fetcher = GOESFetcher()
    delay_1 = fetcher.calculate_backoff_jitter(1)
    delay_3 = fetcher.calculate_backoff_jitter(3)
    assert delay_1 >= 0.5
    assert delay_3 >= 0.5
    assert delay_3 <= 30.0
