"""
Telemetry Health Monitoring Daemon for Aditya-L1 Ingestion Pipelines.
Logs packet loss, frame arrival jitter, and bad CRC occurrences.
"""
from __future__ import annotations
import logging
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TelemetryHealthDaemon:
    def __init__(self, log_path: str = "/tmp/adityscan_health.log") -> None:
        self.log_path = log_path
        self.drop_count = 0
        self.rate_limit_seconds = 10.0
        self.frame_count = 0

    def record_frame(self, frame_id: int, is_valid: bool) -> None:
        self.frame_count += 1
        if not is_valid:
            self.drop_count += 1
            logger.warning("Corrupted telemetry frame detected: ID=%d", frame_id)
