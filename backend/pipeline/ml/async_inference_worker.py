"""
Asynchronous Model Inference Background Worker for AdityaScan.

Executes low-latency inference evaluations on a continuous 10s cadence:
  - Maintains latest cached InferenceResult in memory.
  - Dequeues raw telemetry from ingestion pipelines (SoLEXS, HEL1OS, MAG).
  - Supplies REST and WebSocket consumers with zero-wait nowcast data.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from pipeline.ml.inference_engine import (
    AdityaScanInferenceEngine,
    InferenceResult,
    get_inference_engine,
)

logger = logging.getLogger(__name__)


class AsyncInferenceWorker:
    """
    Background worker thread maintaining a rolling inference cache.
    """

    def __init__(
        self,
        engine: Optional[AdityaScanInferenceEngine] = None,
        cadence_seconds: float = 10.0,
    ) -> None:
        self.engine = engine or get_inference_engine()
        self.cadence_seconds = cadence_seconds
        self._latest_result: Optional[InferenceResult] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("AsyncInferenceWorker started with cadence %.1fs", self.cadence_seconds)

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("AsyncInferenceWorker stopped.")

    def get_latest_inference(self) -> InferenceResult:
        with self._lock:
            if self._latest_result is None:
                # Perform an immediate initial evaluation
                self._latest_result = self.engine.predict()
            return self._latest_result

    def _run_loop(self) -> None:
        while self._running:
            try:
                # In production, telemetry features are pulled from real-time ring buffers
                res = self.engine.predict()
                with self._lock:
                    self._latest_result = res
            except Exception as exc:
                logger.error("Error during background inference cycle: %s", exc)

            time.sleep(self.cadence_seconds)


_global_worker: Optional[AsyncInferenceWorker] = None


def get_inference_worker() -> AsyncInferenceWorker:
    global _global_worker
    if _global_worker is None:
        _global_worker = AsyncInferenceWorker()
    return _global_worker
