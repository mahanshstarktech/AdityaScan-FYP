"""
Aditya-L1 Real-Time Streaming Ring Buffer & Cross-Spectral Ingestion Engine.
Lead Engineer: Sanskriti Raj (Data Ingestion, Physics Feature & Calibration Lead)
Project: AdityaScan-FYP | SKIT Jaipur CSE Department
Deliverable: Phase 3 Operational Telemetry Buffer (Form-2 Sprint Target: Oct 2026 - Mar 2027)

This module provides an in-memory, zero-copy sliding circular buffer (ring buffer)
for synchronized multi-payload solar telemetry (SoLEXS Soft X-rays and HEL1OS Hard X-rays).
It supports:
  1. 90-minute sliding circular window (5,400 1-second frames) with zero file I/O overhead.
  2. Automated detector saturation clipping and spacecraft orbital shadow mask filtering.
  3. Real-time Multi-Frequency Cross-Spectral Density (CSD) and phase lag computation
     between SXR and HXR to isolate non-thermal electron beam injection timing.
  4. Async pub/sub queuing for WebSocket live mission control streaming.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Bitmask Flags for Telemetry Quality & Anomaly Detection ──────────────────
FLAG_CLEAN: int = 0x00
FLAG_ORBITAL_SHADOW: int = 0x01        # Satellite in eclipse / sensor dark pass
FLAG_DETECTOR_SATURATED: int = 0x02     # Telemetry exceeds detector linear range
FLAG_DATA_GAP_INTERPOLATED: int = 0x04  # Synthesized packet for dropped frame
FLAG_PRE_FLARE_BURST: int = 0x08       # Sudden flux derivative spike (Neupert)

# ── Physical Thresholds (Aditya-L1 Specs) ────────────────────────────────────
SOLEXS_SATURATION_FLUX: float = 0.05   # W/m^2 (~X5+ flare saturation limit)
SOLEXS_DARK_CURRENT_FLOOR: float = 1e-9 # Sensor shadow/night floor (W/m^2)
HEL1OS_MAX_COUNT_RATE: float = 50000.0  # Max linear count rate (cts/s) before pileup
DEFAULT_BUFFER_CAPACITY: int = 5400     # 90 minutes at 1 Hz cadence


@dataclass
class TelemetryPacket:
    """Represents a single 1-second synchronized telemetry frame."""
    timestamp: float                    # Unix epoch timestamp (seconds)
    solexs_sdd1: float = 0.0            # SDD1 SXR flux (W/m^2)
    solexs_sdd2: float = 0.0            # SDD2 SXR flux (W/m^2)
    helios_cdte: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0]) # 4 CdTe bands (cts/s)
    helios_czt: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # 4 CZT bands (cts/s)
    quality_score: float = 100.0        # Telemetry confidence score [0..100]
    flags: int = FLAG_CLEAN             # Bitmask anomaly/condition flags

    def to_dict(self) -> Dict[str, Any]:
        """Converts packet to a JSON-serializable dictionary."""
        return {
            "timestamp": float(self.timestamp),
            "iso_time": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            "solexs": {
                "sdd1": float(self.solexs_sdd1),
                "sdd2": float(self.solexs_sdd2),
            },
            "helios": {
                "cdte": [float(x) for x in self.helios_cdte],
                "czt": [float(x) for x in self.helios_czt],
            },
            "quality_score": float(self.quality_score),
            "flags": int(self.flags),
            "is_shadow": bool(self.flags & FLAG_ORBITAL_SHADOW),
            "is_saturated": bool(self.flags & FLAG_DETECTOR_SATURATED),
            "is_interpolated": bool(self.flags & FLAG_DATA_GAP_INTERPOLATED),
        }


class TelemetryRingBuffer:
    """
    High-throughput circular ring buffer maintaining a sliding window of solar telemetry.
    Pre-allocates numpy arrays for zero runtime memory allocations during 1-Hz streaming.
    Thread-safe for concurrent ingestion and analysis.
    """

    def __init__(self, capacity: int = DEFAULT_BUFFER_CAPACITY):
        if capacity <= 0:
            raise ValueError(f"Buffer capacity must be positive, got {capacity}")
        self.capacity: int = capacity
        self._lock = threading.Lock()

        # Circular buffer pointers
        self._head: int = 0
        self._count: int = 0

        # Pre-allocated arrays
        self._timestamps = np.zeros(self.capacity, dtype=np.float64)
        self._solexs_flux = np.zeros((self.capacity, 2), dtype=np.float32)   # [SDD1, SDD2]
        self._helios_cdte = np.zeros((self.capacity, 4), dtype=np.float32)   # [CdTe1..4]
        self._helios_czt = np.zeros((self.capacity, 4), dtype=np.float32)    # [CZT1..4]
        self._flags = np.zeros(self.capacity, dtype=np.uint8)
        self._quality = np.zeros(self.capacity, dtype=np.float32)

    def __len__(self) -> int:
        with self._lock:
            return self._count

    @property
    def is_full(self) -> bool:
        with self._lock:
            return self._count == self.capacity

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return self._count == 0

    def push(self, packet: TelemetryPacket) -> int:
        """
        Inserts a single telemetry packet into the circular buffer with automatic
        shadow mask and detector saturation detection. Returns the insertion index.
        """
        flags = packet.flags

        # Shadow mask filter: check for eclipse or sub-dark floor
        if packet.solexs_sdd2 <= SOLEXS_DARK_CURRENT_FLOOR and packet.solexs_sdd1 <= SOLEXS_DARK_CURRENT_FLOOR:
            flags |= FLAG_ORBITAL_SHADOW

        # Saturation filter: check for sensor clipping
        if (packet.solexs_sdd2 >= SOLEXS_SATURATION_FLUX or
            packet.solexs_sdd1 >= SOLEXS_SATURATION_FLUX or
            any(c >= HEL1OS_MAX_COUNT_RATE for c in packet.helios_cdte) or
            any(c >= HEL1OS_MAX_COUNT_RATE for c in packet.helios_czt)):
            flags |= FLAG_DETECTOR_SATURATED

        with self._lock:
            idx = self._head
            self._timestamps[idx] = packet.timestamp
            self._solexs_flux[idx, 0] = packet.solexs_sdd1
            self._solexs_flux[idx, 1] = packet.solexs_sdd2

            cdte_arr = packet.helios_cdte if len(packet.helios_cdte) == 4 else (packet.helios_cdte + [0.0]*4)[:4]
            czt_arr = packet.helios_czt if len(packet.helios_czt) == 4 else (packet.helios_czt + [0.0]*4)[:4]
            self._helios_cdte[idx, :] = cdte_arr
            self._helios_czt[idx, :] = czt_arr

            self._flags[idx] = flags
            self._quality[idx] = packet.quality_score

            self._head = (self._head + 1) % self.capacity
            if self._count < self.capacity:
                self._count += 1

            return idx

    def push_batch(self, packets: List[TelemetryPacket]) -> int:
        """Batch-pushes multiple packets in chronological order."""
        count = 0
        for pkt in packets:
            self.push(pkt)
            count += 1
        return count

    def get_chronological_slice(self, window_seconds: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Retrieves an unwrapped, strictly chronological view of buffered telemetry.
        If window_seconds is provided, limits output to the most recent window_seconds.
        """
        with self._lock:
            if self._count == 0:
                return {
                    "timestamps": np.array([], dtype=np.float64),
                    "solexs_flux": np.empty((0, 2), dtype=np.float32),
                    "helios_cdte": np.empty((0, 4), dtype=np.float32),
                    "helios_czt": np.empty((0, 4), dtype=np.float32),
                    "flags": np.array([], dtype=np.uint8),
                    "quality": np.array([], dtype=np.float32),
                }

            if self._count < self.capacity:
                # Buffer has not wrapped around yet
                start_idx = 0
                end_idx = self._head
                indices = np.arange(start_idx, end_idx)
            else:
                # Buffer is full, oldest item is at self._head
                indices = (np.arange(self._head, self._head + self.capacity)) % self.capacity

            if window_seconds is not None and window_seconds > 0 and window_seconds < len(indices):
                indices = indices[-window_seconds:]

            return {
                "timestamps": self._timestamps[indices].copy(),
                "solexs_flux": self._solexs_flux[indices].copy(),
                "helios_cdte": self._helios_cdte[indices].copy(),
                "helios_czt": self._helios_czt[indices].copy(),
                "flags": self._flags[indices].copy(),
                "quality": self._quality[indices].copy(),
            }

    def get_latest(self) -> Optional[TelemetryPacket]:
        """Returns the most recent single telemetry packet, or None if buffer is empty."""
        with self._lock:
            if self._count == 0:
                return None
            latest_idx = (self._head - 1) % self.capacity
            return TelemetryPacket(
                timestamp=float(self._timestamps[latest_idx]),
                solexs_sdd1=float(self._solexs_flux[latest_idx, 0]),
                solexs_sdd2=float(self._solexs_flux[latest_idx, 1]),
                helios_cdte=[float(x) for x in self._helios_cdte[latest_idx, :]],
                helios_czt=[float(x) for x in self._helios_czt[latest_idx, :]],
                quality_score=float(self._quality[latest_idx]),
                flags=int(self._flags[latest_idx]),
            )

    def compute_rolling_metrics(self, window_s: int = 60) -> Dict[str, Any]:
        """
        Computes rolling statistics over the most recent window_s seconds:
        - Mean and Peak Soft X-ray (SDD2) & Hard X-ray (CdTe3) flux
        - Numerical Neupert derivative d(SXR)/dt
        - Anomaly counts (shadow, saturated)
        """
        data = self.get_chronological_slice(window_seconds=window_s)
        n = len(data["timestamps"])
        if n == 0:
            return {
                "sample_count": 0,
                "solexs_sdd2_mean": 0.0,
                "solexs_sdd2_peak": 0.0,
                "helios_hxr_mean": 0.0,
                "helios_hxr_peak": 0.0,
                "sxr_derivative": 0.0,
                "shadow_frames": 0,
                "saturated_frames": 0,
            }

        sdd2 = data["solexs_flux"][:, 1]
        cdte3 = data["helios_cdte"][:, 2] # 25-40 keV HXR band
        flags = data["flags"]

        # Central difference slope for derivative
        if n >= 2:
            dt = max(1e-3, data["timestamps"][-1] - data["timestamps"][-2])
            sxr_deriv = float((sdd2[-1] - sdd2[-2]) / dt)
        else:
            sxr_deriv = 0.0

        shadow_count = int(np.sum((flags & FLAG_ORBITAL_SHADOW) != 0))
        sat_count = int(np.sum((flags & FLAG_DETECTOR_SATURATED) != 0))

        return {
            "sample_count": n,
            "solexs_sdd2_mean": float(np.mean(sdd2)),
            "solexs_sdd2_peak": float(np.max(sdd2)),
            "helios_hxr_mean": float(np.mean(cdte3)),
            "helios_hxr_peak": float(np.max(cdte3)),
            "sxr_derivative": sxr_deriv,
            "shadow_frames": shadow_count,
            "saturated_frames": sat_count,
        }

    def compute_cross_spectral_phase_lag(
        self,
        window_seconds: int = 512,
        sxr_channel: int = 1,     # 0: SDD1, 1: SDD2
        hxr_band: int = 2,        # 2: CdTe3 (25-40 keV)
    ) -> Dict[str, Any]:
        """
        Computes Multi-Frequency Cross-Spectral Density (CSD) between SoLEXS SXR and HEL1OS HXR:
            CSD_xy(f) = F{SXR} * conj(F{HXR})
            Phase lag Δφ(f) = atan2(Im(CSD), Re(CSD))
            Time lag τ(f) = Δφ(f) / (2π f)

        This mathematical formulation isolates non-thermal electron beam injection times
        (Neupert effect time lead) directly from the in-memory circular buffer.
        """
        data = self.get_chronological_slice(window_seconds=window_seconds)
        n = len(data["timestamps"])
        if n < 32:
            return {
                "status": "insufficient_data",
                "sample_count": n,
                "mean_phase_lag_rad": 0.0,
                "mean_time_lag_s": 0.0,
                "peak_coherence_freq_hz": 0.0,
            }

        # Zero-mean detrending
        x = data["solexs_flux"][:, sxr_channel].astype(np.float64)
        y = data["helios_cdte"][:, hxr_band].astype(np.float64)

        x = x - np.mean(x)
        y = y - np.mean(y)

        # Windowing (Hanning)
        w = np.hanning(n)
        X = np.fft.rfft(x * w)
        Y = np.fft.rfft(y * w)

        # Cross-Spectral Density P_xy
        P_xy = X * np.conj(Y)
        freqs = np.fft.rfftfreq(n, d=1.0) # 1-Hz sampling

        # Filter to physically relevant solar QPP/reconnection frequencies: 0.005 to 0.1 Hz (10s to 200s periods)
        valid_mask = (freqs >= 0.005) & (freqs <= 0.10)
        if not np.any(valid_mask):
            valid_mask = freqs > 0

        P_xy_filtered = P_xy[valid_mask]
        f_filtered = freqs[valid_mask]

        phase_lags = np.angle(P_xy_filtered) # Radians [-pi, pi]
        # Time lag tau = phase / (2*pi*f)
        time_lags = phase_lags / (2.0 * np.pi * np.maximum(f_filtered, 1e-5))

        # Weight by spectral amplitude
        weights = np.abs(P_xy_filtered)
        if np.sum(weights) > 0:
            weighted_phase = float(np.sum(phase_lags * weights) / np.sum(weights))
            weighted_lag = float(np.sum(time_lags * weights) / np.sum(weights))
            peak_freq = float(f_filtered[np.argmax(weights)])
        else:
            weighted_phase = float(np.mean(phase_lags))
            weighted_lag = float(np.mean(time_lags))
            peak_freq = 0.0

        return {
            "status": "success",
            "sample_count": n,
            "mean_phase_lag_rad": weighted_phase,
            "mean_time_lag_s": weighted_lag,
            "peak_coherence_freq_hz": peak_freq,
            "frequency_band_hz": [0.005, 0.10],
        }


class AsyncTelemetryStreamer:
    """
    Asynchronous pub/sub queue distributor that bridges the TelemetryRingBuffer
    to live WebSocket consumers with backpressure handling.
    """

    def __init__(self, ring_buffer: TelemetryRingBuffer, max_queue_size: int = 120):
        self.ring_buffer: TelemetryRingBuffer = ring_buffer
        self.max_queue_size: int = max_queue_size
        self._subscribers: Set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()
        self._total_broadcast_packets: int = 0

    def subscribe(self) -> asyncio.Queue:
        """Subscribes a new consumer queue for live streaming."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_queue_size)
        self._subscribers.add(queue)
        logger.debug("New telemetry streamer subscriber added. Total: %d", len(self._subscribers))
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Removes a consumer queue."""
        self._subscribers.discard(queue)
        logger.debug("Telemetry streamer subscriber removed. Total: %d", len(self._subscribers))

    async def broadcast_packet(self, packet: TelemetryPacket) -> int:
        """
        Pushes packet to ring buffer and broadcasts to all active subscriber queues.
        Drops oldest frame if a consumer is slow (lossy real-time semantics).
        """
        self.ring_buffer.push(packet)
        self._total_broadcast_packets += 1

        payload = packet.to_dict()
        delivered = 0

        for q in list(self._subscribers):
            try:
                if q.full():
                    try:
                        q.get_nowait() # Evict oldest unconsumed frame
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait(payload)
                delivered += 1
            except Exception as e:
                logger.warning("Failed to deliver frame to subscriber queue: %s", e)

        return delivered

    def get_stats(self) -> Dict[str, Any]:
        """Returns streaming health stats."""
        return {
            "active_subscribers": len(self._subscribers),
            "total_broadcast_packets": self._total_broadcast_packets,
            "buffer_utilization_pct": round(len(self.ring_buffer) / self.ring_buffer.capacity * 100, 2),
            "buffer_count": len(self.ring_buffer),
            "buffer_capacity": self.ring_buffer.capacity,
        }
