"""
Unit Tests for Aditya-L1 Telemetry Ring Buffer & Cross-Spectral Engine.
Lead Engineer: Sanskriti Raj (Data Ingestion, Physics Feature & Calibration Lead)
"""

import asyncio
import time
import unittest
import numpy as np

from backend.pipeline.ingestion.ring_buffer import (
    TelemetryRingBuffer,
    TelemetryPacket,
    AsyncTelemetryStreamer,
    FLAG_CLEAN,
    FLAG_ORBITAL_SHADOW,
    FLAG_DETECTOR_SATURATED,
    SOLEXS_SATURATION_FLUX,
    SOLEXS_DARK_CURRENT_FLOOR,
)


class TestTelemetryRingBuffer(unittest.TestCase):

    def setUp(self):
        self.capacity = 100
        self.buffer = TelemetryRingBuffer(capacity=self.capacity)

    def test_initialization(self):
        self.assertEqual(len(self.buffer), 0)
        self.assertTrue(self.buffer.is_empty)
        self.assertFalse(self.buffer.is_full)
        self.assertIsNone(self.buffer.get_latest())

        # Check chronological slice on empty buffer
        data = self.buffer.get_chronological_slice()
        self.assertEqual(len(data["timestamps"]), 0)
        self.assertEqual(data["solexs_flux"].shape, (0, 2))

    def test_push_and_latest(self):
        t0 = time.time()
        pkt = TelemetryPacket(
            timestamp=t0,
            solexs_sdd1=1.2e-6,
            solexs_sdd2=1.5e-6,
            helios_cdte=[10.0, 5.0, 2.0, 0.5],
            helios_czt=[8.0, 4.0, 1.0, 0.2],
            quality_score=98.5,
        )
        idx = self.buffer.push(pkt)
        self.assertEqual(idx, 0)
        self.assertEqual(len(self.buffer), 1)
        self.assertFalse(self.buffer.is_empty)

        latest = self.buffer.get_latest()
        self.assertIsNotNone(latest)
        self.assertAlmostEqual(latest.timestamp, t0, places=4)
        self.assertAlmostEqual(latest.solexs_sdd2, 1.5e-6, places=8)
        self.assertEqual(len(latest.helios_cdte), 4)
        self.assertAlmostEqual(latest.helios_cdte[2], 2.0, places=4)

    def test_circular_wrap_around_and_ordering(self):
        """Verify that when buffer wraps past capacity, ordering remains strictly chronological."""
        t_base = 1000000.0
        n_items = 150  # Exceeds capacity of 100

        for i in range(n_items):
            pkt = TelemetryPacket(
                timestamp=t_base + i,
                solexs_sdd1=float(i),
                solexs_sdd2=float(i * 2),
                helios_cdte=[float(i)] * 4,
                helios_czt=[float(i)] * 4,
            )
            self.buffer.push(pkt)

        self.assertEqual(len(self.buffer), self.capacity)
        self.assertTrue(self.buffer.is_full)

        # Retrieve chronological slice
        data = self.buffer.get_chronological_slice()
        self.assertEqual(len(data["timestamps"]), self.capacity)

        # Check strictly monotonic increasing timestamps
        timestamps = data["timestamps"]
        self.assertTrue(np.all(np.diff(timestamps) > 0))

        # Check values correspond to the last 100 items (i = 50 to 149)
        expected_first_ts = t_base + (n_items - self.capacity)
        expected_last_ts = t_base + (n_items - 1)
        self.assertAlmostEqual(timestamps[0], expected_first_ts, places=4)
        self.assertAlmostEqual(timestamps[-1], expected_last_ts, places=4)

        # Sub-window query
        sub_data = self.buffer.get_chronological_slice(window_seconds=20)
        self.assertEqual(len(sub_data["timestamps"]), 20)
        self.assertAlmostEqual(sub_data["timestamps"][-1], expected_last_ts, places=4)

    def test_shadow_mask_filter(self):
        """Packets with zero or sub-dark flux should have FLAG_ORBITAL_SHADOW set."""
        pkt = TelemetryPacket(
            timestamp=100.0,
            solexs_sdd1=0.0,
            solexs_sdd2=SOLEXS_DARK_CURRENT_FLOOR / 2.0,
        )
        self.buffer.push(pkt)
        latest = self.buffer.get_latest()
        self.assertTrue(latest.flags & FLAG_ORBITAL_SHADOW)
        self.assertTrue(latest.to_dict()["is_shadow"])

    def test_detector_saturation_filter(self):
        """Packets exceeding saturation limit should set FLAG_DETECTOR_SATURATED."""
        # Extreme X-class saturation
        pkt = TelemetryPacket(
            timestamp=200.0,
            solexs_sdd1=SOLEXS_SATURATION_FLUX + 0.01,
            solexs_sdd2=1e-4,
        )
        self.buffer.push(pkt)
        latest = self.buffer.get_latest()
        self.assertTrue(latest.flags & FLAG_DETECTOR_SATURATED)
        self.assertTrue(latest.to_dict()["is_saturated"])

    def test_rolling_metrics(self):
        """Verify calculation of running mean, peak, derivative and anomalies."""
        for i in range(10):
            pkt = TelemetryPacket(
                timestamp=1000.0 + i,
                solexs_sdd1=1e-5,
                solexs_sdd2=1e-5 + i * 1e-6,
                helios_cdte=[0.0, 0.0, float(i * 10), 0.0],
            )
            self.buffer.push(pkt)

        metrics = self.buffer.compute_rolling_metrics(window_s=10)
        self.assertEqual(metrics["sample_count"], 10)
        self.assertAlmostEqual(metrics["solexs_sdd2_peak"], 1e-5 + 9 * 1e-6, places=8)
        self.assertAlmostEqual(metrics["helios_hxr_peak"], 90.0, places=4)
        self.assertAlmostEqual(metrics["sxr_derivative"], 1e-6, places=8)
        self.assertEqual(metrics["shadow_frames"], 0)
        self.assertEqual(metrics["saturated_frames"], 0)

    def test_cross_spectral_density_phase_lag(self):
        """Test CSD phase lag computation between synthetic sinusoidal SXR and HXR signals."""
        t = np.arange(128, dtype=float)
        freq = 0.02  # 50-second oscillation period within solar QPP band
        time_delay = 5.0  # 5 seconds physical lead time

        sxr = np.sin(2 * np.pi * freq * t)
        hxr = np.sin(2 * np.pi * freq * (t - time_delay))

        buf = TelemetryRingBuffer(capacity=200)
        for i in range(len(t)):
            pkt = TelemetryPacket(
                timestamp=t[i],
                solexs_sdd2=float(sxr[i] + 1.5),
                helios_cdte=[0.0, 0.0, float(hxr[i] + 2.0), 0.0],
            )
            buf.push(pkt)

        csd = buf.compute_cross_spectral_phase_lag(window_seconds=128)
        self.assertEqual(csd["status"], "success")
        self.assertIn("mean_phase_lag_rad", csd)
        self.assertIn("mean_time_lag_s", csd)
        self.assertAlmostEqual(csd["peak_coherence_freq_hz"], freq, places=2)


class TestAsyncTelemetryStreamer(unittest.IsolatedAsyncioTestCase):

    async def test_streaming_broadcast(self):
        buf = TelemetryRingBuffer(capacity=50)
        streamer = AsyncTelemetryStreamer(ring_buffer=buf, max_queue_size=10)

        # Subscribe consumer
        consumer_queue = streamer.subscribe()
        self.assertEqual(len(streamer._subscribers), 1)

        # Broadcast packet
        pkt = TelemetryPacket(timestamp=500.0, solexs_sdd1=1e-6, solexs_sdd2=2e-6)
        delivered = await streamer.broadcast_packet(pkt)
        self.assertEqual(delivered, 1)

        # Read from queue
        received = await asyncio.wait_for(consumer_queue.get(), timeout=1.0)
        self.assertEqual(received["timestamp"], 500.0)
        self.assertEqual(received["solexs"]["sdd2"], 2e-6)

        # Stats
        stats = streamer.get_stats()
        self.assertEqual(stats["active_subscribers"], 1)
        self.assertEqual(stats["total_broadcast_packets"], 1)

        # Unsubscribe
        streamer.unsubscribe(consumer_queue)
        self.assertEqual(len(streamer._subscribers), 0)


if __name__ == "__main__":
    unittest.main()
