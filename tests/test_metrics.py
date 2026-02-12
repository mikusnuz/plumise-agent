"""Tests for plumise_agent.node.metrics -- MetricsCollector and InferenceMetrics."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from plumise_agent.node.metrics import InferenceMetrics, MetricsCollector


# ---------------------------------------------------------------------------
# InferenceMetrics dataclass
# ---------------------------------------------------------------------------

class TestInferenceMetrics:
    """Test the InferenceMetrics value object."""

    def test_defaults(self):
        m = InferenceMetrics()
        assert m.total_tokens_processed == 0
        assert m.total_requests == 0
        assert m.total_latency_ms == 0.0

    def test_avg_latency_zero_requests(self):
        m = InferenceMetrics()
        assert m.avg_latency_ms == 0.0

    def test_avg_latency_computed(self):
        m = InferenceMetrics(total_requests=10, total_latency_ms=500.0)
        assert m.avg_latency_ms == pytest.approx(50.0)

    def test_uptime_seconds(self):
        m = InferenceMetrics(start_time=time.time() - 120)
        # Allow 1 second margin for test execution
        assert 119 <= m.uptime_seconds <= 122

    def test_tokens_per_second_zero_uptime(self):
        m = InferenceMetrics(start_time=time.time(), total_tokens_processed=100)
        assert m.tokens_per_second == 0.0

    def test_tokens_per_second_positive(self):
        m = InferenceMetrics(
            start_time=time.time() - 10,
            total_tokens_processed=100,
        )
        # ~10 tokens/sec, allowing margin
        assert 8.0 <= m.tokens_per_second <= 12.0

    def test_to_dict_keys(self):
        m = InferenceMetrics()
        d = m.to_dict()
        expected_keys = {
            "total_tokens_processed",
            "total_requests",
            "total_latency_ms",
            "avg_latency_ms",
            "uptime_seconds",
            "tokens_per_second",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_type(self):
        m = InferenceMetrics(
            total_tokens_processed=100,
            total_requests=5,
            total_latency_ms=250.0,
        )
        d = m.to_dict()
        assert isinstance(d["total_tokens_processed"], int)
        assert isinstance(d["total_requests"], int)
        assert isinstance(d["total_latency_ms"], float)
        assert isinstance(d["avg_latency_ms"], float)


# ---------------------------------------------------------------------------
# MetricsCollector -- basic recording
# ---------------------------------------------------------------------------

class TestMetricsCollectorRecord:
    """Test record_inference and snapshot."""

    def test_initial_state(self):
        mc = MetricsCollector()
        snap = mc.get_snapshot()
        assert snap.total_tokens_processed == 0
        assert snap.total_requests == 0
        assert snap.total_latency_ms == 0.0

    def test_single_record(self):
        mc = MetricsCollector()
        mc.record_inference(tokens=42, latency_ms=100.5)
        snap = mc.get_snapshot()
        assert snap.total_tokens_processed == 42
        assert snap.total_requests == 1
        assert snap.total_latency_ms == pytest.approx(100.5)

    def test_multiple_records_accumulate(self):
        mc = MetricsCollector()
        mc.record_inference(tokens=10, latency_ms=50.0)
        mc.record_inference(tokens=20, latency_ms=75.0)
        mc.record_inference(tokens=30, latency_ms=100.0)
        snap = mc.get_snapshot()
        assert snap.total_tokens_processed == 60
        assert snap.total_requests == 3
        assert snap.total_latency_ms == pytest.approx(225.0)

    def test_avg_latency_after_records(self):
        mc = MetricsCollector()
        mc.record_inference(tokens=10, latency_ms=100.0)
        mc.record_inference(tokens=10, latency_ms=200.0)
        snap = mc.get_snapshot()
        assert snap.avg_latency_ms == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# MetricsCollector -- snapshot is a copy
# ---------------------------------------------------------------------------

class TestMetricsCollectorSnapshot:
    """Verify that snapshots are independent copies."""

    def test_snapshot_not_affected_by_subsequent_writes(self):
        mc = MetricsCollector()
        mc.record_inference(tokens=10, latency_ms=50.0)
        snap1 = mc.get_snapshot()

        mc.record_inference(tokens=20, latency_ms=100.0)
        snap2 = mc.get_snapshot()

        # snap1 should not have changed
        assert snap1.total_tokens_processed == 10
        assert snap1.total_requests == 1
        assert snap2.total_tokens_processed == 30
        assert snap2.total_requests == 2


# ---------------------------------------------------------------------------
# MetricsCollector -- reset
# ---------------------------------------------------------------------------

class TestMetricsCollectorReset:
    """Test the reset method."""

    def test_reset_returns_snapshot(self):
        mc = MetricsCollector()
        mc.record_inference(tokens=50, latency_ms=200.0)
        snap = mc.reset()
        assert snap.total_tokens_processed == 50
        assert snap.total_requests == 1

    def test_reset_clears_counters(self):
        mc = MetricsCollector()
        mc.record_inference(tokens=50, latency_ms=200.0)
        mc.reset()
        snap = mc.get_snapshot()
        assert snap.total_tokens_processed == 0
        assert snap.total_requests == 0
        assert snap.total_latency_ms == 0.0


# ---------------------------------------------------------------------------
# MetricsCollector -- proof buffer
# ---------------------------------------------------------------------------

class TestMetricsCollectorProofBuffer:
    """Test proof recording and draining."""

    def _make_mock_proof(self, token_count: int = 10) -> MagicMock:
        proof = MagicMock()
        proof.token_count = token_count
        return proof

    def test_drain_empty(self):
        mc = MetricsCollector()
        assert mc.drain_proofs() == []

    def test_record_and_drain(self):
        mc = MetricsCollector()
        p1 = self._make_mock_proof(10)
        p2 = self._make_mock_proof(20)
        mc.record_proof(p1)
        mc.record_proof(p2)
        proofs = mc.drain_proofs()
        assert len(proofs) == 2
        assert proofs[0] is p1
        assert proofs[1] is p2

    def test_drain_clears_buffer(self):
        mc = MetricsCollector()
        mc.record_proof(self._make_mock_proof())
        mc.drain_proofs()
        assert mc.drain_proofs() == []

    def test_buffer_overflow_discards_oldest(self):
        mc = MetricsCollector()
        # Fill beyond max buffer
        for i in range(mc._MAX_PROOF_BUFFER + 50):
            mc.record_proof(self._make_mock_proof(i))
        proofs = mc.drain_proofs()
        assert len(proofs) == mc._MAX_PROOF_BUFFER
        # The oldest 50 should have been discarded; first kept should be 50
        assert proofs[0].token_count == 50


# ---------------------------------------------------------------------------
# MetricsCollector -- thread safety
# ---------------------------------------------------------------------------

class TestMetricsCollectorThreadSafety:
    """Verify correctness under concurrent access."""

    def test_concurrent_record_inference(self):
        """Multiple threads recording concurrently must not lose data."""
        mc = MetricsCollector()
        num_threads = 8
        records_per_thread = 1000
        barrier = threading.Barrier(num_threads)

        def worker():
            barrier.wait()
            for _ in range(records_per_thread):
                mc.record_inference(tokens=1, latency_ms=1.0)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = mc.get_snapshot()
        expected = num_threads * records_per_thread
        assert snap.total_tokens_processed == expected
        assert snap.total_requests == expected
        assert snap.total_latency_ms == pytest.approx(float(expected))

    def test_concurrent_record_and_snapshot(self):
        """Snapshots taken while recording should always be consistent."""
        mc = MetricsCollector()
        stop = threading.Event()
        snapshots: list[tuple[int, int]] = []

        def writer():
            while not stop.is_set():
                mc.record_inference(tokens=1, latency_ms=1.0)

        def reader():
            while not stop.is_set():
                snap = mc.get_snapshot()
                # tokens and requests should always match
                snapshots.append((snap.total_tokens_processed, snap.total_requests))

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)
        writer_thread.start()
        reader_thread.start()

        time.sleep(0.2)
        stop.set()
        writer_thread.join()
        reader_thread.join()

        # Every snapshot: tokens == requests (since we add 1 each time)
        for tokens, requests in snapshots:
            assert tokens == requests

    def test_concurrent_proof_record_and_drain(self):
        """Proof buffer should be consistent under concurrent access."""
        mc = MetricsCollector()
        total_proofs = 500
        drained: list[list] = []
        barrier = threading.Barrier(2)

        def writer():
            barrier.wait()
            for i in range(total_proofs):
                p = MagicMock()
                p.token_count = i
                mc.record_proof(p)

        def drainer():
            barrier.wait()
            for _ in range(50):
                batch = mc.drain_proofs()
                if batch:
                    drained.append(batch)
                time.sleep(0.001)

        w = threading.Thread(target=writer)
        d = threading.Thread(target=drainer)
        w.start()
        d.start()
        w.join()
        d.join()

        # Drain remaining
        remaining = mc.drain_proofs()
        if remaining:
            drained.append(remaining)

        # Total drained should equal total_proofs (no duplicates, no loss)
        total_drained = sum(len(batch) for batch in drained)
        assert total_drained == total_proofs


# ---------------------------------------------------------------------------
# MetricsCollector -- repr
# ---------------------------------------------------------------------------

class TestMetricsCollectorRepr:
    """Ensure repr is informative and does not crash."""

    def test_repr_contains_info(self):
        mc = MetricsCollector()
        mc.record_inference(tokens=42, latency_ms=100.0)
        r = repr(mc)
        assert "42" in r
        assert "1" in r  # 1 request
