"""Metrics reporter to the Plumise Oracle API.

Collects inference metrics from the local ``MetricsCollector`` and
periodically sends signed reports to the oracle endpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING

import aiohttp

from plumise_agent.chain.auth import PlumiseAuth

if TYPE_CHECKING:
    from plumise_agent.node.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class OracleReporter:
    """Periodically report signed metrics to the Plumise oracle.

    Args:
        auth: Authenticated agent identity.
        oracle_url: Base URL of the oracle API.
        interval: Reporting interval in seconds.
    """

    def __init__(
        self,
        auth: PlumiseAuth,
        oracle_url: str,
        interval: int = 60,
    ) -> None:
        self.auth = auth
        self.oracle_url = oracle_url.rstrip("/")
        self.interval = interval

        self._running = False
        self._task: asyncio.Task | None = None
        self._consecutive_failures: int = 0
        self._max_failures: int = 10

        logger.info(
            "OracleReporter configured: url=%s interval=%ds",
            self.oracle_url,
            self.interval,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, metrics_collector: MetricsCollector) -> None:
        """Start the background reporting loop."""
        if self._running:
            logger.warning("OracleReporter is already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._report_loop(metrics_collector))
        logger.info("OracleReporter started")

    async def stop(self) -> None:
        """Stop the background reporting loop and send a final report."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("OracleReporter stopped")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _report_loop(self, metrics_collector: MetricsCollector) -> None:
        """Main loop: sleep -> snapshot -> send."""
        while self._running:
            try:
                await asyncio.sleep(self.interval)
                if not self._running:
                    break

                snapshot = metrics_collector.get_snapshot()
                proofs = metrics_collector.drain_proofs()
                success = await self._send_report(snapshot, proofs=proofs)

                if success:
                    self._consecutive_failures = 0
                else:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= self._max_failures:
                        logger.error(
                            "Reached %d consecutive report failures; "
                            "will keep trying but oracle may be unreachable",
                            self._consecutive_failures,
                        )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Unexpected error in report loop")
                self._consecutive_failures += 1

    async def _send_report(
        self,
        metrics: object,
        proofs: list | None = None,
    ) -> bool:
        """Build, sign and POST a metrics report.

        Args:
            metrics: An ``InferenceMetrics`` snapshot.
            proofs: Optional list of ``ProofData`` instances to include.

        Returns ``True`` on success, ``False`` on any failure.
        """
        ts = int(time.time())

        # Sign only the fields that the Oracle verifies
        sign_data = {
            "agent": self.auth.address,
            "processed_tokens": metrics.total_tokens_processed,
            "timestamp": ts,
        }
        signature = self.auth.sign_payload(sign_data)

        # Build flat body matching Oracle's ReportMetricsDto property names
        body: dict = {
            "wallet": self.auth.address,
            "tokensProcessed": metrics.total_tokens_processed,
            "avgLatencyMs": round(metrics.avg_latency_ms, 2),
            "uptimeSeconds": metrics.uptime_seconds,
            "requestCount": metrics.total_requests,
            "timestamp": ts,
            "signature": signature,
        }

        # Attach inference proofs if available
        if proofs:
            body["proofs"] = [p.to_dict() for p in proofs]

        url = f"{self.oracle_url}/api/v1/metrics/report"

        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=body) as resp:
                    if resp.status < 300:
                        # Validate oracle response structure
                        try:
                            resp_data = await resp.json()
                            if not isinstance(resp_data, dict):
                                logger.warning(
                                    "Oracle returned non-dict response: %s",
                                    str(resp_data)[:200],
                                )
                                return False
                            if resp_data.get("status") == "error":
                                logger.warning(
                                    "Oracle rejected report: %s",
                                    resp_data.get("message", "unknown error")[:200],
                                )
                                return False
                        except (json.JSONDecodeError, aiohttp.ContentTypeError):
                            # Accept non-JSON 2xx responses (some oracles return plain text)
                            pass

                        logger.debug(
                            "Report sent successfully: tokens=%d uptime=%ds",
                            body["tokensProcessed"],
                            body["uptimeSeconds"],
                        )
                        return True
                    else:
                        text = await resp.text()
                        logger.warning(
                            "Oracle returned HTTP %d: %s", resp.status, text[:200]
                        )
                        return False
        except asyncio.TimeoutError:
            logger.warning("Report request timed out")
            return False
        except aiohttp.ClientError as exc:
            logger.warning("Report request failed: %s", exc)
            return False

    async def send_final_report(self, metrics_collector: MetricsCollector) -> None:
        """Send one last report on shutdown."""
        try:
            snapshot = metrics_collector.get_snapshot()
            proofs = metrics_collector.drain_proofs()
            await self._send_report(snapshot, proofs=proofs)
            logger.info("Final report sent")
        except Exception:
            logger.exception("Failed to send final report")
