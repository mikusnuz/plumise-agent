"""Node registration with the Plumise Oracle and layer assignment.

Handles the initial handshake with the Oracle to obtain a layer range
assignment, reports readiness after model loading completes, and
fetches the current pipeline topology for inter-node routing.
"""

from __future__ import annotations

import json
import logging
import socket
import time
from typing import TYPE_CHECKING

import aiohttp

from plumise_agent.chain.auth import PlumiseAuth
from plumise_agent.chain.config import AgentConfig
from plumise_agent.model.layer_range import LayerRange

if TYPE_CHECKING:
    from plumise_agent.pipeline.topology import PipelineTopology

logger = logging.getLogger(__name__)


class OracleRegistry:
    """Handles node registration with the Oracle and layer assignment.

    The registration flow:
    1. ``register()`` -- POST hardware capabilities, receive layer range.
    2. (caller loads model layers)
    3. ``report_ready()`` -- notify Oracle that this node is serving.
    4. ``get_topology()`` -- fetch the full pipeline layout for routing.

    Args:
        auth: Authenticated agent identity (provides address and signing).
        config: Agent configuration (provides URLs, model name, ports).
    """

    def __init__(self, auth: PlumiseAuth, config: AgentConfig) -> None:
        self.auth = auth
        self.config = config

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def register(self) -> LayerRange | None:
        """Register this node with the Oracle and receive a layer assignment.

        Sends hardware capabilities (RAM, VRAM, device) and public
        endpoints to the Oracle.  On success the Oracle responds with
        the layer slice this node should serve.

        Returns:
            A ``LayerRange`` on success, or ``None`` if registration failed.
        """
        ram_mb, vram_mb, device = self._detect_hardware()

        timestamp = int(time.time())
        grpc_endpoint = (
            self.config.grpc_endpoint
            or f"http://{self._get_local_ip()}:{self.config.grpc_port}"
        )
        http_endpoint = (
            self.config.node_endpoint
            or f"http://{self._get_local_ip()}:{self.config.api_port}"
        )

        # Sign JSON message matching Oracle's verifyRegistrationSignature format
        sign_data = {
            "address": self.auth.address,
            "grpcEndpoint": grpc_endpoint,
            "httpEndpoint": http_endpoint,
            "model": self.config.model_name,
            "ramMb": ram_mb,
            "device": device,
            "vramMb": vram_mb,
            "timestamp": timestamp,
        }
        message = json.dumps(sign_data, separators=(",", ":"))
        signature = self.auth.sign_message(message)

        payload = {
            **sign_data,
            "signature": signature,
        }

        url = f"{self.config.oracle_api_url}/api/v1/pipeline/register"
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status in (200, 201):
                        data = await resp.json()
                        if not data.get("success"):
                            logger.error(
                                "Oracle registration rejected: %s",
                                data.get("message", "unknown"),
                            )
                            return None
                        layer_range = LayerRange(
                            start=data["layerStart"],
                            end=data["layerEnd"],
                            total=data["totalLayers"],
                        )
                        logger.info(
                            "Registered with Oracle: layers [%d, %d) of %d",
                            layer_range.start,
                            layer_range.end,
                            layer_range.total,
                        )
                        return layer_range

                    text = await resp.text()
                    logger.error(
                        "Oracle registration failed (%d): %s",
                        resp.status,
                        text[:500],
                    )
                    return None
        except Exception as exc:
            logger.error("Oracle registration request failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Readiness report
    # ------------------------------------------------------------------

    async def report_ready(self) -> bool:
        """Notify the Oracle that this node has loaded its layers and is ready.

        Returns:
            ``True`` on success, ``False`` otherwise.
        """
        timestamp = int(time.time())
        # Sign JSON message matching Oracle's verifyReadySignature format
        sign_data = {
            "address": self.auth.address,
            "model": self.config.model_name,
            "timestamp": timestamp,
        }
        message = json.dumps(sign_data, separators=(",", ":"))
        signature = self.auth.sign_message(message)

        payload = {
            **sign_data,
            "signature": signature,
        }

        url = f"{self.config.oracle_api_url}/api/v1/pipeline/ready"
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status in (200, 201):
                        logger.info("Reported ready to Oracle")
                        return True
                    text = await resp.text()
                    logger.warning(
                        "Oracle ready report failed (%d): %s",
                        resp.status,
                        text[:300],
                    )
                    return False
        except Exception as exc:
            logger.warning("Oracle ready request failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    async def get_topology(self) -> PipelineTopology | None:
        """Fetch the current pipeline topology from the Oracle.

        Returns:
            A ``PipelineTopology`` instance, or ``None`` on failure.
        """
        from plumise_agent.pipeline.topology import PipelineTopology

        try:
            return await PipelineTopology.fetch(
                self.config.oracle_api_url, self.config.model_name
            )
        except Exception as exc:
            logger.warning("Failed to fetch pipeline topology: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Hardware detection
    # ------------------------------------------------------------------

    def _detect_hardware(self) -> tuple[int, int, str]:
        """Detect RAM, VRAM, and compute device.

        Returns:
            Tuple of (ram_mb, vram_mb, device_string).
        """
        import os

        # RAM - prefer env override, then available memory, then total
        env_ram = os.environ.get("RAM_MB")
        if env_ram:
            ram_mb = int(env_ram)
            logger.info("Using RAM_MB from env: %d MB", ram_mb)
        else:
            ram_mb = 16384  # conservative default
            try:
                import psutil  # type: ignore[import-untyped]

                ram_mb = psutil.virtual_memory().available // (1024 * 1024)
                logger.info("Detected available RAM: %d MB", ram_mb)
            except ImportError:
                logger.debug("psutil not installed; using default RAM estimate")

        # Device and VRAM
        device = self.config.device
        vram_mb = 0
        try:
            import torch

            if device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"

            if device.startswith("cuda") and torch.cuda.is_available():
                dev_idx = 0
                if ":" in device:
                    dev_idx = int(device.split(":")[1])
                props = torch.cuda.get_device_properties(dev_idx)
                vram_mb = props.total_mem // (1024 * 1024)
        except ImportError:
            logger.debug("torch not available for device detection")
            if device == "auto":
                device = "cpu"

        return ram_mb, vram_mb, device

    # ------------------------------------------------------------------
    # Network utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _get_local_ip() -> str:
        """Best-effort local IP detection using a UDP probe."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(1)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"
