"""FastAPI HTTP server for the Plumise Agent.

Exposes REST endpoints consumed by the plumise-inference-api gateway
and by monitoring tools:

- ``GET  /health``               -- readiness check (public)
- ``POST /api/v1/generate``      -- text generation (auth required)
- ``GET  /api/v1/pipeline/status`` -- pipeline topology (auth required)
- ``GET  /api/v1/metrics``       -- inference metrics (auth required)
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
from typing import TYPE_CHECKING, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from plumise_agent.node.node import PlumiseAgent

logger = logging.getLogger(__name__)

# Optional auth scheme (auto_error=False so unauthenticated requests get a clear 401)
_bearer_scheme = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class GenerateParams(BaseModel):
    """Generation hyper-parameters."""

    max_new_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0)
    do_sample: bool = True


class GenerateRequest(BaseModel):
    """Text generation request body."""

    inputs: str
    parameters: GenerateParams | None = None
    stream: bool = False


class GenerateResponse(BaseModel):
    """Text generation response body."""

    generated_text: str
    num_tokens: int
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(agent: PlumiseAgent) -> FastAPI:
    """Create the FastAPI application wired to the given agent.

    The ``agent`` instance provides access to the inference engine,
    pipeline executor, layer range, topology, and metrics recording.

    Args:
        agent: A running (or soon-to-be-running) ``PlumiseAgent`` instance.
    """
    app = FastAPI(
        title="Plumise Agent API",
        version="0.1.0",
        description="Distributed LLM inference node for the Plumise chain.",
    )

    api_secret: str = agent.config.api_secret

    # ----- Auth dependency ------------------------------------------------

    async def require_auth(
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    ) -> None:
        """Verify Bearer token if API_SECRET is configured."""
        if not api_secret:
            return  # auth disabled
        if credentials is None:
            raise HTTPException(status_code=401, detail="Authorization header required")
        if not hmac.compare_digest(credentials.credentials, api_secret):
            raise HTTPException(status_code=403, detail="Invalid token")

    # ----- Health ---------------------------------------------------------

    @app.get("/health")
    async def health() -> dict:
        """Readiness check (public, no auth)."""
        ready = agent.engine is not None
        mode = "single"
        if agent.layer_range and not agent.layer_range.is_full:
            mode = "pipeline"

        result: dict = {
            "status": "ok" if ready else "loading",
            "model": agent.config.model_name,
            "mode": mode,
            "address": agent.auth.address,
            "uptime": agent.uptime_seconds,
        }

        if agent.layer_range:
            result["layers"] = {
                "start": agent.layer_range.start,
                "end": agent.layer_range.end,
                "total": agent.layer_range.total,
            }

        return result

    # ----- Generate -------------------------------------------------------

    @app.post(
        "/api/v1/generate",
        dependencies=[Depends(require_auth)],
    )
    async def generate(request: GenerateRequest):
        """Generate text from a prompt (auth required). Supports SSE streaming."""
        if agent.engine is None:
            raise HTTPException(status_code=503, detail="Model is still loading")

        params = request.parameters or GenerateParams()
        gen_params = {
            "max_new_tokens": params.max_new_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "repetition_penalty": params.repetition_penalty,
            "do_sample": params.do_sample,
        }

        # --- Streaming mode ---
        if request.stream and agent.executor is not None:
            async def sse_generator():
                num_tokens = 0
                start = time.time()
                try:
                    async for token in agent.executor.generate_stream(
                        prompt=request.inputs,
                        params=gen_params,
                    ):
                        num_tokens += 1
                        yield f"data: {json.dumps({'token': token})}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as exc:
                    logger.exception("Streaming inference failed")
                    yield f"data: {json.dumps({'error': str(exc)})}\n\n"
                finally:
                    latency_ms = (time.time() - start) * 1000
                    if num_tokens > 0:
                        agent.record_inference(
                            input_data=request.inputs,
                            output_data=f"[streamed {num_tokens} tokens]",
                            token_count=num_tokens,
                            latency_ms=latency_ms,
                        )

            return StreamingResponse(
                sse_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # --- Non-streaming mode ---
        start = time.time()

        try:
            loop = asyncio.get_running_loop()

            if agent.layer_range and agent.layer_range.is_full:
                generated_text, num_tokens = await loop.run_in_executor(
                    None,
                    lambda: agent.engine.forward_full(  # type: ignore[union-attr]
                        prompt=request.inputs,
                        **gen_params,
                    ),
                )
            elif agent.executor is not None:
                generated_text, num_tokens = await agent.executor.generate(
                    prompt=request.inputs,
                    params=gen_params,
                )
            else:
                generated_text, num_tokens = await loop.run_in_executor(
                    None,
                    lambda: agent.engine.forward_full(  # type: ignore[union-attr]
                        prompt=request.inputs,
                        **gen_params,
                    ),
                )

        except RuntimeError as exc:
            logger.warning("Inference runtime error: %s", exc)
            raise HTTPException(status_code=503, detail="Model unavailable")
        except Exception:
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail="Internal inference error")

        latency_ms = (time.time() - start) * 1000

        agent.record_inference(
            input_data=request.inputs,
            output_data=generated_text,
            token_count=num_tokens,
            latency_ms=latency_ms,
        )

        return GenerateResponse(
            generated_text=generated_text,
            num_tokens=num_tokens,
            latency_ms=round(latency_ms, 2),
        )

    # ----- Pipeline Status ------------------------------------------------

    @app.get("/api/v1/pipeline/status", dependencies=[Depends(require_auth)])
    async def pipeline_status() -> dict:
        """Return the current pipeline topology and this node's position."""
        result: dict = {
            "model": agent.config.model_name,
            "address": agent.auth.address,
        }

        if agent.layer_range:
            result["assignment"] = {
                "start": agent.layer_range.start,
                "end": agent.layer_range.end,
                "total": agent.layer_range.total,
                "is_full": agent.layer_range.is_full,
            }

        if agent.topology:
            try:
                result["topology"] = agent.topology.to_dict()
            except Exception:
                result["topology"] = None
        else:
            result["topology"] = None

        return result

    # ----- Metrics --------------------------------------------------------

    @app.get("/api/v1/metrics", dependencies=[Depends(require_auth)])
    async def metrics() -> dict:
        """Return a snapshot of inference metrics (auth required)."""
        snapshot = agent.metrics.get_snapshot()
        return snapshot.to_dict()

    if api_secret:
        logger.info("API authentication enabled (Bearer token)")
    else:
        logger.warning("API authentication DISABLED (no API_SECRET configured)")

    return app


# ---------------------------------------------------------------------------
# Server runner (blocking, for daemon thread)
# ---------------------------------------------------------------------------


def run_api_server(app: FastAPI, host: str = "0.0.0.0", port: int = 31331) -> None:
    """Run the API server via uvicorn (blocking).

    Intended to be called inside a daemon thread from
    ``PlumiseAgent._start_api_server()``.

    Args:
        app: FastAPI application instance.
        host: Bind address.
        port: Bind port.
    """
    logger.info("Starting API server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning")
