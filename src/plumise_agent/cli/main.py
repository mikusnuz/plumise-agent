"""CLI entry point for Plumise Agent.

Provides the ``plumise-agent`` command with subcommands:

- ``start``  -- launch the agent with GPU auto-detection and Oracle
                layer assignment.
- ``status`` -- query a running agent's health and reward summary.

Minimal required argument: ``--private-key`` (or ``PLUMISE_PRIVATE_KEY``
in ``.env``).  Everything else is auto-detected or loaded from the
environment.
"""

from __future__ import annotations

import asyncio
import logging
import os
import stat
import sys

import click
from dotenv import load_dotenv


@click.group()
@click.version_option(package_name="plumise-agent")
def cli() -> None:
    """Plumise Agent -- Distributed LLM inference on the Plumise chain."""


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--model", default=None, help="HuggingFace model name (default: from env or bloom-560m).")
@click.option("--private-key", default=None, help="Agent wallet private key (hex).")
@click.option("--rpc-url", default=None, help="Plumise chain RPC URL.")
@click.option("--oracle-url", default=None, help="Oracle API base URL.")
@click.option("--grpc-port", type=int, default=None, help="gRPC listen port (default: 50051).")
@click.option("--api-port", type=int, default=None, help="HTTP API listen port (default: 31331).")
@click.option("--device", default=None, help="Compute device: auto / cpu / cuda / cuda:0 / mps.")
@click.option("--layer-start", type=int, default=None, help="Manual layer start (inclusive).")
@click.option("--layer-end", type=int, default=None, help="Manual layer end (exclusive).")
@click.option("--node-endpoint", default=None, help="Public HTTP endpoint for this node.")
@click.option("--grpc-endpoint", default=None, help="Public gRPC endpoint for this node.")
@click.option("--hf-token", default=None, help="HuggingFace token for gated models.")
@click.option("--env-file", default=".env", show_default=True, help="Path to .env file.")
@click.option("-v", "--verbose", is_flag=True, help="Enable DEBUG logging.")
def start(
    model: str | None,
    private_key: str | None,
    rpc_url: str | None,
    oracle_url: str | None,
    grpc_port: int | None,
    api_port: int | None,
    device: str | None,
    layer_start: int | None,
    layer_end: int | None,
    node_endpoint: str | None,
    grpc_endpoint: str | None,
    hf_token: str | None,
    env_file: str,
    verbose: bool,
) -> None:
    """Start the Plumise Agent.

    GPU is auto-detected, layers are auto-assigned by the Oracle.
    The only required argument is ``--private-key`` (or set
    ``PLUMISE_PRIVATE_KEY`` in ``.env``).

    \b
    Example:
        plumise-agent start --private-key 0x...
        plumise-agent start --model meta-llama/Llama-3-8B --device cuda
    """
    # 1. Setup logging
    _setup_logging(verbose)

    # 2. Load .env (with permission check)
    if os.path.exists(env_file):
        mode = os.stat(env_file).st_mode
        if mode & (stat.S_IRGRP | stat.S_IROTH):
            click.echo(
                f"Warning: {env_file} is readable by group/others. "
                "Consider: chmod 600 " + env_file,
                err=True,
            )
    load_dotenv(env_file, override=False)

    # 3. Build config overrides from CLI flags
    overrides: dict = {}
    if model is not None:
        overrides["model_name"] = model
    if private_key is not None:
        overrides["plumise_private_key"] = private_key
    if rpc_url is not None:
        overrides["plumise_rpc_url"] = rpc_url
    if oracle_url is not None:
        overrides["oracle_api_url"] = oracle_url
    if grpc_port is not None:
        overrides["grpc_port"] = grpc_port
    if api_port is not None:
        overrides["api_port"] = api_port
    if device is not None:
        overrides["device"] = device
    if layer_start is not None:
        overrides["layer_start"] = layer_start
    if layer_end is not None:
        overrides["layer_end"] = layer_end
    if node_endpoint is not None:
        overrides["node_endpoint"] = node_endpoint
    if grpc_endpoint is not None:
        overrides["grpc_endpoint"] = grpc_endpoint
    if hf_token is not None:
        overrides["hf_token"] = hf_token

    # 4. Create config (env vars fill remaining fields)
    from plumise_agent.chain.config import AgentConfig

    try:
        config = AgentConfig(**overrides)
    except Exception as exc:
        click.echo(f"Configuration error: {exc}", err=True)
        sys.exit(1)

    # 5. Validate private key is present
    if not config.plumise_private_key:
        click.echo(
            "Error: private key is required. Provide --private-key or set "
            "PLUMISE_PRIVATE_KEY in .env",
            err=True,
        )
        sys.exit(1)

    # 6. Create and run agent
    from plumise_agent.node.node import PlumiseAgent

    agent = PlumiseAgent(config)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    agent.install_signal_handlers(loop)

    try:
        loop.run_until_complete(agent.start())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--api-url", default=None, help="Agent HTTP API URL (default: http://localhost:31331).")
@click.option("--env-file", default=".env", show_default=True, help="Path to .env file.")
def status(api_url: str | None, env_file: str) -> None:
    """Show a running agent's status and reward summary."""
    load_dotenv(env_file, override=False)

    url = api_url or os.getenv("AGENT_API_URL", "http://localhost:31331")

    asyncio.run(_fetch_status(url))


async def _fetch_status(base_url: str) -> None:
    """Fetch and print health + metrics from a running agent."""
    import aiohttp

    base_url = base_url.rstrip("/")
    timeout = aiohttp.ClientTimeout(total=10)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Health
            async with session.get(f"{base_url}/health") as resp:
                health = await resp.json()

            # Metrics
            metrics = None
            try:
                async with session.get(f"{base_url}/api/v1/metrics") as resp:
                    if resp.status == 200:
                        metrics = await resp.json()
            except Exception:
                pass

            # Pipeline
            pipeline = None
            try:
                async with session.get(f"{base_url}/api/v1/pipeline/status") as resp:
                    if resp.status == 200:
                        pipeline = await resp.json()
            except Exception:
                pass

    except aiohttp.ClientError as exc:
        click.echo(f"Cannot reach agent at {base_url}: {exc}", err=True)
        raise SystemExit(1)

    # Display
    click.echo("=== Plumise Agent Status ===")
    click.echo(f"  Status:  {health.get('status', 'unknown')}")
    click.echo(f"  Model:   {health.get('model', 'unknown')}")
    click.echo(f"  Mode:    {health.get('mode', 'unknown')}")
    click.echo(f"  Address: {health.get('address', 'unknown')}")
    click.echo(f"  Uptime:  {health.get('uptime', 0)}s")

    layers = health.get("layers")
    if layers:
        click.echo(f"  Layers:  [{layers['start']}, {layers['end']}) of {layers['total']}")

    if metrics:
        click.echo("\n=== Metrics ===")
        click.echo(f"  Requests:     {metrics.get('total_requests', 0)}")
        click.echo(f"  Tokens:       {metrics.get('total_tokens_processed', 0)}")
        click.echo(f"  Avg latency:  {metrics.get('avg_latency_ms', 0):.1f} ms")
        click.echo(f"  Throughput:   {metrics.get('tokens_per_second', 0):.1f} tok/s")
        click.echo(f"  Uptime:       {metrics.get('uptime_seconds', 0)}s")

    if pipeline and pipeline.get("assignment"):
        assign = pipeline["assignment"]
        click.echo("\n=== Pipeline ===")
        click.echo(f"  Full model: {'yes' if assign.get('is_full') else 'no'}")
        click.echo(f"  Assignment: [{assign['start']}, {assign['end']}) of {assign['total']}")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    """Configure root and package loggers."""
    level = logging.DEBUG if verbose else logging.INFO

    # Root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )

    # Suppress noisy third-party loggers
    for name in ("urllib3", "httpcore", "httpx", "asyncio", "aiohttp"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # Keep uvicorn quiet unless verbose
    if not verbose:
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Package entry point (``plumise-agent`` console script)."""
    cli()


if __name__ == "__main__":
    main()
