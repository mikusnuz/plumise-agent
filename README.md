**English** | [한국어](README.ko.md)

# Plumise Agent

> Distributed LLM inference agent for the Plumise blockchain. Provide compute power, earn PLM rewards.

## Overview

Plumise Agent is a Python-based node that contributes GPU or CPU compute to the Plumise decentralized inference network. Agents load a portion of a large language model's transformer layers and collaboratively process inference requests through pipeline parallelism.

- **Pipeline parallelism** -- Multiple agents split a model across its transformer layers. Each agent processes its assigned slice and forwards hidden states to the next via gRPC.
- **Oracle-coordinated** -- The Plumise Oracle assigns layer ranges based on each agent's hardware profile (RAM, VRAM, device type), manages the pipeline topology, and aggregates metrics.
- **On-chain integration** -- Agents register on the Plumise chain through stateful precompiled contracts, submit inference proofs, and earn PLM rewards proportional to tokens processed, uptime, and latency.
- **One-command startup** -- A single `plumise-agent start` auto-detects your hardware, registers with the Oracle, loads the assigned model layers, and begins serving.

## Architecture

```
                         Plumise Inference Pipeline

  User Request
       |
       v
  InferenceAPI (HTTP)
       |
       v
  +------------+    gRPC     +------------+    gRPC     +------------+
  |  Agent A   | ----------> |  Agent B   | ----------> |  Agent C   |
  | layers 0-7 |  hidden     | layers 8-15|  hidden     | layers 16+ |
  | (embed +   |  states     | (middle)   |  states     | (+ norm +  |
  |  first)    |             |            |             |  lm_head)  |
  +------------+             +------------+             +-----+------+
                                                              |
                                                              v
                                                        Generated Text
```

**Key components:**

| Component | Protocol | Purpose |
|-----------|----------|---------|
| HTTP API (FastAPI) | REST | Accepts inference requests, exposes health and status endpoints |
| gRPC Server | gRPC | Transfers hidden-state tensors between pipeline nodes |
| Oracle Client | HTTP | Registers hardware, receives layer assignments, reports metrics |
| Chain Client | JSON-RPC | On-chain registration, heartbeat, proof verification, reward claims |

## Quick Start

### Prerequisites

- Python 3.10+
- 4 GB+ RAM (varies by model)
- A funded Plumise wallet (private key) for on-chain registration
- Access to a Plumise RPC endpoint and the Oracle API

### From Source

```bash
# Clone and install
git clone https://github.com/mikusnuz/plumise-agent.git
cd plumise-agent
pip install -e .

# Configure (copy and edit)
cp .env.example .env
# Edit .env with your private key, RPC URL, and Oracle URL

# Start the agent
plumise-agent start
```

For GPU-accelerated inference with quantization support:

```bash
pip install -e ".[gpu]"
```

### Configuration

All settings are loaded from environment variables or a `.env` file. CLI arguments override environment values.

| Variable | Default | Description |
|----------|---------|-------------|
| `PLUMISE_RPC_URL` | `http://localhost:26902` | Plumise chain JSON-RPC endpoint |
| `PLUMISE_CHAIN_ID` | `41956` | Chain ID |
| `PLUMISE_PRIVATE_KEY` | -- | Agent wallet private key (hex, with or without `0x`) |
| `ORACLE_API_URL` | `http://localhost:3100` | Oracle API base URL |
| `MODEL_NAME` | `bigscience/bloom-560m` | HuggingFace model identifier |
| `DEVICE` | `auto` | Compute device: `auto`, `cpu`, `cuda`, `cuda:0`, `mps` |
| `GRPC_PORT` | `50051` | gRPC server port (inter-node communication) |
| `API_PORT` | `31331` | HTTP API port |
| `REPORT_INTERVAL` | `60` | Metrics reporting interval in seconds |
| `LAYER_START` / `LAYER_END` | _(Oracle assigns)_ | Manual layer range override |
| `VERIFY_ON_CHAIN` | `false` | Submit inference proofs to the 0x20 precompile |
| `CLAIM_THRESHOLD_WEI` | `1000000000000000000` | Auto-claim rewards above this threshold (wei) |

See [`.env.example`](.env.example) for the full list.

## How It Works

1. **Hardware detection** -- The agent probes system RAM, GPU VRAM, and available compute devices (CUDA, MPS, CPU).
2. **Oracle registration** -- Hardware capabilities and public endpoints are sent to the Oracle. The Oracle responds with a transformer layer range assignment.
3. **Model loading** -- Only the assigned layers are kept in memory. The first node retains the embedding layer; the last node retains the final layer-norm and LM head. Unneeded parameters are discarded and garbage-collected.
4. **gRPC + HTTP servers start** -- The gRPC server handles `ForwardPass` and `ForwardPassStream` RPCs for inter-node hidden-state transfer. The HTTP API accepts user-facing inference requests.
5. **On-chain registration** -- The agent registers via the `agentRegister` precompile (0x21) and begins periodic heartbeats via 0x22.
6. **Inference serving** -- Incoming requests flow through the pipeline: first node embeds and processes initial layers, middle nodes continue the forward pass, and the last node samples tokens.
7. **Metrics and proofs** -- After each inference, the agent generates a keccak256 proof (`modelHash || inputHash || outputHash || agent`) and buffers it for the next Oracle report.
8. **Reward claiming** -- When accumulated rewards exceed the configured threshold, the agent automatically calls `claimReward()` on the RewardPool contract.

## Project Structure

```
plumise-agent/
  src/plumise_agent/
    api/             # FastAPI HTTP endpoints
    chain/
      auth.py        # Wallet-based authentication and AgentRegistry queries
      agent.py       # On-chain registration, heartbeat, proof verification (precompiles)
      config.py      # Pydantic-settings configuration (env + .env)
      proof.py       # Inference proof generation (keccak256)
      reporter.py    # Async metrics reporter to Oracle
      rewards.py     # RewardPool tracking and auto-claim
    cli/             # Click-based CLI entry point
    grpc_/           # gRPC server and generated protobuf code
    model/
      engine.py      # Forward pass execution (full / first / middle / last modes)
      layer_range.py # Layer range dataclass with positional queries
      loader.py      # HuggingFace model loading with layer splitting
    node/
      metrics.py     # Thread-safe inference metrics collector
      registry.py    # Oracle registration and topology management
    pipeline/        # Pipeline topology and routing
  proto/
    inference.proto  # gRPC service definition for inter-node communication
  contracts/         # Contract ABIs (AgentRegistry, RewardPool)
  pyproject.toml     # Package metadata and dependencies
```

## gRPC Service

The `InferencePipeline` gRPC service (defined in [`proto/inference.proto`](proto/inference.proto)) provides:

| RPC | Description |
|-----|-------------|
| `ForwardPass` | Send hidden states through this node's layers; returns processed states or final text |
| `ForwardPassStream` | Streaming variant for autoregressive token-by-token generation |
| `GetLayerInfo` | Query which model and layers this node is currently serving |
| `Ping` | Liveness check with agent metadata |

## Supported Models

Any HuggingFace `AutoModelForCausalLM`-compatible model is supported. The loader automatically detects the model architecture and locates the correct embedding, transformer layers, layer-norm, and LM head modules.

Tested architectures:

| Model | Layers | Approx. Size | Architecture |
|-------|--------|-------------|--------------|
| `bigscience/bloom-560m` | 24 | ~1.5 GB | BLOOM |
| `meta-llama/Llama-3.1-8B` | 32 | ~16 GB | LLaMA |

## Chain Integration

| Component | Address | Description |
|-----------|---------|-------------|
| **Plumise v2** | chainId `41956` | Proof-of-Authority chain with agent-aware precompiles |
| `verifyInference` | `0x20` | On-chain inference proof verification |
| `agentRegister` | `0x21` | Agent registration with name, model hash, and capabilities |
| `agentHeartbeat` | `0x22` | Periodic liveness ping (msg.sender-based) |
| AgentRegistry | `0xC9CF...043C` | Post-genesis contract for agent queries |
| RewardPool | _(configurable)_ | Tracks contributions and distributes PLM rewards |

## Network Requirements

- **Plumise RPC** -- Access to a Plumise v2 node (default port 26902)
- **Oracle API** -- Access to the Plumise Oracle (default port 3100)
- **HuggingFace** -- Internet access for model downloads on first run (cached locally after)
- **Inter-agent connectivity** -- gRPC port (default 50051) must be reachable by other pipeline nodes

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Regenerate gRPC stubs from proto
python -m grpc_tools.protoc \
  -I proto \
  --python_out=src/plumise_agent/grpc_/generated \
  --grpc_python_out=src/plumise_agent/grpc_/generated \
  proto/inference.proto
```

## License

MIT
