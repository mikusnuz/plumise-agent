[English](README.md) | **한국어**

# Plumise Agent

> Plumise 블록체인을 위한 분산 LLM 추론 에이전트. 컴퓨팅 파워를 제공하고 PLM 보상을 획득하세요.

## 개요

Plumise Agent는 GPU 또는 CPU 컴퓨팅 파워를 Plumise 분산 추론 네트워크에 기여하는 Python 기반 노드입니다. 에이전트는 대규모 언어 모델의 transformer layer 일부를 로드하고, pipeline parallelism을 통해 추론 요청을 협력적으로 처리합니다.

- **Pipeline parallelism** -- 여러 에이전트가 모델을 transformer layer 단위로 분할합니다. 각 에이전트는 할당된 구간을 처리하고 gRPC를 통해 hidden states를 다음 노드로 전달합니다.
- **오라클 조정** -- Plumise Oracle이 각 에이전트의 하드웨어 프로필(RAM, VRAM, 장치 유형)을 기반으로 layer 범위를 할당하고, pipeline 토폴로지를 관리하며, 메트릭을 수집합니다.
- **온체인 통합** -- 에이전트는 stateful precompiled contract를 통해 Plumise 체인에 등록하고, 추론 증명을 제출하며, 처리한 토큰 수, 가동 시간, 지연 시간에 비례하여 PLM 보상을 획득합니다.
- **원커맨드 시작** -- 단일 `plumise-agent start` 명령으로 하드웨어를 자동 감지하고, Oracle에 등록하며, 할당된 모델 layer를 로드하고, 서비스를 시작합니다.

## 아키텍처

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

**주요 구성 요소:**

| 구성 요소 | 프로토콜 | 목적 |
|-----------|----------|---------|
| HTTP API (FastAPI) | REST | 추론 요청을 수락하고 health 및 status 엔드포인트 제공 |
| gRPC Server | gRPC | pipeline 노드 간 hidden-state tensor 전송 |
| Oracle Client | HTTP | 하드웨어 등록, layer 할당 수신, 메트릭 보고 |
| Chain Client | JSON-RPC | 온체인 등록, heartbeat, 증명 검증, 보상 청구 |

## 빠른 시작

### 사전 요구사항

- Python 3.10+
- 4 GB+ RAM (모델에 따라 다름)
- 온체인 등록을 위한 자금이 있는 Plumise 지갑 (private key)
- Plumise RPC 엔드포인트 및 Oracle API 접근 권한

### 소스 코드로부터 설치

```bash
# Clone 및 설치
git clone https://github.com/mikusnuz/plumise-agent.git
cd plumise-agent
pip install -e .

# 설정 (복사 및 편집)
cp .env.example .env
# .env를 편집하여 private key, RPC URL, Oracle URL 입력

# 에이전트 시작
plumise-agent start
```

양자화를 지원하는 GPU 가속 추론을 위해서는:

```bash
pip install -e ".[gpu]"
```

### 설정

모든 설정은 환경 변수 또는 `.env` 파일에서 로드됩니다. CLI 인수가 환경 변수를 오버라이드합니다.

| 변수 | 기본값 | 설명 |
|----------|---------|-------------|
| `PLUMISE_RPC_URL` | `http://localhost:26902` | Plumise 체인 JSON-RPC 엔드포인트 |
| `PLUMISE_CHAIN_ID` | `41956` | Chain ID |
| `PLUMISE_PRIVATE_KEY` | -- | 에이전트 지갑 private key (hex, `0x` 포함 또는 미포함) |
| `ORACLE_API_URL` | `http://localhost:3100` | Oracle API base URL |
| `MODEL_NAME` | `bigscience/bloom-560m` | HuggingFace 모델 식별자 |
| `DEVICE` | `auto` | 컴퓨팅 장치: `auto`, `cpu`, `cuda`, `cuda:0`, `mps` |
| `GRPC_PORT` | `50051` | gRPC 서버 포트 (노드 간 통신) |
| `API_PORT` | `31331` | HTTP API 포트 |
| `REPORT_INTERVAL` | `60` | 메트릭 보고 주기(초) |
| `LAYER_START` / `LAYER_END` | _(Oracle이 할당)_ | 수동 layer 범위 오버라이드 |
| `VERIFY_ON_CHAIN` | `false` | 0x20 precompile로 추론 증명 제출 |
| `CLAIM_THRESHOLD_WEI` | `1000000000000000000` | 이 임계값(wei)을 초과하면 보상 자동 청구 |

전체 목록은 [`.env.example`](.env.example)을 참조하세요.

## 작동 방식

1. **하드웨어 감지** -- 에이전트가 시스템 RAM, GPU VRAM, 사용 가능한 컴퓨팅 장치(CUDA, MPS, CPU)를 탐지합니다.
2. **오라클 등록** -- 하드웨어 사양과 공개 엔드포인트를 Oracle에 전송합니다. Oracle은 transformer layer 범위 할당으로 응답합니다.
3. **모델 로딩** -- 할당된 layer만 메모리에 유지합니다. 첫 번째 노드는 embedding layer를 유지하고, 마지막 노드는 최종 layer-norm 및 LM head를 유지합니다. 불필요한 파라미터는 삭제되고 가비지 컬렉션됩니다.
4. **gRPC + HTTP 서버 시작** -- gRPC 서버는 노드 간 hidden-state 전송을 위한 `ForwardPass` 및 `ForwardPassStream` RPC를 처리합니다. HTTP API는 사용자 대상 추론 요청을 수락합니다.
5. **온체인 등록** -- 에이전트는 `agentRegister` precompile(0x21)을 통해 등록하고 0x22를 통해 주기적으로 heartbeat를 전송합니다.
6. **추론 서비스** -- 들어오는 요청은 pipeline을 통해 흐릅니다: 첫 번째 노드가 임베딩 및 초기 layer를 처리하고, 중간 노드가 forward pass를 계속하며, 마지막 노드가 토큰을 샘플링합니다.
7. **메트릭 및 증명** -- 각 추론 후 에이전트는 keccak256 증명(`modelHash || inputHash || outputHash || agent`)을 생성하고 다음 Oracle 보고를 위해 버퍼링합니다.
8. **보상 청구** -- 누적 보상이 설정된 임계값을 초과하면 에이전트가 자동으로 RewardPool 컨트랙트의 `claimReward()`를 호출합니다.

## 프로젝트 구조

```
plumise-agent/
  src/plumise_agent/
    api/             # FastAPI HTTP 엔드포인트
    chain/
      auth.py        # 지갑 기반 인증 및 AgentRegistry 쿼리
      agent.py       # 온체인 등록, heartbeat, 증명 검증 (precompiles)
      config.py      # Pydantic-settings 설정 (env + .env)
      proof.py       # 추론 증명 생성 (keccak256)
      reporter.py    # Oracle로 비동기 메트릭 보고
      rewards.py     # RewardPool 추적 및 자동 청구
    cli/             # Click 기반 CLI 진입점
    grpc_/           # gRPC 서버 및 생성된 protobuf 코드
    model/
      engine.py      # Forward pass 실행 (full / first / middle / last 모드)
      layer_range.py # 위치 쿼리가 있는 layer 범위 dataclass
      loader.py      # layer 분할이 있는 HuggingFace 모델 로딩
    node/
      metrics.py     # 스레드 안전 추론 메트릭 수집기
      registry.py    # Oracle 등록 및 토폴로지 관리
    pipeline/        # Pipeline 토폴로지 및 라우팅
  proto/
    inference.proto  # 노드 간 통신을 위한 gRPC 서비스 정의
  contracts/         # 컨트랙트 ABI (AgentRegistry, RewardPool)
  pyproject.toml     # 패키지 메타데이터 및 종속성
```

## gRPC 서비스

[`proto/inference.proto`](proto/inference.proto)에 정의된 `InferencePipeline` gRPC 서비스는 다음을 제공합니다:

| RPC | 설명 |
|-----|-------------|
| `ForwardPass` | 이 노드의 layer를 통해 hidden states를 전송; 처리된 states 또는 최종 텍스트를 반환 |
| `ForwardPassStream` | autoregressive 토큰별 생성을 위한 스트리밍 변형 |
| `GetLayerInfo` | 이 노드가 현재 서비스 중인 모델 및 layer 쿼리 |
| `Ping` | 에이전트 메타데이터가 포함된 liveness 체크 |

## 지원 모델

HuggingFace `AutoModelForCausalLM` 호환 모델이 모두 지원됩니다. 로더는 자동으로 모델 아키텍처를 감지하고 올바른 embedding, transformer layers, layer-norm, LM head 모듈을 찾습니다.

테스트된 아키텍처:

| 모델 | Layer 수 | 대략적 크기 | 아키텍처 |
|-------|--------|-------------|--------------|
| `bigscience/bloom-560m` | 24 | ~1.5 GB | BLOOM |
| `meta-llama/Llama-3.1-8B` | 32 | ~16 GB | LLaMA |

## 체인 통합

| 구성 요소 | 주소 | 설명 |
|-----------|---------|-------------|
| **Plumise v2** | chainId `41956` | 에이전트 인식 precompile이 있는 Proof-of-Authority 체인 |
| `verifyInference` | `0x20` | 온체인 추론 증명 검증 |
| `agentRegister` | `0x21` | 이름, 모델 해시 및 기능으로 에이전트 등록 |
| `agentHeartbeat` | `0x22` | 주기적 liveness ping (msg.sender 기반) |
| AgentRegistry | `0xC9CF...043C` | 에이전트 쿼리를 위한 post-genesis 컨트랙트 |
| RewardPool | _(설정 가능)_ | 기여도 추적 및 PLM 보상 배분 |

## 네트워크 요구사항

- **Plumise RPC** -- Plumise v2 노드 접근 권한 (기본 포트 26902)
- **Oracle API** -- Plumise Oracle 접근 권한 (기본 포트 3100)
- **HuggingFace** -- 첫 실행 시 모델 다운로드를 위한 인터넷 접속 (이후 로컬 캐시)
- **에이전트 간 연결** -- gRPC 포트(기본 50051)가 다른 pipeline 노드에서 접근 가능해야 함

## 개발

```bash
# dev 종속성과 함께 설치
pip install -e ".[dev]"

# 테스트 실행
pytest

# proto로부터 gRPC stub 재생성
python -m grpc_tools.protoc \
  -I proto \
  --python_out=src/plumise_agent/grpc_/generated \
  --grpc_python_out=src/plumise_agent/grpc_/generated \
  proto/inference.proto
```

## 라이선스

MIT
