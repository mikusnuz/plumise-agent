# ──────────────────────────────────────────────
#  plumise-agent  |  Multi-stage Docker build
#  GPU: mount NVIDIA runtime via docker-compose gpu profile
# ──────────────────────────────────────────────

# ── Stage 1: Builder ─────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml setup.py ./
COPY src/ ./src/
COPY contracts/ ./contracts/
COPY proto/ ./proto/
COPY scripts/ ./scripts/

RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir grpcio-tools && \
    bash scripts/generate_proto.sh && \
    pip install --no-cache-dir -e ".[dev]"

# ── Stage 2: Runtime ─────────────────────────
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

VOLUME ["/root/.cache/huggingface"]

EXPOSE 31331 50051

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${API_PORT:-31331}/health || exit 1

ENTRYPOINT ["plumise-agent"]
CMD ["start"]
