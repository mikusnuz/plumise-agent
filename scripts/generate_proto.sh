#!/bin/bash
# Generate Python gRPC stubs from .proto files.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PROTO_DIR="$PROJECT_DIR/proto"
OUT_DIR="$PROJECT_DIR/src/plumise_agent/grpc_/generated"

mkdir -p "$OUT_DIR"

python -m grpc_tools.protoc \
    -I "$PROTO_DIR" \
    --python_out="$OUT_DIR" \
    --grpc_python_out="$OUT_DIR" \
    --pyi_out="$OUT_DIR" \
    "$PROTO_DIR/inference.proto"

# Fix relative imports in generated files
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' 's/^import inference_pb2/from . import inference_pb2/' "$OUT_DIR/inference_pb2_grpc.py"
else
    sed -i 's/^import inference_pb2/from . import inference_pb2/' "$OUT_DIR/inference_pb2_grpc.py"
fi

echo "Proto stubs generated in $OUT_DIR"
