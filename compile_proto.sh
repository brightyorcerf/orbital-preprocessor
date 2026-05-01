#!/usr/bin/env bash
# compile_proto.sh
# ─────────────────────────────────────────────────────────────────────────────
# Install protobuf-compiler and generate Python classes from osp.proto.
#
# Usage:
#   chmod +x compile_proto.sh
#   ./compile_proto.sh
#
# Output:
#   inference/osp_pb2.py   ← generated Python classes (do not edit manually)
#
# Requirements:
#   - Python 3.10+
#   - apt (Debian/Ubuntu) OR brew (macOS) OR manual protoc install
#
# OrbitLab note:
#   Run this once during container build (see Dockerfile).
#   The compiled osp_pb2.py is committed to the repo so protoc is not
#   needed at runtime on MOI-1A.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROTO_FILE="osp.proto"
OUT_DIR="inference"
PROTO_DIR="."

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  OSP Proto Compiler"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. Ensure we're in the osp/ project root ──────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f "$PROTO_FILE" ]]; then
    echo "ERROR: $PROTO_FILE not found in $(pwd)"
    echo "Run this script from the osp/ project root."
    exit 1
fi

# ── 2. Install protobuf Python runtime ───────────────────────────────────────
echo ""
echo "Step 1/3: Installing protobuf Python runtime ..."
pip install --quiet "protobuf>=4.25.0,<6.0.0"
python -c "import google.protobuf; print(f'  protobuf {google.protobuf.__version__} OK')"

# ── 3. Install/locate protoc compiler ────────────────────────────────────────
echo ""
echo "Step 2/3: Locating/installing protoc compiler ..."

PROTOC=""

# Try system protoc first
if command -v protoc &>/dev/null; then
    PROTOC="protoc"
    echo "  Found system protoc: $(protoc --version)"

# Try apt (Debian/Ubuntu/OrbitLab Docker)
elif command -v apt-get &>/dev/null; then
    echo "  Installing via apt-get ..."
    apt-get update -qq && apt-get install -y -qq protobuf-compiler
    PROTOC="protoc"
    echo "  protoc installed: $(protoc --version)"

# Try brew (macOS dev machine)
elif command -v brew &>/dev/null; then
    echo "  Installing via Homebrew ..."
    brew install protobuf
    PROTOC="protoc"

# Try grpc_tools as fallback (pure Python, no system install needed)
else
    echo "  protoc not found via apt/brew — trying grpc_tools Python fallback ..."
    pip install --quiet grpcio-tools
    PROTOC="python -m grpc_tools.protoc"
    echo "  Using grpc_tools.protoc"
fi

# ── 4. Compile proto → Python ─────────────────────────────────────────────────
echo ""
echo "Step 3/3: Compiling $PROTO_FILE → ${OUT_DIR}/osp_pb2.py ..."

mkdir -p "$OUT_DIR"

if [[ "$PROTOC" == "python -m grpc_tools.protoc" ]]; then
    # grpc_tools signature differs slightly
    python -m grpc_tools.protoc \
        -I"$PROTO_DIR" \
        --python_out="$OUT_DIR" \
        "$PROTO_FILE"
else
    $PROTOC \
        -I"$PROTO_DIR" \
        --python_out="$OUT_DIR" \
        "$PROTO_FILE"
fi

# Verify output was created
if [[ ! -f "${OUT_DIR}/osp_pb2.py" ]]; then
    echo "ERROR: Compilation failed — osp_pb2.py not created."
    exit 1
fi

LINES=$(wc -l < "${OUT_DIR}/osp_pb2.py")
echo "  Generated: ${OUT_DIR}/osp_pb2.py (${LINES} lines)"

# ── 5. Smoke-test generated classes ──────────────────────────────────────────
echo ""
echo "Smoke-testing generated classes ..."
python - <<'PYEOF'
import sys
sys.path.insert(0, "inference")

from osp_pb2 import SceneBrief, Anomaly, TileFootprint, AnomalyType

# Build a minimal SceneBrief and round-trip it
brief = SceneBrief(
    scene_id      = "TEST-001",
    timestamp_utc = "2026-04-24T09:00:00Z",
    cloud_cover   = 0.05,
    inference_ms  = 312.0,
    model_version = "osp-yolov8n-int8-v1",
    tile_footprint= TileFootprint(lat_min=8.0, lat_max=9.0, lon_min=77.0, lon_max=78.0),
    anomalies=[
        Anomaly(
            type=AnomalyType.SHIP,
            lat=8.412, lon=77.821,
            confidence=0.87,
            bbox_px=[300, 200, 360, 240],
        )
    ],
    compression_ratio=85000,
)

# Serialize → binary
binary = brief.SerializeToString()

# Deserialize back
brief2 = SceneBrief()
brief2.ParseFromString(binary)

assert brief2.scene_id == "TEST-001"
assert brief2.anomalies[0].type == AnomalyType.SHIP
assert abs(brief2.anomalies[0].lat - 8.412) < 1e-9
assert list(brief2.anomalies[0].bbox_px) == [300, 200, 360, 240]

print(f"  Round-trip OK: {len(binary)} bytes binary | {len(brief2.scene_id)} chars scene_id")
print(f"  AnomalyType.SHIP = {AnomalyType.SHIP}")
PYEOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✓ Proto compilation complete"
echo "  Output: ${OUT_DIR}/osp_pb2.py"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo "  python inference/serialization_utils.py   # compression report"
echo "  python test_pipeline.py                    # full pipeline verification"