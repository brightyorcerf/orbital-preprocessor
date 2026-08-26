# ─────────────────────────────────────────────────────────────────────────────
# OSP OrbitLab Container — Orbital Scene Preprocessor
# Target: MOI-1A (100TOPS GPU / 4GB VRAM / OrbitLab Environment)
#
# Two targets, because the payload and the workstation need different things.
#
#   payload  (default)  inference only: an already-quantised ONNX graph, the
#                       preprocessor and the propagator. ~1.5 GB.
#   training            adds torch, ultralytics and sentence-transformers
#                       for train.py / satellite_export.py / the RAG layer.
#
# The single-manifest image this replaces was ~10 GB: it installed the whole
# training stack into the flight container, plus onnxruntime-gpu with CPU
# onnxruntime layered over it, plus scipy, which nothing imports. Shipping a
# 10 GB image to a 4 GB VRAM target was not a size problem so much as a
# statement that nobody had run it.
#
# Build:  docker build -t osp:latest .                       # payload
#         docker build --target training -t osp:training .   # full stack
#         docker build --build-arg ONNX_RUNTIME=onnxruntime-gpu -t osp:flight .
#
# Run:    docker run --gpus 1 --memory 4g --cpus 2 \
#           -v /data/input:/input -v /data/output:/output osp:flight
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim AS base

# libgl1-mesa-glx was removed in Debian 12, and python:3.10-slim now resolves to
# a trixie base, so the old package list failed at apt with "no installation
# candidate". opencv-python-headless needs libglib2.0-0 and nothing more.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PYTHONPATH is what makes `python inference/engine.py` resolve the sibling
# packages. Running a file by path puts *its* directory on sys.path, not the
# repo root, which fifteen modules used to work around with a hand-written
# sys.path.insert preamble each. One env var replaces all of them; outside a
# container, `pip install -e .` does the same job.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    OMP_NUM_THREADS=2


# ── Full stack: training, export, RAG ────────────────────────────────────────
# Declared before `payload` so that a bare `docker build` produces the payload.
FROM base AS training

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the RAG embedding model so the ground segment does not reach for the
# network on first use. Training-only: the payload has no RAG layer.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN mkdir -p /app/model/artifacts /app/rag/vector_store

COPY . .
CMD ["python", "train.py", "--help"]


# ── Flight payload: inference only. Last stage, so it is the default target. ──
FROM base AS payload

# CPU by default so the image builds and the reproduction check runs on any
# machine. The flight build passes onnxruntime-gpu.
ARG ONNX_RUNTIME=onnxruntime
RUN pip install --no-cache-dir "${ONNX_RUNTIME}>=1.17,<2.0"

COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

RUN mkdir -p /app/model/artifacts

COPY . .

VOLUME ["/input", "/output"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import onnxruntime; print('EP:', onnxruntime.get_available_providers())"

# Default: batch inference on /input → /output. Override to run a stage:
#   docker run ... osp:latest python tools/generate_briefs.py --help
#   docker run ... osp:training python train.py --quick
CMD ["python", "inference/engine.py", "--model", "/app/model/artifacts/osp_yolov8n_int8.onnx", "--tiles", "/input"]
