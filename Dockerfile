# ─────────────────────────────────────────────────────────────────────────────
# OSP OrbitLab Container — Orbital Scene Preprocessor
# Target: MOI-1A (100TOPS GPU / 4GB VRAM / OrbitLab Environment)
#
# Build:  docker build -t osp:latest .
# Run:    docker run --gpus 1 --memory 4g --cpus 2 \
#           -v /data/input:/input \
#           -v /data/output:/output \
#           osp:latest
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir \
    onnxruntime-gpu \
    numpy \
    opencv-python-headless \
    scipy \
    && pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN mkdir -p /app/model/artifacts /app/rag/vector_store

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Mount points (OrbitLab spec) ──────────────────────────────────────────────
VOLUME ["/input", "/output"]

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=2

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import onnxruntime; print('EP:', onnxruntime.get_available_providers())"

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Default: run batch inference on /input → /output
# Override CMD to run specific steps:
#   docker run ... osp:latest python train.py --quick
#   docker run ... osp:latest python model/export.py --weights best.pt
#   docker run ... osp:latest streamlit run ground/dashboard.py

CMD ["python", "inference/engine.py", "--model", "/app/model/artifacts/osp_yolov8n_int8.onnx", "--tiles", "/input"]