# ─────────────────────────────────────────────────────────────────────────────
# OSP Command Centre — public dashboard image
#
# This is NOT the mission container. The root Dockerfile targets the OrbitLab
# GPU payload and installs the full stack (torch, ultralytics, onnxruntime-gpu)
# because it has to train, export and infer. This image only has to serve the
# committed brief corpus in data/briefs/, so it installs the dashboard subset
# and stays small enough for a free Space or a Cloud Run instance.
#
# Build:  docker build -f deploy/Dockerfile -t osp-dashboard .
# Run:    docker run --rm -p 8501:8501 osp-dashboard
# Then:   http://localhost:8501
#
# Hugging Face Spaces (Docker SDK) uses port 7860; PORT is honoured below so
# the same image serves both without an edit.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# opencv-python-headless still needs libglib2.0-0. libgl1-mesa-glx does not
# exist on a trixie base, which is what broke the root Dockerfile's build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY deploy/requirements-dashboard.txt .
RUN pip install --no-cache-dir -r requirements-dashboard.txt

COPY . .

# No API key is baked in. ORION reads GEMINI_API_KEY if the host sets one, and
# otherwise takes a key the visitor types into the sidebar for their own
# session. A public deployment therefore cannot spend the owner's quota.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    PORT=8501

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",8501)}/_stcore/health')"

CMD ["sh", "-c", "streamlit run ground/dashboard.py --server.port=${PORT} --server.address=0.0.0.0"]
