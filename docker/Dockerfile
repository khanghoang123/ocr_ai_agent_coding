# ============================================================
# Dockerfile — Vietnamese Handwritten OCR API
# Multi-stage build for production FastAPI service
# ============================================================

# ----- Stage 1: Base Python environment -----
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ----- Stage 2: Install Python dependencies -----
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ----- Stage 3: Application -----
FROM deps AS app

# Copy source code
COPY src/ ./src/
COPY app/ ./app/
COPY models/model_registry.yaml ./models/model_registry.yaml

# Mount point for model weights (inject via volume or Dockerfile build arg)
# models/experiment_B/best_model.pth — expected at runtime
RUN mkdir -p models/experiment_B models/baseline

# Install package in editable mode
COPY pyproject.toml .
RUN pip install -e . --no-deps

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:8000/health'); exit(0 if r.ok else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
