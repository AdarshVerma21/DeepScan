# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS builder
ENV PYTHONUNBUFFERED=1 HF_HOME=/opt/huggingface
WORKDIR /build
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt --target /install
COPY app ./app
RUN PYTHONPATH=/install:/build python -c "from app.models import load_detectors; load_detectors()"

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 HF_HOME=/opt/huggingface PORT=7860
RUN apt-get update && \
    apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /install /usr/local/lib/python3.11/site-packages
COPY --from=builder /install/bin /usr/local/bin
COPY --from=builder /opt/huggingface /opt/huggingface
COPY . .
RUN useradd --create-home --uid 10001 deepscan && mkdir -p /app/reports && chown -R deepscan:deepscan /app /opt/huggingface
USER deepscan
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.getenv(\"PORT\", \"7860\")}/api/health')"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
