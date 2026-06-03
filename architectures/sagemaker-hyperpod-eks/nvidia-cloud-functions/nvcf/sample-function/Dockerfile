# ---------------------------------------------------------------------------
# Dockerfile for NVCF Echo Function
#
# This builds a minimal container compatible with NVIDIA Cloud Functions.
# It exposes:
#   - Health endpoint:    GET  /health  (port 8000)
#   - Inference endpoint: POST /echo    (port 8000)
#
# NVCF requirements:
#   - Must NOT run as root
#   - Ports 8080 and 8010 are reserved by NVCF
#   - /config/ directory must remain read-only
#
# Build:
#   docker buildx build . -t nvcf-echo:1.0.0 --platform linux/amd64
#
# Test locally:
#   docker run -p 8000:8000 nvcf-echo:1.0.0
#   curl http://localhost:8000/health
#   curl -X POST http://localhost:8000/echo -H "Content-Type: application/json" \
#        -d '{"message": "hello from HyperPod"}'
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm

# NVCF requirement: do NOT run as root
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

ENV WORKER_COUNT=10
ENV PORT=8000

WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application
COPY server.py ./

# Ensure /config is read-only compatible (NVCF uses this for caching)
RUN mkdir -p /config && chown appuser:appuser /config

# Switch to non-root user
USER appuser

# Expose the inference port (not 8080 or 8010 -- those are reserved by NVCF)
EXPOSE ${PORT}

# Start the server
CMD ["python", "server.py"]
