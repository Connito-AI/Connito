# syntax=docker/dockerfile:1
#
# Connito Telemetry Gateway container image.
#
# Build:
#     docker build -t connito-telemetry-gateway .
#
# Run:
#     docker run -p 8400:8400 \
#         -e VALIDATOR_URLS="http://10.0.0.1:8300" \
#         -e POLL_INTERVAL_SECONDS=12 \
#         connito-telemetry-gateway
#
# Healthcheck endpoint: GET /healthz   (does not require validator reachability)
# Data endpoint:        GET /api/leaderboard

FROM python:3.12-slim

# Run as a non-root user.
RUN useradd --system --uid 1001 --create-home --shell /bin/false connito
WORKDIR /app

# Install deps first so the layer caches between code edits.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

USER connito

ENV PORT=8400
EXPOSE 8400

# --proxy-headers lets us trust X-Forwarded-* if running behind nginx / a load balancer.
# Use sh -c so $PORT is expanded at container start, not at image build.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --proxy-headers"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{__import__(\"os\").environ.get(\"PORT\",\"8400\")}/healthz', timeout=4)" || exit 1
