# =============================================================================
# mirage-mcp — Horizon MCP server
# =============================================================================
# Preprocessing (image decode + validate + resize) runs locally in this
# container. GPU inference is dispatched to the Modal MIRAGE serverless
# endpoint at runtime — no model weights are baked into this image.
#
# Required environment variables (set in Horizon, not here):
#   MODAL_ENDPOINT_URL   Full Modal endpoint URL,
#                        e.g. https://mathgcloud--mirage-api.modal.run
# =============================================================================

FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
COPY server.py .

RUN useradd --no-create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

CMD ["python", "server.py"]
