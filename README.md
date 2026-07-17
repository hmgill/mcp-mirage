# mcp-mirage

A [FastMCP](https://github.com/jlowin/fastmcp) server that exposes **MIRAGE** —
a multimodal retinal OCT/SLO foundation model — as MCP tools, deployed via
Prefect Horizon.

The server is deliberately lightweight: image preprocessing (decode, validate,
resize) runs locally on CPU, while GPU inference is dispatched to a **Modal**
serverless endpoint. No model weights or CUDA are baked into the container, so it
stays small and cheap to run.

**Deployed MCP endpoint:** `https://mirage.fastmcp.app/mcp`

## Architecture

```
MCP client
    │  base64 OCT B-scan (+ optional SLO)
    ▼
mcp-mirage  @ https://mirage.fastmcp.app/mcp   (this server — CPU only)
    │  decode · validate · resize to 512px
    ▼
Modal serverless endpoint  (GPU: MIRAGE ViT-B / ViT-L)
    │  embeddings · reconstruction · layer map
    ▼
JSON response
```

Two URLs are involved and serve different roles: clients connect to the
**MCP endpoint** (`https://mirage.fastmcp.app/mcp`); that server in turn
dispatches GPU work to the **Modal endpoint** (`MODAL_ENDPOINT_URL`, below).

Preprocessing and inference are separated so the always-on MCP layer needs no
GPU, and the expensive model only spins up on Modal when a request arrives.

## Tools

| Tool | Purpose | Returns |
|------|---------|---------|
| `extract_features` | Encoder-only ViT token embeddings for an OCT B-scan (and optional SLO) | `(n_tokens, embed_dim)` matrix — 768-dim (base) or 1024-dim (large) |
| `reconstruct_oct` | Full encoder+decoder multi-task reconstruction | Reconstructed B-scan, plus SLO and layer map when their inputs are supplied |
| `segment_layers` | Retinal layer segmentation from a B-scan | Integer layer map, 13 classes, with dimensions |
| `health` | Liveness probe | Status and Modal endpoint configuration |

Embeddings are suited to transfer learning / linear probing for disease staging,
similarity search across OCT volumes, and multimodal OCT+SLO fusion.
Reconstruction supports quality/artifact detection (input vs. reconstruction)
and cross-modal synthesis (e.g. predicting SLO from OCT).

### Common arguments

- `bscan_b64` — base64 grayscale OCT B-scan (PNG or JPEG), minimum 32×32.
- `image_id` — identifier echoed back for tracing.
- `slo_b64` — optional SLO fundus image; enables multimodal encoding.
- `model_size` — `"base"` (ViT-B, 86M params, default) or `"large"`
  (ViT-L, 307M params, higher accuracy).
- `max_side` — longest edge to resize to before dispatch (default 512).

MIRAGE natively operates at 512×512, so larger inputs are downscaled with no
loss of effective resolution. Tools return a JSON object with a `success` flag;
failures report a `reason` (validation) or `error` (runtime) rather than raising.

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `MODAL_ENDPOINT_URL` | yes | Modal endpoint URL, e.g. `https://mathgcloud--mirage-api.modal.run` |
| `MAX_SIDE` | no | Default resize edge (default `512`) |

## Connecting

The server is hosted at `https://mirage.fastmcp.app/mcp`. Point any MCP client at
that URL — for example, in a client config:

```json
{
  "mcpServers": {
    "mirage": {
      "url": "https://mirage.fastmcp.app/mcp"
    }
  }
}
```

To self-host instead, run the server yourself as below.

## Running

### Docker

```bash
docker build -t mcp-mirage .
docker run -p 8080:8080 \
  -e MODAL_ENDPOINT_URL=https://<your-modal-endpoint>.modal.run \
  mcp-mirage
```

The image is a multi-stage `python:3.11-slim` build that runs as a non-root user
and exposes port 8080.

### Local

```bash
pip install -r requirements.txt
export MODAL_ENDPOINT_URL=https://<your-modal-endpoint>.modal.run
python server.py
```

The server runs over stateless HTTP with JSON responses and accepts request
bodies up to 64 MB (OCT payloads can be large).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
