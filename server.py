"""
server.py — mirage-mcp
=======================
FastMCP server exposing MIRAGE (multimodal retinal OCT/SLO foundation model)
as MCP tools, deployed via Prefect Horizon.

Image preprocessing runs locally (decode + validate + resize); GPU inference
is dispatched to the Modal serverless endpoint — Horizon needs no GPU or
model weights.

Required environment variables:
    MODAL_ENDPOINT_URL   Full Modal endpoint URL,
                         e.g. https://mathgcloud--mirage-api.modal.run

Optional environment variables:
    MAX_SIDE             Resize longest edge before sending (default: 512).
                         MIRAGE natively works at 512×512; larger values are
                         downscaled by the Modal worker anyway.

Tools:
    extract_features(bscan_b64, ..., model_size, ...)
        → (N_tokens, D) embedding array for downstream tasks

    reconstruct_oct(bscan_b64, ..., model_size, ...)
        → reconstructed bscan + optional bscanlayermap prediction

    segment_layers(bscan_b64, ..., model_size, ...)
        → integer layer-class map (13 classes)

    health()
        → liveness check
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from datetime import datetime, timezone
from typing import Literal, Optional

import numpy as np
import requests
from fastmcp import FastMCP
from PIL import Image

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

MODAL_ENDPOINT_URL = os.environ.get("MODAL_ENDPOINT_URL", "").rstrip("/")

if not MODAL_ENDPOINT_URL:
    logger.warning("MODAL_ENDPOINT_URL is not set — inference calls will fail.")

# MIRAGE natively processes 512×512; no benefit in sending larger images
DEFAULT_MAX_SIDE = int(os.environ.get("MAX_SIDE", "512"))


# ─── Modal client ────────────────────────────────────────────────────────────

def _modal_dispatch(payload: dict, image_id: str) -> dict:
    """POST the payload to the Modal endpoint and return the parsed response."""
    if not MODAL_ENDPOINT_URL:
        raise RuntimeError("MODAL_ENDPOINT_URL is not set.")

    logger.info(f"[{image_id}] Dispatching to Modal: {MODAL_ENDPOINT_URL}")
    resp = requests.post(
        MODAL_ENDPOINT_URL,
        json=payload,
        timeout=300,   # A10G cold-start ~60 s; ViT-Large inference up to ~2 min
    )
    resp.raise_for_status()
    return resp.json()


# ─── Preprocessing — runs locally, no GPU needed ─────────────────────────────

def _preprocess_grayscale(
    image_b64: str,
    image_id: str,
    tag: str = "bscan",
    max_side: int = DEFAULT_MAX_SIDE,
) -> tuple[str, int, int, int, int]:
    """
    Decode a base64 PNG/JPEG grayscale image (OCT bscan or SLO), validate it,
    optionally downscale it, and return a cleaned base64 PNG string.

    Returns:
        (clean_b64, sent_w, sent_h, orig_w, orig_h)
    """
    logger.info(f"[{image_id}] Preprocessing {tag} (len={len(image_b64)})")
    try:
        raw = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(raw)).convert("L")  # grayscale
    except Exception as e:
        raise ValueError(f"[{image_id}] Could not decode {tag}: {e}")

    orig_w, orig_h = img.size
    if orig_w < 32 or orig_h < 32:
        raise ValueError(
            f"[{image_id}] {tag} too small ({orig_w}×{orig_h}). "
            "Minimum 32×32 expected."
        )

    if max(orig_w, orig_h) > max_side:
        scale  = max_side / max(orig_w, orig_h)
        sent_w = max(1, int(orig_w * scale))
        sent_h = max(1, int(orig_h * scale))
        img    = img.resize((sent_w, sent_h), Image.LANCZOS)
        logger.info(f"[{image_id}] {tag}: resized {orig_w}×{orig_h} → {sent_w}×{sent_h}")
    else:
        sent_w, sent_h = orig_w, orig_h
        logger.info(f"[{image_id}] {tag}: {orig_w}×{orig_h}, no resize needed")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    clean_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return clean_b64, sent_w, sent_h, orig_w, orig_h


def _validate_model_size(model_size: str, image_id: str) -> None:
    if model_size not in ("base", "large"):
        raise ValueError(
            f"[{image_id}] Invalid model_size '{model_size}'. "
            "Must be 'base' or 'large'."
        )


# ─── FastMCP app ─────────────────────────────────────────────────────────────

mcp = FastMCP("mirage-mcp")


# ─── Tools ───────────────────────────────────────────────────────────────────

@mcp.tool()
async def extract_features(
    bscan_b64: str,
    image_id: str,
    slo_b64: Optional[str] = None,
    model_size: Literal["base", "large"] = "base",
    max_side: int = DEFAULT_MAX_SIDE,
) -> str:
    """
    Extract ViT token embeddings from a retinal OCT B-scan (and optionally SLO).

    Runs MIRAGE's encoder without the decoder, returning a flat token embedding
    matrix suitable for downstream classification, retrieval, or clustering.
    MIRAGE-Base produces 768-dim tokens; MIRAGE-Large produces 1024-dim tokens.

    Suitable for:
        - Transfer learning / linear probing for retinal disease staging
        - Zero-shot similarity search across OCT volumes
        - Multi-modal fusion (OCT + SLO embeddings)
        - Feeding into a downstream classification head

    Args:
        bscan_b64:   Base64-encoded grayscale OCT B-scan (PNG or JPEG).
        image_id:    Identifier for logging / tracing.
        slo_b64:     Optional base64-encoded SLO fundus image (PNG or JPEG).
                     When provided, MIRAGE performs multimodal encoding.
        model_size:  "base" (ViT-B, 86M params, default) or
                     "large" (ViT-L, 307M params, higher accuracy).
        max_side:    Pre-resize longest edge to this value (default 512).

    Returns:
        JSON with:
          features        — list[list[float]] of shape (N_tokens, embed_dim)
          n_tokens        — number of patch tokens
          embed_dim       — embedding dimension (768 or 1024)
          model_size      — which weights were used
          image_id        — echoed for tracing
          image_width/height — original image dimensions
    """
    try:
        _validate_model_size(model_size, image_id)
        bscan_clean, _, _, orig_w, orig_h = _preprocess_grayscale(
            bscan_b64, image_id, tag="bscan", max_side=max_side
        )

        payload: dict = {
            "bscan_b64":     bscan_clean,
            "model_size":    model_size,
            "features_only": True,
        }
        if slo_b64:
            slo_clean, *_ = _preprocess_grayscale(slo_b64, image_id, tag="slo", max_side=max_side)
            payload["slo_b64"] = slo_clean

        result = _modal_dispatch(payload, image_id)

        features = result.get("features", [])
        n_tokens  = len(features)
        embed_dim = len(features[0]) if n_tokens else 0

        out = json.dumps({
            "success":       True,
            "image_id":      image_id,
            "features":      features,
            "n_tokens":      n_tokens,
            "embed_dim":     embed_dim,
            "model_size":    model_size,
            "image_width":   orig_w,
            "image_height":  orig_h,
            "created_at":    datetime.now(timezone.utc).isoformat(),
        })
        logger.info(
            f"extract_features: {image_id}  model={model_size}  "
            f"tokens={n_tokens}  dim={embed_dim}  payload={len(out)/1024:.1f}KB"
        )
        return out

    except ValueError as e:
        return json.dumps({"success": False, "reason": str(e), "image_id": image_id})
    except Exception as e:
        logger.error(f"extract_features failed: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e), "image_id": image_id})


@mcp.tool()
async def reconstruct_oct(
    bscan_b64: str,
    image_id: str,
    slo_b64: Optional[str] = None,
    layermap_b64: Optional[str] = None,
    model_size: Literal["base", "large"] = "base",
    max_side: int = DEFAULT_MAX_SIDE,
) -> str:
    """
    Run MIRAGE multi-task reconstruction on a retinal OCT B-scan.

    Runs the full encoder + decoder to reconstruct the B-scan (and optionally
    the SLO and layer segmentation map). The reconstructed outputs reflect
    MIRAGE's internal representation of retinal structure and are useful for
    quality assessment, anomaly detection, and cross-modal synthesis.

    Suitable for:
        - OCT image quality / artifact detection (compare input vs. reconstruction)
        - Cross-modal generation (predict SLO from OCT)
        - Pretraining evaluation and representation probing

    Args:
        bscan_b64:    Base64-encoded grayscale OCT B-scan (PNG or JPEG).
        image_id:     Identifier for logging / tracing.
        slo_b64:      Optional SLO image; when provided, MIRAGE reconstructs
                      both the B-scan and SLO in parallel.
        layermap_b64: Optional layer segmentation map (PNG or NPY, 128×128 int).
                      When provided, layer reconstruction is also returned.
        model_size:   "base" (default) or "large".
        max_side:     Pre-resize longest edge to this value (default 512).

    Returns:
        JSON with:
          predictions.bscan         — reconstructed B-scan as 2D float list (H×W, [0,1])
          predictions.slo           — reconstructed SLO (if slo_b64 was given)
          predictions.bscanlayermap — reconstructed layer map (if layermap_b64 was given)
          model_size, image_id, image_width, image_height
    """
    try:
        _validate_model_size(model_size, image_id)
        bscan_clean, _, _, orig_w, orig_h = _preprocess_grayscale(
            bscan_b64, image_id, tag="bscan", max_side=max_side
        )

        payload: dict = {
            "bscan_b64":     bscan_clean,
            "model_size":    model_size,
            "features_only": False,
        }
        if slo_b64:
            slo_clean, *_ = _preprocess_grayscale(slo_b64, image_id, tag="slo", max_side=max_side)
            payload["slo_b64"] = slo_clean
        if layermap_b64:
            payload["layermap_b64"] = layermap_b64  # passed through as-is (NPY)

        result = _modal_dispatch(payload, image_id)
        predictions = result.get("predictions", {})

        out = json.dumps({
            "success":      True,
            "image_id":     image_id,
            "predictions":  predictions,
            "model_size":   model_size,
            "image_width":  orig_w,
            "image_height": orig_h,
            "created_at":   datetime.now(timezone.utc).isoformat(),
        })
        keys = list(predictions.keys())
        logger.info(
            f"reconstruct_oct: {image_id}  model={model_size}  "
            f"outputs={keys}  payload={len(out)/1024:.1f}KB"
        )
        return out

    except ValueError as e:
        return json.dumps({"success": False, "reason": str(e), "image_id": image_id})
    except Exception as e:
        logger.error(f"reconstruct_oct failed: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e), "image_id": image_id})


@mcp.tool()
async def segment_layers(
    bscan_b64: str,
    image_id: str,
    slo_b64: Optional[str] = None,
    model_size: Literal["base", "large"] = "base",
    max_side: int = DEFAULT_MAX_SIDE,
) -> str:
    """
    Predict a retinal layer segmentation map from an OCT B-scan.

    Runs MIRAGE in reconstruction mode and extracts the bscanlayermap output,
    which assigns each pixel to one of 13 retinal layer classes. The map is
    returned at MIRAGE's native 128×128 resolution as a flat integer array
    suitable for further processing or visualization.

    Layer class indices (0–12) follow the MIRAGE/MultiMAE convention:
        0  Background
        1  RNFL (Retinal Nerve Fiber Layer)
        2  GCL+IPL
        3  INL
        4  OPL
        5  ONL
        6  ELM
        7  IS/OS
        8  RPE
        9  BM
        10 Choroid
        11 Vitreous
        12 Other

    Suitable for:
        - Automated retinal layer thickness measurement
        - AMD / DME / glaucoma staging from layer maps
        - Preprocessing for downstream segmentation pipelines

    Args:
        bscan_b64:   Base64-encoded grayscale OCT B-scan (PNG or JPEG).
        image_id:    Identifier for logging / tracing.
        slo_b64:     Optional SLO image for multimodal conditioning.
        model_size:  "base" (default) or "large".
        max_side:    Pre-resize longest edge to this value (default 512).

    Returns:
        JSON with:
          layermap         — flat list of 128×128 = 16 384 integer class indices
          layermap_h/w     — dimensions (128, 128)
          n_classes        — 13
          model_size, image_id, image_width, image_height
    """
    try:
        _validate_model_size(model_size, image_id)
        bscan_clean, _, _, orig_w, orig_h = _preprocess_grayscale(
            bscan_b64, image_id, tag="bscan", max_side=max_side
        )

        payload: dict = {
            "bscan_b64":     bscan_clean,
            "model_size":    model_size,
            "features_only": False,
        }
        if slo_b64:
            slo_clean, *_ = _preprocess_grayscale(slo_b64, image_id, tag="slo", max_side=max_side)
            payload["slo_b64"] = slo_clean

        result = _modal_dispatch(payload, image_id)
        predictions = result.get("predictions", {})

        if "bscanlayermap" not in predictions:
            return json.dumps({
                "success":  False,
                "reason":   "Modal endpoint did not return a bscanlayermap. "
                            "Ensure the Modal worker has output adapters enabled.",
                "image_id": image_id,
            })

        # predictions["bscanlayermap"] is a 2D list (H×W) from the Modal worker
        layermap_2d = predictions["bscanlayermap"]
        h = len(layermap_2d)
        w = len(layermap_2d[0]) if h else 0
        flat = [v for row in layermap_2d for v in row]

        out = json.dumps({
            "success":      True,
            "image_id":     image_id,
            "layermap":     flat,
            "layermap_h":   h,
            "layermap_w":   w,
            "n_classes":    13,
            "model_size":   model_size,
            "image_width":  orig_w,
            "image_height": orig_h,
            "created_at":   datetime.now(timezone.utc).isoformat(),
        })
        logger.info(
            f"segment_layers: {image_id}  model={model_size}  "
            f"map={h}×{w}  payload={len(out)/1024:.1f}KB"
        )
        return out

    except ValueError as e:
        return json.dumps({"success": False, "reason": str(e), "image_id": image_id})
    except Exception as e:
        logger.error(f"segment_layers failed: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e), "image_id": image_id})


@mcp.tool()
async def health() -> str:
    """Liveness probe. Reports Modal endpoint configuration status."""
    return json.dumps({
        "status":  "ok",
        "service": "mirage-mcp",
        "modal": {
            "endpoint_url": MODAL_ENDPOINT_URL or "(not set)",
            "configured":   bool(MODAL_ENDPOINT_URL),
        },
        "defaults": {
            "max_side":   DEFAULT_MAX_SIDE,
            "model_size": "base",
        },
    })


if __name__ == "__main__":
    mcp.run(
        stateless_http=True,
        json_response=True,
        max_request_body_size=64 * 1024 * 1024,   # 64 MB — OCT volumes can be large
    )
