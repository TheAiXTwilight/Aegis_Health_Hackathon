"""
vision/gradcam.py — best-effort X-ray heatmap artifact generation.

This module exists so `vision.xray_processor` can safely import
`generate_xray_heatmap` when Grad-CAM is enabled.

Important:
    A true Grad-CAM implementation needs access to the loaded model,
    target layer activations, gradients, and the inference tensor. The current
    `XRayProcessor` calls this helper with only `(image_path, output_dir,
    finding)`, so this file provides a robust fallback heatmap artifact that
    does not crash the pipeline. It uses image contrast/edge saliency plus a
    center prior to create a visual overlay.

    This is intentionally best-effort and non-fatal. If dependencies or image
    decoding fail, the function returns None and the report still completes.

Upgrade path:
    When true Grad-CAM is wired, keep this public function name but pass model
    context or add a second function that accepts `(model, tensor, target_idx)`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from loguru import logger


def _safe_slug(value: str) -> str:
    value = (value or "xray").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value or "xray"


def _normalise_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.percentile(arr, 1))
    hi = float(np.percentile(arr, 99))
    if hi <= lo:
        hi = float(arr.max()) if arr.size else 1.0
        lo = float(arr.min()) if arr.size else 0.0
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def _load_grayscale(path: Path) -> np.ndarray | None:
    suffix = path.suffix.lower()

    # DICOM path: optional dependency. If unavailable, return None.
    if suffix in {".dcm", ".dicom"}:
        try:
            import pydicom  # type: ignore[import]

            ds = pydicom.dcmread(str(path))
            arr = ds.pixel_array.astype(np.float32)
            # Apply common DICOM photometric inversion if needed.
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                arr = arr.max() - arr
            return _normalise_to_uint8(arr)
        except Exception as exc:
            logger.warning(
                "gradcam · failed to decode DICOM image",
                path=str(path),
                error=str(exc),
            )
            return None

    # Standard image path.
    try:
        from PIL import Image

        img = Image.open(path).convert("L")
        return np.asarray(img, dtype=np.uint8)
    except Exception as exc:
        logger.warning(
            "gradcam · failed to decode image",
            path=str(path),
            error=str(exc),
        )
        return None


def _make_saliency_heatmap(gray: np.ndarray) -> np.ndarray:
    """
    Create a Grad-CAM-style fallback saliency map.

    This is not a model-gradient heatmap. It highlights high-contrast regions
    with a mild center prior so the artifact is useful for UI development and
    demo plumbing until true Grad-CAM receives model activations.
    """
    arr = gray.astype(np.float32) / 255.0

    # Edge/contrast saliency.
    gy, gx = np.gradient(arr)
    edge = np.sqrt(gx * gx + gy * gy)
    edge = edge / (edge.max() + 1e-6)

    # Center prior: chest X-rays usually have anatomy centered.
    h, w = arr.shape[:2]
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    center = np.exp(-2.2 * (x * x + y * y))
    center = center / (center.max() + 1e-6)

    heat = 0.70 * edge + 0.30 * center
    heat = heat / (heat.max() + 1e-6)
    return heat.astype(np.float32)


def _overlay_heatmap(gray: np.ndarray, heat: np.ndarray):
    """Return a PIL Image with the heatmap overlay applied."""
    from PIL import Image

    base = Image.fromarray(gray).convert("RGB")

    # Red/yellow heatmap without matplotlib dependency.
    heat_u8 = np.clip(heat * 255, 0, 255).astype(np.uint8)
    rgba = np.zeros((heat_u8.shape[0], heat_u8.shape[1], 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = np.clip(heat_u8 * 0.65, 0, 255).astype(np.uint8)
    rgba[..., 2] = 0
    rgba[..., 3] = np.clip(heat_u8 * 0.48, 0, 150).astype(np.uint8)

    overlay = Image.fromarray(rgba, mode="RGBA")
    base_rgba = base.convert("RGBA")
    return Image.alpha_composite(base_rgba, overlay).convert("RGB")


def generate_xray_heatmap(image_path: str, output_dir: str, finding: str) -> str | None:
    """
    Generate a best-effort X-ray heatmap PNG.

    Args:
        image_path: Path to DICOM/PNG/JPEG X-ray image.
        output_dir: Directory where the PNG should be written.
        finding: Top finding label, used in filename only.

    Returns:
        Saved PNG path as string, or None when heatmap generation fails.
    """
    try:
        src = Path(image_path)
        if not src.is_file():
            logger.warning("gradcam · source image missing", path=str(src))
            return None

        gray = _load_grayscale(src)
        if gray is None:
            return None

        heat = _make_saliency_heatmap(gray)
        overlay_image = _overlay_heatmap(gray, heat)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{src.stem}_{_safe_slug(finding)}_heatmap.png"
        overlay_image.save(out)

        logger.info(
            "gradcam · heatmap artifact generated",
            source=str(src),
            output=str(out),
            finding=finding,
        )
        return str(out)
    except Exception as exc:
        logger.warning(
            "gradcam · heatmap generation failed",
            image_path=image_path,
            finding=finding,
            error=str(exc),
        )
        return None
