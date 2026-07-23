"""
vision/xray_processor.py — X-ray classification tool.

Real implementation target: torchxrayvision DenseNet
(densenet121-res224-all), CPU-safe, Jetson-friendly, non-fatal on all
runtime failures.

Responsibilities:
    - Accept one or more X-ray image paths from AegisState.xray_image_path.
    - Support DICOM (.dcm/.dicom), PNG, JPG, and JPEG.
    - Run torchxrayvision DenseNet inference when dependencies/weights exist.
    - Map torchxrayvision pathology labels to Aegis checklist findings.
    - Merge findings across multiple images.
    - Merge clinician-selected checklist findings from state.xray_findings_raw.
    - Preserve clinician free-text from state.xray_free_text_raw.
    - Optionally attach a Grad-CAM artifact if vision.gradcam is implemented.
    - Return ToolError(fatal=False) for failures; never raise to pipeline.

Contract:
    async def run(self, state: AegisState) -> XRayResult | ToolError

Notes:
    - This file does not mutate state. The pipeline owns state.xray_result.
    - If no image is provided but clinician findings/free-text exist, it returns
      those as a valid XRayResult. This makes the tool robust even if the
      pipeline gate is later changed from image-only to image-or-findings.
    - Grad-CAM is intentionally a best-effort optional artifact. A heatmap
      failure must never fail the X-ray classification or the full pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
from loguru import logger

from schemas.errors import ToolError
from schemas.state import AegisState
from schemas.xray import XRayResult
from tools.tool_names import TOOL_XRAY_PROCESSOR

try:
    import torch
    import torchxrayvision.models as xrv_models
except ImportError:
    torch = None
    xrv_models = None


# ── Configuration ─────────────────────────────────────────────────

_DEFAULT_MODEL_DIR = Path("data/xray")
_MODEL_DIR_ENV = "AEGIS_XRAY_MODEL_DIR"
_THRESHOLD_ENV = "AEGIS_XRAY_THRESHOLD"
_GRADCAM_ENV = "AEGIS_XRAY_GRADCAM"

_MODEL_WEIGHTS = "densenet121-res224-all"
_INPUT_RESOLUTION = 224
_DEFAULT_THRESHOLD = 0.5
_MAX_FINDINGS = 4

_DICOM_SUFFIXES = {".dcm", ".dicom"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_ALL_SUFFIXES = _DICOM_SUFFIXES | _IMAGE_SUFFIXES

_NORMAL_FINDING = "Normal / No significant findings"


# torchxrayvision pathology label → Aegis checklist finding.
_LABEL_MAP: dict[str, str] = {
    "Atelectasis": "Atelectasis",
    "Consolidation": "Consolidation",
    "Infiltration": "Infiltrates",
    "Pneumothorax": "Pneumothorax",
    "Edema": "Pulmonary Edema",
    "Effusion": "Pleural Effusion",
    "Pneumonia": "Pneumonia",
    "Cardiomegaly": "Cardiomegaly",
    "Nodule": "Nodule / Mass",
    "Mass": "Nodule / Mass",
    "Fracture": "Fracture",
}

# Per-label thresholds reduce noisy multi-label over-detection.
_LABEL_THRESHOLDS: dict[str, float] = {
    "Pneumothorax": 0.85,
    "Effusion": 0.80,
    "Consolidation": 0.80,
    "Pneumonia": 0.80,
    "Nodule": 0.85,
    "Mass": 0.85,
    "Fracture": 0.85,
    "Atelectasis": 0.75,
    "Infiltration": 0.75,
    "Edema": 0.75,
    "Cardiomegaly": 0.75,
}


# ── Singleton model ───────────────────────────────────────────────

_MODEL: Any | None = None


def _get_model_dir() -> Path:
    raw = os.getenv(_MODEL_DIR_ENV, "").strip()
    return Path(raw) if raw else _DEFAULT_MODEL_DIR


def _get_threshold() -> float:
    raw = os.getenv(_THRESHOLD_ENV, "").strip()
    if not raw:
        return _DEFAULT_THRESHOLD
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "xray_processor · invalid AEGIS_XRAY_THRESHOLD, using default",
            raw=raw,
            default=_DEFAULT_THRESHOLD,
        )
        return _DEFAULT_THRESHOLD
    if not 0.0 < value < 1.0:
        logger.warning(
            "xray_processor · threshold out of range, using default",
            value=value,
            default=_DEFAULT_THRESHOLD,
        )
        return _DEFAULT_THRESHOLD
    return value


def _gradcam_enabled() -> bool:
    return os.getenv(_GRADCAM_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def _load_model() -> Any:
    """Lazy-load torchxrayvision DenseNet model."""
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    import torchxrayvision.models as xrv_models

    model_dir = _get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "xray_processor · loading DenseNet model",
        weights=_MODEL_WEIGHTS,
        model_dir=str(model_dir),
    )

    _MODEL = xrv_models.DenseNet(
        weights=_MODEL_WEIGHTS,
        cache_dir=str(model_dir),
    )
    _MODEL.eval()

    logger.info(
        "xray_processor · model loaded",
        n_pathologies=len(getattr(_MODEL, "pathologies", [])),
    )
    return _MODEL


# ── Preprocessing and inference ───────────────────────────────────


def _load_and_preprocess(path: Path) -> Any:
    """Load DICOM/PNG/JPEG and preprocess for torchxrayvision DenseNet."""
    import torch
    import torchxrayvision.datasets as xrv_datasets
    import torchxrayvision.utils as xrv_utils

    img: np.ndarray = xrv_utils.load_image(str(path))

    # torchxrayvision usually returns shape [1, H, W]. Be defensive.
    if img.ndim == 2:
        img = img[None, :, :]
    if img.ndim != 3:
        raise ValueError(f"Unexpected X-ray image tensor shape: {img.shape}")

    img = xrv_datasets.XRayCenterCrop()(img)
    img = xrv_datasets.XRayResizer(_INPUT_RESOLUTION)(img)

    tensor = torch.from_numpy(img).float().unsqueeze(0)
    return tensor


def _run_inference(tensor: Any) -> dict[str, float]:
    """Run DenseNet and return {pathology_label: probability}."""
    import torch

    model = _load_model()
    with torch.no_grad():
        preds = model(tensor)

    probs = preds[0].detach().cpu().numpy().astype(float)

    # Most torchxrayvision DenseNet weights already emit probabilities.
    # If a model variant emits logits, squash defensively.
    if np.nanmin(probs) < 0.0 or np.nanmax(probs) > 1.0:
        probs = 1.0 / (1.0 + np.exp(-probs))

    labels = list(getattr(model, "pathologies", []))
    return dict(zip(labels, probs.tolist()))


# ── Finding selection and merge helpers ───────────────────────────


def _select_findings(probs: dict[str, float], threshold: float) -> list[str]:
    """Apply thresholds, map labels, deduplicate, and cap findings."""
    found: dict[str, float] = {}

    for txv_label, prob in probs.items():
        mapped = _LABEL_MAP.get(txv_label)
        if mapped is None:
            continue

        required = threshold
        if prob < required:
            continue

        previous = found.get(mapped)
        if previous is None or prob > previous:
            found[mapped] = float(prob)

    if not found:
        return [_NORMAL_FINDING]

    sorted_findings = sorted(found.items(), key=lambda item: item[1], reverse=True)
    top_findings = [finding for finding, _prob in sorted_findings[:_MAX_FINDINGS]]
    return sorted(top_findings)


def _normalise_finding(value: str) -> str:
    value = (value or "").strip()
    canonical = {
        "normal": _NORMAL_FINDING,
        "normal / no significant findings": _NORMAL_FINDING,
        "pulmonary oedema": "Pulmonary Edema",
        "effusion": "Pleural Effusion",
        "nodule": "Nodule / Mass",
        "mass": "Nodule / Mass",
    }
    return canonical.get(value.lower(), value)


def _merge_findings(existing: list[str], new_findings: list[str]) -> list[str]:
    merged = list(existing)
    for raw in new_findings:
        finding = _normalise_finding(raw)
        if finding and finding not in merged:
            merged.append(finding)

    # If any abnormal finding exists, remove the normal placeholder.
    if len(merged) > 1 and _NORMAL_FINDING in merged:
        merged = [f for f in merged if f != _NORMAL_FINDING]

    return merged


def _as_path_list(value: str | list[str] | None) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, list):
        return [Path(v) for v in value if v]
    return [Path(value)]


# ── Optional Grad-CAM integration ─────────────────────────────────


def _maybe_generate_heatmap(path: Path, finding: str, session_id: str) -> str | None:
    """Best-effort Grad-CAM hook. Returns heatmap path or None."""
    if not _gradcam_enabled():
        return None

    try:
        from vision.gradcam import generate_xray_heatmap

        output_dir = Path("/tmp/aegis_uploads") / session_id
        return generate_xray_heatmap(str(path), str(output_dir), finding)
    except Exception as exc:
        logger.warning(
            "xray_processor · Grad-CAM generation skipped/failed",
            path=str(path),
            finding=finding,
            error=str(exc),
            session_id=session_id,
        )
        return None


# ── Single image processing ───────────────────────────────────────


def _process_one_image(path: Path, threshold: float, session_id: str) -> tuple[XRayResult | None, ToolError | None, str | None]:
    if not path.is_file():
        return None, ToolError(
            tool=TOOL_XRAY_PROCESSOR,
            code="file_not_found",
            reason=f"X-ray image file not found: {path}",
            fatal=False,
        ), None

    if path.suffix.lower() not in _ALL_SUFFIXES:
        return None, ToolError(
            tool=TOOL_XRAY_PROCESSOR,
            code="unsupported_format",
            reason=(
                f"Unsupported X-ray image format '{path.suffix}'. "
                "Accepted: DICOM (.dcm/.dicom), PNG, JPG, JPEG."
            ),
            fatal=False,
        ), None

    logger.info(
        "xray_processor · processing image",
        path=str(path),
        threshold=threshold,
        session_id=session_id,
    )

    tensor = _load_and_preprocess(path)
    probs = _run_inference(tensor)
    findings = _select_findings(probs, threshold)

    top_probs = sorted(
        [(label, float(prob)) for label, prob in probs.items()],
        key=lambda item: item[1],
        reverse=True,
    )[:8]

    logger.info(
        "xray_processor · inference complete",
        path=str(path),
        findings=findings,
        top_probs=top_probs,
        session_id=session_id,
    )

    heatmap_path = _maybe_generate_heatmap(path, findings[0], session_id) if findings else None
    heatmap_url = f"/queue/heatmap/{session_id}/{Path(heatmap_path).name}" if heatmap_path else None
    return XRayResult(findings=findings, free_text=None, heatmap_path=heatmap_path, heatmap_url=heatmap_url), None, heatmap_path


# ── Public tool ───────────────────────────────────────────────────


class XRayProcessor:
    """Classifies chest X-ray images and returns structured findings."""

    TOOL_NAME = TOOL_XRAY_PROCESSOR

    async def run(self, state: AegisState) -> XRayResult | ToolError:
        try:
            paths = _as_path_list(state.xray_image_path)
            threshold = _get_threshold()

            merged_findings: list[str] = []
            failures: list[str] = []
            heatmaps: list[str] = []

            # Merge clinician-entered findings first, if present.
            if state.xray_findings_raw:
                merged_findings = _merge_findings(merged_findings, state.xray_findings_raw)

            # If no image path exists but clinician findings/free-text exist,
            # return them as a valid non-inference XRayResult.
            if not paths:
                if merged_findings or state.xray_free_text_raw:
                    if not merged_findings:
                        merged_findings = [_NORMAL_FINDING]
                    return XRayResult(
                        findings=merged_findings,
                        free_text=state.xray_free_text_raw,
                    )
                return ToolError(
                    tool=TOOL_XRAY_PROCESSOR,
                    code="missing_input",
                    reason="No X-ray image path or X-ray findings supplied.",
                    fatal=False,
                )

            # Process every uploaded image, keeping successful images even if
            # some images fail.
            for path in paths:
                try:
                    result, err, heatmap_path = _process_one_image(path, threshold, state.session_id)
                except Exception as exc:
                    result = None
                    err = ToolError(
                        tool=TOOL_XRAY_PROCESSOR,
                        code="inference_error",
                        reason=f"{path}: {type(exc).__name__}: {exc}",
                        fatal=False,
                    )
                    heatmap_path = None

                if err is not None:
                    failures.append(err.reason)
                    logger.warning(
                        "xray_processor · image failed, continuing",
                        path=str(path),
                        reason=err.reason,
                        session_id=state.session_id,
                    )
                    continue

                if result is not None:
                    merged_findings = _merge_findings(merged_findings, result.findings)
                if heatmap_path:
                    heatmaps.append(heatmap_path)

            if not merged_findings and failures:
                return ToolError(
                    tool=TOOL_XRAY_PROCESSOR,
                    code="all_images_failed",
                    reason="; ".join(failures),
                    fatal=False,
                )

            if not merged_findings:
                merged_findings = [_NORMAL_FINDING]

            free_text_parts: list[str] = []
            if state.xray_free_text_raw:
                free_text_parts.append(state.xray_free_text_raw)
            if failures:
                free_text_parts.append(f"{len(failures)} X-ray image(s) could not be processed.")

            logger.info(
                "xray_processor · combined result complete",
                images_total=len(paths),
                images_failed=len(failures),
                findings=merged_findings,
                heatmaps=len(heatmaps),
                session_id=state.session_id,
            )

            return XRayResult(
                findings=merged_findings,
                free_text="\n".join(free_text_parts) if free_text_parts else None,
                heatmap_path=heatmaps[0] if heatmaps else None,
                heatmap_url=f"/queue/heatmap/{state.session_id}/{Path(heatmaps[0]).name}" if heatmaps else None,
            )

        except Exception as exc:
            logger.exception(
                "xray_processor · unexpected error",
                session_id=getattr(state, "session_id", None),
            )
            return ToolError(
                tool=TOOL_XRAY_PROCESSOR,
                code="internal_error",
                reason=f"{type(exc).__name__}: {exc}",
                fatal=False,
            )


async def process(state: AegisState) -> XRayResult | ToolError:
    """Canonical functional entrypoint."""
    return await XRayProcessor().run(state)
