"""
vision/dicom_reader.py — DICOM utility helpers for Aegis Health.

Purpose:
    - Detect whether a file is probably DICOM.
    - Read non-PHI DICOM metadata safely.
    - Read DICOM pixel data as a normalized uint8 numpy array when pydicom is available.
    - Avoid crashing the pipeline when pydicom or pixel handlers are missing.

Design rules:
    - This module is not a pipeline tool.
    - It never raises for normal caller usage; public helpers return safe fallback values.
    - Patient-identifying DICOM fields are not returned raw. PatientName and PatientID
      are hashed when present.
    - XRayProcessor may use torchxrayvision directly for image loading; this module is
      mainly for metadata, validation, future previews, and safer DICOM-specific handling.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


DICOM_MAGIC_OFFSET = 128
DICOM_MAGIC = b"DICM"
DICOM_SUFFIXES = {".dcm", ".dicom"}

# Tags that are useful for audit/debugging without exposing raw PHI.
_SAFE_TAGS: dict[str, str] = {
    "Modality": "modality",
    "StudyDate": "study_date",
    "StudyTime": "study_time",
    "BodyPartExamined": "body_part_examined",
    "ViewPosition": "view_position",
    "PhotometricInterpretation": "photometric_interpretation",
    "Rows": "rows",
    "Columns": "columns",
    "PixelSpacing": "pixel_spacing",
    "BitsAllocated": "bits_allocated",
    "BitsStored": "bits_stored",
    "Manufacturer": "manufacturer",
    "ManufacturerModelName": "manufacturer_model_name",
}

# These fields may identify the patient; hash only if needed for audit correlation.
_PHI_TAGS_TO_HASH: dict[str, str] = {
    "PatientName": "patient_name_hash",
    "PatientID": "patient_id_hash",
    "AccessionNumber": "accession_number_hash",
}


def _hash_value(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    if not text:
        return None
    return sha256(text.encode("utf-8")).hexdigest()[:16]


def is_dicom_file(path: str | Path) -> bool:
    """
    Return True if the file appears to be DICOM.

    Detection order:
        1. DICM magic at byte offset 128.
        2. DICOM-like suffix as a weak fallback.

    The suffix fallback is intentionally weak. Some valid DICOM files have no
    .dcm suffix, and some .dcm files may be malformed. Call read_dicom_metadata
    or read_dicom_pixels for definitive parsing.
    """
    p = Path(path)
    if not p.is_file():
        return False

    try:
        with p.open("rb") as f:
            f.seek(DICOM_MAGIC_OFFSET)
            if f.read(4) == DICOM_MAGIC:
                return True
    except OSError:
        return False

    return p.suffix.lower() in DICOM_SUFFIXES


def read_dicom_metadata(path: str | Path, *, include_phi_hashes: bool = True) -> dict[str, Any]:
    """
    Read safe DICOM metadata.

    Returns a dictionary with:
        - path
        - is_dicom
        - parse_ok
        - selected safe tags
        - optional hashes of selected PHI tags
        - error when parsing fails

    Raw PHI values are never returned.
    """
    p = Path(path)
    metadata: dict[str, Any] = {
        "path": str(p),
        "filename": p.name,
        "is_dicom": is_dicom_file(p),
        "parse_ok": False,
    }

    if not p.is_file():
        metadata["error"] = "file_not_found"
        return metadata

    try:
        import pydicom  # type: ignore[import]

        # stop_before_pixels=True keeps this lightweight and avoids loading large pixel arrays.
        ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)

        for dicom_name, output_name in _SAFE_TAGS.items():
            value = getattr(ds, dicom_name, None)
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                metadata[output_name] = [str(v) for v in value]
            else:
                metadata[output_name] = str(value)

        if include_phi_hashes:
            for dicom_name, output_name in _PHI_TAGS_TO_HASH.items():
                hashed = _hash_value(getattr(ds, dicom_name, None))
                if hashed:
                    metadata[output_name] = hashed

        metadata["parse_ok"] = True
        return metadata

    except ImportError:
        metadata["error"] = "pydicom_not_installed"
        logger.warning("dicom_reader · pydicom not installed", path=str(p))
        return metadata
    except Exception as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("dicom_reader · metadata parse failed", path=str(p), error=str(exc))
        return metadata


def read_dicom_pixels(path: str | Path, *, normalize: bool = True) -> np.ndarray | None:
    """
    Read DICOM pixel data.

    Returns:
        - np.ndarray uint8 when normalize=True
        - raw numpy array when normalize=False
        - None on any failure

    This helper is intentionally tolerant. Some compressed DICOM files require
    additional pixel handlers that may not be installed on Jetson.
    """
    p = Path(path)
    if not p.is_file():
        return None

    try:
        import pydicom  # type: ignore[import]

        ds = pydicom.dcmread(str(p), force=True)
        arr = ds.pixel_array

        # Apply rescale slope/intercept when present.
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        arr = arr.astype(np.float32) * slope + intercept

        # MONOCHROME1 images are inverted by convention.
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr

        if not normalize:
            return arr

        return normalize_pixels(arr)

    except ImportError:
        logger.warning("dicom_reader · pydicom not installed", path=str(p))
        return None
    except Exception as exc:
        logger.warning("dicom_reader · pixel read failed", path=str(p), error=str(exc))
        return None


def normalize_pixels(arr: np.ndarray) -> np.ndarray:
    """Normalize an arbitrary numeric image array to uint8 [0, 255]."""
    data = np.asarray(arr, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    if data.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    lo = float(np.percentile(data, 1))
    hi = float(np.percentile(data, 99))
    if hi <= lo:
        lo = float(data.min())
        hi = float(data.max())

    if hi <= lo:
        return np.zeros(data.shape, dtype=np.uint8)

    data = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    return (data * 255).astype(np.uint8)


def save_dicom_preview_png(path: str | Path, output_path: str | Path) -> str | None:
    """
    Save a normalized PNG preview for a DICOM file.

    Returns the output path as string, or None on failure.
    """
    pixels = read_dicom_pixels(path, normalize=True)
    if pixels is None:
        return None

    try:
        from PIL import Image

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(out)
        return str(out)
    except Exception as exc:
        logger.warning(
            "dicom_reader · failed to save preview",
            path=str(path),
            output=str(output_path),
            error=str(exc),
        )
        return None
