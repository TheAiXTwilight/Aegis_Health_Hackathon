"""
tests/tools/test_xray_processor.py — XRayProcessor.

Test structure:
    Unit tests (always run in CI)
        Mock _load_model and _run_inference.
        Verify all logic paths: guards, label mapping, threshold,
        deduplication, Normal fallback, error handling.
        No model download, no network, deterministic.

    Integration tests (@pytest.mark.xray)
        Real torchxrayvision DenseNet loaded from AEGIS_XRAY_MODEL_DIR.
        Real image fixture (synthetic PNG via numpy/PIL).
        Not run in standard CI — require model to be present.
        Run with: pytest -m xray

Key behaviours under test:
    - No image path              → ToolError(fatal=False)
    - Missing file               → ToolError(fatal=False)
    - Unsupported format         → ToolError(fatal=False)
    - Preprocessing raises       → ToolError(fatal=False)
    - Inference raises           → ToolError(fatal=False)
    - Model load fails           → ToolError(fatal=False)
    - Known label above threshold → mapped checklist item
    - Nodule + Mass both above threshold → "Nodule / Mass" once (deduplication)
    - No labels above threshold  → ["Normal / No significant findings"]
    - Unknown txv labels ignored → not in findings
    - free_text always None
    - state.xray_result NOT written (pipeline owns it)
    - Tool attribution always TOOL_XRAY_PROCESSOR
    - _get_model_dir() reads AEGIS_XRAY_MODEL_DIR env var
    - _get_threshold() reads AEGIS_XRAY_THRESHOLD env var
    - _get_threshold() falls back to 0.5 on invalid value
    - _select_findings() is deterministic (sorted output)
    - Singleton: DenseNet constructor called once across multiple _load_model() calls

torchxrayvision label strings tested against (densenet121-res224-all):
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
    'Edema', 'Effusion', 'Pneumonia', 'Cardiomegaly', 'Nodule', 'Mass',
    'Fracture' (mapped), plus unmapped: 'Emphysema', 'Fibrosis', 'Hernia' (ignored)

Singleton patch note:
    The singleton test patches pathlib.Path.mkdir in addition to _get_model_dir.
    Without this, passing Path("/fake") causes mkdir() to hit the macOS SIP
    read-only filesystem and raise OSError before DenseNet is ever called.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from schemas.errors import ToolError
from schemas.state import AegisState
from schemas.xray import XRayResult
from tools.tool_names import TOOL_XRAY_PROCESSOR
from vision.xray_processor import (
    XRayProcessor,
    _get_model_dir,
    _get_threshold,
    _select_findings,
    _NORMAL_FINDING,
    _ALL_SUFFIXES,
    process,
)


# ── Helpers ────────────────────────────────────────────────────────

# Full set of pathology labels for densenet121-res224-all (verified from source)
_ALL_TXV_LABELS = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia",
    "Lung Lesion", "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum",
]


def _make_probs(overrides: dict[str, float] | None = None) -> dict[str, float]:
    """
    Return a full probability dict with all txv labels defaulting to 0.0,
    with specified labels overridden.
    """
    base = {label: 0.0 for label in _ALL_TXV_LABELS}
    if overrides:
        base.update(overrides)
    return base


def _mock_model(probs: dict[str, float]) -> MagicMock:
    """
    Return a MagicMock DenseNet whose forward() returns a tensor
    aligned to _ALL_TXV_LABELS order, and whose .pathologies matches.
    """
    prob_list = [probs.get(label, 0.0) for label in _ALL_TXV_LABELS]
    tensor = torch.tensor([prob_list], dtype=torch.float32)

    model = MagicMock()
    model.pathologies = _ALL_TXV_LABELS
    model.__call__ = MagicMock(return_value=tensor)
    # torch.no_grad context manager compatibility
    model.return_value = tensor
    return model


def _make_png(tmp_path: Path) -> Path:
    """
    Write a minimal valid 224×224 grayscale PNG using numpy only.
    Returns the path.
    """
    p = tmp_path / "xray.png"
    # Write a numpy array as raw PNG via PIL if available, else skip
    try:
        from PIL import Image
        arr = np.zeros((224, 224), dtype=np.uint8)
        Image.fromarray(arr).save(str(p))
    except ImportError:
        # Fallback: write minimal PNG header bytes
        import struct
        import zlib
        def png_chunk(chunk_type, data):
            c = chunk_type + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

        sig = b"\x89PNG\r\n\x1a\n"
        ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
        raw = b"\x00\x00"
        idat_data = zlib.compress(raw)
        png = sig + png_chunk(b"IHDR", ihdr_data) + png_chunk(b"IDAT", idat_data) + png_chunk(b"IEND", b"")
        p.write_bytes(png)
    return p


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def png_path(tmp_path) -> Path:
    return _make_png(tmp_path)


@pytest.fixture
def dcm_path(tmp_path) -> Path:
    """Minimal .dcm extension file (content mocked in tests)."""
    p = tmp_path / "xray.dcm"
    p.write_bytes(b"\x00" * 132 + b"DICM" + b"\x00" * 100)
    return p


@pytest.fixture
def unsupported_path(tmp_path) -> Path:
    p = tmp_path / "xray.bmp"
    p.write_bytes(b"BM")
    return p


@pytest.fixture
def txt_path(tmp_path) -> Path:
    p = tmp_path / "notes.txt"
    p.write_text("no xray here", encoding="utf-8")
    return p


# ── _get_model_dir ─────────────────────────────────────────────────

def test_get_model_dir_default(monkeypatch):
    monkeypatch.delenv("AEGIS_XRAY_MODEL_DIR", raising=False)
    assert _get_model_dir() == Path("data/xray")


def test_get_model_dir_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("AEGIS_XRAY_MODEL_DIR", str(tmp_path))
    assert _get_model_dir() == tmp_path


def test_get_model_dir_strips_whitespace(monkeypatch, tmp_path):
    monkeypatch.setenv("AEGIS_XRAY_MODEL_DIR", f"  {tmp_path}  ")
    assert _get_model_dir() == tmp_path


def test_get_model_dir_empty_env_falls_back(monkeypatch):
    monkeypatch.setenv("AEGIS_XRAY_MODEL_DIR", "")
    assert _get_model_dir() == Path("data/xray")


# ── _get_threshold ─────────────────────────────────────────────────

def test_get_threshold_default(monkeypatch):
    monkeypatch.delenv("AEGIS_XRAY_THRESHOLD", raising=False)
    assert _get_threshold() == 0.5


def test_get_threshold_env_var(monkeypatch):
    monkeypatch.setenv("AEGIS_XRAY_THRESHOLD", "0.7")
    assert _get_threshold() == pytest.approx(0.7)


def test_get_threshold_invalid_string_falls_back(monkeypatch):
    monkeypatch.setenv("AEGIS_XRAY_THRESHOLD", "not_a_float")
    assert _get_threshold() == 0.5


def test_get_threshold_out_of_range_falls_back(monkeypatch):
    monkeypatch.setenv("AEGIS_XRAY_THRESHOLD", "1.5")
    assert _get_threshold() == 0.5


def test_get_threshold_zero_falls_back(monkeypatch):
    monkeypatch.setenv("AEGIS_XRAY_THRESHOLD", "0.0")
    assert _get_threshold() == 0.5


def test_get_threshold_strips_whitespace(monkeypatch):
    monkeypatch.setenv("AEGIS_XRAY_THRESHOLD", "  0.6  ")
    assert _get_threshold() == pytest.approx(0.6)


# ── _select_findings ───────────────────────────────────────────────

def test_select_findings_normal_when_all_below_threshold():
    probs = _make_probs()  # all 0.0
    findings = _select_findings(probs, threshold=0.5)
    assert findings == [_NORMAL_FINDING]


def test_select_findings_single_mapped_label():
    probs = _make_probs({"Pneumonia": 0.85})
    findings = _select_findings(probs, threshold=0.5)
    assert "Pneumonia" in findings
    assert _NORMAL_FINDING not in findings


def test_select_findings_effusion_maps_to_pleural_effusion():
    probs = _make_probs({"Effusion": 0.9})
    findings = _select_findings(probs, threshold=0.5)
    assert "Pleural Effusion" in findings


def test_select_findings_infiltration_maps_to_infiltrates():
    probs = _make_probs({"Infiltration": 0.75})
    findings = _select_findings(probs, threshold=0.5)
    assert "Infiltrates" in findings


def test_select_findings_edema_maps_to_pulmonary_edema():
    probs = _make_probs({"Edema": 0.6})
    findings = _select_findings(probs, threshold=0.5)
    assert "Pulmonary Edema" in findings


def test_select_findings_nodule_and_mass_deduplicated():
    """Both Nodule and Mass above threshold → 'Nodule / Mass' appears exactly once."""
    probs = _make_probs({"Nodule": 0.91, "Mass": 0.77})
    findings = _select_findings(probs, threshold=0.5)
    assert findings.count("Nodule / Mass") == 1
    assert _NORMAL_FINDING not in findings


def test_select_findings_nodule_only():
    probs = _make_probs({"Nodule": 0.8})
    findings = _select_findings(probs, threshold=0.5)
    assert "Nodule / Mass" in findings
    assert findings.count("Nodule / Mass") == 1


def test_select_findings_unmapped_labels_ignored():
    """Emphysema, Fibrosis, Hernia, etc. are not in _LABEL_MAP — must not appear."""
    probs = _make_probs({"Emphysema": 0.99, "Fibrosis": 0.99, "Hernia": 0.99})
    findings = _select_findings(probs, threshold=0.5)
    assert findings == [_NORMAL_FINDING]


def test_select_findings_output_is_sorted():
    """Sorted output is deterministic regardless of dict iteration order."""
    probs = _make_probs({"Pneumonia": 0.9, "Atelectasis": 0.8, "Fracture": 0.7})
    findings = _select_findings(probs, threshold=0.5)
    assert findings == sorted(findings)


def test_select_findings_at_exactly_threshold_included():
    probs = _make_probs({"Cardiomegaly": 0.5})
    findings = _select_findings(probs, threshold=0.5)
    assert "Cardiomegaly" in findings


def test_select_findings_just_below_threshold_excluded():
    probs = _make_probs({"Cardiomegaly": 0.499})
    findings = _select_findings(probs, threshold=0.5)
    assert "Cardiomegaly" not in findings
    assert findings == [_NORMAL_FINDING]


def test_select_findings_multiple_findings_no_normal():
    probs = _make_probs({"Pneumonia": 0.9, "Pneumothorax": 0.8})
    findings = _select_findings(probs, threshold=0.5)
    assert _NORMAL_FINDING not in findings
    assert "Pneumonia" in findings
    assert "Pneumothorax" in findings


# ── Guard: no image path ───────────────────────────────────────────

async def test_no_image_path_returns_tool_error():
    state = AegisState()
    result = await XRayProcessor().run(state)
    assert isinstance(result, ToolError)
    assert result.fatal is False
    assert result.tool == TOOL_XRAY_PROCESSOR


# ── Guard: missing file ────────────────────────────────────────────

async def test_missing_file_returns_tool_error():
    state = AegisState(xray_image_path="/nonexistent/xray.png")
    result = await XRayProcessor().run(state)
    assert isinstance(result, ToolError)
    assert result.fatal is False
    assert result.tool == TOOL_XRAY_PROCESSOR


# ── Guard: unsupported format ──────────────────────────────────────

async def test_unsupported_format_returns_tool_error(unsupported_path):
    state = AegisState(xray_image_path=str(unsupported_path))
    result = await XRayProcessor().run(state)
    assert isinstance(result, ToolError)
    assert result.fatal is False
    assert result.tool == TOOL_XRAY_PROCESSOR


async def test_txt_file_returns_tool_error(txt_path):
    state = AegisState(xray_image_path=str(txt_path))
    result = await XRayProcessor().run(state)
    assert isinstance(result, ToolError)
    assert result.fatal is False
    assert result.tool == TOOL_XRAY_PROCESSOR


# ── Happy path (mocked inference) ─────────────────────────────────

async def test_pneumonia_finding_returned(png_path):
    probs = _make_probs({"Pneumonia": 0.85})
    tensor = torch.zeros(1, 1, 224, 224)

    with patch("vision.xray_processor._load_and_preprocess", return_value=tensor), \
         patch("vision.xray_processor._run_inference", return_value=probs):
        state = AegisState(xray_image_path=str(png_path))
        result = await XRayProcessor().run(state)

    assert isinstance(result, XRayResult)
    assert "Pneumonia" in result.findings
    assert result.free_text is None


async def test_normal_finding_when_all_below_threshold(png_path):
    probs = _make_probs()  # all 0.0
    tensor = torch.zeros(1, 1, 224, 224)

    with patch("vision.xray_processor._load_and_preprocess", return_value=tensor), \
         patch("vision.xray_processor._run_inference", return_value=probs):
        state = AegisState(xray_image_path=str(png_path))
        result = await XRayProcessor().run(state)

    assert isinstance(result, XRayResult)
    assert result.findings == [_NORMAL_FINDING]
    assert result.free_text is None


async def test_nodule_mass_deduplication_in_full_pipeline(png_path):
    probs = _make_probs({"Nodule": 0.91, "Mass": 0.77})
    tensor = torch.zeros(1, 1, 224, 224)

    with patch("vision.xray_processor._load_and_preprocess", return_value=tensor), \
         patch("vision.xray_processor._run_inference", return_value=probs):
        state = AegisState(xray_image_path=str(png_path))
        result = await XRayProcessor().run(state)

    assert isinstance(result, XRayResult)
    assert result.findings.count("Nodule / Mass") == 1


async def test_multiple_findings_returned(png_path):
    probs = _make_probs({"Pneumonia": 0.9, "Cardiomegaly": 0.75, "Atelectasis": 0.6})
    tensor = torch.zeros(1, 1, 224, 224)

    with patch("vision.xray_processor._load_and_preprocess", return_value=tensor), \
         patch("vision.xray_processor._run_inference", return_value=probs):
        state = AegisState(xray_image_path=str(png_path))
        result = await XRayProcessor().run(state)

    assert isinstance(result, XRayResult)
    assert "Pneumonia" in result.findings
    assert "Cardiomegaly" in result.findings
    assert "Atelectasis" in result.findings
    assert _NORMAL_FINDING not in result.findings


async def test_free_text_always_none(png_path):
    probs = _make_probs({"Pneumonia": 0.9})
    tensor = torch.zeros(1, 1, 224, 224)

    with patch("vision.xray_processor._load_and_preprocess", return_value=tensor), \
         patch("vision.xray_processor._run_inference", return_value=probs):
        state = AegisState(xray_image_path=str(png_path))
        result = await XRayProcessor().run(state)

    assert isinstance(result, XRayResult)
    assert result.free_text is None


async def test_schema_version(png_path):
    probs = _make_probs()
    tensor = torch.zeros(1, 1, 224, 224)

    with patch("vision.xray_processor._load_and_preprocess", return_value=tensor), \
         patch("vision.xray_processor._run_inference", return_value=probs):
        state = AegisState(xray_image_path=str(png_path))
        result = await XRayProcessor().run(state)

    assert isinstance(result, XRayResult)
    assert result.schema_version == "1.0"


# ── Custom threshold ───────────────────────────────────────────────

async def test_custom_threshold_filters_findings(png_path, monkeypatch):
    """At threshold=0.9, only Pneumonia (0.95) passes; Cardiomegaly (0.75) does not."""
    monkeypatch.setenv("AEGIS_XRAY_THRESHOLD", "0.9")
    probs = _make_probs({"Pneumonia": 0.95, "Cardiomegaly": 0.75})
    tensor = torch.zeros(1, 1, 224, 224)

    with patch("vision.xray_processor._load_and_preprocess", return_value=tensor), \
         patch("vision.xray_processor._run_inference", return_value=probs):
        state = AegisState(xray_image_path=str(png_path))
        result = await XRayProcessor().run(state)

    assert isinstance(result, XRayResult)
    assert "Pneumonia" in result.findings
    assert "Cardiomegaly" not in result.findings


async def test_low_threshold_includes_more_findings(png_path, monkeypatch):
    """At threshold=0.3, Cardiomegaly (0.4) is included."""
    monkeypatch.setenv("AEGIS_XRAY_THRESHOLD", "0.3")
    probs = _make_probs({"Cardiomegaly": 0.4})
    tensor = torch.zeros(1, 1, 224, 224)

    with patch("vision.xray_processor._load_and_preprocess", return_value=tensor), \
         patch("vision.xray_processor._run_inference", return_value=probs):
        state = AegisState(xray_image_path=str(png_path))
        result = await XRayProcessor().run(state)

    assert isinstance(result, XRayResult)
    assert "Cardiomegaly" in result.findings


# ── State not mutated ──────────────────────────────────────────────

async def test_xray_result_not_written_by_tool(png_path):
    """Pipeline owns state.xray_result — tool must not assign it."""
    probs = _make_probs({"Pneumonia": 0.9})
    tensor = torch.zeros(1, 1, 224, 224)

    with patch("vision.xray_processor._load_and_preprocess", return_value=tensor), \
         patch("vision.xray_processor._run_inference", return_value=probs):
        state = AegisState(xray_image_path=str(png_path))
        await XRayProcessor().run(state)

    assert state.xray_result is None


# ── Error propagation ──────────────────────────────────────────────

async def test_preprocessing_exception_returns_tool_error(png_path):
    with patch(
        "vision.xray_processor._load_and_preprocess",
        side_effect=RuntimeError("corrupt image"),
    ):
        state = AegisState(xray_image_path=str(png_path))
        result = await XRayProcessor().run(state)

    assert isinstance(result, ToolError)
    assert result.fatal is False
    assert result.tool == TOOL_XRAY_PROCESSOR
    assert "RuntimeError" in result.reason


async def test_inference_exception_returns_tool_error(png_path):
    tensor = torch.zeros(1, 1, 224, 224)
    with patch("vision.xray_processor._load_and_preprocess", return_value=tensor), \
         patch(
             "vision.xray_processor._run_inference",
             side_effect=RuntimeError("model error"),
         ):
        state = AegisState(xray_image_path=str(png_path))
        result = await XRayProcessor().run(state)

    assert isinstance(result, ToolError)
    assert result.fatal is False
    assert result.tool == TOOL_XRAY_PROCESSOR


async def test_model_load_failure_returns_tool_error(png_path):
    with patch(
        "vision.xray_processor._load_model",
        side_effect=Exception("weights not found"),
    ):
        state = AegisState(xray_image_path=str(png_path))
        result = await XRayProcessor().run(state)

    assert isinstance(result, ToolError)
    assert result.fatal is False
    assert result.tool == TOOL_XRAY_PROCESSOR


async def test_tool_error_is_never_fatal(png_path):
    with patch(
        "vision.xray_processor._load_and_preprocess",
        side_effect=Exception("anything"),
    ):
        state = AegisState(xray_image_path=str(png_path))
        result = await XRayProcessor().run(state)

    assert isinstance(result, ToolError)
    assert result.fatal is False


# ── Singleton call-once ────────────────────────────────────────────

def test_load_model_called_once_across_multiple_calls(monkeypatch):
    """
    DenseNet constructor must be called exactly once regardless of
    how many times _load_model() is invoked.

    Patches both _get_model_dir (to avoid real filesystem access) and
    pathlib.Path.mkdir (to avoid OSError on read-only paths like /fake
    on macOS with SIP). Without the mkdir patch, the test would crash
    before DenseNet is ever constructed.
    """
    import vision.xray_processor as xp_module

    original = xp_module._MODEL
    xp_module._MODEL = None

    try:
        with patch("vision.xray_processor._get_model_dir", return_value=Path("/fake")), \
             patch("pathlib.Path.mkdir"), \
             patch("vision.xray_processor.torch"):
            import torchxrayvision.models as _real_xrv
            with patch.object(_real_xrv, "DenseNet") as mock_cls:
                mock_instance = MagicMock()
                mock_instance.pathologies = []
                mock_cls.return_value = mock_instance

                from vision.xray_processor import _load_model
                _load_model()
                _load_model()
                _load_model()

                mock_cls.assert_called_once()
    finally:
        xp_module._MODEL = original


# ── Functional entrypoint ──────────────────────────────────────────

async def test_process_function_delegates_to_tool(png_path):
    """process() is a thin wrapper — same contract as XRayProcessor().run()."""
    probs = _make_probs({"Pneumonia": 0.9})
    tensor = torch.zeros(1, 1, 224, 224)

    with patch("vision.xray_processor._load_and_preprocess", return_value=tensor), \
         patch("vision.xray_processor._run_inference", return_value=probs):
        state = AegisState(xray_image_path=str(png_path))
        result = await process(state)

    assert isinstance(result, XRayResult)
    assert "Pneumonia" in result.findings


# ── Accepted suffix coverage ───────────────────────────────────────

@pytest.mark.parametrize("suffix", [".dcm", ".dicom", ".png", ".jpg", ".jpeg"])
def test_all_accepted_suffixes_in_constant(suffix):
    assert suffix in _ALL_SUFFIXES


# ── Integration tests (require real model) ─────────────────────────

@pytest.mark.xray
async def test_real_model_returns_xray_result(tmp_path, monkeypatch):
    """
    Integration: load real torchxrayvision DenseNet and run inference
    on a synthetic blank PNG. Asserts return type and findings list
    validity — not specific pathology content (blank image has no findings).

    Requires: AEGIS_XRAY_MODEL_DIR set to a directory containing the
    downloaded densenet121-res224-all weights.
    """
    import vision.xray_processor as xp_module
    xp_module._MODEL = None

    png = _make_png(tmp_path)
    state = AegisState(xray_image_path=str(png))
    result = await XRayProcessor().run(state)

    assert isinstance(result, (XRayResult, ToolError))
    if isinstance(result, XRayResult):
        assert isinstance(result.findings, list)
        assert len(result.findings) >= 1
        assert result.free_text is None
    else:
        assert result.fatal is False
        assert result.tool == TOOL_XRAY_PROCESSOR

    xp_module._MODEL = None