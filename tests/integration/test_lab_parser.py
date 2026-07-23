"""
tests/integration/test_lab_parser.py — LabReportParser integration.

Tests alias normalisation, abnormal detection, extra_measurements,
PDF magic byte rejection, and canonical key usage.

All tests use text fixture files written to tmp_path — no real PDFs.
PDF magic byte test creates a minimal PDF header to verify rejection.
"""

from __future__ import annotations

from pathlib import Path


from schemas.errors import ToolError
from schemas.lab import LabReportResult
from schemas.state import AegisState
from tools.lab_constants import (
    LAB_KEY_GLUCOSE,
    LAB_KEY_HAEMOGLOBIN,
    LAB_KEY_POTASSIUM,
    LAB_KEY_TROPONIN,
    LAB_KEY_WBC,
)
from tools.lab_report_parser import LabReportParser
from tools.lab_thresholds import (
    ABNORMAL_HIGH_GLUCOSE_MG_DL,
    ABNORMAL_HIGH_POTASSIUM_MMOL_L,
    ABNORMAL_HIGH_TROPONIN_NG_ML,
    ABNORMAL_LOW_HAEMOGLOBIN_G_DL,
)


# ── Helpers ────────────────────────────────────────────────────────

async def _parse(
    content: str, tmp_path: Path
) -> LabReportResult | ToolError:
    p = tmp_path / "lab.txt"
    p.write_text(content, encoding="utf-8")
    state = AegisState(lab_pdf_path=str(p))
    return await LabReportParser().run(state)


# ── Guard conditions ───────────────────────────────────────────────

async def test_no_path_returns_nonfatal_tool_error():
    state = AegisState()
    result = await LabReportParser().run(state)
    assert isinstance(result, ToolError)
    assert result.fatal is False


async def test_missing_file_returns_nonfatal_tool_error(tmp_path):
    state = AegisState(
        lab_pdf_path=str(tmp_path / "nonexistent.txt")
    )
    result = await LabReportParser().run(state)
    assert isinstance(result, ToolError)
    assert result.fatal is False


async def test_real_pdf_returns_nonfatal_tool_error(tmp_path):
    """Files starting with %PDF magic bytes are rejected."""
    p = tmp_path / "real.pdf"
    p.write_bytes(b"%PDF-1.4 fake pdf content")
    state = AegisState(lab_pdf_path=str(p))
    result = await LabReportParser().run(state)
    assert isinstance(result, ToolError)
    assert result.fatal is False


# ── British canonical key: haemoglobin ────────────────────────────

async def test_british_haemoglobin_canonical(tmp_path):
    result = await _parse("Haemoglobin: 14.5", tmp_path)
    assert isinstance(result, LabReportResult)
    assert LAB_KEY_HAEMOGLOBIN in result.measurements
    assert abs(result.measurements[LAB_KEY_HAEMOGLOBIN] - 14.5) < 1e-9


async def test_us_hemoglobin_normalises_to_british(tmp_path):
    """US 'hemoglobin' must normalise to British canonical 'haemoglobin'."""
    result = await _parse("hemoglobin: 13.0", tmp_path)
    assert isinstance(result, LabReportResult)
    assert LAB_KEY_HAEMOGLOBIN in result.measurements
    assert "hemoglobin" not in result.measurements


async def test_hb_alias_normalises_to_haemoglobin(tmp_path):
    result = await _parse("Hb: 12.5", tmp_path)
    assert isinstance(result, LabReportResult)
    assert LAB_KEY_HAEMOGLOBIN in result.measurements


async def test_hgb_alias_normalises_to_haemoglobin(tmp_path):
    result = await _parse("HGB: 11.0", tmp_path)
    assert isinstance(result, LabReportResult)
    assert LAB_KEY_HAEMOGLOBIN in result.measurements


# ── Troponin aliases ───────────────────────────────────────────────

async def test_troponin_canonical(tmp_path):
    result = await _parse("Troponin: 0.02", tmp_path)
    assert isinstance(result, LabReportResult)
    assert LAB_KEY_TROPONIN in result.measurements


async def test_troponin_i_alias(tmp_path):
    result = await _parse("Troponin I: 0.01", tmp_path)
    assert isinstance(result, LabReportResult)
    assert LAB_KEY_TROPONIN in result.measurements


# ── Potassium aliases ──────────────────────────────────────────────

async def test_potassium_canonical(tmp_path):
    result = await _parse("Potassium: 4.2", tmp_path)
    assert isinstance(result, LabReportResult)
    assert LAB_KEY_POTASSIUM in result.measurements


async def test_k_alias_normalises_to_potassium(tmp_path):
    """
    'K' alias normalises to canonical potassium key.

    Note: 'K+' is also in the alias map but the placeholder parser's
    regex does not include '+' in its key character class. The real
    PDF parser (Phase 3) will handle '+' correctly.
    """
    result = await _parse("K: 5.0", tmp_path)
    assert isinstance(result, LabReportResult)
    assert LAB_KEY_POTASSIUM in result.measurements


# ── Glucose aliases ────────────────────────────────────────────────

async def test_glucose_canonical(tmp_path):
    result = await _parse("Glucose: 95", tmp_path)
    assert isinstance(result, LabReportResult)
    assert LAB_KEY_GLUCOSE in result.measurements


async def test_fasting_glucose_alias(tmp_path):
    result = await _parse("Fasting Glucose: 110", tmp_path)
    assert isinstance(result, LabReportResult)
    assert LAB_KEY_GLUCOSE in result.measurements


# ── WBC aliases ────────────────────────────────────────────────────

async def test_wbc_canonical(tmp_path):
    result = await _parse("WBC: 7.5", tmp_path)
    assert isinstance(result, LabReportResult)
    assert LAB_KEY_WBC in result.measurements


async def test_leucocytes_alias_british(tmp_path):
    result = await _parse("Leucocytes: 8.0", tmp_path)
    assert isinstance(result, LabReportResult)
    assert LAB_KEY_WBC in result.measurements


async def test_leukocytes_alias_us(tmp_path):
    result = await _parse("Leukocytes: 9.0", tmp_path)
    assert isinstance(result, LabReportResult)
    assert LAB_KEY_WBC in result.measurements


# ── Abnormal detection ─────────────────────────────────────────────

async def test_low_haemoglobin_flagged_as_abnormal(tmp_path):
    low_hb = ABNORMAL_LOW_HAEMOGLOBIN_G_DL - 1.0
    result = await _parse(f"Haemoglobin: {low_hb}", tmp_path)
    assert isinstance(result, LabReportResult)
    assert len(result.abnormal_values) > 0
    assert any("haemoglobin" in v.lower() for v in result.abnormal_values)


async def test_normal_haemoglobin_not_flagged(tmp_path):
    normal_hb = ABNORMAL_LOW_HAEMOGLOBIN_G_DL + 1.0
    result = await _parse(f"Haemoglobin: {normal_hb}", tmp_path)
    assert isinstance(result, LabReportResult)
    assert not any("haemoglobin" in v.lower() for v in result.abnormal_values)


async def test_high_troponin_flagged_as_abnormal(tmp_path):
    high_trop = ABNORMAL_HIGH_TROPONIN_NG_ML + 0.01
    result = await _parse(f"Troponin: {high_trop}", tmp_path)
    assert isinstance(result, LabReportResult)
    assert any("troponin" in v.lower() for v in result.abnormal_values)


async def test_high_glucose_flagged_as_abnormal(tmp_path):
    high_glu = ABNORMAL_HIGH_GLUCOSE_MG_DL + 10.0
    result = await _parse(f"Glucose: {high_glu}", tmp_path)
    assert isinstance(result, LabReportResult)
    assert any("glucose" in v.lower() for v in result.abnormal_values)


async def test_high_potassium_flagged_as_abnormal(tmp_path):
    high_k = ABNORMAL_HIGH_POTASSIUM_MMOL_L + 0.5
    result = await _parse(f"Potassium: {high_k}", tmp_path)
    assert isinstance(result, LabReportResult)
    assert any("potassium" in v.lower() for v in result.abnormal_values)


# ── extra_measurements ─────────────────────────────────────────────

async def test_unknown_key_goes_to_extra_measurements(tmp_path):
    result = await _parse("crp: 12.5", tmp_path)
    assert isinstance(result, LabReportResult)
    assert "crp" in result.extra_measurements
    assert abs(result.extra_measurements["crp"] - 12.5) < 1e-9


async def test_unknown_key_not_in_measurements(tmp_path):
    result = await _parse("crp: 12.5", tmp_path)
    assert isinstance(result, LabReportResult)
    assert "crp" not in result.measurements


async def test_recognised_key_not_in_extra_measurements(tmp_path):
    result = await _parse("Troponin: 0.02", tmp_path)
    assert isinstance(result, LabReportResult)
    assert LAB_KEY_TROPONIN not in result.extra_measurements


# ── Multiple values ────────────────────────────────────────────────

async def test_multiple_measurements_parsed(tmp_path):
    content = "Haemoglobin: 14.0\nPotassium: 4.5\nGlucose: 90"
    result = await _parse(content, tmp_path)
    assert isinstance(result, LabReportResult)
    assert LAB_KEY_HAEMOGLOBIN in result.measurements
    assert LAB_KEY_POTASSIUM in result.measurements
    assert LAB_KEY_GLUCOSE in result.measurements


# ── Schema compliance ──────────────────────────────────────────────

async def test_schema_version(tmp_path):
    result = await _parse("Troponin: 0.01", tmp_path)
    assert isinstance(result, LabReportResult)
    assert result.schema_version == "1.0"


# ── Functional entrypoint ──────────────────────────────────────────

async def test_parse_functional_entrypoint(tmp_path):
    from tools.lab_report_parser import parse
    p = tmp_path / "lab.txt"
    p.write_text("Haemoglobin: 14.0", encoding="utf-8")
    state = AegisState(lab_pdf_path=str(p))
    result = await parse(state)
    assert isinstance(result, LabReportResult)