"""Unit tests for the current structured lab parser internals.

The parser now returns six values from _parse_text: measurements, extras,
abnormal values, units, ranges, and text findings. The old test file assumed
the retired three-value contract and a removed parse() wrapper.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from schemas.errors import ToolError
from schemas.lab import LabReportResult
from schemas.state import AegisState
from tools.lab_constants import LAB_KEY_GLUCOSE, LAB_KEY_HAEMOGLOBIN, LAB_KEY_POTASSIUM
from tools.lab_report_parser import LabReportParser, _ocr_enabled, _parse_text


def fake_pdf(path: Path) -> Path:
    """A magic-byte fixture is sufficient because extraction is mocked."""
    path.write_bytes(b"%PDF-1.7 synthetic fixture")
    return path


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "True", "yes", "YES"])
def test_ocr_enabled_recognizes_supported_values(value):
    with patch.dict(os.environ, {"AEGIS_OCR": value}):
        assert _ocr_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "enabled"])
def test_ocr_enabled_rejects_other_values(value):
    with patch.dict(os.environ, {"AEGIS_OCR": value}):
        assert _ocr_enabled() is False


def test_parse_text_returns_current_six_part_contract_with_metadata():
    measurements, extras, abnormal, units, ranges, text_findings = _parse_text(
        "Haemoglobin: 9.0 g/dL\nGlucose: 220 mg/dL\nCRP: 8.5 mg/L"
    )
    assert measurements[LAB_KEY_HAEMOGLOBIN] == pytest.approx(9.0)
    assert measurements[LAB_KEY_GLUCOSE] == pytest.approx(220.0)
    assert extras.get("crp", measurements.get("crp")) == pytest.approx(8.5)
    assert units[LAB_KEY_HAEMOGLOBIN] == "g/dL"
    assert LAB_KEY_HAEMOGLOBIN in ranges
    assert any("haemoglobin" in item.lower() for item in abnormal)
    assert isinstance(text_findings, list)


def test_parse_text_first_value_wins_for_duplicate_canonical_measurement():
    measurements, *_rest = _parse_text(
        "Haemoglobin: 11.0 g/dL\nHaemoglobin: 14.5 g/dL"
    )
    assert measurements[LAB_KEY_HAEMOGLOBIN] == pytest.approx(11.0)


def test_parse_text_preserves_unknown_measurement_in_documented_output():
    _measurements, extras, _abnormal, units, _ranges, _findings = _parse_text("Unknown Marker: 5.0 U/L")
    assert extras["unknown_marker"] == pytest.approx(5.0)
    assert units["unknown_marker"] == "U/L"


def test_parse_text_empty_input_has_empty_structures():
    measurements, extras, abnormal, units, ranges, findings = _parse_text("")
    assert measurements == {}
    assert extras == {}
    assert abnormal == []
    assert units == {}
    assert ranges == {}
    assert findings == []


async def test_pdf_waterfall_uses_pdfminer_when_pymupdf_returns_no_text(tmp_path):
    path = fake_pdf(tmp_path / "lab.pdf")
    state = AegisState(lab_pdf_path=str(path))
    extracted = "Potassium: 4.2 mmol/L"

    with patch("tools.lab_report_parser._extract_via_pymupdf", return_value=None), \
         patch("tools.lab_report_parser._extract_via_pdfminer", return_value=extracted) as pdfminer, \
         patch("tools.lab_report_parser._extract_via_easyocr") as ocr:
        result = await LabReportParser().run(state)

    assert isinstance(result, LabReportResult)
    assert result.measurements[LAB_KEY_POTASSIUM] == pytest.approx(4.2)
    pdfminer.assert_called_once()
    ocr.assert_not_called()


async def test_pdf_waterfall_uses_ocr_only_when_enabled_and_other_extractors_fail(tmp_path):
    path = fake_pdf(tmp_path / "lab.pdf")
    state = AegisState(lab_pdf_path=str(path))

    with patch("tools.lab_report_parser._extract_via_pymupdf", return_value=None), \
         patch("tools.lab_report_parser._extract_via_pdfminer", return_value=None), \
         patch("tools.lab_report_parser._extract_via_easyocr", return_value="Glucose: 100 mg/dL") as ocr, \
         patch.dict(os.environ, {"AEGIS_OCR": "1"}):
        result = await LabReportParser().run(state)

    assert isinstance(result, LabReportResult)
    assert result.measurements[LAB_KEY_GLUCOSE] == pytest.approx(100.0)
    ocr.assert_called_once()


async def test_pdf_extraction_failure_is_nonfatal_and_explains_ocr_option(tmp_path):
    path = fake_pdf(tmp_path / "lab.pdf")
    state = AegisState(lab_pdf_path=str(path))

    with patch("tools.lab_report_parser._extract_via_pymupdf", return_value=None), \
         patch("tools.lab_report_parser._extract_via_pdfminer", return_value=None), \
         patch.dict(os.environ, {}, clear=True):
        result = await LabReportParser().run(state)

    assert isinstance(result, ToolError)
    assert result.fatal is False
    assert "AEGIS_OCR=1" in result.reason


async def test_multiple_paths_merge_current_structured_result(tmp_path):
    first = tmp_path / "one.txt"
    second = tmp_path / "two.txt"
    first.write_text("Haemoglobin: 13.5 g/dL", encoding="utf-8")
    second.write_text("Potassium: 4.5 mmol/L", encoding="utf-8")

    result = await LabReportParser().run(AegisState(lab_pdf_path=[str(first), str(second)]))
    assert isinstance(result, LabReportResult)
    assert result.measurements[LAB_KEY_HAEMOGLOBIN] == pytest.approx(13.5)
    assert result.measurements[LAB_KEY_POTASSIUM] == pytest.approx(4.5)
