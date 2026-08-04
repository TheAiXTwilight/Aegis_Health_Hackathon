"""Refreshed LabReportParser contract tests.

The old file called text fixtures “integration”, asserted a removed parse()
entrypoint, and relied on ambiguous bare Hb/K aliases. This version separates
fast structured-text contract tests from an optional real-PDF extraction smoke
test and makes desired alias expansion visible as a clinical gate.
"""
from __future__ import annotations

from pathlib import Path

import pytest

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


async def parse_text(content: str, tmp_path: Path) -> LabReportResult | ToolError:
    path = tmp_path / "synthetic_lab.txt"
    path.write_text(content, encoding="utf-8")
    return await LabReportParser().run(AegisState(lab_pdf_path=str(path)))


async def test_no_path_is_nonfatal_tool_error():
    result = await LabReportParser().run(AegisState())
    assert isinstance(result, ToolError)
    assert result.fatal is False


async def test_missing_path_is_nonfatal_tool_error(tmp_path):
    result = await LabReportParser().run(
        AegisState(lab_pdf_path=str(tmp_path / "missing.pdf"))
    )
    assert isinstance(result, ToolError)
    assert result.fatal is False


@pytest.mark.parametrize(
    "line,key,value,unit",
    [
        ("Haemoglobin: 14.5 g/dL", LAB_KEY_HAEMOGLOBIN, 14.5, "g/dL"),
        ("Hemoglobin: 13.0 g/dL", LAB_KEY_HAEMOGLOBIN, 13.0, "g/dL"),
        ("HGB 11.0 g/dL", LAB_KEY_HAEMOGLOBIN, 11.0, "g/dL"),
        ("Potassium: 4.2 mmol/L", LAB_KEY_POTASSIUM, 4.2, "mmol/L"),
        ("Glucose: 95 mg/dL", LAB_KEY_GLUCOSE, 95.0, "mg/dL"),
        ("WBC 7.5 10^9/L", LAB_KEY_WBC, 7.5, "K/µL"),
        ("Troponin: 0.02 ng/mL", LAB_KEY_TROPONIN, 0.02, "ng/mL"),
    ],
)
async def test_recognized_measurements_are_canonicalized_with_units(tmp_path, line, key, value, unit):
    result = await parse_text(line, tmp_path)
    assert isinstance(result, LabReportResult)
    assert result.measurements[key] == pytest.approx(value)
    assert result.units[key] == unit
    assert key in result.reference_ranges


async def test_known_noncanonical_biomarker_is_preserved_as_structured_measurement(tmp_path):
    result = await parse_text("CRP: 12.5 mg/L", tmp_path)
    assert isinstance(result, LabReportResult)
    # The current universal knowledge map recognizes CRP, so it belongs in
    # measurements rather than the startup-era extra_measurements assertion.
    assert result.measurements["crp"] == pytest.approx(12.5)
    assert result.units["crp"] == "mg/L"


async def test_abnormal_values_are_flagged_from_structured_value_and_threshold(tmp_path):
    result = await parse_text(
        "Haemoglobin: 11.0 g/dL\nPotassium: 5.8 mmol/L\nGlucose: 220 mg/dL",
        tmp_path,
    )
    assert isinstance(result, LabReportResult)
    joined = " ".join(result.abnormal_values).lower()
    assert "haemoglobin" in joined
    assert "potassium" in joined
    assert "glucose" in joined


async def test_multiple_lab_paths_merge_measurements(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("Haemoglobin: 14.0 g/dL", encoding="utf-8")
    second.write_text("Potassium: 4.5 mmol/L", encoding="utf-8")

    result = await LabReportParser().run(AegisState(lab_pdf_path=[str(first), str(second)]))
    assert isinstance(result, LabReportResult)
    assert result.measurements[LAB_KEY_HAEMOGLOBIN] == pytest.approx(14.0)
    assert result.measurements[LAB_KEY_POTASSIUM] == pytest.approx(4.5)


async def test_corrupt_pdf_returns_nonfatal_error_instead_of_crashing(tmp_path):
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.7 this is not a valid PDF")

    result = await LabReportParser().run(AegisState(lab_pdf_path=str(corrupt)))
    assert isinstance(result, ToolError)
    assert result.fatal is False


async def test_real_selectable_text_pdf_uses_pdf_extraction_path(tmp_path):
    pytest.importorskip("fitz")
    reportlab = pytest.importorskip("reportlab.pdfgen.canvas")

    pdf_path = tmp_path / "lab_report.pdf"
    canvas = reportlab.Canvas(str(pdf_path))
    canvas.drawString(72, 720, "Haemoglobin: 13.5 g/dL")
    canvas.drawString(72, 700, "Potassium: 4.8 mmol/L")
    canvas.save()

    result = await LabReportParser().run(AegisState(lab_pdf_path=str(pdf_path)))
    assert isinstance(result, LabReportResult)
    assert result.measurements[LAB_KEY_HAEMOGLOBIN] == pytest.approx(13.5)
    assert result.measurements[LAB_KEY_POTASSIUM] == pytest.approx(4.8)


@pytest.mark.clinical_gate
@pytest.mark.xfail(strict=True, reason="Desired clinical alias support: bare Hb/K+ is not parsed by current safe parser.")
@pytest.mark.parametrize(
    "line,key",
    [
        ("Hb: 12.5 g/dL", LAB_KEY_HAEMOGLOBIN),
        ("K+ 5.0 mmol/L", LAB_KEY_POTASSIUM),
    ],
)
async def test_common_short_lab_aliases_are_supported_when_unit_is_present(tmp_path, line, key):
    result = await parse_text(line, tmp_path)
    assert isinstance(result, LabReportResult)
    assert key in result.measurements
