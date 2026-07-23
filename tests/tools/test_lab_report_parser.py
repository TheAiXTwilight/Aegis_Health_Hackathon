"""
tests/tools/test_lab_report_parser.py — LabReportParser real PDF extraction tests.

Tests the PyMuPDF → pdfminer → EasyOCR waterfall introduced in Phase 3 Commit 4.

Scope of THIS file:
    - Real PDF extraction waterfall (mock each tier at the boundary)
    - OCR env-var gate
    - Waterfall ordering (pdfminer runs before OCR regardless of failure mode)
    - first-wins duplicate canonical key policy
    - K+/Na+ aliases now reachable with the wider regex

The existing tests/integration/test_lab_parser.py covers:
    - Text fixture path (unchanged, all pass)
    - Alias normalisation, abnormal detection, extra_measurements, schema_version
    - Guard conditions (no path, missing file, PDF magic byte)

Do NOT duplicate those tests here.

Mocking strategy:
    We mock the individual _extract_via_* functions at the module level
    (tools.lab_report_parser._extract_via_pymupdf etc.) to control
    what each tier returns, without importing fitz/pdfminer/easyocr.
    This keeps the test suite runnable without optional dependencies.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from schemas.errors import ToolError
from schemas.lab import LabReportResult
from schemas.state import AegisState
from tools.lab_constants import (
    LAB_KEY_GLUCOSE,
    LAB_KEY_HAEMOGLOBIN,
    LAB_KEY_POTASSIUM,
    LAB_KEY_SODIUM,
    LAB_KEY_TROPONIN,
)
from tools.lab_report_parser import LabReportParser, _parse_text, _ocr_enabled


# ── Fixture: minimal synthetic PDF file ──────────────────────────
#
# Generates a real one-page PDF via PyMuPDF (already a hard runtime
# dependency — no extra package required).
#
# A bare b"%PDF-1.4 synthetic" byte stub is intentionally NOT used here.
# Rationale (locked decision 79): if future tests stop mocking the
# extraction layer, the fixture must survive real PyMuPDF/pdfminer
# parsing. A stub would cause those tiers to raise or return blank,
# masking bugs instead of exposing them.
#
# text_content is written into the page body so any test that
# accidentally reaches real extraction still gets parseable output.

def _write_fake_pdf(path: Path, text_content: str = "Glucose: 100") -> Path:
    """
    Write a real one-page PDF to path using PyMuPDF.

    The file passes _is_real_pdf() detection and survives real
    PyMuPDF/pdfminer extraction if mocks are ever removed.
    """
    import fitz  # PyMuPDF — already a hard runtime dependency

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text_content)
    doc.save(str(path))
    doc.close()
    return path


# ── _ocr_enabled unit tests ───────────────────────────────────────
# Pure function — no I/O, no mocks. Tests all accepted and rejected values.

class TestOcrEnabled:

    def test_value_1_enables(self):
        with patch.dict(os.environ, {"AEGIS_OCR": "1"}):
            assert _ocr_enabled() is True

    def test_value_true_lowercase_enables(self):
        with patch.dict(os.environ, {"AEGIS_OCR": "true"}):
            assert _ocr_enabled() is True

    def test_value_true_uppercase_enables(self):
        with patch.dict(os.environ, {"AEGIS_OCR": "TRUE"}):
            assert _ocr_enabled() is True

    def test_value_true_mixed_case_enables(self):
        with patch.dict(os.environ, {"AEGIS_OCR": "True"}):
            assert _ocr_enabled() is True

    def test_value_yes_lowercase_enables(self):
        with patch.dict(os.environ, {"AEGIS_OCR": "yes"}):
            assert _ocr_enabled() is True

    def test_value_yes_uppercase_enables(self):
        with patch.dict(os.environ, {"AEGIS_OCR": "YES"}):
            assert _ocr_enabled() is True

    def test_value_0_disables(self):
        with patch.dict(os.environ, {"AEGIS_OCR": "0"}):
            assert _ocr_enabled() is False

    def test_value_false_disables(self):
        with patch.dict(os.environ, {"AEGIS_OCR": "false"}):
            assert _ocr_enabled() is False

    def test_value_no_disables(self):
        with patch.dict(os.environ, {"AEGIS_OCR": "no"}):
            assert _ocr_enabled() is False

    def test_empty_string_disables(self):
        with patch.dict(os.environ, {"AEGIS_OCR": ""}):
            assert _ocr_enabled() is False

    def test_unset_disables(self):
        env = {k: v for k, v in os.environ.items() if k != "AEGIS_OCR"}
        with patch.dict(os.environ, env, clear=True):
            assert _ocr_enabled() is False

    def test_arbitrary_string_disables(self):
        with patch.dict(os.environ, {"AEGIS_OCR": "enabled"}):
            assert _ocr_enabled() is False


# ── _parse_text unit tests ────────────────────────────────────────
# These test the shared parser logic in isolation — no I/O, no mocks.

class TestParseText:

    def test_canonical_key_stored(self):
        m, _, _ = _parse_text("Haemoglobin: 14.5")
        assert LAB_KEY_HAEMOGLOBIN in m
        assert abs(m[LAB_KEY_HAEMOGLOBIN] - 14.5) < 1e-9

    def test_k_plus_alias_now_reachable(self):
        """K+ was unreachable in placeholder due to narrow regex. Now resolved."""
        m, _, _ = _parse_text("K+: 4.8")
        assert LAB_KEY_POTASSIUM in m
        assert abs(m[LAB_KEY_POTASSIUM] - 4.8) < 1e-9

    def test_na_plus_alias_now_reachable(self):
        m, _, _ = _parse_text("Na+: 138")
        assert LAB_KEY_SODIUM in m

    def test_first_wins_duplicate_canonical_key(self):
        """
        If Hb and Haemoglobin both appear, the first parsed value wins.
        Input order: Hb (11.0) then Haemoglobin (14.5).
        Expected: 11.0 stored, 14.5 dropped.
        """
        m, _, _ = _parse_text("Hb: 11.0\nHaemoglobin: 14.5")
        assert abs(m[LAB_KEY_HAEMOGLOBIN] - 11.0) < 1e-9

    def test_first_wins_preserves_first_not_last(self):
        """Converse: Haemoglobin first, then Hb second — 14.5 wins."""
        m, _, _ = _parse_text("Haemoglobin: 14.5\nHb: 11.0")
        assert abs(m[LAB_KEY_HAEMOGLOBIN] - 14.5) < 1e-9

    def test_unknown_key_first_wins(self):
        """Duplicate unrecognised key: first value preserved."""
        _, extra, _ = _parse_text("crp: 5.0\ncrp: 10.0")
        assert abs(extra["crp"] - 5.0) < 1e-9

    def test_abnormal_low_haemoglobin_detected(self):
        _, _, abnormal = _parse_text("Haemoglobin: 9.0")
        assert any("haemoglobin" in v.lower() for v in abnormal)

    def test_normal_haemoglobin_not_flagged(self):
        _, _, abnormal = _parse_text("Haemoglobin: 14.0")
        assert not any("haemoglobin" in v.lower() for v in abnormal)

    def test_arbitrary_punctuation_in_key_not_matched(self):
        """
        'Hb***:' must NOT match — '*' is not in the allowed character class.
        The lazy quantifier stops before '*', leaving no valid key boundary.
        """
        m, extra, _ = _parse_text("Hb***: 11.0")
        # Either nothing is parsed, or only the bare "hb" prefix up to the
        # first '*' is attempted — but since the regex requires [:=] immediately
        # after the key (with optional whitespace), "hb***" never forms a valid
        # match. No canonical or extra key should appear.
        assert LAB_KEY_HAEMOGLOBIN not in m
        assert "hb***" not in extra

    def test_empty_text_returns_empty_dicts(self):
        m, extra, abnormal = _parse_text("")
        assert m == {}
        assert extra == {}
        assert abnormal == []


# ── Waterfall unit tests ──────────────────────────────────────────
# Mock _extract_via_* at module level to control each tier precisely.

LAB_TEXT = "Troponin: 0.10\nGlucose: 200\n"  # Two abnormal values


class TestExtractionWaterfall:
    """
    Tests the _extract_pdf_text waterfall logic via LabReportParser.run().

    All tests use a fake PDF file (passes _is_real_pdf) with the
    extraction functions mocked so no real PyMuPDF/pdfminer needed.
    """

    def _fake_pdf(self, tmp_path: Path) -> Path:
        return _write_fake_pdf(tmp_path / "lab.pdf")

    async def test_pymupdf_success_used_directly(self, tmp_path):
        """When PyMuPDF returns text, pdfminer and OCR are never called."""
        path = self._fake_pdf(tmp_path)
        state = AegisState(lab_pdf_path=str(path))

        with patch("tools.lab_report_parser._extract_via_pymupdf", return_value=LAB_TEXT) as m_mupdf, \
             patch("tools.lab_report_parser._extract_via_pdfminer") as m_pdf, \
             patch("tools.lab_report_parser._extract_via_easyocr") as m_ocr:

            result = await LabReportParser().run(state)

        assert isinstance(result, LabReportResult)
        m_mupdf.assert_called_once()
        m_pdf.assert_not_called()
        m_ocr.assert_not_called()
        assert LAB_KEY_TROPONIN in result.measurements

    async def test_pymupdf_blank_falls_to_pdfminer(self, tmp_path):
        """PyMuPDF blank → pdfminer tried."""
        path = self._fake_pdf(tmp_path)
        state = AegisState(lab_pdf_path=str(path))

        with patch("tools.lab_report_parser._extract_via_pymupdf", return_value=None), \
             patch("tools.lab_report_parser._extract_via_pdfminer", return_value=LAB_TEXT) as m_pdf, \
             patch("tools.lab_report_parser._extract_via_easyocr") as m_ocr:

            result = await LabReportParser().run(state)

        assert isinstance(result, LabReportResult)
        m_pdf.assert_called_once()
        m_ocr.assert_not_called()

    async def test_pymupdf_exception_falls_to_pdfminer(self, tmp_path):
        """
        PyMuPDF exception → pdfminer tried (not OCR directly).
        _extract_via_pymupdf already catches its own exceptions and
        returns None. This test verifies that the None causes fallback,
        not a direct jump to OCR.
        """
        path = self._fake_pdf(tmp_path)
        state = AegisState(lab_pdf_path=str(path))

        # Simulate _extract_via_pymupdf returning None after internal exception
        with patch("tools.lab_report_parser._extract_via_pymupdf", return_value=None), \
             patch("tools.lab_report_parser._extract_via_pdfminer", return_value=LAB_TEXT) as m_pdf, \
             patch("tools.lab_report_parser._extract_via_easyocr") as m_ocr:

            result = await LabReportParser().run(state)

        assert isinstance(result, LabReportResult)
        m_pdf.assert_called_once()  # pdfminer ran
        m_ocr.assert_not_called()   # OCR did not

    async def test_both_blank_ocr_disabled_returns_tool_error(self, tmp_path):
        """PyMuPDF blank + pdfminer blank + AEGIS_OCR not set → ToolError."""
        path = self._fake_pdf(tmp_path)
        state = AegisState(lab_pdf_path=str(path))

        with patch("tools.lab_report_parser._extract_via_pymupdf", return_value=None), \
             patch("tools.lab_report_parser._extract_via_pdfminer", return_value=None), \
             patch.dict(os.environ, {}, clear=True):

            result = await LabReportParser().run(state)

        assert isinstance(result, ToolError)
        assert result.fatal is False
        assert "AEGIS_OCR=1" in result.reason

    async def test_both_blank_ocr_enabled_ocr_called(self, tmp_path):
        """PyMuPDF blank + pdfminer blank + AEGIS_OCR=1 → OCR called."""
        path = self._fake_pdf(tmp_path)
        state = AegisState(lab_pdf_path=str(path))

        with patch("tools.lab_report_parser._extract_via_pymupdf", return_value=None), \
             patch("tools.lab_report_parser._extract_via_pdfminer", return_value=None), \
             patch("tools.lab_report_parser._extract_via_easyocr", return_value=LAB_TEXT) as m_ocr, \
             patch.dict(os.environ, {"AEGIS_OCR": "1"}):

            result = await LabReportParser().run(state)

        assert isinstance(result, LabReportResult)
        m_ocr.assert_called_once()

    @pytest.mark.parametrize("ocr_value", ["true", "TRUE", "True", "yes", "YES"])
    async def test_both_blank_ocr_enabled_via_true_yes(self, tmp_path, ocr_value):
        """AEGIS_OCR accepts 'true' and 'yes' (case-insensitive) in addition to '1'."""
        path = self._fake_pdf(tmp_path)
        state = AegisState(lab_pdf_path=str(path))

        with patch("tools.lab_report_parser._extract_via_pymupdf", return_value=None), \
             patch("tools.lab_report_parser._extract_via_pdfminer", return_value=None), \
             patch("tools.lab_report_parser._extract_via_easyocr", return_value=LAB_TEXT) as m_ocr, \
             patch.dict(os.environ, {"AEGIS_OCR": ocr_value}):

            result = await LabReportParser().run(state)

        assert isinstance(result, LabReportResult), \
            f"Expected LabReportResult with AEGIS_OCR={ocr_value!r}"
        m_ocr.assert_called_once()

    async def test_ocr_also_blank_returns_tool_error(self, tmp_path):
        """All three tiers blank → ToolError even with AEGIS_OCR=1."""
        path = self._fake_pdf(tmp_path)
        state = AegisState(lab_pdf_path=str(path))

        with patch("tools.lab_report_parser._extract_via_pymupdf", return_value=None), \
             patch("tools.lab_report_parser._extract_via_pdfminer", return_value=None), \
             patch("tools.lab_report_parser._extract_via_easyocr", return_value=None), \
             patch.dict(os.environ, {"AEGIS_OCR": "1"}):

            result = await LabReportParser().run(state)

        assert isinstance(result, ToolError)
        assert result.fatal is False

    async def test_pdfminer_success_skips_ocr(self, tmp_path):
        """pdfminer succeeds → OCR never called even if AEGIS_OCR=1."""
        path = self._fake_pdf(tmp_path)
        state = AegisState(lab_pdf_path=str(path))

        with patch("tools.lab_report_parser._extract_via_pymupdf", return_value=None), \
             patch("tools.lab_report_parser._extract_via_pdfminer", return_value=LAB_TEXT), \
             patch("tools.lab_report_parser._extract_via_easyocr") as m_ocr, \
             patch.dict(os.environ, {"AEGIS_OCR": "1"}):

            result = await LabReportParser().run(state)

        assert isinstance(result, LabReportResult)
        m_ocr.assert_not_called()

    async def test_tool_error_hint_absent_when_ocr_enabled(self, tmp_path):
        """When OCR is enabled and exhausted, the 'set AEGIS_OCR=1' hint is not shown."""
        path = self._fake_pdf(tmp_path)
        state = AegisState(lab_pdf_path=str(path))

        with patch("tools.lab_report_parser._extract_via_pymupdf", return_value=None), \
             patch("tools.lab_report_parser._extract_via_pdfminer", return_value=None), \
             patch("tools.lab_report_parser._extract_via_easyocr", return_value=None), \
             patch.dict(os.environ, {"AEGIS_OCR": "true"}):

            result = await LabReportParser().run(state)

        assert isinstance(result, ToolError)
        assert "AEGIS_OCR=1" not in result.reason


# ── PDF path produces correct LabReportResult ─────────────────────

class TestPdfPathParsing:
    """
    Tests that parsed LabReportResult from a PDF path is correct.
    Uses mocked PyMuPDF so no real PDF library needed.
    """

    async def test_measurements_populated_from_pdf(self, tmp_path):
        path = _write_fake_pdf(tmp_path / "lab.pdf")
        state = AegisState(lab_pdf_path=str(path))

        pdf_text = "Haemoglobin: 14.2\nPotassium: 4.1\nGlucose: 95\n"

        with patch("tools.lab_report_parser._extract_via_pymupdf", return_value=pdf_text):
            result = await LabReportParser().run(state)

        assert isinstance(result, LabReportResult)
        assert LAB_KEY_HAEMOGLOBIN in result.measurements
        assert LAB_KEY_POTASSIUM in result.measurements
        assert LAB_KEY_GLUCOSE in result.measurements

    async def test_abnormal_values_populated_from_pdf(self, tmp_path):
        path = _write_fake_pdf(tmp_path / "lab.pdf")
        state = AegisState(lab_pdf_path=str(path))

        # Troponin 0.10 > ABNORMAL_HIGH_TROPONIN_NG_ML (0.04)
        pdf_text = "Troponin: 0.10\n"

        with patch("tools.lab_report_parser._extract_via_pymupdf", return_value=pdf_text):
            result = await LabReportParser().run(state)

        assert isinstance(result, LabReportResult)
        assert any("troponin" in v.lower() for v in result.abnormal_values)

    async def test_schema_version_from_pdf_path(self, tmp_path):
        path = _write_fake_pdf(tmp_path / "lab.pdf")
        state = AegisState(lab_pdf_path=str(path))

        with patch("tools.lab_report_parser._extract_via_pymupdf", return_value="WBC: 7.5"):
            result = await LabReportParser().run(state)

        assert isinstance(result, LabReportResult)
        assert result.schema_version == "1.0"

    async def test_extra_measurements_from_pdf(self, tmp_path):
        path = _write_fake_pdf(tmp_path / "lab.pdf")
        state = AegisState(lab_pdf_path=str(path))

        with patch("tools.lab_report_parser._extract_via_pymupdf", return_value="crp: 8.5"):
            result = await LabReportParser().run(state)

        assert isinstance(result, LabReportResult)
        assert "crp" in result.extra_measurements


# ── Functional entrypoint ─────────────────────────────────────────

async def test_parse_functional_entrypoint_pdf(tmp_path):
    """parse() functional entrypoint works for PDF path."""
    from tools.lab_report_parser import parse

    path = _write_fake_pdf(tmp_path / "lab.pdf")
    state = AegisState(lab_pdf_path=str(path))

    with patch("tools.lab_report_parser._extract_via_pymupdf", return_value="Glucose: 95"):
        result = await parse(state)

    assert isinstance(result, LabReportResult)
    assert LAB_KEY_GLUCOSE in result.measurements