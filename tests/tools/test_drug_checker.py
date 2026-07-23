"""
tests/tools/test_drug_checker.py — DrugInteractionChecker.

Tests structured DrugInteraction output, severity enum values,
confidence calculation, deduplication, and ToolError guard paths.

All tests use real AegisState instances — no mocks required.
"""

from __future__ import annotations


from schemas.drugs import DrugInteractionResult, DrugInteractionSeverity
from schemas.errors import ToolError
from schemas.state import AegisState
from tools.drug_checker import DrugInteractionChecker


# ── Helpers ────────────────────────────────────────────────────────

async def _check(medications: list[str]) -> DrugInteractionResult | ToolError:
    state = AegisState(medications_raw=medications)
    return await DrugInteractionChecker().run(state)


# ── Empty / no medications ─────────────────────────────────────────

async def test_empty_medications_returns_result_not_error():
    result = await _check([])
    assert isinstance(result, DrugInteractionResult)


async def test_empty_medications_zero_confidence():
    result = await _check([])
    assert isinstance(result, DrugInteractionResult)
    assert result.confidence == 0.0


async def test_empty_medications_no_interactions():
    result = await _check([])
    assert isinstance(result, DrugInteractionResult)
    assert result.interactions == []


# ── Known drugs resolve ────────────────────────────────────────────

async def test_known_drug_resolves():
    result = await _check(["warfarin"])
    assert isinstance(result, DrugInteractionResult)
    assert "warfarin" in result.resolved


async def test_unknown_drug_goes_to_unresolved():
    result = await _check(["AEGIS_TEST_UNRESOLVABLE_DRUG_XYZ"])
    assert isinstance(result, DrugInteractionResult)
    assert "aegis_test_unresolvable_drug_xyz" in result.unresolved


async def test_mixed_resolved_and_unresolved():
    result = await _check(["warfarin", "UNKNOWN_DRUG_123"])
    assert isinstance(result, DrugInteractionResult)
    assert "warfarin" in result.resolved
    assert "unknown_drug_123" in result.unresolved


# ── Confidence formula ─────────────────────────────────────────────

async def test_confidence_all_resolved():
    result = await _check(["warfarin", "aspirin"])
    assert isinstance(result, DrugInteractionResult)
    assert result.confidence == 1.0


async def test_confidence_none_resolved():
    result = await _check(["UNKNOWN_A", "UNKNOWN_B"])
    assert isinstance(result, DrugInteractionResult)
    assert result.confidence == 0.0


async def test_confidence_partial_resolution():
    result = await _check(["warfarin", "UNKNOWN_XYZ"])
    assert isinstance(result, DrugInteractionResult)
    # 1 resolved / 2 total = 0.5
    assert abs(result.confidence - 0.5) < 1e-9


# ── Interaction detection ──────────────────────────────────────────

async def test_warfarin_aspirin_severe_interaction():
    result = await _check(["warfarin", "aspirin"])
    assert isinstance(result, DrugInteractionResult)
    assert len(result.interactions) == 1
    interaction = result.interactions[0]
    assert interaction.severity == DrugInteractionSeverity.SEVERE
    assert set(interaction.drugs) == {"warfarin", "aspirin"}


async def test_aspirin_ibuprofen_moderate_interaction():
    result = await _check(["aspirin", "ibuprofen"])
    assert isinstance(result, DrugInteractionResult)
    assert len(result.interactions) == 1
    assert result.interactions[0].severity == DrugInteractionSeverity.MODERATE


async def test_digoxin_amiodarone_severe_interaction():
    result = await _check(["digoxin", "amiodarone"])
    assert isinstance(result, DrugInteractionResult)
    assert any(
        i.severity == DrugInteractionSeverity.SEVERE
        for i in result.interactions
    )


async def test_no_interaction_between_unrelated_drugs():
    result = await _check(["metformin", "warfarin"])
    assert isinstance(result, DrugInteractionResult)
    assert not any(
        set(i.drugs) == {"metformin", "warfarin"}
        for i in result.interactions
    )


async def test_metformin_contrast_dye_moderate():
    result = await _check(["metformin", "contrast dye"])
    assert isinstance(result, DrugInteractionResult)
    assert len(result.interactions) == 1
    assert result.interactions[0].severity == DrugInteractionSeverity.MODERATE


# ── Interaction description propagates ────────────────────────────

async def test_interaction_description_nonempty():
    result = await _check(["warfarin", "aspirin"])
    assert isinstance(result, DrugInteractionResult)
    assert result.interactions[0].description != ""


# ── Warnings — structured assertions ──────────────────────────────

async def test_unresolved_drug_produces_warning():
    """Unresolved drugs produce at least one warning."""
    result = await _check(["warfarin", "UNKNOWN_DRUG"])
    assert isinstance(result, DrugInteractionResult)
    assert len(result.unresolved) > 0
    assert len(result.warnings) > 0


async def test_detected_interaction_produces_warning():
    """Detected interactions produce at least one warning."""
    result = await _check(["warfarin", "aspirin"])
    assert isinstance(result, DrugInteractionResult)
    assert len(result.interactions) > 0
    assert len(result.warnings) > 0


async def test_no_warnings_for_safe_single_known_drug():
    """Single resolved drug with no interactions produces no warnings."""
    result = await _check(["metformin"])
    assert isinstance(result, DrugInteractionResult)
    assert len(result.unresolved) == 0
    assert len(result.interactions) == 0
    assert len(result.warnings) == 0


# ── Deduplication ─────────────────────────────────────────────────

async def test_duplicate_drug_does_not_create_duplicate_interactions():
    """Warfarin submitted twice should not generate a self-interaction."""
    result = await _check(["warfarin", "warfarin", "aspirin"])
    assert isinstance(result, DrugInteractionResult)
    assert len(result.interactions) == 1


async def test_duplicate_drug_resolved_once():
    result = await _check(["warfarin", "warfarin"])
    assert isinstance(result, DrugInteractionResult)
    assert result.resolved.count("warfarin") == 1


# ── Case normalisation ─────────────────────────────────────────────

async def test_drug_name_case_insensitive():
    result = await _check(["Warfarin", "ASPIRIN"])
    assert isinstance(result, DrugInteractionResult)
    assert "warfarin" in result.resolved
    assert "aspirin" in result.resolved


# ── Whitespace handling ────────────────────────────────────────────

async def test_drug_name_strips_whitespace():
    result = await _check(["  warfarin  ", " aspirin "])
    assert isinstance(result, DrugInteractionResult)
    assert "warfarin" in result.resolved


# ── Multiple pairs ─────────────────────────────────────────────────

async def test_three_drugs_with_two_severe_one_moderate():
    """
    warfarin + aspirin (SEVERE)
    warfarin + ibuprofen (SEVERE)
    aspirin + ibuprofen (MODERATE)
    """
    result = await _check(["warfarin", "aspirin", "ibuprofen"])
    assert isinstance(result, DrugInteractionResult)
    assert len(result.interactions) == 3

    severities = [i.severity for i in result.interactions]
    assert severities.count(DrugInteractionSeverity.SEVERE) == 2
    assert severities.count(DrugInteractionSeverity.MODERATE) == 1


# ── Schema compliance ──────────────────────────────────────────────

async def test_schema_version():
    result = await _check(["warfarin"])
    assert isinstance(result, DrugInteractionResult)
    assert result.schema_version == "1.0"


# ── Functional entrypoint ──────────────────────────────────────────

async def test_check_functional_entrypoint():
    from tools.drug_checker import check
    state = AegisState(medications_raw=["warfarin"])
    result = await check(state)
    assert isinstance(result, DrugInteractionResult)