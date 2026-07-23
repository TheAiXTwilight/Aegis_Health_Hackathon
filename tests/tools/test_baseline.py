"""
tests/tools/test_baseline.py — Tests for z-score and baseline summary.

Place at: tests/tools/test_baseline.py
"""
from __future__ import annotations

import pytest
from backend.baseline import z_score, baseline_summary


class TestZScore:
    def test_normal_distribution(self):
        history = [70.0, 72.0, 74.0, 71.0, 73.0]
        result = z_score(80.0, history)
        assert result is not None
        assert result > 2.0  # well above mean of ~72

    def test_current_equals_mean(self):
        history = [70.0, 72.0, 74.0]
        result = z_score(72.0, history)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_insufficient_history(self):
        assert z_score(100.0, [100.0]) is None
        assert z_score(100.0, [100.0, 101.0]) is None

    def test_exactly_three_values(self):
        result = z_score(80.0, [70.0, 72.0, 74.0])
        assert result is not None  # 3 = minimum viable

    def test_zero_variance(self):
        """When all values are identical, z=0 if current matches."""
        result = z_score(100.0, [100.0, 100.0, 100.0, 100.0])
        assert result == 0.0

    def test_zero_variance_different_current(self):
        """All history = 100, current = 120: infinite deviation."""
        result = z_score(120.0, [100.0, 100.0, 100.0, 100.0])
        assert result is None  # can't compute meaningful z-score

    def test_negative_z(self):
        history = [70.0, 72.0, 74.0, 71.0, 73.0]
        result = z_score(66.0, history)
        assert result is not None
        assert result < -1.0  # far below mean


class TestBaselineSummary:
    def test_full_summary(self):
        summary = baseline_summary(
            current=88.0,
            history=[70.0, 71.0, 72.0, 73.0, 70.0, 72.0, 71.0, 73.0],
            vital_name="heart_rate",
        )
        assert summary["vital"] == "heart_rate"
        assert summary["current"] == 88.0
        assert summary["mean"] is not None
        assert summary["z_score"] is not None
        assert summary["z_score"] > 3.0  # very elevated
        assert summary["interpretation"] == "elevated"
        assert summary["insufficient_history"] is False
        assert summary["sample_size"] == 8

    def test_insufficient_history(self):
        summary = baseline_summary(
            current=72.0,
            history=[70.0, 71.0],  # only 2
            vital_name="spo2",
        )
        assert summary["z_score"] is None
        assert summary["insufficient_history"] is True
        assert summary["current"] == 72.0
        assert summary["mean"] == pytest.approx(70.5)

    def test_normal_interpretation(self):
        summary = baseline_summary(
            current=72.0,
            history=[70.0, 71.0, 73.0, 72.0, 71.0, 70.0],
            vital_name="hr",
        )
        assert summary["interpretation"] == "normal"

    def test_mild_deviation(self):
        summary = baseline_summary(
            current=76.0,
            history=[70.0, 71.0, 72.0, 71.0],
            vital_name="hr",
        )
        assert "mild deviation" in summary["interpretation"]

    def test_low_interpretation(self):
        summary = baseline_summary(
            current=58.0,
            history=[70.0, 71.0, 72.0, 73.0, 70.0, 72.0],
            vital_name="hr",
        )
        assert summary["interpretation"] == "low"
