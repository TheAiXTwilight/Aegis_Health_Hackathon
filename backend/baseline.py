"""
backend/baseline.py — Personal baseline risk z-score calculator.

Computes per-user z-scores by comparing a current vital value against
that user's historical mean and standard deviation.

Requires at least 3 historical values — returns None ("insufficient history")
when fewer exist, matching the spec requirement.
"""
from __future__ import annotations


def z_score(current: float, history: list[float]) -> float | None:
    """
    Compute the z-score of `current` against `history`.

        z = (current - μ) / σ

    Returns None when:
        - Fewer than 3 historical values (insufficient baseline)
        - All historical values are identical and current differs (infinite deviation)
    
    Returns 0.0 when σ = 0 and current equals the mean.
    """
    if len(history) < 3:
        return None

    mu = sum(history) / len(history)
    variance = sum((x - mu) ** 2 for x in history) / (len(history) - 1)

    if variance == 0:
        return 0.0 if current == mu else None

    return (current - mu) / (variance ** 0.5)


def baseline_summary(
    *,
    current: float,
    history: list[float],
    vital_name: str,
) -> dict:
    """
    Build a human-readable baseline summary for a single vital.

    Returns:
        {
            "vital": "heart_rate",
            "current": 88.0,
            "mean": 72.0,
            "std": 6.5,
            "z_score": 2.46,
            "sample_size": 10,
            "interpretation": "elevated",
            "insufficient_history": false
        }
    """
    z = z_score(current, history)
    mu = sum(history) / len(history) if history else None
    std = _std(history) if len(history) >= 3 else None

    interpretation = None
    if z is not None:
        if abs(z) < 1.0:
            interpretation = "normal"
        elif abs(z) < 2.0:
            interpretation = "mild deviation"
        elif z > 0:
            interpretation = "elevated"
        else:
            interpretation = "low"

    return {
        "vital": vital_name,
        "current": current,
        "mean": mu,
        "std": std,
        "z_score": z,
        "sample_size": len(history),
        "interpretation": interpretation,
        "insufficient_history": z is None and len(history) > 0,
    }


def _std(values: list[float]) -> float | None:
    """Sample standard deviation. Returns None if < 3 values."""
    if len(values) < 3:
        return None
    mu = sum(values) / len(values)
    var = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return var ** 0.5