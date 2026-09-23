"""Pure scoring helpers shared by the optimiser and its tests."""

from typing import Any


def tone_accuracy(score: float, target: float) -> float:
    """Return a 0..1 reward that penalises large misses disproportionately."""
    linear = max(0.0, 1.0 - abs(score - target) / 100.0)
    return linear * linear


def combined_metric(
    score: float,
    target: float,
    meaning_probability: float,
    *,
    tone_weight: float = 0.85,
    meaning_weight: float = 0.15,
) -> float:
    """Tone dominates; Jev's meaning probability is the smaller constraint."""
    if not 0 <= meaning_probability <= 1:
        raise ValueError("meaning_probability must be between 0 and 1")
    if tone_weight < 0 or meaning_weight < 0 or tone_weight + meaning_weight == 0:
        raise ValueError("metric weights must be non-negative and non-zero")
    total = tone_weight + meaning_weight
    return (
        tone_accuracy(score, target) * tone_weight
        + meaning_probability * meaning_weight
    ) / total


def parse_jev_review(payload: dict[str, Any], dimension: str) -> tuple[float, float]:
    """Extract percentage tone score and semantic-retention probability."""
    result = payload.get("result", payload)
    if isinstance(result, dict) and result.get("state") == "Completed":
        result = result.get("result", {})
    answers = result.get("answers", {}) if isinstance(result, dict) else {}
    tone = answers.get(dimension, {})
    meaning = answers.get("meaning_retained", {})
    raw_score = tone.get("score")
    raw_meaning = meaning.get("noul")
    if not isinstance(raw_score, (int, float)) or isinstance(raw_score, bool):
        raise ValueError("Jev review has no tone score")
    if not isinstance(raw_meaning, (int, float)) or isinstance(raw_meaning, bool):
        raise ValueError("Jev review has no meaning probability")
    return float(raw_score) * 25, float(raw_meaning)
