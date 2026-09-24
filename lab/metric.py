"""Pure scoring helpers shared by the optimiser and its tests."""

from typing import Any


def pair_evaluations(
    baseline: list[dict[str, Any]], selected: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Pair the same held-out examples before and after prompt optimisation."""
    selected_by_id = {row["id"]: row for row in selected}
    if set(selected_by_id) != {row["id"] for row in baseline}:
        raise ValueError("baseline and selected evaluations must contain the same ids")

    shared = ["output", "score", "distance", "meaning", "metric"]
    if all(
        "quality" in row and "quality" in selected_by_id[row["id"]] for row in baseline
    ):
        shared.append("quality")
    return [
        {
            "id": row["id"],
            "source": row["source"],
            "dimension": row["dimension"],
            "target": row["target"],
            "before": {key: row[key] for key in shared},
            "after": {key: selected_by_id[row["id"]][key] for key in shared},
        }
        for row in baseline
    ]


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


def sarcasm_metric(
    score: float,
    target: float,
    meaning_probability: float,
    quality_probability: float,
) -> float:
    """Reward dial accuracy only when both meaning and writing quality survive."""
    if not 0 <= quality_probability <= 1:
        raise ValueError("quality_probability must be between 0 and 1")
    if abs(score - target) > 35:
        return 0.0
    weighted = (
        tone_accuracy(score, target) * 0.75
        + meaning_probability * 0.10
        + quality_probability * 0.15
    )
    return weighted * min(meaning_probability, quality_probability)


def parse_jev_review(
    payload: dict[str, Any], dimension: str
) -> tuple[float, float, float]:
    """Extract tone, semantic retention, and optional specialist quality."""
    result = payload.get("result", payload)
    if isinstance(result, dict) and result.get("state") == "Completed":
        result = result.get("result", {})
    answers = result.get("answers", {}) if isinstance(result, dict) else {}
    tone = answers.get(dimension, {})
    meaning = answers.get("meaning_retained", {})
    raw_score = tone.get("score")
    raw_meaning = meaning.get("noul")
    quality = answers.get("sarcasm_quality", {})
    raw_quality = quality.get("noul", 1.0)
    if not isinstance(raw_score, (int, float)) or isinstance(raw_score, bool):
        raise ValueError("Jev review has no tone score")
    if not isinstance(raw_meaning, (int, float)) or isinstance(raw_meaning, bool):
        raise ValueError("Jev review has no meaning probability")
    if not isinstance(raw_quality, (int, float)) or isinstance(raw_quality, bool):
        raise ValueError("Jev review has no sarcasm quality probability")
    return float(raw_score) * 25, float(raw_meaning), float(raw_quality)
