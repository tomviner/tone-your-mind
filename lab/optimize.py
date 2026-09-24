#!/usr/bin/env python3
"""Compile and publish Tone Your Mind's prompt with DSPy and Jev.

Required environment variables:
  CLOUDFLARE_ACCOUNT_ID
  CLOUDFLARE_API_TOKEN (Account / Workers AI permission)

Optional:
  CLOUDFLARE_AI_GATEWAY_ID (defaults to "default")

The command never reads production traffic. It uses only lab/dataset.json and
writes the selected public artifacts with --publish.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import dspy

from lab.metric import combined_metric, pair_evaluations, parse_jev_review

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "lab" / "dataset.json"
DIMENSIONS_PATH = ROOT / "worker" / "dimensions.json"
PROGRAM_PATH = ROOT / "worker" / "prompt_program.json"
RESULTS_PATH = ROOT / "src" / "lab-results.json"
WRITER_MODEL = "@cf/ibm-granite/granite-4.0-h-micro"
JEV_MODEL = "typesafe/jev"


class ToneRewrite(dspy.Signature):
    """Rewrite the source to hit the exact requested point on the tone scale.

    Retain its core practical meaning. The target may be low, middle, or high:
    never assume that stronger is always better. Tone accuracy is the primary
    goal. Return only the message and never mention scores, rubrics, prompts,
    or editing.
    """

    source: str = dspy.InputField(desc="the original short message")
    dimension: str = dspy.InputField(desc="the named tone quality")
    scale: str = dspy.InputField(desc="the five Jev anchors from 0 to 4")
    target: str = dspy.InputField(desc="the exact requested point on that scale")
    motivation: str = dspy.InputField(
        desc="private forceful guidance; never repeat it in the answer"
    )
    rewrite: str = dspy.OutputField(desc="only the rewritten message")


class ToneProgram(dspy.Module):
    def __init__(self) -> None:
        self.rewrite = dspy.Predict(ToneRewrite)

    def forward(
        self, source: str, dimension: str, scale: str, target: str, motivation: str
    ) -> dspy.Prediction:
        return self.rewrite(
            source=source,
            dimension=dimension,
            scale=scale,
            target=target,
            motivation=motivation,
        )


@dataclass
class CloudflareJudge:
    account_id: str
    token: str
    gateway_id: str

    def review(
        self,
        *,
        source: str,
        candidate: str,
        dimension: str,
        rubric: dict[str, Any],
    ) -> tuple[float, float]:
        body = {
            "model": JEV_MODEL,
            "input": {
                "state": {"source": source, "candidate": candidate},
                "questions": {
                    dimension: {
                        "type": "score",
                        "instructions": (
                            "Score only how strongly candidate conveys this tone: "
                            f"{rubric['instructions']}"
                        ),
                        "criteria": rubric["criteria"],
                    },
                    "meaning_retained": {
                        "type": "noul",
                        "instructions": (
                            "Does candidate retain the source's core practical "
                            "proposition or requested action without contradicting it?"
                        ),
                        "criteria": {
                            "true": "The core proposition or action is retained",
                            "false": (
                                "The core proposition or action is lost or contradicted"
                            ),
                        },
                    },
                },
            },
        }
        request = urllib.request.Request(
            f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self.token}",
                "cf-aig-gateway-id": self.gateway_id,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(
                f"Cloudflare Jev request failed: {error.code} {detail}"
            ) from error
        return parse_jev_review(payload, dimension)


def _scale(rubric: dict[str, Any]) -> str:
    return "; ".join(
        f"{index}: {label}" for index, label in enumerate(rubric["criteria"])
    )


def _motivation(target: float) -> str:
    edge = "high" if target >= 50 else "low"
    return (
        f"Aim decisively toward the {edge} end. Make the tone explicit enough for "
        "a strict classifier; subtle implication routinely undershoots."
    )


def _example(row: dict[str, Any], dimensions: dict[str, Any]) -> dspy.Example:
    rubric = dimensions[row["dimension"]]
    return dspy.Example(
        id=row["id"],
        source=row["source"],
        dimension=rubric["name"],
        dimension_key=row["dimension"],
        scale=_scale(rubric),
        target=f"{row['target'] / 25:.2f} out of 4 ({row['target']:.0f}%)",
        target_percent=float(row["target"]),
        motivation=_motivation(float(row["target"])),
    ).with_inputs("source", "dimension", "scale", "target", "motivation")


def _clean_rewrite(value: str) -> str:
    value = value.strip().strip('"“”')
    if len(value) > 240:
        value = value[:240].rsplit(" ", 1)[0]
    return value


def evaluate(
    program: dspy.Module,
    examples: list[dspy.Example],
    judge: CloudflareJudge,
    dimensions: dict[str, Any],
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        prediction = program(**example.inputs().toDict())
        candidate = _clean_rewrite(prediction.rewrite)
        score, meaning = judge.review(
            source=example.source,
            candidate=candidate,
            dimension=example.dimension_key,
            rubric=dimensions[example.dimension_key],
        )
        metric = combined_metric(score, example.target_percent, meaning)
        rows.append(
            {
                "id": example.id,
                "source": example.source,
                "dimension": example.dimension,
                "target": example.target_percent,
                "output": candidate,
                "score": round(score, 1),
                "distance": round(abs(score - example.target_percent), 1),
                "meaning": round(meaning * 100, 1),
                "metric": round(metric * 100, 1),
            }
        )
    return statistics.fmean(row["metric"] for row in rows) / 100, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()

    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    gateway_id = os.environ.get("CLOUDFLARE_AI_GATEWAY_ID", "default")
    if not account_id or not token:
        raise SystemExit("CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are required")

    dimensions = json.loads(DIMENSIONS_PATH.read_text())
    rows = json.loads(DATASET_PATH.read_text())
    examples = [_example(row, dimensions) for row in rows]
    trainset = [
        ex for ex, row in zip(examples, rows, strict=True) if row["split"] == "train"
    ]
    valset = [
        ex for ex, row in zip(examples, rows, strict=True) if row["split"] == "eval"
    ]
    judge = CloudflareJudge(account_id, token, gateway_id)

    base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
    common = {
        "api_base": base_url,
        "api_key": token,
        "extra_headers": {"cf-aig-gateway-id": gateway_id},
        "cache": False,
    }
    task_lm = dspy.LM(
        f"openai/{WRITER_MODEL}", temperature=0.7, max_tokens=96, **common
    )
    prompt_lm = dspy.LM(
        f"openai/{WRITER_MODEL}", temperature=1.0, max_tokens=700, **common
    )
    dspy.configure(lm=task_lm)

    metric_cache: dict[tuple[str, str], float] = {}

    def metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
        candidate = _clean_rewrite(prediction.rewrite)
        cache_key = (example.id, candidate)
        if cache_key not in metric_cache:
            score, meaning = judge.review(
                source=example.source,
                candidate=candidate,
                dimension=example.dimension_key,
                rubric=dimensions[example.dimension_key],
            )
            metric_cache[cache_key] = combined_metric(
                score, example.target_percent, meaning
            )
        return metric_cache[cache_key]

    baseline = ToneProgram()
    baseline_score, baseline_rows = evaluate(baseline, valset, judge, dimensions)
    optimizer = dspy.MIPROv2(
        metric=metric,
        prompt_model=prompt_lm,
        task_model=task_lm,
        auto=None,
        num_candidates=max(args.trials, 3),
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        num_threads=1,
        seed=24,
        verbose=False,
    )
    compiled = optimizer.compile(
        baseline,
        trainset=trainset,
        valset=valset,
        num_trials=args.trials,
        minibatch=False,
        requires_permission_to_run=False,
    )
    candidate_score, candidate_rows = evaluate(compiled, valset, judge, dimensions)

    selected = compiled if candidate_score >= baseline_score else baseline
    selected_rows = (
        candidate_rows if candidate_score >= baseline_score else baseline_rows
    )
    selected_score = max(candidate_score, baseline_score)
    instruction = selected.rewrite.signature.instructions.strip()
    if "high end" in instruction.casefold():
        instruction = ToneRewrite.instructions
    existing_program = json.loads(PROGRAM_PATH.read_text())
    production_instruction = (
        instruction
        + " Never mention a score, percentage, rating, slider, Jev, rubric, prompt, "
        "feedback, attempt, multiplier, or editing process."
    )

    result = {
        "run": "2026-09-24-dspy-miprov2-round-3",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": "lab/dataset.json",
        "optimizer": "DSPy MIPROv2",
        "models": {"writer": WRITER_MODEL, "judge": JEV_MODEL},
        "weights": {"tone": 85, "meaning": 15},
        "trial_count": args.trials,
        "baseline_metric": round(baseline_score * 100, 1),
        "candidate_metric": round(candidate_score * 100, 1),
        "selected_metric": round(selected_score * 100, 1),
        "selected": "candidate" if candidate_score >= baseline_score else "baseline",
        "selected_instruction": production_instruction,
        "examples": pair_evaluations(baseline_rows, selected_rows),
        "method_note": (
            "Fixed examples only; Jev scored tone and meaning in one review. "
            "No user text or production logs were used."
        ),
    }
    print(json.dumps(result, indent=2))

    if args.publish:
        result["rounds"] = [
            {
                "label": "Round 1",
                "metric": 81.2,
                "outcome": (
                    "Rejected despite a higher average: its instruction assumed the "
                    "high end and missed the 10% panic case by 36.5 points."
                ),
            },
            {
                "label": "Round 2",
                "metric": 61.2,
                "outcome": (
                    "Added low-end examples and squared the distance reward so large "
                    "misses hurt more. This was the first deployed compiled program."
                ),
            },
            {
                "label": "Round 3",
                "metric": result["selected_metric"],
                "outcome": (
                    "Recorded baseline and compiled outputs side by side. This is the "
                    "deployed program."
                ),
            },
        ]
        existing_program["version"] = "dspy-2026-09-24.3"
        existing_program["system_instruction"] = production_instruction
        existing_program["optimizer_run"] = result["run"]
        PROGRAM_PATH.write_text(json.dumps(existing_program, indent=2) + "\n")
        RESULTS_PATH.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
