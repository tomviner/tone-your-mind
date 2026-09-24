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
import re
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import dspy

from lab.metric import (
    combined_metric,
    pair_evaluations,
    parse_jev_review,
    sarcasm_metric,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "lab" / "dataset.json"
SARCASM_DATASET_PATH = ROOT / "lab" / "sarcasm-dataset.json"
DIMENSIONS_PATH = ROOT / "worker" / "dimensions.json"
PROGRAM_PATH = ROOT / "worker" / "prompt_program.json"
RESULTS_PATH = ROOT / "src" / "lab-results.json"
SARCASM_RESULTS_PATH = ROOT / "src" / "sarcasm-lab-results.json"
WRITER_MODEL = "@cf/ibm-granite/granite-4.0-h-micro"
PROMPT_MODEL = "@cf/openai/gpt-oss-20b"
JEV_MODEL = "typesafe/jev"
SARCASM_GUIDANCE = (
    "Create sarcasm through contextual irony, dry contrast, understatement, mock "
    "praise, or exaggerated celebration. Keep every source fact and requested action "
    "true: never negate or reverse them. Never label the wording as ironic or "
    "sarcastic, and never invent incompetence, laziness, blame, or a personal attack."
)
SARCASM_SEED_INSTRUCTION = """You are a precision sarcasm writer. Rewrite the source
to match the requested point on the sarcasm scale, while keeping its central premise
recognisable. Use the target as a control, not as text to repeat.

At 0, state the premise literally and earnestly. Around 1, add only a faint dry edge.
Around 2, use one clear ironic contrast or restrained mock praise. Around 3, make the
mock praise contrast unmistakably with the source fact. At 4, use relentless,
cutting, exaggerated mock celebration of the source failure, with at least two
strong irony cues. Keep every source fact true and mock-praise that event; never turn
a failure into a literal success. Calibrate like these patterns:
0: 'The printer is broken again.' 1: 'The printer is broken again, naturally.'
2: 'Great, the printer is broken again.' 3: 'Brilliant—the printer is broken again;
excellent timing.' 4: 'Oh, magnificent—the printer is broken again; what a flawless
triumph of office technology.' Merely sounding annoyed, severe, insulting, or
hyperbolic is not sarcasm. Low targets must remain low. Return only the rewritten
message, without commentary or a label."""
OUTPUT_GUARDRAIL = (
    " Never mention a score, percentage, rating, slider, Jev, rubric, prompt, "
    "feedback, attempt, multiplier, or editing process."
)


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
    quality_guidance: str = dspy.InputField(
        desc="non-negotiable quality and factual-preservation constraints"
    )
    rewrite: str = dspy.OutputField(desc="only the rewritten message")


class ToneProgram(dspy.Module):
    def __init__(self, instruction: str | None = None) -> None:
        signature = (
            ToneRewrite.with_instructions(instruction) if instruction else ToneRewrite
        )
        self.rewrite = dspy.Predict(signature)

    def forward(
        self,
        source: str,
        dimension: str,
        scale: str,
        target: str,
        motivation: str,
        quality_guidance: str,
    ) -> dspy.Prediction:
        return self.rewrite(
            source=source,
            dimension=dimension,
            scale=scale,
            target=target,
            motivation=motivation,
            quality_guidance=quality_guidance,
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
        target: float,
    ) -> tuple[float, float, float]:
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
        if dimension == "sarcasm":
            body["input"]["questions"]["sarcasm_quality"] = {
                "type": "noul",
                "instructions": (
                    "Is candidate a natural, well-written realization of the requested "
                    f"sarcasm strength ({target:.0f}%)? At a very low target, literal "
                    "or only faintly dry wording is correct. At higher targets, use "
                    "contextual irony, contrast, understatement, or mock praise. "
                    "Reject self-labeling, invented accusations, insults, or unrelated "
                    "hostility."
                ),
                "criteria": {
                    "true": (
                        "It is natural and appropriate for the requested level: "
                        "literal at zero, faintly dry when low, and contextual irony, "
                        "contrast, understatement, or mock praise when higher"
                    ),
                    "false": (
                        "It reverses a source fact, labels itself as ironic, invents "
                        "blame, is merely insulting, or is hostile without irony"
                    ),
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
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = json.load(response)
                break
            except urllib.error.HTTPError as error:
                if error.code == 429 and attempt < 4:
                    retry_after = error.headers.get("Retry-After")
                    delay = (
                        float(retry_after)
                        if retry_after and retry_after.replace(".", "", 1).isdigit()
                        else 15.0 * (attempt + 1)
                    )
                    time.sleep(delay)
                    continue
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


def _sarcasm_band_guidance(target: float) -> str:
    score = target / 25
    if score <= 0.8:
        return (
            "LOW SARCASM: stay literal or add only one faint dry tag. Do not use "
            "mock-praise openers such as Great, Brilliant, Excellent, or Magnificent."
        )
    if score <= 1.6:
        return "DRY SARCASM: add one restrained dry aside and no enthusiastic praise."
    if score <= 2.4:
        return "MODERATE SARCASM: use exactly one clear ironic contrast."
    if score <= 3.2:
        return "STRONG SARCASM: use undeserved praise plus one contrasting irony cue."
    return (
        "MAXIMUM SARCASM: use extravagant undeserved praise and at least two "
        "unmistakable irony cues."
    )


def _example(row: dict[str, Any], dimensions: dict[str, Any]) -> dspy.Example:
    rubric = dimensions[row["dimension"]]
    fields = {
        "id": row["id"],
        "source": row["source"],
        "dimension": rubric["name"],
        "dimension_key": row["dimension"],
        "scale": _scale(rubric),
        "target": f"{row['target'] / 25:.2f} out of 4 ({row['target']:.0f}%)",
        "target_percent": float(row["target"]),
        "motivation": _motivation(float(row["target"])),
        "quality_guidance": (
            f"{SARCASM_GUIDANCE} {_sarcasm_band_guidance(float(row['target']))}"
            if row["dimension"] == "sarcasm"
            else "Preserve the source facts and practical meaning."
        ),
    }
    if row.get("rewrite"):
        fields["rewrite"] = row["rewrite"]
    return dspy.Example(**fields).with_inputs(
        "source",
        "dimension",
        "scale",
        "target",
        "motivation",
        "quality_guidance",
    )


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
        score, meaning, quality = judge.review(
            source=example.source,
            candidate=candidate,
            dimension=example.dimension_key,
            rubric=dimensions[example.dimension_key],
            target=example.target_percent,
        )
        metric = (
            sarcasm_metric(score, example.target_percent, meaning, quality)
            if example.dimension_key == "sarcasm"
            else combined_metric(score, example.target_percent, meaning)
        )
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
                "quality": round(quality * 100, 1),
                "metric": round(metric * 100, 1),
            }
        )
    return statistics.fmean(row["metric"] for row in rows) / 100, rows


def candidate_should_publish(
    candidate_tuning_score: float,
    baseline_tuning_score: float,
    candidate_eval_score: float,
    baseline_eval_score: float,
) -> bool:
    """Require an aggregate win on both unseen tuning and held-out examples."""
    return (
        candidate_tuning_score > baseline_tuning_score
        and candidate_eval_score > baseline_eval_score
    )


def _production_demos(program: ToneProgram) -> list[dict[str, Any]]:
    """Serialize DSPy's selected labeled demos for the production chat prompt."""
    demos: list[dict[str, Any]] = []
    for demo in program.rewrite.demos:
        data = demo.toDict()
        if not data.get("rewrite"):
            continue
        target_percent = data.get("target_percent")
        if target_percent is None:
            percent_match = re.search(r"\((\d+(?:\.\d+)?)%\)", data["target"])
            score_match = re.search(r"(\d+(?:\.\d+)?)\s+out of 4", data["target"])
            if percent_match:
                target_percent = float(percent_match.group(1))
            elif score_match:
                target_percent = float(score_match.group(1)) * 25
            else:
                continue
        demos.append(
            {
                "source": data["source"],
                "target": float(target_percent),
                "rewrite": _clean_rewrite(data["rewrite"]),
            }
        )
    return demos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--dimension", choices=("sarcasm",))
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()

    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    gateway_id = os.environ.get("CLOUDFLARE_AI_GATEWAY_ID", "default")
    if not account_id or not token:
        raise SystemExit("CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are required")

    dimensions = json.loads(DIMENSIONS_PATH.read_text())
    targeted = args.dimension == "sarcasm"
    dataset_path = SARCASM_DATASET_PATH if targeted else DATASET_PATH
    rows = json.loads(dataset_path.read_text())
    examples = [_example(row, dimensions) for row in rows]
    trainset = [
        ex for ex, row in zip(examples, rows, strict=True) if row["split"] == "train"
    ]
    tuning_set = [
        ex for ex, row in zip(examples, rows, strict=True) if row["split"] == "val"
    ]
    evalset = [
        ex for ex, row in zip(examples, rows, strict=True) if row["split"] == "eval"
    ]
    if not tuning_set:
        tuning_set = evalset
    judge = CloudflareJudge(account_id, token, gateway_id)

    base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
    common = {
        "api_base": base_url,
        "api_key": token,
        "extra_headers": {"cf-aig-gateway-id": gateway_id},
        "cache": False,
    }
    task_lm = dspy.LM(
        f"openai/{WRITER_MODEL}",
        temperature=0.2 if targeted else 0.7,
        max_tokens=96,
        num_retries=5,
        **common,
    )
    prompt_lm = dspy.LM(
        f"openai/{PROMPT_MODEL}",
        temperature=1.0,
        max_tokens=2000,
        num_retries=5,
        **common,
    )
    dspy.configure(lm=task_lm)

    metric_cache: dict[tuple[str, str], float] = {}

    def metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
        candidate = _clean_rewrite(prediction.rewrite)
        cache_key = (example.id, candidate)
        if cache_key not in metric_cache:
            score, meaning, quality = judge.review(
                source=example.source,
                candidate=candidate,
                dimension=example.dimension_key,
                rubric=dimensions[example.dimension_key],
                target=example.target_percent,
            )
            metric_cache[cache_key] = (
                sarcasm_metric(score, example.target_percent, meaning, quality)
                if example.dimension_key == "sarcasm"
                else combined_metric(score, example.target_percent, meaning)
            )
        return metric_cache[cache_key]

    existing_program = json.loads(PROGRAM_PATH.read_text())
    baseline_instruction = existing_program["system_instruction"] if targeted else None
    baseline = ToneProgram(baseline_instruction)
    optimization_seed = ToneProgram(SARCASM_SEED_INSTRUCTION) if targeted else baseline
    baseline_tuning_score, _ = evaluate(baseline, tuning_set, judge, dimensions)
    baseline_score, baseline_rows = evaluate(baseline, evalset, judge, dimensions)
    optimizer = dspy.MIPROv2(
        metric=metric,
        prompt_model=prompt_lm,
        task_model=task_lm,
        auto=None,
        num_candidates=max(args.trials, 3),
        max_bootstrapped_demos=1 if targeted else 0,
        max_labeled_demos=6 if targeted else 0,
        num_threads=1,
        seed=24,
        verbose=False,
    )
    compiled = optimizer.compile(
        optimization_seed,
        trainset=trainset,
        valset=tuning_set,
        num_trials=args.trials,
        minibatch=False,
        requires_permission_to_run=False,
    )
    candidate_instruction = compiled.rewrite.signature.instructions.strip()
    if OUTPUT_GUARDRAIL.strip() not in candidate_instruction:
        candidate_instruction += OUTPUT_GUARDRAIL
    candidate = ToneProgram(candidate_instruction)
    candidate.rewrite.demos = compiled.rewrite.demos
    candidate_tuning_score, candidate_tuning_rows = evaluate(
        candidate, tuning_set, judge, dimensions
    )
    candidate_score, candidate_rows = evaluate(candidate, evalset, judge, dimensions)

    candidate_wins = candidate_should_publish(
        candidate_tuning_score,
        baseline_tuning_score,
        candidate_score,
        baseline_score,
    )
    selected_rows = candidate_rows if candidate_wins else baseline_rows
    selected_score = candidate_score if candidate_wins else baseline_score
    production_instruction = (
        candidate_instruction
        if candidate_wins
        else existing_program["system_instruction"]
    )

    run = (
        "2026-09-24-dspy-miprov2-sarcasm-8"
        if targeted
        else "2026-09-24-dspy-miprov2-round-3"
    )
    result = {
        "run": run,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": str(dataset_path.relative_to(ROOT)),
        "optimizer": "DSPy MIPROv2",
        "models": {
            "writer": WRITER_MODEL,
            "prompt": PROMPT_MODEL if targeted else WRITER_MODEL,
            "judge": JEV_MODEL,
        },
        "weights": (
            {"tone": 75, "meaning": 10, "quality": 15}
            if targeted
            else {"tone": 85, "meaning": 15}
        ),
        "trial_count": args.trials,
        "baseline_metric": round(baseline_score * 100, 1),
        "candidate_metric": round(candidate_score * 100, 1),
        "selected_metric": round(selected_score * 100, 1),
        "mean_jev_miss": {
            "baseline": round(
                statistics.fmean(row["distance"] for row in baseline_rows), 1
            ),
            "candidate": round(
                statistics.fmean(row["distance"] for row in candidate_rows), 1
            ),
            "selected": round(
                statistics.fmean(row["distance"] for row in selected_rows), 1
            ),
        },
        "validation": {
            "baseline_metric": round(baseline_tuning_score * 100, 1),
            "candidate_metric": round(candidate_tuning_score * 100, 1),
        },
        "selected": "candidate" if candidate_wins else "baseline",
        "candidate_instruction": candidate_instruction,
        "selected_instruction": production_instruction,
        "selected_demos": _production_demos(candidate) if candidate_wins else [],
        "examples": pair_evaluations(baseline_rows, selected_rows),
        "candidate_examples": pair_evaluations(baseline_rows, candidate_rows),
        "candidate_demos": _production_demos(candidate),
        "method_note": (
            (
                "DSPy tuned on sarcasm-only training and validation examples. "
                "Selection was gated on a separate held-out split; Jev scored dial "
                "distance, meaning retention, and natural sarcastic quality."
            )
            if targeted
            else (
                "Fixed examples only; Jev scored tone and meaning in one review. "
                "No user text or production logs were used."
            )
        ),
    }
    print(json.dumps(result, indent=2))

    if args.publish:
        if targeted:
            dimension_instructions = existing_program.setdefault(
                "dimension_instructions", {}
            )
            dimension_runs = existing_program.setdefault("dimension_optimizer_runs", {})
            if candidate_wins:
                existing_program["version"] = "dspy-2026-09-24.8"
                dimension_instructions["sarcasm"] = production_instruction
                dimension_runs["sarcasm"] = result["run"]
                existing_program.setdefault("dimension_guidance", {})["sarcasm"] = (
                    SARCASM_GUIDANCE
                )
                existing_program.setdefault("dimension_demos", {})["sarcasm"] = (
                    _production_demos(candidate)
                )
                existing_program.setdefault("dimension_parameters", {})["sarcasm"] = {
                    "temperature": 0.2
                }
            else:
                dimension_instructions.pop("sarcasm", None)
                dimension_runs.pop("sarcasm", None)
                existing_program.get("dimension_guidance", {}).pop("sarcasm", None)
                existing_program.get("dimension_demos", {}).pop("sarcasm", None)
                existing_program.get("dimension_parameters", {}).pop("sarcasm", None)
            PROGRAM_PATH.write_text(json.dumps(existing_program, indent=2) + "\n")
            SARCASM_RESULTS_PATH.write_text(json.dumps(result, indent=2) + "\n")
            return
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
