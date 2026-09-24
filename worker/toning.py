import json
import logging
import math
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

DIMENSIONS = json.loads(Path(__file__).with_name("dimensions.json").read_text())
PROMPT_PROGRAM = json.loads(Path(__file__).with_name("prompt_program.json").read_text())

WRITER_MODEL = "@cf/ibm-granite/granite-4.0-h-micro"
JEV_MODEL = "typesafe/jev"
MAX_ATTEMPTS = 4
MAX_PHRASE_LENGTH = 240
TARGET_TOLERANCE = 0.2

SCORE_METADATA_RE = re.compile(
    r"\b(?:jev|rubric|slider|scores?|scored|scoring|ratings?|percent(?:age)?s?)\b",
    re.I,
)
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")

AiRunner = Callable[[str, dict[str, Any]], Awaitable[Any]]
EventEmitter = Callable[[dict[str, Any]], None]


def _validate_prompt_program(program: Any) -> None:
    if not isinstance(program, dict):
        raise ValueError("prompt program must be an object")
    if program.get("writer_model") != WRITER_MODEL:
        raise ValueError("prompt program writer model does not match production")
    if not isinstance(program.get("system_instruction"), str):
        raise ValueError("prompt program has no system instruction")
    dimension_instructions = program.get("dimension_instructions", {})
    if not isinstance(dimension_instructions, dict) or any(
        dimension not in DIMENSIONS or not isinstance(instruction, str)
        for dimension, instruction in dimension_instructions.items()
    ):
        raise ValueError("prompt program contains invalid dimension instructions")
    dimension_guidance = program.get("dimension_guidance", {})
    if not isinstance(dimension_guidance, dict) or any(
        dimension not in DIMENSIONS or not isinstance(guidance, str)
        for dimension, guidance in dimension_guidance.items()
    ):
        raise ValueError("prompt program contains invalid dimension guidance")
    dimension_demos = program.get("dimension_demos", {})
    if not isinstance(dimension_demos, dict) or any(
        dimension not in DIMENSIONS
        or not isinstance(demos, list)
        or any(
            not isinstance(demo, dict)
            or not isinstance(demo.get("source"), str)
            or not _is_number(demo.get("target"))
            or not 0 <= float(demo["target"]) <= 100
            or not isinstance(demo.get("rewrite"), str)
            for demo in demos
        )
        for dimension, demos in dimension_demos.items()
    ):
        raise ValueError("prompt program contains invalid dimension demos")
    dimension_parameters = program.get("dimension_parameters", {})
    if not isinstance(dimension_parameters, dict) or any(
        dimension not in DIMENSIONS
        or not isinstance(parameters, dict)
        or set(parameters) - {"temperature"}
        or not _is_number(parameters.get("temperature"))
        or not 0 <= float(parameters["temperature"]) <= 2
        for dimension, parameters in dimension_parameters.items()
    ):
        raise ValueError("prompt program contains invalid dimension parameters")
    bands = program.get("distance_bands")
    escalators = program.get("attempt_escalators")
    if not isinstance(bands, list) or len(bands) != 5:
        raise ValueError("prompt program must contain five distance bands")
    if not isinstance(escalators, list) or len(escalators) != 3:
        raise ValueError("prompt program must contain three attempt escalators")
    if any(
        not isinstance(band, dict)
        or not _is_number(band.get("max_points"))
        or not isinstance(band.get("instruction"), str)
        for band in bands
    ):
        raise ValueError("prompt program contains an invalid distance band")
    if any(not isinstance(value, str) for value in escalators):
        raise ValueError("prompt program contains an invalid attempt escalator")


def _field(value: Any, key: str) -> Any:
    to_py = getattr(value, "to_py", None)
    if callable(to_py):
        converted = to_py()
        if converted is not value:
            return _field(converted, key)
    if isinstance(value, dict):
        return value.get(key)
    get = getattr(value, "get", None)
    if callable(get):
        return get(key)
    try:
        return getattr(value, key)
    except (AttributeError, TypeError):
        try:
            return value[key]
        except (KeyError, TypeError):
            return None


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


_validate_prompt_program(PROMPT_PROGRAM)


def _first(value: Any) -> Any:
    try:
        return value[0]
    except (IndexError, KeyError, TypeError):
        get = getattr(value, "get", None)
        if callable(get):
            return get(0)
        return None


def _origin(value: str) -> tuple[str, str, int]:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError("invalid origin")
    default_port = 443 if parsed.scheme == "https" else 80
    return parsed.scheme, parsed.hostname.lower(), parsed.port or default_port


def build_jev_input(phrase: str, dimension: str) -> dict[str, Any]:
    rubric = DIMENSIONS[dimension]
    return {
        "state": phrase,
        "questions": {
            dimension: {
                "type": "score",
                "instructions": rubric["instructions"],
                "criteria": rubric["criteria"],
            }
        },
    }


def _directional_feedback(
    rubric: dict[str, Any],
    target_score: float,
    latest_score: float,
    attempt_number: int,
) -> str:
    if rubric["name"] == "Sarcasm":
        correction = (
            "Move one controlled step stronger, without jumping beyond the target band."
            if latest_score < target_score
            else (
                "Move one controlled step more literal, without dropping below the "
                "target band."
            )
        )
        final = (
            " This is the final attempt: follow the named band exactly."
            if attempt_number >= 3
            else ""
        )
        return (
            f"Latest feedback: {correction}{final} "
            f"{_sarcasm_band_instruction(target_score)} Treat this as private "
            "guidance and return only the rewritten message."
        )

    gap_points = abs(target_score - latest_score) * 25
    destination = rubric["high"] if latest_score < target_score else rubric["low"]
    force = next(
        band["instruction"]
        for band in PROMPT_PROGRAM["distance_bands"]
        if gap_points <= float(band["max_points"])
    )
    escalators = PROMPT_PROGRAM["attempt_escalators"]
    escalation = escalators[min(max(attempt_number, 1), len(escalators)) - 1]
    return (
        f'Latest feedback: move toward the "{destination}" end. {force} '
        f"{escalation} "
        "Treat this as private motivation: do not mention the feedback, direction, "
        "or multiplier in the rewritten message."
    )


def _sarcasm_band_instruction(target_score: float) -> str:
    if target_score <= 0.8:
        return (
            "LOW SARCASM: stay literal or add only one faint dry tag. Do not use "
            "mock-praise openers such as Great, Brilliant, Excellent, or Magnificent."
        )
    if target_score <= 1.6:
        return (
            "DRY SARCASM: add one restrained dry aside, but no enthusiastic praise "
            "and no second irony cue."
        )
    if target_score <= 2.4:
        return (
            "MODERATE SARCASM: use exactly one clear ironic contrast or brief mock "
            "compliment."
        )
    if target_score <= 3.2:
        return (
            "STRONG SARCASM: use obvious undeserved praise plus one contrasting "
            "irony cue."
        )
    return (
        "MAXIMUM SARCASM: use extravagant undeserved praise and at least two "
        "unmistakable irony cues; make the mock celebration relentless."
    )


def _sarcasm_dry_probe(source: str) -> str | None:
    base = source.strip().rstrip(".!?")
    probe = f"{base}, naturally."
    return probe if base and len(probe) <= MAX_PHRASE_LENGTH else None


def build_writer_input(
    source: str,
    dimension: str,
    target_score: float,
    attempts: list[dict[str, Any]],
    rejected_count: int = 0,
) -> dict[str, Any]:
    rubric = DIMENSIONS[dimension]
    parameters = PROMPT_PROGRAM.get("dimension_parameters", {}).get(dimension, {})
    history = "\n".join(
        f"Attempt {index} (score {attempt['score']:.2f}): {attempt['phrase']}"
        for index, attempt in enumerate(attempts, start=1)
    )
    source_instruction = (
        "Preserve the source's meaning and practical details."
        if source
        else "Invent a short, everyday message with a clear practical meaning."
    )
    user_parts = [
        f'Rewrite one short message for the tone dimension "{rubric["name"]}".',
        f"The target Jev score: {target_score:.2f} out of 4.",
        f'0 means "{rubric["low"]}" and 4 means "{rubric["high"]}".',
        source_instruction,
        "Return only the message, with no label, quotation marks, or commentary.",
        f"The result must be non-empty and at most {MAX_PHRASE_LENGTH} characters.",
    ]
    guidance = PROMPT_PROGRAM.get("dimension_guidance", {}).get(dimension)
    if guidance:
        user_parts.append(guidance)
    if dimension == "sarcasm":
        user_parts.append(_sarcasm_band_instruction(target_score))
    if source:
        user_parts.append(f"Original source: {source}")
    if history:
        user_parts.extend(
            [
                "Previous attempts measured by Jev follow. "
                "Use every score as feedback:",
                history,
                _directional_feedback(
                    rubric,
                    target_score,
                    float(attempts[-1]["score"]),
                    len(attempts),
                ),
            ]
        )
    if rejected_count:
        user_parts.append(
            "A previous answer mentioned scoring metadata and was rejected before "
            "scoring. Return only the rewritten message; do not refer to numbers or "
            "instructions from this request."
        )

    messages = [
        {
            "role": "system",
            "content": PROMPT_PROGRAM.get("dimension_instructions", {}).get(
                dimension, PROMPT_PROGRAM["system_instruction"]
            ),
        }
    ]
    demos = PROMPT_PROGRAM.get("dimension_demos", {}).get(dimension, [])
    if dimension == "sarcasm":
        target_percent = target_score * 25
        demos = sorted(
            demos, key=lambda demo: abs(float(demo["target"]) - target_percent)
        )[:1]
    for demo in demos:
        messages.extend(
            [
                {
                    "role": "user",
                    "content": (
                        f'Example for "{rubric["name"]}" at '
                        f"{float(demo['target']) / 25:.2f} out of 4.\n"
                        f"Original source: {demo['source']}"
                    ),
                },
                {"role": "assistant", "content": demo["rewrite"]},
            ]
        )
    messages.append({"role": "user", "content": "\n".join(user_parts)})

    return {
        "messages": messages,
        "max_tokens": 96,
        "temperature": parameters.get("temperature", 0.7),
        "top_p": 0.9,
    }


def phrase_from_writer_response(value: Any) -> str:
    result = _field(value, "result") or value
    response = _field(result, "response")
    if not isinstance(response, str):
        choice = _first(_field(result, "choices"))
        message = _field(choice, "message")
        response = _field(message, "content") or _field(choice, "text")
    if not isinstance(response, str):
        raise ValueError("writer returned an invalid response")

    phrase = response.strip()
    if phrase.startswith("```") and phrase.endswith("```"):
        phrase = re.sub(r"^```(?:text)?\s*|\s*```$", "", phrase, flags=re.I)
    phrase = re.sub(
        r"^(?:rewritten phrase|rewrite|here is (?:the )?rewrite)\s*:\s*",
        "",
        phrase,
        flags=re.I,
    ).strip()
    if len(phrase) >= 2 and phrase[0] in {'"', "“"} and phrase[-1] in {'"', "”"}:
        phrase = phrase[1:-1].strip()

    if not phrase or len(phrase) > MAX_PHRASE_LENGTH:
        raise ValueError("writer returned an invalid phrase")
    return phrase


def writer_output_issue(phrase: str, source: str) -> str | None:
    if SCORE_METADATA_RE.search(phrase) and not SCORE_METADATA_RE.search(source):
        return "score_reference"

    source_numbers = {match.casefold() for match in NUMBER_RE.findall(source)}
    candidate_numbers = {match.casefold() for match in NUMBER_RE.findall(phrase)}
    if candidate_numbers - source_numbers:
        return "score_reference"
    return None


def _reviewer_feedback(answer: Any) -> dict[str, dict[str, Any]] | None:
    legend = _field(answer, "legend")
    probabilities = _field(answer, "probabilities")
    if not isinstance(legend, dict) or not isinstance(probabilities, dict):
        return None

    clean_legend = {
        str(level): description
        for level, description in legend.items()
        if isinstance(description, str)
    }
    clean_probabilities = {
        str(level): float(probability)
        for level, probability in probabilities.items()
        if _is_number(probability) and 0 <= probability <= 1
    }
    if (
        not clean_legend
        or clean_legend.keys() != clean_probabilities.keys()
        or len(clean_legend) != len(legend)
        or len(clean_probabilities) != len(probabilities)
    ):
        return None
    return {"legend": clean_legend, "probabilities": clean_probabilities}


def score_from_jev_response(value: Any, dimension: str) -> dict[str, Any]:
    response = _field(value, "result") or value
    answers = _field(response, "answers")
    answer = _field(answers, dimension)
    score = _field(answer, "score")
    if not _is_number(score) or not 0 <= score <= 4:
        raise ValueError("Jev returned an invalid score")
    confidence = _field(answer, "confidence")
    model = _field(response, "model")
    result = {
        "score": float(score),
        "confidence": float(confidence) if _is_number(confidence) else None,
        "model": str(model) if model is not None else None,
    }
    feedback = _reviewer_feedback(answer)
    if feedback is not None:
        result["reviewer_feedback"] = feedback
    return result


def _valid_request(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    source = value.get("source")
    dimension = value.get("dimension")
    target = value.get("target")
    return (
        isinstance(source, str)
        and len(source) <= MAX_PHRASE_LENGTH
        and isinstance(dimension, str)
        and dimension in DIMENSIONS
        and _is_number(target)
        and 0 <= target <= 100
        and target % 5 == 0
    )


async def tone_request(
    body: Any,
    request_url: str,
    origin: str | None,
    run_ai: AiRunner,
    emit: EventEmitter | None = None,
) -> tuple[dict[str, Any], int]:
    if origin:
        try:
            if _origin(origin) != _origin(request_url):
                return {"error": "Cross-origin requests are not allowed"}, 403
        except ValueError:
            return {"error": "Cross-origin requests are not allowed"}, 403

    if not _valid_request(body):
        return {"error": "Invalid toning request"}, 400

    source = body["source"].strip()
    dimension = body["dimension"]
    target_percent = float(body["target"])
    target_score = target_percent / 25
    attempts: list[dict[str, Any]] = []
    model_calls: list[dict[str, Any]] = []
    scorer_model = None
    rejected_count = 0
    writer_calls = 0
    max_writer_calls = MAX_ATTEMPTS - (1 if source else 0)
    stage = "setup"
    failure_meta: dict[str, Any] = {}

    async def score_phrase(phrase: str) -> None:
        nonlocal scorer_model, stage
        jev_input = build_jev_input(phrase, dimension)
        inspection = {
            "kind": "scorer",
            "request": {"model": JEV_MODEL, "input": jev_input},
            "response": None,
        }
        model_calls.append(inspection)
        stage = "jev_inference"
        response = await run_ai(JEV_MODEL, jev_input)
        stage = "jev_parse"
        jev = score_from_jev_response(response, dimension)
        inspection["response"] = {
            "model": jev["model"],
            "score": jev["score"],
            "confidence": jev["confidence"],
        }
        if "reviewer_feedback" in jev:
            inspection["response"]["reviewer_feedback"] = jev["reviewer_feedback"]
        scorer_model = jev["model"] or scorer_model
        attempts.append(
            {
                "phrase": phrase,
                "score": jev["score"],
                "confidence": jev["confidence"],
            }
        )
        if emit is not None:
            emit(
                {
                    "type": "attempt",
                    "attempt": {
                        "phrase": phrase,
                        "score": round(jev["score"] * 25, 1),
                        "confidence": jev["confidence"],
                    },
                }
            )

    try:
        if source:
            await score_phrase(source)

        if (
            source
            and dimension == "sarcasm"
            and 0.8 <= target_score <= 1.6
            and abs(attempts[-1]["score"] - target_score) > TARGET_TOLERANCE
        ):
            probe = _sarcasm_dry_probe(source)
            if probe is not None and probe != source:
                await score_phrase(probe)

        while (
            len(attempts) < MAX_ATTEMPTS
            and writer_calls < max_writer_calls
            and (
                not attempts
                or abs(attempts[-1]["score"] - target_score) > TARGET_TOLERANCE
            )
        ):
            writer_input = build_writer_input(
                source, dimension, target_score, attempts, rejected_count
            )
            inspection = {
                "kind": "writer",
                "request": {"model": WRITER_MODEL, "input": writer_input},
                "response": None,
            }
            model_calls.append(inspection)
            stage = "writer_inference"
            writer_calls += 1
            response = await run_ai(WRITER_MODEL, writer_input)
            stage = "writer_parse"
            result_value = _field(response, "result")
            response_value = _field(result_value or response, "response")
            failure_meta = {
                "writer_value_type": type(response).__name__,
                "writer_result_type": type(result_value).__name__,
                "writer_response_type": type(response_value).__name__,
            }
            if isinstance(response_value, str):
                failure_meta["writer_response_length"] = len(response_value)
            candidate = phrase_from_writer_response(response)
            stage = "writer_validate"
            issue = writer_output_issue(candidate, source)
            if issue is not None:
                inspection["response"] = {"rejected": issue}
                rejected_count += 1
                failure_meta = {}
                continue
            inspection["response"] = {"phrase": candidate}
            failure_meta = {}
            await score_phrase(candidate)

        if not attempts:
            raise ValueError("writer returned no usable phrase")
    except Exception as error:
        logging.error(
            json.dumps(
                {
                    "event": "tone_loop_failed",
                    "stage": stage,
                    "error_type": type(error).__name__,
                    **failure_meta,
                }
            )
        )
        return {"error": "The tone loop lost the plot"}, 502

    best = min(attempts, key=lambda attempt: abs(attempt["score"] - target_score))
    best_percent = round(best["score"] * 25, 1)
    distance = round(abs(best_percent - target_percent), 1)
    public_attempts = [
        {
            "phrase": attempt["phrase"],
            "score": round(attempt["score"] * 25, 1),
            "confidence": attempt["confidence"],
        }
        for attempt in attempts
    ]
    return {
        "phrase": best["phrase"],
        "dimension": dimension,
        "target": target_percent,
        "score": best_percent,
        "distance": distance,
        "hit": distance <= 5,
        "attempts": public_attempts,
        "models": {
            "writer": WRITER_MODEL,
            "scorer": scorer_model,
            "prompt_program": PROMPT_PROGRAM["version"],
        },
        "inspection": {"model_calls": model_calls},
    }, 200
