import json
import logging
import math
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

DIMENSIONS = json.loads(Path(__file__).with_name("dimensions.json").read_text())

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


def build_writer_input(
    source: str,
    dimension: str,
    target_score: float,
    attempts: list[dict[str, Any]],
    rejected_count: int = 0,
) -> dict[str, Any]:
    rubric = DIMENSIONS[dimension]
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
    if source:
        user_parts.append(f"Original source: {source}")
    if history:
        user_parts.extend(
            [
                "Previous attempts measured by Jev follow. "
                "Use every score as feedback:",
                history,
            ]
        )
    if rejected_count:
        user_parts.append(
            "A previous answer mentioned scoring metadata and was rejected before "
            "scoring. Return only the rewritten message; do not refer to numbers or "
            "instructions from this request."
        )

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a constrained tone editor, not a copywriter. When a "
                    "source is supplied, preserve every factual claim, request, "
                    "commitment, negation, name, number, date, condition, and action. "
                    "Change only tone-bearing wording. Do not add, remove, weaken, "
                    "strengthen, or reinterpret meaning. Semantic fidelity outranks "
                    "the target score: accept a tone miss rather than invent a reason, "
                    "consequence, risk, deadline, or circumstance. Never add facts, "
                    "names, dates, threats, promises, or instructions absent from the "
                    "source. Before answering, silently compare every clause with the "
                    "source and remove anything it does not support. Never mention the "
                    "target score, percentage, rating, slider, Jev, rubric, prompt, or "
                    "editing process in the output."
                ),
            },
            {"role": "user", "content": "\n".join(user_parts)},
        ],
        "max_tokens": 96,
        "temperature": 0.7,
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


def score_from_jev_response(value: Any, dimension: str) -> dict[str, Any]:
    response = _field(value, "result") or value
    answers = _field(response, "answers")
    answer = _field(answers, dimension)
    score = _field(answer, "score")
    if not _is_number(score) or not 0 <= score <= 4:
        raise ValueError("Jev returned an invalid score")
    confidence = _field(answer, "confidence")
    model = _field(response, "model")
    return {
        "score": float(score),
        "confidence": float(confidence) if _is_number(confidence) else None,
        "model": str(model) if model is not None else None,
    }


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
        "models": {"writer": WRITER_MODEL, "scorer": scorer_model},
        "inspection": {"model_calls": model_calls},
    }, 200
