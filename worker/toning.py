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

AiRunner = Callable[[str, dict[str, Any]], Awaitable[Any]]


def _field(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
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

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise rewriting engine. Adjust tone without adding "
                    "facts, names, dates, threats, or instructions absent from "
                    "the source."
                ),
            },
            {"role": "user", "content": "\n".join(user_parts)},
        ],
        "max_tokens": 96,
        "temperature": 0.7,
        "top_p": 0.9,
    }


def phrase_from_writer_response(value: Any) -> str:
    response = _field(_field(value, "result") or value, "response")
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
    scorer_model = None

    async def score_phrase(phrase: str) -> None:
        nonlocal scorer_model
        jev = score_from_jev_response(
            await run_ai(JEV_MODEL, build_jev_input(phrase, dimension)), dimension
        )
        scorer_model = jev["model"] or scorer_model
        attempts.append(
            {
                "phrase": phrase,
                "score": jev["score"],
                "confidence": jev["confidence"],
            }
        )

    try:
        if source:
            await score_phrase(source)

        while len(attempts) < MAX_ATTEMPTS and (
            not attempts or abs(attempts[-1]["score"] - target_score) > TARGET_TOLERANCE
        ):
            writer_input = build_writer_input(source, dimension, target_score, attempts)
            candidate = phrase_from_writer_response(
                await run_ai(WRITER_MODEL, writer_input)
            )
            await score_phrase(candidate)
    except Exception as error:
        logging.error(
            json.dumps(
                {
                    "event": "tone_loop_failed",
                    "error_type": type(error).__name__,
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
    }, 200
