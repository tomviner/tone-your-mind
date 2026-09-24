import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from worker.toning import (
    DIMENSIONS,
    MAX_ATTEMPTS,
    PROMPT_PROGRAM,
    WRITER_MODEL,
    build_jev_input,
    build_writer_input,
    phrase_from_writer_response,
    score_from_jev_response,
    tone_request,
)


def jev_response(score, confidence=0.9):
    return {
        "model": "jev-1.13.0",
        "answers": {"panic": {"score": score, "confidence": confidence}},
    }


class ToneContractTests(unittest.TestCase):
    def test_uses_the_fast_low_cost_writer(self):
        self.assertEqual(WRITER_MODEL, "@cf/ibm-granite/granite-4.0-h-micro")
        self.assertEqual(PROMPT_PROGRAM["writer_model"], WRITER_MODEL)
        self.assertGreater(
            PROMPT_PROGRAM["metric"]["tone_weight"],
            PROMPT_PROGRAM["metric"]["meaning_weight"],
        )

    def test_catalogue_keeps_original_dimensions_and_adds_requested_fun(self):
        self.assertIn("urgency", DIMENSIONS)
        self.assertIn("whimsy", DIMENSIONS)
        self.assertEqual(DIMENSIONS["sarcasm"]["low"], "Literal")
        self.assertEqual(DIMENSIONS["sarcasm"]["high"], "Sarcastic")
        self.assertEqual(DIMENSIONS["panic"]["low"], "Unruffled")
        self.assertEqual(DIMENSIONS["panic"]["high"], "Full panic")
        self.assertTrue(
            {"passive_aggression", "smugness", "drama", "ominousness"}
            <= DIMENSIONS.keys()
        )

    def test_jev_receives_the_phrase_as_state_and_one_score_question(self):
        result = build_jev_input("Please read the manual.", "panic")

        self.assertEqual(result["state"], "Please read the manual.")
        self.assertEqual(
            result["questions"]["panic"],
            {
                "type": "score",
                "instructions": DIMENSIONS["panic"]["instructions"],
                "criteria": DIMENSIONS["panic"]["criteria"],
            },
        )

    def test_writer_prompt_contains_bounded_history_and_target(self):
        result = build_writer_input(
            "Please read the manual.",
            "panic",
            2.4,
            [
                {"phrase": "Please read the manual.", "score": 0.5},
                {"phrase": "Read it soon.", "score": 1.6},
            ],
        )

        self.assertEqual(result["max_tokens"], 96)
        self.assertEqual(result["temperature"], 0.7)
        prompt = result["messages"][-1]["content"]
        self.assertIn("target Jev score: 2.40 out of 4", prompt)
        self.assertIn("Attempt 1 (score 0.50): Please read the manual.", prompt)
        self.assertIn("Attempt 2 (score 1.60): Read it soon.", prompt)
        self.assertIn("240 characters", prompt)
        system = result["messages"][0]["content"]
        self.assertIn("precisely convey the specified sentiment", system)
        self.assertIn("primary focus", system)
        self.assertIn("accurate tone representation", system)
        self.assertIn("original practical meaning", system)
        self.assertNotIn(
            "Never add facts, names, dates, threats, promises, or instructions",
            system,
        )
        self.assertIn("Never mention a score", system)
        self.assertIn("percentage, rating, slider, Jev", system)

    def test_writer_uses_the_compiled_sarcasm_instruction_only_for_sarcasm(self):
        sarcasm_input = build_writer_input(
            "Please read the manual.", "sarcasm", 3.2, []
        )
        panic_input = build_writer_input("Please read the manual.", "panic", 3.2, [])
        sarcasm = sarcasm_input["messages"][0]["content"]
        panic = panic_input["messages"][0]["content"]

        self.assertEqual(sarcasm, PROMPT_PROGRAM["dimension_instructions"]["sarcasm"])
        self.assertEqual(panic, PROMPT_PROGRAM["system_instruction"])
        self.assertNotEqual(sarcasm, panic)
        guidance = PROMPT_PROGRAM["dimension_guidance"]["sarcasm"]
        self.assertIn(guidance, sarcasm_input["messages"][-1]["content"])
        self.assertNotIn(guidance, panic_input["messages"][-1]["content"])

    def test_sarcasm_prompt_spells_out_the_requested_band(self):
        low = build_writer_input("The printer is broken.", "sarcasm", 0.4, [])
        high = build_writer_input("The printer is broken.", "sarcasm", 3.8, [])

        self.assertIn("LOW SARCASM", low["messages"][-1]["content"])
        self.assertIn("Do not use mock-praise openers", low["messages"][-1]["content"])
        self.assertIn("MAXIMUM SARCASM", high["messages"][-1]["content"])
        self.assertIn(
            "at least two unmistakable irony cues", high["messages"][-1]["content"]
        )

    def test_writer_uses_a_compiled_dimension_instruction_when_available(self):
        with patch.dict(
            PROMPT_PROGRAM,
            {"dimension_instructions": {"sarcasm": "Sarcasm specialist prompt."}},
        ):
            result = build_writer_input("The meeting ran late.", "sarcasm", 3.0, [])

        self.assertEqual(result["messages"][0]["content"], "Sarcasm specialist prompt.")

    def test_writer_includes_compiled_sarcasm_demonstrations(self):
        with patch.dict(
            PROMPT_PROGRAM,
            {
                "dimension_demos": {
                    "sarcasm": [
                        {
                            "source": "The train was cancelled.",
                            "target": 75,
                            "rewrite": "Excellent news: the train was cancelled.",
                        }
                    ]
                }
            },
        ):
            result = build_writer_input("The lift is broken.", "sarcasm", 2.0, [])

        self.assertEqual(result["messages"][1]["role"], "user")
        self.assertIn("The train was cancelled.", result["messages"][1]["content"])
        self.assertIn("3.00 out of 4", result["messages"][1]["content"])
        self.assertEqual(
            result["messages"][2],
            {
                "role": "assistant",
                "content": "Excellent news: the train was cancelled.",
            },
        )
        self.assertIn("The lift is broken.", result["messages"][-1]["content"])

    def test_writer_uses_dimension_specific_sampling_parameters(self):
        with patch.dict(
            PROMPT_PROGRAM,
            {"dimension_parameters": {"sarcasm": {"temperature": 0.2}}},
        ):
            sarcasm = build_writer_input("The lift is broken.", "sarcasm", 3, [])
            panic = build_writer_input("The lift is broken.", "panic", 3, [])

        self.assertEqual(sarcasm["temperature"], 0.2)
        self.assertEqual(panic["temperature"], 0.7)

    def test_writer_uses_only_the_nearest_sarcasm_demonstration(self):
        with patch.dict(
            PROMPT_PROGRAM,
            {
                "dimension_demos": {
                    "sarcasm": [
                        {"source": "Low.", "target": 0, "rewrite": "Low."},
                        {
                            "source": "High.",
                            "target": 100,
                            "rewrite": "Oh, perfect—High.",
                        },
                    ]
                }
            },
        ):
            result = build_writer_input("Current.", "sarcasm", 0.4, [])

        conversation = "\n".join(message["content"] for message in result["messages"])
        self.assertIn("Low.", conversation)
        self.assertNotIn("High.", conversation)

    def test_writer_prompt_escalates_plain_language_feedback_after_each_miss(self):
        cases = [
            (2.1, "noticeably further"),
            (1.8, "several times more obvious"),
            (1.4, "roughly twentyfold"),
            (0.9, "fifty times stronger"),
            (0.4, "one hundred times harder"),
        ]

        for score, expected in cases:
            with self.subTest(score=score):
                result = build_writer_input(
                    "Please read the manual.",
                    "panic",
                    2.4,
                    [{"phrase": "Please read the manual.", "score": score}],
                )
                prompt = result["messages"][-1]["content"]
                self.assertIn(expected, prompt)
                self.assertIn('toward the "Full panic" end', prompt)

    def test_sarcasm_retry_uses_the_target_band_without_generic_overcorrection(self):
        result = build_writer_input(
            "The coffee machine is broken again.",
            "sarcasm",
            1.2,
            [{"phrase": "The coffee machine is broken again.", "score": 0.4}],
        )

        prompt = result["messages"][-1]["content"]
        self.assertIn("one controlled step stronger", prompt)
        self.assertIn("DRY SARCASM", prompt)
        self.assertNotIn("one hundred times harder", prompt)

    def test_writer_prompt_multiplies_emphasis_by_iteration(self):
        attempts = [
            {"phrase": "Please read the manual.", "score": 0.2},
            {"phrase": "Please, read the manual soon.", "score": 0.4},
            {"phrase": "Read the manual now!", "score": 0.8},
        ]

        first = build_writer_input("Please read the manual.", "panic", 4, attempts[:1])[
            "messages"
        ][-1]["content"]
        second = build_writer_input(
            "Please read the manual.", "panic", 4, attempts[:2]
        )["messages"][-1]["content"]
        final = build_writer_input("Please read the manual.", "panic", 4, attempts)[
            "messages"
        ][-1]["content"]

        self.assertIn("firm correction", first)
        self.assertIn("previous correction failed", second)
        self.assertIn("FINAL ATTEMPT", final)
        self.assertIn("melodramatic, superlative, relentless", final)

    def test_writer_prompt_pushes_toward_the_low_end_after_overshooting(self):
        result = build_writer_input(
            "Please read the manual.",
            "panic",
            1.0,
            [{"phrase": "READ IT NOW!", "score": 3.0}],
        )

        prompt = result["messages"][-1]["content"]
        self.assertIn('toward the "Unruffled" end', prompt)
        self.assertIn("one hundred times harder", prompt)

    def test_writer_response_removes_common_wrappers_but_rejects_bad_output(self):
        self.assertEqual(
            phrase_from_writer_response(
                {"response": 'Rewritten phrase: "Please read it right now."'}
            ),
            "Please read it right now.",
        )
        self.assertEqual(
            phrase_from_writer_response(
                SimpleNamespace(response="Here is the rewrite:\nDo it, please."),
            ),
            "Do it, please.",
        )
        for response in ({}, {"response": ""}, {"response": "x" * 241}):
            with self.subTest(response=response), self.assertRaises(ValueError):
                phrase_from_writer_response(response)

    def test_writer_response_converts_cloudflare_js_dictionary(self):
        class JsDict:
            def __init__(self, value):
                self.value = value

            def get(self, key):
                return self.value.get(key)

        self.assertEqual(
            phrase_from_writer_response(
                JsDict(
                    {
                        "choices": [
                            JsDict(
                                {
                                    "message": JsDict(
                                        {"content": "Please read it immediately."}
                                    )
                                }
                            )
                        ]
                    }
                )
            ),
            "Please read it immediately.",
        )

    def test_parses_direct_enveloped_and_property_backed_jev_responses(self):
        direct = jev_response(2.4, 0.81)
        expected = {"score": 2.4, "confidence": 0.81, "model": "jev-1.13.0"}

        self.assertEqual(score_from_jev_response(direct, "panic"), expected)
        self.assertEqual(score_from_jev_response({"result": direct}, "panic"), expected)
        self.assertEqual(
            score_from_jev_response(
                SimpleNamespace(
                    model="jev-1.13.0",
                    answers=SimpleNamespace(
                        panic=SimpleNamespace(score=2.4, confidence=0.81)
                    ),
                ),
                "panic",
            ),
            expected,
        )

    def test_preserves_jev_distribution_as_reviewer_feedback(self):
        result = score_from_jev_response(
            {
                "model": "jev-1.13.0",
                "answers": {
                    "panic": {
                        "type": "score",
                        "score": 3.2,
                        "confidence": 0.67,
                        "legend": {
                            "0": "Unruffled",
                            "1": "Slight concern",
                            "2": "Clearly worried",
                            "3": "Strong panic",
                            "4": "Full panic",
                        },
                        "probabilities": {
                            "0": 0.0,
                            "1": 0.0,
                            "2": 0.1,
                            "3": 0.6,
                            "4": 0.3,
                        },
                    }
                },
            },
            "panic",
        )

        self.assertEqual(
            result["reviewer_feedback"],
            {
                "legend": {
                    "0": "Unruffled",
                    "1": "Slight concern",
                    "2": "Clearly worried",
                    "3": "Strong panic",
                    "4": "Full panic",
                },
                "probabilities": {
                    "0": 0.0,
                    "1": 0.0,
                    "2": 0.1,
                    "3": 0.6,
                    "4": 0.3,
                },
            },
        )

    def test_rejects_missing_non_finite_and_out_of_range_jev_scores(self):
        for score in (None, math.nan, -0.1, 4.1, True):
            with self.subTest(score=score), self.assertRaises(ValueError):
                score_from_jev_response(jev_response(score), "panic")


class ToneRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_invalid_requests_before_inference(self):
        calls = []

        async def run(model, value):
            calls.append((model, value))

        invalid = [
            None,
            {},
            {"source": 3, "dimension": "panic", "target": 60},
            {"source": "x" * 241, "dimension": "panic", "target": 60},
            {"source": "Hello", "dimension": "invented", "target": 60},
            {"source": "Hello", "dimension": "panic", "target": True},
            {"source": "Hello", "dimension": "panic", "target": -5},
            {"source": "Hello", "dimension": "panic", "target": 101},
            {"source": "Hello", "dimension": "panic", "target": 62},
        ]

        for body in invalid:
            with self.subTest(body=body):
                payload, status = await tone_request(
                    body,
                    "https://tone-jev.tomv.uk/api/tone",
                    "https://tone-jev.tomv.uk",
                    run,
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload, {"error": "Invalid toning request"})
        self.assertEqual(calls, [])

    async def test_rejects_cross_origin_before_inference(self):
        calls = []

        async def run(model, value):
            calls.append((model, value))

        payload, status = await tone_request(
            {"source": "Hello", "dimension": "panic", "target": 60},
            "https://tone-jev.tomv.uk/api/tone",
            "http://tone-jev.tomv.uk",
            run,
        )

        self.assertEqual(status, 403)
        self.assertEqual(payload, {"error": "Cross-origin requests are not allowed"})
        self.assertEqual(calls, [])

    async def test_scores_source_first_and_skips_writer_when_already_close(self):
        calls = []

        async def run(model, value):
            calls.append((model, value))
            return jev_response(2.4, 0.92)

        payload, status = await tone_request(
            {
                "source": "Please read the manual.",
                "dimension": "panic",
                "target": 60,
            },
            "https://tone-jev.tomv.uk/api/tone",
            "https://tone-jev.tomv.uk",
            run,
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["phrase"], "Please read the manual.")
        self.assertEqual(payload["score"], 60.0)
        self.assertEqual(payload["distance"], 0.0)
        self.assertTrue(payload["hit"])
        self.assertEqual(len(payload["attempts"]), 1)
        self.assertEqual([model for model, _ in calls], ["typesafe/jev"])

    async def test_dry_sarcasm_scores_a_calibrated_probe_before_calling_writer(self):
        calls = []

        async def run(model, value):
            calls.append((model, value))
            score = 0.4 if len(calls) == 1 else 1.2
            return {
                "model": "jev-1.13.0",
                "answers": {
                    "sarcasm": {"score": score, "confidence": 0.8},
                },
            }

        payload, status = await tone_request(
            {
                "source": "The coffee machine is broken again.",
                "dimension": "sarcasm",
                "target": 30,
            },
            "https://tone-jev.tomv.uk/api/tone",
            "https://tone-jev.tomv.uk",
            run,
        )

        self.assertEqual(status, 200)
        self.assertEqual(
            payload["phrase"], "The coffee machine is broken again, naturally."
        )
        self.assertEqual(payload["score"], 30.0)
        self.assertTrue(payload["hit"])
        self.assertEqual([model for model, _ in calls], ["typesafe/jev"] * 2)

    async def test_blank_source_generates_then_scores_and_stops_on_hit(self):
        calls = []
        events = []

        async def run(model, value):
            calls.append((model, value))
            if model == WRITER_MODEL:
                return {"response": "Where is the emergency manual?!"}
            return jev_response(2.5)

        payload, status = await tone_request(
            {"source": "", "dimension": "panic", "target": 60},
            "https://tone-jev.tomv.uk/api/tone",
            None,
            run,
            events.append,
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["phrase"], "Where is the emergency manual?!")
        self.assertEqual(payload["score"], 62.5)
        self.assertEqual(payload["distance"], 2.5)
        self.assertTrue(payload["hit"])
        self.assertEqual([model for model, _ in calls], [WRITER_MODEL, "typesafe/jev"])
        self.assertEqual(calls[0][1]["messages"][-1]["content"].count("Attempt"), 0)
        self.assertEqual(
            events,
            [
                {
                    "type": "attempt",
                    "attempt": {
                        "phrase": "Where is the emergency manual?!",
                        "score": 62.5,
                        "confidence": 0.9,
                    },
                }
            ],
        )
        self.assertEqual(
            payload["inspection"]["model_calls"],
            [
                {
                    "kind": "writer",
                    "request": {"model": WRITER_MODEL, "input": calls[0][1]},
                    "response": {"phrase": "Where is the emergency manual?!"},
                },
                {
                    "kind": "scorer",
                    "request": {"model": "typesafe/jev", "input": calls[1][1]},
                    "response": {
                        "model": "jev-1.13.0",
                        "score": 2.5,
                        "confidence": 0.9,
                    },
                },
            ],
        )

    async def test_rejects_score_leaks_before_they_are_scored_or_streamed(self):
        writer_phrases = iter(
            [
                (
                    "Chloe, this is no small matter, like a 0, but a dinner "
                    "finale, like a 4."
                ),
                "Chloe, dinner awaits your most dramatic entrance!",
            ]
        )
        calls = []
        events = []

        async def run(model, value):
            calls.append((model, value))
            if model == WRITER_MODEL:
                return {"response": next(writer_phrases)}
            return jev_response(0.2 if len(calls) == 1 else 4.0)

        payload, status = await tone_request(
            {
                "source": "Chloe, could you please eat your dinner?",
                "dimension": "panic",
                "target": 100,
            },
            "https://tone-jev.tomv.uk/api/tone",
            "https://tone-jev.tomv.uk",
            run,
            events.append,
        )

        self.assertEqual(status, 200)
        self.assertEqual(
            payload["phrase"], "Chloe, dinner awaits your most dramatic entrance!"
        )
        self.assertNotIn("like a 0", str(payload))
        self.assertEqual(
            [model for model, _ in calls],
            ["typesafe/jev", WRITER_MODEL, WRITER_MODEL, "typesafe/jev"],
        )
        self.assertEqual(
            [event["attempt"]["phrase"] for event in events],
            [
                "Chloe, could you please eat your dinner?",
                "Chloe, dinner awaits your most dramatic entrance!",
            ],
        )
        rejected = payload["inspection"]["model_calls"][1]
        self.assertEqual(rejected["response"], {"rejected": "score_reference"})
        retry_prompt = calls[2][1]["messages"][-1]["content"]
        self.assertIn("A previous answer mentioned scoring metadata", retry_prompt)

    async def test_uses_full_history_and_returns_closest_after_attempt_cap(self):
        writer_phrases = iter(["candidate one", "candidate two", "candidate three"])
        jev_scores = iter([0.2, 1.0, 2.0, 1.8])
        writer_inputs = []

        async def run(model, value):
            if model == WRITER_MODEL:
                writer_inputs.append(value)
                return {"response": next(writer_phrases)}
            return jev_response(next(jev_scores))

        payload, status = await tone_request(
            {"source": "original", "dimension": "panic", "target": 80},
            "https://tone-jev.tomv.uk/api/tone",
            "https://tone-jev.tomv.uk",
            run,
        )

        self.assertEqual(status, 200)
        self.assertEqual(MAX_ATTEMPTS, 4)
        self.assertEqual(len(payload["attempts"]), MAX_ATTEMPTS)
        self.assertEqual(payload["phrase"], "candidate two")
        self.assertEqual(payload["score"], 50.0)
        self.assertEqual(payload["distance"], 30.0)
        self.assertFalse(payload["hit"])
        final_prompt = writer_inputs[-1]["messages"][-1]["content"]
        self.assertIn("Attempt 1 (score 0.20): original", final_prompt)
        self.assertIn("Attempt 3 (score 2.00): candidate two", final_prompt)

    async def test_keeps_earlier_attempt_when_distances_tie(self):
        writer_phrases = iter(["too much", "same distance", "still off"])
        jev_scores = iter([1.0, 3.0, 1.0, 0.0])

        async def run(model, value):
            if model == WRITER_MODEL:
                return {"response": next(writer_phrases)}
            return jev_response(next(jev_scores))

        payload, status = await tone_request(
            {"source": "first", "dimension": "panic", "target": 50},
            "https://tone-jev.tomv.uk/api/tone",
            "https://tone-jev.tomv.uk",
            run,
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["phrase"], "first")

    async def test_maps_writer_and_jev_failures_to_fixed_error_without_partial_text(
        self,
    ):
        async def writer_failure(model, value):
            raise RuntimeError("private provider detail")

        async def invalid_jev(model, value):
            if model == WRITER_MODEL:
                return {"response": "A partial candidate"}
            return {"answers": {}}

        for run in (writer_failure, invalid_jev):
            with self.subTest(run=run), self.assertLogs(level="ERROR"):
                payload, status = await tone_request(
                    {"source": "", "dimension": "panic", "target": 60},
                    "https://tone-jev.tomv.uk/api/tone",
                    "https://tone-jev.tomv.uk",
                    run,
                )
                self.assertEqual(status, 502)
                self.assertEqual(payload, {"error": "The tone loop lost the plot"})
                self.assertNotIn("partial", str(payload).lower())

    async def test_logs_failed_boundary_without_phrase_content(self):
        async def invalid_writer(model, value):
            return {"unexpected": "Please do not log this phrase"}

        with self.assertLogs(level="ERROR") as writer_logs:
            await tone_request(
                {"source": "", "dimension": "panic", "target": 60},
                "https://tone-jev.tomv.uk/api/tone",
                "https://tone-jev.tomv.uk",
                invalid_writer,
            )
        self.assertIn('"stage": "writer_parse"', writer_logs.output[0])
        self.assertNotIn("Please do not log", writer_logs.output[0])

        async def invalid_jev(model, value):
            return {"answers": {}}

        with self.assertLogs(level="ERROR") as jev_logs:
            await tone_request(
                {"source": "private source", "dimension": "panic", "target": 60},
                "https://tone-jev.tomv.uk/api/tone",
                "https://tone-jev.tomv.uk",
                invalid_jev,
            )
        self.assertIn('"stage": "jev_parse"', jev_logs.output[0])
        self.assertNotIn("private source", jev_logs.output[0])


if __name__ == "__main__":
    unittest.main()
