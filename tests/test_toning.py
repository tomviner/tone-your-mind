import math
import unittest
from types import SimpleNamespace

from worker.toning import (
    DIMENSIONS,
    MAX_ATTEMPTS,
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
    def test_catalogue_keeps_original_dimensions_and_adds_requested_fun(self):
        self.assertIn("urgency", DIMENSIONS)
        self.assertIn("whimsy", DIMENSIONS)
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
        self.assertIn("constrained tone editor, not a copywriter", system)
        self.assertIn("preserve every factual claim, request, commitment", system)
        self.assertIn("negation, name, number, date, condition, and action", system)
        self.assertIn("Change only tone-bearing wording", system)

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
