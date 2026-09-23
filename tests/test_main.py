import json
import unittest
from types import SimpleNamespace

from worker.tone_http import api_dispatch, response_headers


class FakeHeaders(dict):
    def get(self, key, default=None):
        return super().get(key.lower(), default)


class FakeRequest:
    def __init__(
        self,
        *,
        url="https://tone-jev.tomv.uk/api/tone",
        method="POST",
        headers=None,
        body=None,
        json_error=None,
    ):
        self.url = url
        self.method = method
        self.headers = FakeHeaders(
            {key.lower(): value for key, value in (headers or {}).items()}
        )
        self.body = body
        self.json_error = json_error

    async def json(self):
        if self.json_error:
            raise self.json_error
        return self.body


class FakeLimiter:
    def __init__(self, success=True):
        self.success = success
        self.keys = []

    async def limit(self, value):
        self.keys.append(value["key"])
        return SimpleNamespace(success=self.success)


class FakeAI:
    def __init__(self):
        self.calls = []

    async def run(self, model, value):
        self.calls.append((model, value))
        return {
            "model": "jev-1.13.0",
            "answers": {"panic": {"score": 2.4, "confidence": 0.9}},
        }


def fake_env(*, session_success=True, global_success=True):
    return SimpleNamespace(
        AI=FakeAI(),
        SESSION_LIMITER=FakeLimiter(session_success),
        GLOBAL_LIMITER=FakeLimiter(global_success),
    )


def valid_request(**changes):
    values = {
        "headers": {
            "origin": "https://tone-jev.tomv.uk",
            "x-tone-session": "123e4567-e89b-12d3-a456-426614174000",
        },
        "body": {
            "source": "Please read the manual.",
            "dimension": "panic",
            "target": 60,
        },
    }
    values.update(changes)
    return FakeRequest(**values)


class HttpBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_non_api_routes_fall_through_to_assets(self):
        result = await api_dispatch(
            FakeRequest(url="https://tone-jev.tomv.uk/about", method="GET"),
            fake_env(),
        )

        self.assertIsNone(result)

    async def test_rejects_method_bad_json_and_invalid_session_before_inference(self):
        cases = [
            (valid_request(method="GET"), 405, "Method not allowed"),
            (
                valid_request(body=None, json_error=ValueError("bad json")),
                400,
                "Invalid JSON",
            ),
            (
                valid_request(
                    headers={
                        "origin": "https://tone-jev.tomv.uk",
                        "x-tone-session": "too short!",
                    }
                ),
                400,
                "Invalid session",
            ),
        ]

        for request, expected_status, message in cases:
            env = fake_env()
            with self.subTest(message=message):
                result = await api_dispatch(request, env)
                self.assertEqual(result, ({"error": message}, expected_status))
                self.assertEqual(env.AI.calls, [])
                self.assertEqual(env.SESSION_LIMITER.keys, [])

    async def test_rejects_oversized_body_before_parsing_or_rate_limiting(self):
        env = fake_env()
        request = valid_request(
            headers={
                "origin": "https://tone-jev.tomv.uk",
                "x-tone-session": "123e4567-e89b-12d3-a456-426614174000",
                "content-length": "4097",
            }
        )

        payload, status = await api_dispatch(request, env)

        self.assertEqual((payload, status), ({"error": "Request too large"}, 413))
        self.assertEqual(env.SESSION_LIMITER.keys, [])
        self.assertEqual(env.AI.calls, [])

    async def test_rejects_cross_origin_without_inference(self):
        env = fake_env()
        request = valid_request(
            headers={
                "origin": "https://attacker.example",
                "x-tone-session": "123e4567-e89b-12d3-a456-426614174000",
            }
        )

        payload, status = await api_dispatch(request, env)

        self.assertEqual(status, 403)
        self.assertEqual(payload["error"], "Cross-origin requests are not allowed")
        self.assertEqual(env.AI.calls, [])

    async def test_applies_session_then_global_rate_limits(self):
        session_limited = fake_env(session_success=False)
        payload, status = await api_dispatch(valid_request(), session_limited)
        self.assertEqual(
            (payload, status), ({"error": "Easy, tiger. Try again soon."}, 429)
        )
        self.assertEqual(
            session_limited.SESSION_LIMITER.keys,
            ["123e4567-e89b-12d3-a456-426614174000"],
        )
        self.assertEqual(session_limited.GLOBAL_LIMITER.keys, [])

        globally_limited = fake_env(global_success=False)
        payload, status = await api_dispatch(valid_request(), globally_limited)
        self.assertEqual(
            (payload, status),
            ({"error": "The tone queue is full. Try again soon."}, 429),
        )
        self.assertEqual(globally_limited.GLOBAL_LIMITER.keys, ["tone-jev"])
        self.assertEqual(globally_limited.AI.calls, [])

    async def test_runs_valid_request_and_returns_tone_payload(self):
        env = fake_env()

        payload, status = await api_dispatch(valid_request(), env)

        self.assertEqual(status, 200)
        self.assertEqual(payload["phrase"], "Please read the manual.")
        self.assertEqual(payload["score"], 60.0)
        self.assertEqual(len(env.AI.calls), 1)
        self.assertEqual(env.AI.calls[0][0], "typesafe/jev")

    def test_json_responses_are_private_and_sniff_safe(self):
        self.assertEqual(
            response_headers(),
            {
                "cache-control": "no-store",
                "content-type": "application/json; charset=utf-8",
                "x-content-type-options": "nosniff",
            },
        )
        json.dumps(response_headers())


if __name__ == "__main__":
    unittest.main()
