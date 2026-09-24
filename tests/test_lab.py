import json
import unittest
import urllib.error
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import dspy

from lab.metric import (
    combined_metric,
    pair_evaluations,
    parse_jev_review,
    sarcasm_metric,
    tone_accuracy,
)
from lab.optimize import (
    CloudflareJudge,
    ToneProgram,
    _production_demos,
    candidate_should_publish,
)


class JsonResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class LabMetricTests(unittest.TestCase):
    @patch("lab.optimize.time.sleep")
    @patch("lab.optimize.urllib.request.urlopen")
    def test_jev_review_retries_a_rate_limited_request(self, urlopen, sleep):
        rate_limit = urllib.error.HTTPError(
            "https://example.test",
            429,
            "Rate limited",
            {"Retry-After": "0"},
            BytesIO(b'{"errors": []}'),
        )
        success = JsonResponse(
            json.dumps(
                {
                    "result": {
                        "answers": {
                            "sarcasm": {"score": 2.4},
                            "meaning_retained": {"noul": 0.9},
                            "sarcasm_quality": {"noul": 0.8},
                        }
                    }
                }
            ).encode()
        )
        urlopen.side_effect = [rate_limit, success]

        result = CloudflareJudge("account", "token", "gateway").review(
            source="The train is late.",
            candidate="Wonderful, the train is late.",
            dimension="sarcasm",
            rubric={"instructions": "sarcasm", "criteria": ["low", "high"]},
            target=60,
        )

        self.assertEqual(result, (60.0, 0.9, 0.8))
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(0.0)

    def test_published_program_and_result_share_the_optimizer_run(self):
        root = Path(__file__).resolve().parents[1]
        program = json.loads((root / "worker" / "prompt_program.json").read_text())
        results = json.loads((root / "src" / "lab-results.json").read_text())

        self.assertEqual(program["optimizer_run"], results["run"])
        self.assertEqual(program["system_instruction"], results["selected_instruction"])
        self.assertEqual(len(program["distance_bands"]), 5)
        self.assertEqual(len(program["attempt_escalators"]), 3)
        self.assertEqual(results["weights"], {"tone": 85, "meaning": 15})

        sarcasm = json.loads((root / "src" / "sarcasm-lab-results.json").read_text())
        self.assertEqual(program["dimension_optimizer_runs"]["sarcasm"], sarcasm["run"])
        self.assertIn("sarcasm level", program["dimension_instructions"]["sarcasm"])
        self.assertIn(
            "every factual element remains truthful",
            program["dimension_instructions"]["sarcasm"],
        )
        self.assertEqual(
            program["dimension_parameters"]["sarcasm"], {"temperature": 0.2}
        )
        self.assertEqual(sarcasm["selected"], "candidate")

    def test_sarcasm_dataset_has_separate_tuning_and_held_out_splits(self):
        root = Path(__file__).resolve().parents[1]
        rows = json.loads((root / "lab" / "sarcasm-dataset.json").read_text())

        self.assertEqual({row["dimension"] for row in rows}, {"sarcasm"})
        self.assertEqual({row["split"] for row in rows}, {"train", "val", "eval"})
        for split in ("train", "val", "eval"):
            targets = [row["target"] for row in rows if row["split"] == split]
            self.assertLessEqual(min(targets), 20)
            self.assertGreaterEqual(max(targets), 80)

        train_rows = [row for row in rows if row["split"] == "train"]
        self.assertTrue(all(row.get("rewrite") for row in train_rows))

    def test_tone_accuracy_is_percentage_distance_reward(self):
        self.assertEqual(tone_accuracy(60, 60), 1)
        self.assertEqual(tone_accuracy(20, 60), 0.36)
        self.assertEqual(tone_accuracy(0, 100), 0)

    def test_combined_metric_weights_tone_substantially_higher(self):
        accurate_bad_meaning = combined_metric(60, 60, 0)
        inaccurate_good_meaning = combined_metric(20, 60, 1)

        self.assertEqual(accurate_bad_meaning, 0.85)
        self.assertAlmostEqual(inaccurate_good_meaning, 0.456)
        self.assertGreater(accurate_bad_meaning, inaccurate_good_meaning)

    def test_sarcasm_metric_rejects_insults_that_only_hit_the_tone_score(self):
        accurate_insult = sarcasm_metric(80, 80, 1, 0)
        natural_near_hit = sarcasm_metric(72, 80, 1, 1)
        accurate_meaning_loss = sarcasm_metric(80, 80, 0.05, 1)
        faithful_but_far = sarcasm_metric(50, 80, 1, 1)
        catastrophic_miss = sarcasm_metric(35, 80, 1, 1)

        self.assertGreater(natural_near_hit, accurate_insult)
        self.assertLess(accurate_insult, 0.1)
        self.assertLess(accurate_meaning_loss, 0.1)
        self.assertGreater(faithful_but_far, accurate_meaning_loss)
        self.assertEqual(catastrophic_miss, 0)

    def test_sarcasm_publication_uses_both_unseen_aggregate_scores(self):
        self.assertTrue(candidate_should_publish(51.5, 34.1, 44.1, 34.7))
        self.assertFalse(candidate_should_publish(51.5, 34.1, 33.0, 34.7))
        self.assertFalse(candidate_should_publish(33.0, 34.1, 44.1, 34.7))

    def test_bootstrapped_demo_can_be_serialized_without_target_percent(self):
        program = ToneProgram()
        program.rewrite.demos = [
            dspy.Example(
                source="The printer is broken.",
                target="3.00 out of 4 (75%)",
                rewrite="Perfect—the printer is broken.",
            )
        ]

        self.assertEqual(
            _production_demos(program),
            [
                {
                    "source": "The printer is broken.",
                    "target": 75.0,
                    "rewrite": "Perfect—the printer is broken.",
                }
            ],
        )

    def test_parses_jev_universal_endpoint_response(self):
        score, meaning, quality = parse_jev_review(
            {
                "result": {
                    "state": "Completed",
                    "result": {
                        "answers": {
                            "panic": {"score": 2.4},
                            "meaning_retained": {"noul": 0.91},
                            "sarcasm_quality": {"noul": 0.83},
                        }
                    },
                }
            },
            "panic",
        )

        self.assertEqual(score, 60)
        self.assertEqual(meaning, 0.91)
        self.assertEqual(quality, 0.83)

    def test_sarcasm_dataset_covers_the_full_dial_in_both_splits(self):
        root = Path(__file__).resolve().parents[1]
        rows = json.loads((root / "lab" / "dataset.json").read_text())

        for split in ("train", "eval"):
            targets = {
                row["target"]
                for row in rows
                if row["dimension"] == "sarcasm" and row["split"] == split
            }
            self.assertTrue(any(target <= 20 for target in targets))
            self.assertTrue(any(40 <= target <= 60 for target in targets))
            self.assertTrue(any(target >= 80 for target in targets))

    def test_pairs_baseline_and_selected_evaluations_for_public_comparison(self):
        baseline = [
            {
                "id": "example-1",
                "source": "Please read this.",
                "dimension": "Panic",
                "target": 60.0,
                "output": "Please read this soon.",
                "score": 25.0,
                "distance": 35.0,
                "meaning": 99.0,
                "metric": 55.0,
            }
        ]
        selected = [
            {
                "id": "example-1",
                "source": "Please read this immediately.",
                "dimension": "Panic",
                "target": 60.0,
                "output": "Please read this immediately.",
                "score": 62.5,
                "distance": 2.5,
                "meaning": 98.0,
                "metric": 95.0,
            }
        ]

        self.assertEqual(
            pair_evaluations(baseline, selected),
            [
                {
                    "id": "example-1",
                    "source": "Please read this.",
                    "dimension": "Panic",
                    "target": 60.0,
                    "before": {
                        "output": "Please read this soon.",
                        "score": 25.0,
                        "distance": 35.0,
                        "meaning": 99.0,
                        "metric": 55.0,
                    },
                    "after": {
                        "output": "Please read this immediately.",
                        "score": 62.5,
                        "distance": 2.5,
                        "meaning": 98.0,
                        "metric": 95.0,
                    },
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
