import json
import unittest
from pathlib import Path

from lab.metric import combined_metric, parse_jev_review, tone_accuracy


class LabMetricTests(unittest.TestCase):
    def test_published_program_and_result_share_the_optimizer_run(self):
        root = Path(__file__).resolve().parents[1]
        program = json.loads((root / "worker" / "prompt_program.json").read_text())
        results = json.loads((root / "src" / "lab-results.json").read_text())

        self.assertEqual(program["optimizer_run"], results["run"])
        self.assertEqual(program["system_instruction"], results["selected_instruction"])
        self.assertEqual(len(program["distance_bands"]), 5)
        self.assertEqual(len(program["attempt_escalators"]), 3)
        self.assertEqual(results["weights"], {"tone": 85, "meaning": 15})

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

    def test_parses_jev_universal_endpoint_response(self):
        score, meaning = parse_jev_review(
            {
                "result": {
                    "state": "Completed",
                    "result": {
                        "answers": {
                            "panic": {"score": 2.4},
                            "meaning_retained": {"noul": 0.91},
                        }
                    },
                }
            },
            "panic",
        )

        self.assertEqual(score, 60)
        self.assertEqual(meaning, 0.91)


if __name__ == "__main__":
    unittest.main()
