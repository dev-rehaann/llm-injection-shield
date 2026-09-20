"""Offline comparison check: python -m unittest discover -s tests."""

import json
import tempfile
import unittest
from pathlib import Path

from matplotlib import image as mpimg

from src.compare_models import METRICS, MODEL_REPORTS, compare_models


class ComparisonTest(unittest.TestCase):
    def test_outputs_and_reject_mismatched_or_invalid_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = []
            for index, (_, filename) in enumerate(MODEL_REPORTS):
                report = {
                    metric: 0.80 + index * 0.05 + number * 0.01
                    for number, metric in enumerate(METRICS)
                }
                report.update({
                    "evaluation_split": "test", "positive_label": 1,
                    "test_samples": 100, "data_sha256": {"test": "a" * 64},
                })
                (root / filename).write_text(json.dumps(report), encoding="utf-8")
                reports.append(report)
            table_path, chart_path = compare_models(root)
            text = table_path.read_text(encoding="utf-8")
            self.assertIn("**100 prompts**", text)
            for (name, filename), report in zip(MODEL_REPORTS, reports):
                expected = " | ".join(f"{report[metric]:.2%}" for metric in METRICS)
                self.assertIn(f"| [{name}]({filename}) | {expected} |", text)
            self.assertIn("(comparison_chart.png)", text)
            pixels = mpimg.imread(chart_path)
            self.assertGreater(pixels.shape[0], 500)
            self.assertGreater(pixels.shape[1], 1000)
            self.assertGreater(float(pixels.max() - pixels.min()), 0.5)

            original_table = table_path.read_bytes()
            original_chart = chart_path.read_bytes()
            last_path = root / MODEL_REPORTS[-1][1]
            original_report = reports[-1]
            for field, value, message in (
                ("data_sha256", {"test": "b" * 64}, "same test split"),
                ("test_samples", 101, "same test split"),
                ("evaluation_split", "val", "expected test metrics"),
                ("positive_label", 0, "malicious=1"),
                ("f1", 1.5, "finite number"),
                ("accuracy", float("nan"), "finite number"),
                ("recall", True, "finite number"),
            ):
                with self.subTest(field=field):
                    last_path.write_text(json.dumps({**original_report, field: value}), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        compare_models(root)
                    self.assertEqual(table_path.read_bytes(), original_table)
                    self.assertEqual(chart_path.read_bytes(), original_chart)
            last_path.unlink()
            with self.assertRaises(FileNotFoundError):
                compare_models(root)
            self.assertEqual(table_path.read_bytes(), original_table)
            self.assertEqual(chart_path.read_bytes(), original_chart)
