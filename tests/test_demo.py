"""Offline CLI check: python -m unittest discover -s tests -p test_demo.py."""

import contextlib
import io
import math
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from src.demo import interactive_loop, load_models, main, predict_prompt, print_results
from src.preprocess import preprocess_text


class DemoTest(unittest.TestCase):
    def test_preprocessing_confidence_multiline_and_errors(self):
        text = "THE rain!\nIgnore RULES."
        baseline = Mock(classes_=[1, 0])  # Reversed order catches probability-column mistakes.
        baseline.predict.return_value = [0]
        baseline.predict_proba.return_value = [[0.1, 0.9]]
        lstm = Mock(return_value=torch.tensor([math.log(0.2 / 0.8)]))
        checkpoint = {"max_length": 2, "vocab": {"the": 2, "rain": 3}, "threshold": 0.5}
        tokenizer = Mock(model_max_length=6)
        tokenizer.tokenize.return_value = text.split()
        tokenizer.num_special_tokens_to_add.return_value = 2
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2]])}
        transformer = Mock(return_value=SimpleNamespace(logits=torch.tensor([[0.0, math.log(4.0)]])))
        models = {"baseline": baseline, "lstm": (lstm, checkpoint), "transformer": (tokenizer, transformer)}

        results = predict_prompt(text, models)
        self.assertEqual([r["prediction"] for r in results], ["benign", "benign", "malicious"])
        for result, expected in zip(results, (0.9, 0.8, 0.8)):
            self.assertAlmostEqual(result["confidence"], expected, places=6)
        baseline.predict.assert_called_once_with([preprocess_text(text)["clean_text"]])
        self.assertEqual(lstm.call_args.args[0].tolist(), [[2, 3]])
        self.assertEqual(lstm.call_args.args[1].tolist(), [2])
        tokenizer.assert_called_once_with(text, truncation=True, max_length=6, return_tensors="pt")
        self.assertTrue(results[1]["note"])
        self.assertFalse(results[2]["note"])  # Exactly at the limit.
        tokenizer.tokenize.return_value = ["word"] * 7
        self.assertTrue(predict_prompt(text, {"transformer": (tokenizer, transformer)})[0]["note"])
        with self.assertRaises(ValueError):
            predict_prompt(" \n ", models)
        baseline.predict_proba.return_value = [[float("nan"), float("nan")]]
        with self.assertRaisesRegex(ValueError, "invalid probability"):
            predict_prompt(text, {"baseline": baseline})

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_results(results)
        self.assertIn("90.00%", output.getvalue())
        self.assertIn("Models disagree", output.getvalue())
        self.assertIn("uncalibrated", output.getvalue())

        # Blank lines remain part of the pasted prompt; /clear discards only that buffer.
        with patch("src.demo.predict_prompt", return_value=results) as predict, \
             patch("builtins.input", side_effect=["/run", "discard", "/clear",
                                                  "first line", "", "second line", "/run", "/quit"]), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            interactive_loop(models)
        predict.assert_called_once_with("first line\n\nsecond line", models)
        self.assertIn("non-empty prompt", output.getvalue())
        with patch("src.demo.predict_prompt", return_value=results) as predict, \
             patch("builtins.input", side_effect=["unfinished", EOFError]), \
             contextlib.redirect_stdout(io.StringIO()):
            interactive_loop(models)
        predict.assert_called_once_with("unfinished", models)

        with patch("src.demo.load_models", return_value=models) as load, \
             patch("src.demo.predict_prompt", return_value=results[:1]) as predict, \
             patch("sys.stdin", io.StringIO(text)), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--model", "baseline"]), 0)
        load.assert_called_once_with("baseline")
        predict.assert_called_once_with(text, models)
        with patch("src.demo.load_models", side_effect=FileNotFoundError("Train the model first")), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(main(["--prompt", text]), 1)
        self.assertIn("Train the model first", errors.getvalue())
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaisesRegex(FileNotFoundError, "src.model_baseline"):
                load_models("baseline", empty)
