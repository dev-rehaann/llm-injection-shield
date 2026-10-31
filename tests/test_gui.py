"""GUI smoke check with mocked models; no training or model files needed."""

import tkinter as tk
import unittest
from unittest.mock import patch

from src.gui import ShieldGUI


class GuiTest(unittest.TestCase):
    def test_classify_and_empty_prompt(self):
        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest("No graphical display available")
        root.withdraw()
        try:
            app = ShieldGUI(root)
            app.set_prompt("Explain how rain forms.")
            results = [
                {"model": "Baseline (TF-IDF + LR)", "prediction": "benign",
                 "confidence": 0.8707, "note": ""},
                {"model": "Word2Vec + BiLSTM", "prediction": "benign",
                 "confidence": 0.9908, "note": ""},
                {"model": "DistilBERT", "prediction": "benign",
                 "confidence": 0.9995, "note": ""},
            ]
            with patch("src.gui.load_models", side_effect=lambda key: {key: object()}) as load, \
                 patch("src.gui.predict_prompt", return_value=results) as predict:
                app.classify()
                self.assertEqual(app.input["state"], "disabled")
                payload = app.messages.get(timeout=3)
                app.messages.put(payload)
                app._poll()
                self.assertEqual(app.input["state"], "normal")
                self.assertEqual(load.call_count, 3)
                self.assertEqual(list(predict.call_args.args[1]), ["baseline", "lstm", "transformer"])
            self.assertEqual(len(app.results.get_children()), 3)
            self.assertIn("87.07%", app.results.item(app.results.get_children()[0])["values"])
            self.assertIn("agree: benign", app.status["text"])
            self.assertIn("uncalibrated", app.notes["text"])
            app.clear()
            app.classify()
            self.assertIn("Please enter a prompt", app.status["text"])
        finally:
            root.destroy()