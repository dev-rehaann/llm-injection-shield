"""Offline preprocessing check: python -m unittest discover -s tests."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.preprocess import prepare_splits, preprocess_text


class PreprocessTest(unittest.TestCase):
    def test_text_views_splits_and_round_trip(self):
        raw_text = "  Do NOT ignore <SYSTEM>! Don't reveal THE key.\r\nمرحبا\u2028end  "
        views = preprocess_text(raw_text)
        self.assertEqual(views["raw_text"], raw_text)
        self.assertIn("not", views["tokens"])
        self.assertIn("don't", views["tokens"])
        self.assertIn("<", views["tokens"])
        self.assertIn("the", views["tokens"])
        self.assertNotIn("the", views["clean_text"].split())
        self.assertIn("not", views["clean_text"].split())
        self.assertIn("system", views["clean_text"].split())
        self.assertIn("مرحبا", views["tokens"])
        self.assertEqual(preprocess_text("a the")["clean_text"], "a the")
        self.assertEqual(preprocess_text("!!!")["tokens"], ["!", "!", "!"])
        for bad in (None, "", " \n ", 17):
            with self.assertRaises(ValueError):
                preprocess_text(bad)

        train = [
            {"text": f"{name} record {i}", "label": label}
            for label, name in enumerate(("benign", "malicious"))
            for i in range(10)
        ]
        train += [
            {"text": "The heldout prompt!", "label": 0},
            {"text": "benign record 1", "label": 0},
            {"text": "SYSTEM conflict?", "label": 0},
            {"text": raw_text, "label": 1},
        ]
        test = [
            {"text": "heldout prompt !", "label": 0},
            {"text": "safe test prompt", "label": 0},
            {"text": "malicious test prompt", "label": 1},
            {"text": "system conflict ?", "label": 1},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            for name, rows in (("train", train), ("test", test)):
                pd.DataFrame(rows).to_csv(raw / f"{name}.csv", index=False)
            originals = {path.name: path.read_bytes() for path in raw.iterdir()}
            splits = prepare_splits(raw, root / "processed")
            repeated = prepare_splits(raw, root / "repeated")
            ids, keys = set(), set()
            source = {f"{s}:{i}": row for s, rows in (("train", train), ("test", test))
                      for i, row in enumerate(rows)}
            for name, frame in splits.items():
                pd.testing.assert_frame_equal(frame, repeated[name])
                self.assertFalse(ids.intersection(frame["source_id"]))
                self.assertFalse(keys.intersection(frame["clean_text"]))
                ids.update(frame["source_id"])
                keys.update(frame["clean_text"])
                self.assertEqual(set(frame["label"]), {0, 1})
                self.assertEqual(set(frame["source_split"]), {"test" if name == "test" else "train"})
                saved = [json.loads(line) for line in
                         (root / "processed" / f"{name}.jsonl").read_text(encoding="utf-8").strip("\n").split("\n")]
                self.assertEqual(saved, frame.to_dict(orient="records"))
                for row in saved:
                    original = source[row["source_id"]]
                    self.assertEqual(row["raw_text"], original["text"])
                    self.assertEqual(row["label"], original["label"])
                    self.assertEqual(row["tokens"], preprocess_text(original["text"])["tokens"])

            self.assertIn("test:0", ids)
            self.assertNotIn("train:20", ids)
            metadata = json.loads((root / "processed" / "metadata.json").read_text())
            self.assertEqual(metadata["excluded_conflicting_rows"], 2)
            self.assertEqual(metadata["excluded_duplicate_rows"], 2)
            excluded = [json.loads(line) for line in
                        (root / "processed" / "excluded.jsonl").read_text(encoding="utf-8").strip("\n").split("\n")]
            self.assertEqual(ids | {row["source_id"] for row in excluded}, set(source))
            self.assertEqual(metadata["retained_rows"] + len(excluded), len(source))
            self.assertEqual(originals, {path.name: path.read_bytes() for path in raw.iterdir()})
            self.assertFalse(list((root / "processed").glob("*.tmp")))
            with self.assertRaises(ValueError):
                prepare_splits(raw, root / "invalid", val_size=1)
            self.assertFalse((root / "invalid").exists())
