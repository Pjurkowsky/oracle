"""cardid.compare (spec 001 AC-6 to AC-8): same queries for both bundles, PASS/FAIL rules."""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import compare


def result(correct: int, count: int) -> dict:
    return {"correct": correct, "count": count, "top1": correct / count if count else None}


class VerdictTest(unittest.TestCase):
    def results(self, photos=((3, 10), (5, 10)), binder=((40, 100), (60, 100)), realistic=((80, 100), (79, 100))):
        return {
            "binder_photos": [result(*photos[0]), result(*photos[1])],
            "synthetic_binder": [result(*binder[0]), result(*binder[1])],
            "synthetic_realistic": [result(*realistic[0]), result(*realistic[1])],
        }

    def test_pass(self):
        passed, lines = compare.verdict(self.results())
        self.assertTrue(passed, lines)
        self.assertEqual([line.split()[0] for line in lines], ["PASS", "PASS", "PASS"])

    def test_candidate_must_beat_baseline_on_photos_and_binder(self):
        self.assertFalse(compare.verdict(self.results(photos=((5, 10), (5, 10))))[0])
        self.assertFalse(compare.verdict(self.results(binder=((60, 100), (59, 100))))[0])

    def test_realistic_may_drop_one_point(self):
        self.assertTrue(compare.verdict(self.results(realistic=((80, 100), (79, 100))))[0])
        passed, lines = compare.verdict(self.results(realistic=((80, 100), (78, 100))))
        self.assertFalse(passed)
        self.assertTrue(lines[-1].startswith("FAIL synthetic_realistic"))

    def test_no_photos_is_skipped_not_failed(self):
        passed, lines = compare.verdict(self.results(photos=((0, 0), (0, 0))))
        self.assertTrue(passed)
        self.assertTrue(lines[0].startswith("SKIP"))


class FakeBundle:
    """Answers from a fixed table: art id seen in the scene's red channel -> predicted id."""

    def __init__(self, answers: dict[int, str]):
        self.answers = answers
        self.clicks = []

    def identify(self, img, click):
        self.clicks.append(click)
        return {"results": [{"id": self.answers[int(img[0, 0, 0])]}]}


class ScoreScenesTest(unittest.TestCase):
    def test_same_scenes_and_frame_centre_click(self):
        scene = lambda k: np.full((640, 640, 3), k, np.uint8)  # noqa: E731
        cases = [(scene(1), "a"), (scene(2), "b"), (scene(3), "unknown-to-b")]
        a = FakeBundle({1: "a", 2: "x", 3: "unknown-to-b"})
        b = FakeBundle({1: "a", 2: "b"})
        indexes = [{"a": 0, "b": 1, "x": 2, "unknown-to-b": 3}, {"a": 0, "b": 1, "x": 2}]
        base, cand = compare.score_scenes([a, b], indexes, iter(cases))
        self.assertEqual((base["correct"], base["count"]), (1, 2))
        self.assertEqual((cand["correct"], cand["count"]), (2, 2))
        self.assertEqual(a.clicks, [(320.0, 320.0)] * 2)


class EvalCardsTest(unittest.TestCase):
    def test_only_plain_eval_split_cards(self):
        arts = [
            {"id": "e1", "split": "eval", "layout": "normal", "face": 0},
            {"id": "e2", "split": "eval", "layout": "normal", "face": 0},
            {"id": "t1", "split": "train", "layout": "normal", "face": 0},
            {"id": "tok", "split": "eval", "layout": "token", "face": 0},
            {"id": "e3-1", "split": "eval", "layout": "normal", "face": 1},
            {"id": "alias", "split": "eval", "layout": "normal", "alias_of": "e1"},
        ]
        picked = compare.eval_cards(arts, 10, seed=1)
        self.assertEqual(sorted(a["id"] for a in picked), ["e1", "e2"])
        self.assertEqual(compare.eval_cards(arts, 10, seed=1), picked)
        self.assertEqual(len(compare.eval_cards(arts, 1, seed=1)), 1)


class CompareTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.paths = []
        for name in ("base", "cand"):
            path = self.root / name
            path.mkdir()
            (path / "manifest.json").write_text("{}")
            (path / "arts.json").write_text(json.dumps([{"id": "a"}, {"id": "b"}]))
            self.paths.append(path)

    def test_binder_photos_only_and_report(self):
        rows = [
            {"capture_id": "p1", "label": "a", "split": "eval", "source": "binder-photos"},
            {"capture_id": "p2", "label": "zzz", "split": "eval", "source": "binder-photos"},
            {"capture_id": "w1", "label": "b", "split": "eval"},
        ]
        scored = []

        def scorer(path, photos, real):
            scored.append((path.name, [r["capture_id"] for r in photos]))
            return result(0 if path.name == "base" else 1, len(photos))

        scenes = [(np.full((640, 640, 3), 1, np.uint8), "a"), (np.full((640, 640, 3), 2, np.uint8), "b")]
        bundles = {"base": FakeBundle({1: "a", 2: "a"}), "cand": FakeBundle({1: "a", 2: "b"})}
        with patch.object(compare, "scenes", side_effect=lambda *a: iter(scenes)), redirect_stdout(io.StringIO()):
            report = compare.compare(self.paths, rows, self.root, [], 1, 0, load=lambda p: bundles[p.name], scorer=scorer)
        self.assertEqual(scored, [("base", ["p1"]), ("cand", ["p1"])])
        self.assertTrue(report["passed"], report["rules"])
        self.assertEqual(report["results"]["synthetic_binder"][1]["correct"], 2)
        out = io.StringIO()
        with redirect_stdout(out):
            compare.print_report(report)
        self.assertIn("synthetic_realistic", out.getvalue())
        self.assertTrue(out.getvalue().strip().endswith("PASS"))


if __name__ == "__main__":
    unittest.main()
