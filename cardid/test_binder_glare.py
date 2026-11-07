"""Binder glare (spec 001 AC-1, AC-2): the augmentation, its profiles and the preview."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from . import degrade
from .degrade import BINDER_KINDS, PROFILES, binder_glare, degraded_view


def art(h: int = 120, w: int = 160) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.dstack([xx / w * 200 + 20, yy / h * 180 + 30, np.sin(xx / 9) * 60 + 100]).astype(np.uint8)
    cv2.circle(img, (50, 60), 30, (200, 40, 40), -1)
    return img


class BinderGlareTest(unittest.TestCase):
    def test_kinds_are_known_and_output_is_clipped(self):
        rng = np.random.default_rng(0)
        seen = set()
        for _ in range(60):
            out, kinds = binder_glare(art().astype(np.float32), rng)
            self.assertTrue(1 <= len(kinds) <= 2)
            self.assertTrue(set(kinds) <= set(BINDER_KINDS))
            self.assertGreaterEqual(out.min(), 0)
            self.assertLessEqual(out.max(), 255)
            seen.update(kinds)
        self.assertEqual(seen, set(BINDER_KINDS))

    def test_glare_brightens_the_card(self):
        rng = np.random.default_rng(1)
        img = art().astype(np.float32)
        brighter = [binder_glare(img, rng)[0].mean() > img.mean() for _ in range(30)]
        self.assertGreater(sum(brighter), 25)

    def test_hot_spot_clips_part_of_the_art_to_white(self):
        rng = np.random.default_rng(2)
        clipped = []
        for _ in range(80):
            out, kinds = binder_glare(art().astype(np.float32), rng)
            if kinds == ["hot_spot"]:
                clipped.append(float((out.min(axis=2) >= 254.5).mean()))
        self.assertTrue(clipped)
        self.assertGreater(np.median(clipped), 0.01)

    def test_mask_keeps_glare_on_the_card(self):
        rng = np.random.default_rng(3)
        img = np.full((100, 100, 3), 60, np.float32)
        mask = np.zeros((100, 100), np.float32)
        mask[30:70, 30:70] = 1
        for _ in range(20):
            out, _ = binder_glare(img, rng, mask)
            np.testing.assert_array_equal(out[mask == 0], img[mask == 0])


class ProfilesTest(unittest.TestCase):
    def test_profiles(self):
        self.assertEqual(PROFILES["harsh"].binder_p, 0.0)
        self.assertEqual(PROFILES["realistic"].binder_p, 0.0)
        self.assertEqual(PROFILES["harsh-binder"].binder_p, 0.3)
        self.assertEqual(PROFILES["binder"].binder_p, 1.0)
        self.assertEqual(PROFILES["binder"].min_width, PROFILES["realistic"].min_width)

    def test_harsh_is_unchanged(self):
        """Without binder glare the random stream, and so every cached eval query, stays as before."""
        img = art()
        with patch.object(degrade, "binder_glare", side_effect=AssertionError("binder glare on harsh")):
            a, info = degraded_view(img, np.random.default_rng(5), PROFILES["harsh"])
        self.assertEqual(info["binder"], [])
        b, _ = degraded_view(img, np.random.default_rng(5), PROFILES["harsh"])
        np.testing.assert_array_equal(a, b)

    def test_share_of_glared_samples(self):
        rng = np.random.default_rng(6)
        img = art()
        glared = [bool(degraded_view(img, rng, PROFILES["harsh-binder"])[1]["binder"]) for _ in range(300)]
        self.assertAlmostEqual(np.mean(glared), 0.3, delta=0.08)
        self.assertTrue(all(degraded_view(img, rng, PROFILES["binder"])[1]["binder"] for _ in range(20)))


class PreviewTest(unittest.TestCase):
    def test_binder_preview_sheet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "art").mkdir()
            cv2.imwrite(str(root / "art" / "x.jpg"), art())
            with patch("cardid.ART_DIR", root / "art"), patch("cardid.DATA_DIR", root), redirect_stdout(io.StringIO()):
                degrade.preview("x", n=8, profile="binder")
            self.assertTrue((root / "preview-x-binder.png").is_file())


if __name__ == "__main__":
    unittest.main()
