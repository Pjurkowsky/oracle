"""Compare two exported bundles on binder glare before one replaces the other (spec 001).

    uv run python -m cardid.compare --bundle <baseline> --bundle <candidate>
    uv run python -m cardid.compare --bundle ~/manavault-bundle --bundle data/bundles/<version> --json data/compare.json

Both bundles answer the same queries end to end (detector.onnx, embed.onnx, search.onnx):

- **binder photos**: held-out captures labelled with `cardid.capture --photos` (source
  `binder-photos`), on the labels both galleries know;
- **synthetic binder**: ManaVault-style phone scenes (`render_phone_scene`) of eval-split
  cards with binder glare (`degrade.binder_glare`) on the card;
- **synthetic realistic**: the same kind of scenes without binder glare.

Scenes click the frame centre, as MTG Vault's scanner does. Their cards come from the eval
split only, so neither model trained on them; their full-card scans are downloaded once to
data/compare-cards/ (kept apart from data/cards, which detector training renders).

PASS when the candidate (second bundle) gets more right than the baseline on binder photos and
on synthetic binder, and is at most 1 point below it on synthetic realistic. With no binder
photos labelled, that rule is skipped. Exits 1 on FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np

from . import DATA_DIR
from .constants import SCENE
from .degrade import binder_glare
from .sources import BINDER_PHOTOS, row_source

COMPARE_CARD_DIR = DATA_DIR / "compare-cards"
REALISTIC_SLACK = 0.01  # the candidate may lose at most 1 point of top-1 on synthetic realistic
SETS = ("binder_photos", "synthetic_binder", "synthetic_realistic")


def eval_cards(arts: list[dict], n: int, seed: int) -> list[dict]:
    """`n` eval-split arts of plain single-faced cards, in a fixed random order."""
    picked = [a for a in arts if a.get("split") == "eval" and a.get("layout") == "normal" and not a.get("face") and not a.get("alias_of")]
    picked.sort(key=lambda a: a["id"])
    np.random.default_rng(seed).shuffle(picked)
    return picked[:n]


def download_scans(cards: list[dict], dest: Path = COMPARE_CARD_DIR) -> list[tuple[Path, str]]:
    """(scan path, art id) for every card whose full-card scan is (or now is) on disk."""
    import httpx

    from .catalog import card_image_url
    from .scryfall import HEADERS, fetch_image

    dest.mkdir(parents=True, exist_ok=True)
    missing = [c for c in cards if not (dest / f"{c['id']}.jpg").exists()]
    if missing:
        print(f"downloading {len(missing)} eval-split card scans to {dest}", flush=True)
        with httpx.Client(headers=HEADERS, follow_redirects=True) as client:
            for card in missing:
                fetch_image(client, {"id": card["id"], "card_url": card_image_url(card["url"])}, dest, "card_url")
    return [(dest / f"{c['id']}.jpg", c["id"]) for c in cards if (dest / f"{c['id']}.jpg").exists()]


def card_mask(quad: np.ndarray, size: int) -> np.ndarray:
    """The card's area in a scene, softened by a pixel so the glare has no hard outline."""
    mask = np.zeros((size, size), np.float32)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 1.0, lineType=cv2.LINE_AA)
    return cv2.GaussianBlur(mask, (0, 0), 1.0)


def scenes(scans: list[tuple[Path, str]], per_card: int, seed: int, glare: bool) -> Iterator[tuple[np.ndarray, str]]:
    """(640 px RGB phone scene, art id), deterministic in `seed` so both bundles see the same pixels."""
    from .image_bank import ArtBank, CardBank
    from .scene_renderer import render_phone_scene

    bank, backgrounds = CardBank([p for p, _ in scans]), ArtBank()
    rng = np.random.default_rng(seed)
    for i, (_, art_id) in enumerate(scans):
        for _ in range(per_card):
            scene, quad = render_phone_scene(rng, bank, backgrounds, out=SCENE, target_index=i)
            if glare:
                scene, _ = binder_glare(scene.astype(np.float32), rng, card_mask(quad, scene.shape[0]))
                scene = scene.astype(np.uint8)
            yield scene, art_id


def score_scenes(bundles: list, indexes: list[dict[str, int]], cases: Iterator[tuple[np.ndarray, str]]) -> list[dict]:
    """Top-1 of each bundle on the same scenes, clicking the frame centre like the scanner."""
    correct, count = [0] * len(bundles), 0
    for scene, art_id in cases:
        if not all(art_id in index for index in indexes):
            continue
        count += 1
        click = (scene.shape[1] / 2, scene.shape[0] / 2)
        for k, (bundle, index) in enumerate(zip(bundles, indexes, strict=True)):
            top = bundle.identify(scene, click)["results"][0]["id"]
            correct[k] += index.get(top) == index[art_id]
    return [{"correct": c, "count": count, "top1": c / count if count else None} for c in correct]


def binder_photo_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if row_source(r) == BINDER_PHOTOS and r.get("split") == "eval" and r.get("label")]


def verdict(results: dict[str, list[dict]]) -> tuple[bool, list[str]]:
    """(passed, one line per rule) for results[set] = [baseline, candidate]."""
    lines, passed = [], True
    for name in ("binder_photos", "synthetic_binder"):
        base, cand = results[name]
        if not base["count"]:
            if name == "binder_photos":
                lines.append("SKIP binder photos: none labelled (cardid.capture --photos)")
                continue
            passed = False
            lines.append(f"FAIL {name}: no queries")
            continue
        ok = cand["correct"] > base["correct"]
        passed &= ok
        lines.append(f"{'PASS' if ok else 'FAIL'} {name}: candidate {cand['correct']}/{cand['count']} must beat baseline {base['correct']}/{base['count']}")
    base, cand = results["synthetic_realistic"]
    if not base["count"]:
        passed = False
        lines.append("FAIL synthetic_realistic: no queries")
    else:
        ok = cand["top1"] >= base["top1"] - REALISTIC_SLACK
        passed &= ok
        lines.append(
            f"{'PASS' if ok else 'FAIL'} synthetic_realistic: candidate {cand['top1']:.3f} must be at most {REALISTIC_SLACK:.2f} below baseline {base['top1']:.3f}"
        )
    return passed, lines


def compare(
    paths: list[Path],
    rows: list[dict],
    real_dir: Path,
    scans: list[tuple[Path, str]],
    per_card: int,
    seed: int,
    load: Callable[[Path], object] | None = None,
    scorer: Callable | None = None,
) -> dict:
    from .gallery import bundle_index
    from .workflow import comparable_rows, score

    if load is None:
        from .bundle import Bundle

        load = Bundle
    scorer = scorer or score
    bundles = [load(p) for p in paths]
    indexes = [bundle_index(p) for p in paths]
    results: dict[str, list[dict]] = {}
    photos, dropped = comparable_rows(binder_photo_rows(rows), *paths)
    if dropped:
        print(f"WARNING: {len(dropped)} binder photos are unknown to one bundle and are not compared", flush=True)
    empty = {"correct": 0, "count": 0, "top1": None}
    results["binder_photos"] = [scorer(p, photos, real_dir) for p in paths] if photos else [dict(empty), dict(empty)]
    for name, glare in (("synthetic_binder", True), ("synthetic_realistic", False)):
        results[name] = score_scenes(bundles, indexes, scenes(scans, per_card, seed, glare))
    passed, lines = verdict(results)
    return {"bundles": [str(p) for p in paths], "results": results, "passed": passed, "rules": lines}


def print_report(report: dict) -> None:
    base, cand = report["bundles"]
    print(f"baseline:  {base}\ncandidate: {cand}")
    print(f"{'set':<22}{'baseline':>18}{'candidate':>18}")
    for name in SETS:
        cells = []
        for r in report["results"][name]:
            cells.append(f"{r['top1']:.3f} ({r['correct']}/{r['count']})" if r["count"] else "-")
        print(f"{name:<22}{cells[0]:>18}{cells[1]:>18}")
    for line in report["rules"]:
        print(line)
    print("PASS" if report["passed"] else "FAIL")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", action="append", type=Path, required=True, help="baseline first, then candidate")
    parser.add_argument("--cards", type=int, default=300, help="eval-split cards to render scenes of (default 300)")
    parser.add_argument("--per-card", type=int, default=1, help="scenes per card and set (default 1)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json", type=Path, help="also write the report here")
    args = parser.parse_args(argv)
    if len(args.bundle) != 2:
        parser.error("pass exactly two --bundle: the baseline, then the candidate")
    for path in args.bundle:
        if not (path / "manifest.json").is_file():
            parser.error(f"{path} is not a bundle (no manifest.json)")

    from .data import load_arts
    from .real import REAL_DIR, load_labels

    scans = download_scans(eval_cards(load_arts(), args.cards, args.seed))
    if not scans:
        raise SystemExit("no eval-split card scans could be downloaded; check the network and data/arts.json")
    report = compare(args.bundle, load_labels("eval", all_sources=True), REAL_DIR, scans, args.per_card, args.seed)
    print_report(report)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
