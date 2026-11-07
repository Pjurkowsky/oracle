"""Synthetic webcam degradation of a clean Scryfall art crop.

The model input is what the recognizer sees *after* the detector has found a card quad,
perspective-warped it to a canonical card, and cut out the fixed art box. So a degraded
sample simulates: detector misalignment, residual perspective, the camera's low pixel
density on the art (60-140 px wide on a 1080p table shot), optics blur, lighting and white
balance, glare off sleeves, sensor noise, video compression, and partial occlusion by
dice/counters/overlapping cards. Profiles with `binder_p` add binder glare (`binder_glare`):
the hot spots, pocket haze and overexposure of a phone over a glossy binder page.

    uv run python -m cardid.degrade <art_id>                          # data/preview-<id>.png, 12 samples
    uv run python -m cardid.degrade <art_id> --degradation binder     # every sample with binder glare
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import cv2
import numpy as np
from PIL import Image

INPUT_SIZE = 128  # model input (square); aspect is squashed identically for gallery and query


@dataclass(frozen=True)
class Degradation:
    """Knobs so evaluation can bucket by difficulty. Widths are the simulated art width in px."""

    min_width: int = 56
    max_width: int = 140
    strong_perspective_p: float = 0.25
    occlusion_p: float = 0.25
    glare_p: float = 0.4
    motion_blur_p: float = 0.2
    binder_p: float = 0.0  # share of samples with binder glare (see `binder_glare`)


# Training uses the harsh defaults above for robustness. Evaluation also reports this profile,
# which assumes the detector returned a quad that was perspective-warped, so only crop jitter
# and mild residual perspective remain, and the art is at least ~70 px wide (a 1080p camera
# over a 4-player table; smaller than that and the card is unreadable to a human too).
PROFILES = {
    "harsh": Degradation(),
    "realistic": Degradation(min_width=70, max_width=140, strong_perspective_p=0.0, occlusion_p=0.15, glare_p=0.4, motion_blur_p=0.1),
}
# Binder pockets: training adds binder glare to ~30% of the harsh pairs (MTG Vault's scanner
# line); the `binder` eval profile is `realistic` with binder glare on every query.
PROFILES["harsh-binder"] = replace(PROFILES["harsh"], binder_p=0.3)
PROFILES["binder"] = replace(PROFILES["realistic"], binder_p=1.0)

BINDER_KINDS = ("hot_spot", "haze", "overexposed")


def _box(mask: np.ndarray | None, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """(x0, y0, width, height) of the glare's target: the mask's bounding box, else the image."""
    if mask is None:
        return 0.0, 0.0, float(shape[1]), float(shape[0])
    ys, xs = np.nonzero(mask > 0.5)
    if not len(xs):
        return 0.0, 0.0, float(shape[1]), float(shape[0])
    return float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)


def binder_glare(img: np.ndarray, rng: np.random.Generator, mask: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    """Binder-pocket glare on a float32 RGB image (0-255), clipped at 255 like a saturated sensor.

    One kind, or two in a third of the samples:
    - hot_spot: a lamp or window reflected in the pocket, a sharp-edged patch or streak clipped
      to white where the art is lost;
    - haze: the pocket's plastic as a pale veil over the card, flattening contrast and colour;
    - overexposed: phone auto-exposure metering the dark page, so the whole card is too bright.

    `mask` (H x W, 0-1) limits the glare to the card in a scene; without it the whole image is
    the card (an art crop). Returns the image and the kinds applied."""
    out = img.astype(np.float32, copy=True)
    h, w = out.shape[:2]
    weight = np.ones((h, w), np.float32) if mask is None else mask.astype(np.float32)
    bx, by, bw, bh = _box(mask, (h, w))
    count = 2 if rng.random() < 1 / 3 else 1
    kinds = [BINDER_KINDS[int(i)] for i in sorted(rng.choice(len(BINDER_KINDS), size=count, replace=False))]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for kind in kinds:
        if kind == "hot_spot":
            if rng.random() < 0.5:  # a patch with a steep edge (super-Gaussian)
                gx, gy = bx + rng.uniform(0.1, 0.9) * bw, by + rng.uniform(0.1, 0.9) * bh
                sx, sy = rng.uniform(0.08, 0.35) * bw, rng.uniform(0.06, 0.3) * bh
                r2 = ((xx - gx) / sx) ** 2 + ((yy - gy) / sy) ** 2
                spot = np.exp(-(r2 ** rng.uniform(1.5, 3.0)))
            else:  # a streak across the pocket, any direction
                ang = rng.uniform(0, np.pi)
                cx, cy = bx + rng.uniform(0.2, 0.8) * bw, by + rng.uniform(0.2, 0.8) * bh
                d = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
                spot = np.exp(-((np.abs(d) / (rng.uniform(0.03, 0.15) * min(bw, bh))) ** rng.uniform(2.0, 4.0)))
            # a little bloom around the clipped core
            spot = np.clip(spot * 1.15, 0, 1)
            out += (spot * weight * rng.uniform(150, 320))[..., None]
        elif kind == "haze":
            veil = np.float32(rng.uniform(200, 250, size=3)) * rng.uniform(0.97, 1.03, size=3).astype(np.float32)
            # strongest towards one side of the pocket, where the light comes from
            ang = rng.uniform(0, 2 * np.pi)
            ramp = ((xx - bx) / max(bw, 1) - 0.5) * np.cos(ang) + ((yy - by) / max(bh, 1) - 0.5) * np.sin(ang)
            amount = np.clip(rng.uniform(0.15, 0.5) * (1 + rng.uniform(0, 0.8) * ramp), 0, 0.75) * weight
            out = out * (1 - amount[..., None]) + veil * amount[..., None]
        else:
            gain = rng.uniform(1.3, 2.0)
            lifted = out * gain + rng.uniform(0, 30)
            out = out + (lifted - out) * weight[..., None]
    return np.clip(out, 0, 255), kinds


def clean_view(img: np.ndarray) -> np.ndarray:
    """Gallery view: the art crop resized straight to model input."""
    return cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)


def degraded_view(img: np.ndarray, rng: np.random.Generator, cfg: Degradation = PROFILES["harsh"]) -> tuple[np.ndarray, dict]:
    """Return (128x128 RGB uint8, info) where info records the sampled difficulty knobs."""
    h, w = img.shape[:2]
    width = int(rng.integers(cfg.min_width, cfg.max_width + 1))
    height = max(8, round(width * h / w))

    # 1. Geometry: crop jitter (detector error) + residual perspective + slight rotation,
    #    rendered directly at the low capture resolution so downscaling is part of the warp.
    strong = rng.random() < cfg.strong_perspective_p
    persp = 0.18 if strong else 0.05
    scale = rng.uniform(0.90, 1.12)
    shift = rng.uniform(-0.06, 0.06, size=2)
    angle = rng.uniform(-4, 4)
    # A sensor integrates over its pixel area, so downscale with INTER_AREA to ~1.5x the target
    # first; warpPerspective only offers point-sampling interpolation and would alias.
    pre_w, pre_h = int(width * 1.5), max(8, int(height * 1.5))
    pre = cv2.resize(img, (pre_w, pre_h), interpolation=cv2.INTER_AREA)
    src = np.float32([[0, 0], [pre_w, 0], [pre_w, pre_h], [0, pre_h]])
    # destination box in low-res space, then perturbed
    cx, cy = width / 2 + shift[0] * width, height / 2 + shift[1] * height
    hw, hh = width * scale / 2, height * scale / 2
    dst = np.float32([[cx - hw, cy - hh], [cx + hw, cy - hh], [cx + hw, cy + hh], [cx - hw, cy + hh]])
    dst += rng.uniform(-persp, persp, size=(4, 2)).astype(np.float32) * np.float32([width, height])
    rot = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    dst = (rot @ np.c_[dst, np.ones(4)].T).T.astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    # border: what leaks in when the crop overshoots the art box is card frame, so a dark
    # or parchment-ish flat colour rather than a mirrored art edge.
    frames = [(20, 20, 20), (30, 28, 25), (200, 190, 165), (60, 60, 70), (235, 230, 220)]
    frame = tuple(float(c) for c in frames[int(rng.integers(0, len(frames)))])
    low = cv2.warpPerspective(pre, H, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=frame)
    low = low.astype(np.float32)

    # 2. Photometrics: exposure, contrast, white balance, gamma, saturation.
    gain = rng.uniform(0.85, 1.15, size=3).astype(np.float32)
    low = low * gain
    low = (low - 128) * rng.uniform(0.7, 1.3) + 128 + rng.uniform(-35, 35)
    low = np.clip(low, 0, 255)
    low = 255 * (low / 255) ** rng.uniform(0.8, 1.25)
    gray = low.mean(axis=2, keepdims=True)
    low = gray + (low - gray) * rng.uniform(0.65, 1.2)

    # 3. Glare: soft elliptical highlight or a sleeve band.
    if rng.random() < cfg.glare_p:
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        gx, gy = rng.uniform(0, width), rng.uniform(0, height)
        sx, sy = rng.uniform(0.15, 0.6) * width, rng.uniform(0.1, 0.6) * height
        blob = np.exp(-(((xx - gx) / sx) ** 2 + ((yy - gy) / sy) ** 2))
        low = low + blob[..., None] * rng.uniform(40, 140)
    binder: list[str] = []
    if cfg.binder_p and rng.random() < cfg.binder_p:
        low, binder = binder_glare(low, rng)

    # 4. Occlusion: a die/counter (circle) or an overlapping card edge (rectangle from a side).
    if rng.random() < cfg.occlusion_p:
        color = tuple(float(c) for c in rng.integers(0, 255, size=3))
        if rng.random() < 0.5:
            r = int(rng.uniform(0.08, 0.18) * width)
            cv2.circle(low, (int(rng.uniform(0, width)), int(rng.uniform(0, height))), r, color, -1)
        else:
            frac = rng.uniform(0.08, 0.22)
            side = rng.integers(0, 4)
            if side == 0:
                low[:, : int(width * frac)] = color
            elif side == 1:
                low[:, width - int(width * frac) :] = color
            elif side == 2:
                low[: int(height * frac), :] = color
            else:
                low[height - int(height * frac) :, :] = color

    # 5. Optics: defocus and occasionally motion blur.
    sigma = rng.uniform(0, 1.3)
    if sigma > 0.2:
        low = cv2.GaussianBlur(low, (0, 0), sigma)
    if rng.random() < cfg.motion_blur_p:
        k = int(rng.integers(3, 8))
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0), (k, k))
        low = cv2.filter2D(low, -1, kernel / max(kernel.sum(), 1e-6))

    # 6. Sensor noise, then video-codec style compression at the low resolution.
    low = low + rng.normal(0, rng.uniform(1.5, 9), size=low.shape).astype(np.float32)
    low = np.clip(low, 0, 255).astype(np.uint8)
    quality = int(rng.integers(35, 90))
    _, enc = cv2.imencode(".jpg", cv2.cvtColor(low, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    low = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    # 7. Upsample to model input as the recognizer would.
    out = cv2.resize(low, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    return out, {"width": width, "strong_perspective": strong, "jpeg_quality": quality, "blur_sigma": round(sigma, 2), "binder": binder}


def load_rgb(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preview(art_id: str, n: int = 12, profile: str = "harsh") -> None:
    from . import ART_DIR, DATA_DIR

    img = load_rgb(ART_DIR / f"{art_id}.jpg")
    rng = np.random.default_rng(0)
    tiles = [clean_view(img)] + [degraded_view(img, rng, PROFILES[profile])[0] for _ in range(n - 1)]
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * (INPUT_SIZE + 4), cols * (INPUT_SIZE + 4), 3), 40, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * (INPUT_SIZE + 4) : r * (INPUT_SIZE + 4) + INPUT_SIZE, c * (INPUT_SIZE + 4) : c * (INPUT_SIZE + 4) + INPUT_SIZE] = t
    out = DATA_DIR / (f"preview-{art_id}.png" if profile == "harsh" else f"preview-{art_id}-{profile}.png")
    Image.fromarray(sheet).save(out)
    print(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a sheet of degraded samples of one art.")
    parser.add_argument("art_id")
    parser.add_argument("--degradation", choices=sorted(PROFILES), default="harsh")
    parser.add_argument("--n", type=int, default=12)
    args = parser.parse_args()
    preview(args.art_id, args.n, args.degradation)


if __name__ == "__main__":
    main()
