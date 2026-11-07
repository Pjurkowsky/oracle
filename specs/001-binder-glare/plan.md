# Plan 001: Training the scanner model on binder glare

Spec: [spec.md](spec.md)

| Topic | Choice |
|---|---|
| Glare | `degrade.binder_glare(img, rng, mask)`: hot spot (super-Gaussian blob or streak, +150-320, clipped at 255), haze (blend towards a pale veil, 15-50%), overexposure (gain 1.2-1.8, clipped). One kind, sometimes two. Applied after photometrics, before optics. |
| Profiles | `Degradation.binder_p`; `harsh-binder` = harsh + 0.3 (training), `binder` = realistic + 1.0 (eval). `train --degradation`. |
| CUDA | `cuda` extra on the PyTorch cu126 index, conflicting with `cpu` and `rocm`. |
| Photo mode | `capture --photos`: `/mode` tells the page; page draws the picked file letterboxed into a 640 canvas (rgb 18) and crops from it. Server saves `source: binder-photos`, `split: eval`. |
| Compare | `cardid.compare`: real rows = eval labels with source `binder-photos`, scored with `workflow.score` on labels both bundles know. Synthetic: `render_phone_scene(out=640)` of eval-split scans, plus `binder_glare` masked to the card quad for `binder`; same seed for both bundles; `Bundle.identify` at the frame centre, as the scanner reads it. `--fetch` downloads a baseline with `workflow.snapshot_bundle`. |
| Detector reuse | `export --detector-onnx`: copy the file over `detector.onnx`, manifest `detector.onnx_source`/`onnx_sha256`; verify compares recognition on the bundle's own quad. |
| Profile | `profiles/mtgvault.env.example`; retrain reads `CARDID_DEGRADATION` and `CARDID_DETECTOR_ONNX`. |

## Files

- `pyproject.toml`, `uv.lock`: `cuda` extra
- `cardid/degrade.py`: binder glare, profiles, preview flag
- `cardid/train.py`: `--degradation`
- `cardid/capture.py`, `cardid/capture.html`, `cardid/real.py`, `cardid/sources.py`: photo mode
- `cardid/compare.py`: new
- `cardid/export.py`: `--detector-onnx`
- `cardid/retrain.py`: profile options
- `profiles/mtgvault.env.example`, `mise.toml` (`setup-mtgvault`, `mtgvault`, `compare`, `label-photos`)
- `docs/binder-glare.md`, `docs/training.md`, `README.md`

## Tests

| Test | ACs |
|---|---|
| `cardid/test_binder_glare.py` | AC-1, AC-2 |
| `cardid/test_capture.py` | AC-4, AC-5 |
| `cardid/test_compare.py` | AC-6, AC-7, AC-8 |
| `cardid/test_manifest_contract.py` | AC-9 |
| `cardid/test_retrain.py` | AC-10 |

AC-3 is checked by `uv lock` resolving the extra and by running on the desktop.

## Order

Glare and profiles, CUDA extra, photo mode, compare, detector reuse, profile and
retrain, docs.
