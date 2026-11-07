# Spec 001: Training the scanner model on binder glare

Status: Done
Followed by: MTG Vault 020-scanner-model-source

## Goal

The phone-scanner model recognises cards behind glossy binder pockets: hard
glare spots, a milky haze and overexposed frames. We train it on an NVIDIA
desktop, measure it against ManaVault's bundle on synthetic glare and on real
photos of a binder, and write passing bundles to a local folder that MTG Vault
can install from.

## Terms

- **Binder glare**: one or two of: a hot spot (a sharp patch or streak clipped
  to white), pocket haze (a pale veil that flattens contrast and colour), an
  overexposed frame (the whole card too bright, highlights clipped).
- **Binder photos**: phone photos of binder pages, labelled card by card in
  `cardid.capture`, source `binder-photos`.

## Acceptance criteria

### Training

- **AC-1** `train --degradation harsh-binder` adds binder glare to about 30% of
  training pairs on top of today's degradation. Without the flag training is
  unchanged, and The Gathering's profile does not use it.
- **AC-2** `python -m cardid.degrade <art_id> --degradation binder` writes a
  preview sheet of binder-glare samples.
- **AC-3** `uv sync --extra cuda` installs a CUDA build of PyTorch that runs on
  an RTX 2070, and `--device auto` trains on it. The `cpu` and `rocm` extras
  are unchanged.

### Binder photos

- **AC-4** `cardid.capture --photos` opens phone photos picked in the page
  instead of the webcam. Each photo is fitted into a 640 px square with dark
  bars, as MTG Vault's scanner does. You click and label each card as usual;
  N and P move to the next and previous photo.
- **AC-5** Labels saved in photo mode have source `binder-photos` and are
  always in the eval split, so no training run ever trains on them.

### Comparing bundles

- **AC-6** `python -m cardid.compare --bundle A --bundle B` prints top-1 for
  each bundle on: binder photos, synthetic `binder` phone scenes and synthetic
  `realistic` phone scenes, over the same queries for both.
- **AC-7** It prints PASS when B beats A on binder photos and on synthetic
  `binder`, and is at most 1 point below A on `realistic`; otherwise FAIL and
  which rule failed. It exits non-zero on FAIL.
- **AC-8** Synthetic scenes use only eval-split cards, so neither model has
  trained on them.

### Bundles for MTG Vault

- **AC-9** `export --detector-onnx <file>` puts that detector graph in the
  bundle unchanged (ManaVault's), and the manifest records where it came from.
  The export check still compares recognition with the torch pipeline.
- **AC-10** A `mtgvault` profile trains with binder glare, keeps ManaVault's
  detector graph, gates on binder photos and writes bundles to a local folder
  (`<folder>/<version>`, with `current`). `mise run new-set -- --profile mtgvault`
  refreshes the gallery for new sets without training.
- **AC-11** The docs say how to set up the NVIDIA desktop, take and label
  binder photos, train, compare and hand the folder to MTG Vault.

## Out of scope

- Copying bundles from the training desktop to the MTG Vault server
- Binder page scenes for the detector, retraining the detector
- The Gathering's model line, publishing to GitHub

## Assumptions (filled in during writing, not grilled)

- The `binder` eval profile puts binder glare on every query; `realistic` is
  today's profile (AC-6).
- Synthetic comparison renders ManaVault-style phone scenes of 300 eval-split
  cards by default (`--cards`), downloaded once to `data/compare-cards/`, and
  runs each bundle end to end; both bundles share the detector graph, so the
  difference is recognition (AC-6, AC-8).
- "Beats" means strictly more correct answers on the same queries; with no
  binder photos labelled, that rule is reported as skipped and does not fail
  (AC-7).
- Photo mode labels need the photos in the browser only; the server never reads
  your photo folder (AC-4).
