# Spec 001 decisions: training the scanner model on binder glare

## Requests

1. "i would like to help this model in detecing cards that are placed in the binder"
2. "so cards are seen by the camera with some glare or overlighting"
3. "use skills from my mtgvault repository" (spec workflow, grilling)

## Research

- MTG Vault installs ManaVault's bundles: GitHub releases `scanner-bundle-*` in
  cfbender/manavault (`mtgvault/backend/scanner/bundles.py`). Oracle trains them with the
  `manavault` profile (`CARDID_PUBLISH_TO=github:cfbender/manavault`, phone scenes).
- Embedder training pairs a clean art crop with `degrade.degraded_view` (profile `harsh`, the
  same for both apps). Glare there is one soft additive white blob in 40% of samples
  (`degrade.py` step 3). Nothing clips the art to white, veils the whole card or overexposes it.
- Detector phone scenes (`scene_renderer.render_phone_scene`) add sleeve/foil gloss (milky tint
  of 3-15% plus a band or blob) to sleeved cards and 45% of bare cards.
- MTG Vault spec 019 handles glare in the app (glare hint, combine frames, guesses) and lists
  retraining the model as out of scope. Its research: glare on the art box breaks a match from
  about 20% washed out; glare on the text box barely matters.
- Profiles pass training options through env files (`CARDID_SCENE_PROFILE=phone`), so a
  manavault-only option fits there.
- This laptop: no `data/` (no gallery, no captures), no profiles, Intel iGPU, 15 GB RAM.

## Round 1

- Spec lives in Oracle: `specs/001-binder-glare/`; tests are Oracle's pytest suite.
- Only the ManaVault scanner line learns binder glare; The Gathering's training is unchanged.
- Glare kinds: hard clipped hot spots and streaks, milky pocket haze, overexposed frame.
  Not a binder page detector scene.
- Delivery: train our own bundle and let MTG Vault install it from our own source
  (not an upstream PR to cfbender/oracle).

## Round 2

- Training runs on a Linux desktop with an RTX 2070 (8 GB, CUDA); this laptop has no usable GPU.
- Bundle reaches MTG Vault through a local folder install, not a GitHub release.
- Measure with a synthetic `binder` evaluation profile plus 30-50 phone photos of the user's
  binder pages, labelled and used for evaluation only.
- Binder glare on about 30% of training pairs, on top of today's soft blob.

## Round 3

- The RTX 2070 machine is a Linux desktop.
- Oracle gets a `cuda` extra (PyTorch CUDA 12.x wheels) beside `cpu` and `rocm`, with docs.
- Photos: taken with the phone, copied to a folder; `cardid.capture` gets an option to show
  those files instead of the webcam. Labelled as source `binder-photos`, always eval, never trained on.
- "if this would be better we will replace theirs with ours"

## Research (round 4)

- ManaVault's published `.pt` checkpoints are not public; the installed bundle
  (`mtgvault/backend/var/scanner/retrain-20261026T052522727512Z`) holds only ONNX graphs.
  `models/` in Oracle are The Gathering's promoted weights (neither hash matches ManaVault's).
- `cardid.evaluate` scores `.pt` checkpoints only, not a bundle folder.

## Round 4

- MTG Vault gets a setting for where the model is taken from (in MTG Vault's own spec).
- Fine-tune from Oracle's `models/recogniser.pt`.
- Gate: higher top-1 than ManaVault's installed bundle on the binder photos and on the
  synthetic `binder` set, and at most 1 point lower on synthetic `realistic`.
- Our bundle reuses ManaVault's `detector.onnx` unchanged; only the embedder and gallery change.

## Research (round 5)

- MTG Vault: `sync_scanner_bundle --from-dir` installs one bundle folder by hand; the daily
  Celery task (04:00) always installs the newest cfbender/manavault release. Staff users have an
  Admin menu item that opens Django admin.

## Round 5

- The model source is set in the app, not an env var.
- A folder source holds one subfolder per bundle; sync installs the newest by manifest `created`.
- Oracle gets `cardid.compare` (bundle vs bundle, PASS/FAIL); installing stays the user's call.
- `new-set` and `retrain` can write to a local folder (`CARDID_PUBLISH_TO=dir:<path>`, an
  Oracle profile `mtgvault`), no training for new-set, no GitHub publish.

## Round 6

- The model source controls are a staff page in the app, not Django admin.
- Saving a new source installs right away; the daily sync follows the saved source.
- Getting bundles from the training desktop to the server: decided later, out of scope.

## Round 7

- A staff-only "Scanner model" section on the Settings page.
- It shows: Source (GitHub releases + owner/repo, default cfbender/manavault; or Server folder +
  path), installed version and date, last sync result with its error, and Sync now. A failed
  sync shows a red alert and keeps the previous bundle.
