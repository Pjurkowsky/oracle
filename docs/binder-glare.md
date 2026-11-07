# Binder glare: MTG Vault's scanner model

MTG Vault scans cards in binder pockets, where the camera sees hot spots, a milky haze and
overexposed frames. The `mtgvault` model line ([spec 001](../specs/001-binder-glare/spec.md))
fine-tunes ManaVault's phone-scanner pipeline on that glare, measures it against ManaVault's
bundle, and writes passing bundles to a local folder that MTG Vault installs from (Settings →
Scanner model → Server folder).

| Piece | What it does |
|---|---|
| `degrade.binder_glare` | hot spots and streaks clipped to white, pocket haze, overexposure |
| `train --degradation harsh-binder` | binder glare on ~30% of the training pairs |
| `capture --photos` (`mise run label-photos`) | label phone photos of your binder, card by card; eval only |
| `compare` (`mise run compare`) | baseline vs candidate on binder photos, synthetic binder and realistic; PASS/FAIL |
| `export --detector-onnx` | ship ManaVault's detector graph unchanged |
| `mtgvault` profile (`mise run mtgvault`) | all of the above in one retrain, written to a folder |

## 1. Set up the NVIDIA desktop

Needs Linux, an NVIDIA card from the RTX 20 series on (an RTX 2070 works) and driver 525 or
newer (`nvidia-smi` shows the version). About 10 GB of disk for the gallery and runs.

```sh
curl https://mise.run | sh                         # once; then open a new shell
git clone <this repository> oracle && cd oracle    # see "Getting the code" below
mise install
mise exec -- uv sync --extra cuda
export UV_NO_SYNC=1                                # keep the CUDA torch (add it to ~/.bashrc)
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True NVIDIA GeForce RTX 2070` (or your card). Then download the
gallery and the card scans that export and comparison render:

```sh
uv run python -m cardid.scryfall --all          # ~52k art crops, ~3 GB, an hour or two
uv run python -m cardid.scryfall --cards 3000   # full-card scans for scenes and hub penalties
```

**Getting the code.** The spec 001 commits live in your laptop's checkout. Either push them to
a fork you own and clone that, or clone straight from the laptop:
`git clone ssh://<laptop>/home/<you>/repos/oracle`.

## 2. Get ManaVault's bundle (the baseline)

```sh
sudo pacman -S github-cli   # or your distro's gh package
gh auth login
mise run compare -- --fetch github:cfbender/manavault
```

It prints `data/bundles/<manavault-version>`. That bundle is what MTG Vault runs today: the
baseline to beat, and the source of the detector graph we ship.

## 3. Photograph and label your binder

1. With your phone, photograph 30-50 binder pages the way you scan them: same lamp, same
   distance, glare included. Vary the pages and the angle a little.
2. Copy the photos to the desktop (any folder).
3. `mise run label-photos`, open http://localhost:8765, click **Choose Files** and pick them.
4. Click a card, confirm it with `1`-`5`, or press `/` and search when none is right
   (`forest fin 280` narrows by set and number). Do every card on the page, then `N` for
   the next photo (`P` goes back, `S` skips a card).

Each photo is fitted into a 640 px square with dark bars, as MTG Vault's scanner does with its
camera frames. Labels are stored in `data/real/` as source `binder-photos`, always in the eval
split: no run ever trains on them. Nine cards a page gives a few hundred test cards.

## 4. Train

```sh
mise run setup-mtgvault          # creates ~/.config/cardid/mtgvault.env
${EDITOR:-nano} ~/.config/cardid/mtgvault.env
mkdir -p ~/mtgvault-bundles
```

In the profile set `CARDID_PUBLISH_TO` to that folder (`/home/<you>/mtgvault-bundles`) and
`CARDID_DETECTOR_ONNX` to `data/bundles/<manavault-version>/detector.onnx` from step 2.

```sh
mise run mtgvault -- --no-publish
```

This fine-tunes `models/recogniser.pt` for 6 epochs with binder glare on ~30% of the pairs,
exports `data/bundles/retrain-<time>/` with ManaVault's detector graph, checks it against the
torch pipeline and prints the synthetic `realistic` and `binder` scores. The augmentation runs
on the CPU and is usually the bottleneck; `uv run python -m cardid.bench_loader --workers 12 8`
finds the best worker count, then add `-- --workers N`. The report is in `data/retrain/`.

## 5. Compare, then hand over

```sh
mise run compare -- --bundle data/bundles/<manavault-version> --bundle data/bundles/retrain-<time>
```

```
set                             baseline         candidate
binder_photos            0.612 (186/304)   0.743 (226/304)
synthetic_binder         0.410 (123/300)   0.560 (168/300)
synthetic_realistic      0.873 (262/300)   0.867 (260/300)
PASS binder_photos: candidate 226/304 must beat baseline 186/304
...
PASS
```

(Example numbers.) PASS means the new model wins on your binder photos and on synthetic glare
and loses at most 1 point on glare-free scenes. On FAIL, try more epochs
(`CARDID_RETRAIN_EPOCHS`) or more photos before changing anything else. On PASS:

```sh
mise run publish -- data/bundles/retrain-<time> --to ~/mtgvault-bundles
```

The folder now holds `<version>/` and a `current` link. Copy it to the MTG Vault server however
suits you (rsync, a shared drive). With MTG Vault's Docker setup, put it in `./data/app/bundles`
(the container sees `/data/bundles`):

```sh
rsync -a ~/mtgvault-bundles/ <server>:<mtgvault>/data/app/bundles/
```

Then as a staff user open MTG Vault's **Settings → Scanner model**, choose **Server folder**,
enter the path as the server sees it (`/data/bundles`) and **Save**: it installs the newest
bundle in the folder at once and the daily sync keeps following it.
Switching back to **GitHub releases** (`cfbender/manavault`) restores ManaVault's models.

Later runs can publish straight away: `mise run mtgvault` publishes only when the new bundle is
at least as good as the one in the folder on your binder photos.

## New sets

New cards need gallery embeddings, not training:

```sh
mise run new-set -- --profile mtgvault
```

It refreshes the gallery, re-exports the published checkpoints with ManaVault's detector graph,
checks your binder photos still score as well, and writes a new version to the folder. Copy it
to the server; MTG Vault picks it up at its next daily sync, or press **Sync now**.
