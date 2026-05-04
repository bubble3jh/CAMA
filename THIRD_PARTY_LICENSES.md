# Third-Party Code and Licenses

This repository vendors or adapts code from several open-source projects.
All third-party components retain their original licenses; the relevant
files and their upstream sources are listed below. Anonymous reviewers
who wish to verify attribution can compare any specific file against the
upstream repository.

---

## 1. BATCLIP framework
**Upstream**: BATCLIP test-time-adaptation framework (Maharana et al., 2025).
**License**: MIT.

The following files are unmodified or lightly modified from BATCLIP's
`classification/` directory and retain their original copyright notices:

- `conf.py`
- `test_time.py`
- `methods/base.py`
- `methods/source.py`, `methods/tent.py`, `methods/rotta.py`,
  `methods/rpl.py`, `methods/sar.py`, `methods/tpt.py`, `methods/vte.py`
- `methods/ours.py` (the BATCLIP method itself; "ours" is the upstream name,
  retained verbatim for reproducibility — this is **not** the proposed CAMA method)
- `models/`, `datasets/`, `augmentations/`, `utils/`

Modifications relative to upstream:
- Methods unrelated to the paper baselines have been removed.
- Per-dataset YAML configs have been pruned to the eight methods used.
- Architecture is forced to `ViT-B-16` across all configs for consistency
  with the paper protocol.

---

## 2. RobustBench
**Upstream**: <https://github.com/RobustBench/robustbench>.
**License**: MIT.

A subset of `robustbench/` is vendored under the directory of the same name,
used only for model loading utilities and architecture enumeration.
Files retain their original headers where present.

---

## 3. open_clip
Imported from PyPI (`open_clip_torch==2.20.0`). Not vendored; see
`requirements.txt`. Original license: MIT.

The 2.20.0 pin is required because later releases changed the default
activation from QuickGELU to GELU, which silently degrades the OpenAI
ViT-B/16 zero-shot baseline by approximately 3pp.

---

## 4. yacs, iopath, timm, webdataset, packaging, numpy, Pillow
Standard PyPI dependencies, see `requirements.txt`. All MIT or
BSD-licensed.

---

## 5. Newly authored
- `main_cama.py`
- `status_writer.py`
- `scripts/run_cama_*.sh`, `scripts/run_baselines.sh`
- `README.md`, `THIRD_PARTY_LICENSES.md`, `LICENSE` (MIT)

These files are released under the MIT License (`LICENSE`).
