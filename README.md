# CAMA: Corruption-Aware Marginal Alignment for Test-Time Adaptation of CLIP

Anonymous code submission accompanying the paper *CAMA: Corruption-Aware Marginal Alignment for Test-Time Adaptation of CLIP*. This repository contains the reference implementation of CAMA and the baselines reported in the main table on CIFAR-10-C, CIFAR-100-C, and ImageNet-C.

> **Note.** Author identifiers, internal experiment IDs, and project-specific paths have been stripped for double-blind review. All numbers below come from the camera-ready hyperparameters; reproducing them requires the same datasets and CLIP weights described in the *Setup* section.

---

## 1. Overview

CAMA is a test-time adaptation method for CLIP under corruption. It runs in two phases per corruption:

1. **Calibration (one pass over `N_cal` batches).** From pooled logits, harmonic-rank weighting yields a corruption reference distribution `pi`. Per-batch gradient norms of the entropy and KL components are measured to set a per-corruption regularization strength `lambda`. An effective-dimension shrinkage attenuates `lambda` toward 1 in proportion to finite-sample noise. The escort target `p_dag = pi^(lambda/(lambda-1))` is then derived in closed form.

2. **Online adaptation.** For every test batch, the loss
```
L = - I_batch(p_bar)  +  (lambda - 1) * KL(p_bar || p_dag)
I_batch = H(p_bar) - mean_i H(p_i)
```
is minimized by a single AdamW/Adam step that updates only the LayerNorm affine parameters of both encoders. `pi`, `lambda`, and `p_dag` are held fixed within a corruption; only the batch marginal `p_bar` varies.

The full mathematical statement, the Gibbs-Renyi decomposition, and the escort-robustness proposition are in the paper.

---

## 2. Repository structure

```
.
├── README.md                  # this file
├── LICENSE                    # MIT
├── requirements.txt           # Python dependencies
├── main_cama.py               # CAMA runner (calibration + adaptation)
├── test_time.py               # baseline runner (TENT, RoTTA, RPL, SAR, TPT, VTE, BATCLIP, ZS)
├── conf.py                    # YACS-based config loader
├── status_writer.py           # optional status-file writer used by main_cama.py
├── methods/                   # baseline TTA methods
│   ├── __init__.py            #   registry imports
│   ├── base.py                #   shared TTAMethod base class
│   ├── source.py              #   zero-shot baseline (ZS in paper)
│   ├── tent.py                #   Wang et al., 2020
│   ├── rotta.py               #   Yuan et al., 2023
│   ├── rpl.py                 #   Rusak et al., 2022
│   ├── sar.py                 #   Niu et al., 2023
│   ├── tpt.py                 #   Shu et al., 2022
│   ├── vte.py                 #   Dobler et al., 2024
│   └── ours.py                #   BATCLIP baseline (Maharana et al., 2025)
├── models/                    # CLIP loader, custom prompt-tuning wrapper
├── datasets/                  # CIFAR-10-C / CIFAR-100-C / ImageNet-C loaders
├── augmentations/             # transforms used by RoTTA / TPT
├── utils/                     # registry, losses, eval helpers
├── robustbench/               # vendored subset for model loading
├── cfgs/                      # YAML configs per dataset and per method
│   ├── cifar10_c/             #   ours.yaml, tent.yaml, rotta.yaml, ...
│   ├── cifar100_c/
│   └── imagenet_c/
└── scripts/                   # reproducible launch scripts
    ├── run_cama_cifar10c.sh
    ├── run_cama_cifar100c.sh
    ├── run_cama_imagenet_c.sh
    └── run_baselines.sh
```

---

## 3. Setup

### 3.1 Environment

Tested with Python 3.10, PyTorch 2.1, CUDA 12.1.

```bash
conda create -n cama python=3.10 -y
conda activate cama
pip install -r requirements.txt
```

**Reproducibility-critical: `open_clip_torch` must be pinned to `2.20.0`.** OpenAI's ViT-B/16 checkpoint uses QuickGELU activations; `open_clip_torch` releases newer than 2.20.0 default to GELU and silently load the same weights against the wrong activation, dropping zero-shot CIFAR-10-C accuracy by roughly 3pp and shifting every adapted number downstream. The pin is already in `requirements.txt`; verify before running anything:

```bash
python -c "import open_clip; print(open_clip.__version__)"   # must print 2.20.0
```

### 3.2 Data

Place the corruption datasets under `./data/`. The expected layout is:

```
data/
├── cifar-10-c/
│   ├── gaussian_noise.npy
│   ├── shot_noise.npy
│   ├── ... (15 corruption .npy files + labels.npy)
├── cifar-100-c/
│   └── (same 15 corruptions + labels.npy)
└── imagenet-c/
    ├── gaussian_noise/
    │   └── 5/                 # severity 5 only is needed for the main table
    │       ├── n01440764/
    │       └── ...
    └── ... (14 more corruptions)
```

Download links:

- **CIFAR-10-C** and **CIFAR-100-C**: <https://zenodo.org/records/2535967> and <https://zenodo.org/records/3555552>.
- **ImageNet-C**: <https://zenodo.org/records/2235448> (only severity 5 is required for the headline numbers; the full archive is ~70 GB).

CLIP weights (ViT-B/16, OpenAI release) are fetched automatically by `open_clip` on first run; set `HF_HOME` or `TORCH_HOME` if a custom cache directory is desired.

### 3.3 Quick smoke test

```bash
python main_cama.py \
    --dataset cifar10_c --phase main \
    --corruptions gaussian_noise \
    --output-dir outputs/smoke \
    --cfg cfgs/cifar10_c/ours.yaml DATA_DIR ./data
```

Expected runtime on a single RTX 3090: under one minute. Final online accuracy should be approximately 67% (single-corruption zero-shot baseline ≈ 38%).

---

## 4. Reproducing the main table

The reported numbers in the paper come from a single fixed hyperparameter pair `(zeta, kappa) = (0.05, 0.1)` applied uniformly across all three benchmarks. Per-dataset settings (batch size, learning rate, weight decay, optimizer, calibration batches) live in `cfgs/<dataset>/ours.yaml` and the `_CFGS` table in `main_cama.py`; CLI flags override either layer.

> **Note on YAML usage.** When invoked through `main_cama.py`, the YAML config supplies only `CORRUPTION.DATASET`, `CORRUPTION.TYPE`, and `DATA_DIR`; per-method optimizer / learning-rate / batch-size defaults live in the `_CFGS` table inside `main_cama.py` (overridable via CLI flags such as `--lr`, `--wd`, `--batch-size`). The same YAML, when invoked through `test_time.py`, drives the full BATCLIP-style pipeline including `MODEL.ADAPTATION` registry dispatch.

### 4.1 CAMA

```bash
# CIFAR-10-C  (B=200, lr=1e-3, wd=0.01, AdamW, N_cal=3) → 79.49 mean over 15 corruptions
bash scripts/run_cama_cifar10c.sh

# CIFAR-100-C (B=200, lr=5e-4, wd=0.0,  Adam,  N_cal=3) → 48.55
bash scripts/run_cama_cifar100c.sh

# ImageNet-C  (B=64,  lr=5e-4, wd=0.01, AdamW, N_cal=16) → 36.26
bash scripts/run_cama_imagenet_c.sh
```

Each script writes per-corruption JSON / CSV under `outputs/<dataset>/` and a final aggregated table with mean online and offline accuracy.

### 4.2 Baselines (TENT, RoTTA, RPL, SAR, TPT, VTE, BATCLIP, ZS)

```bash
# Run all in-tree baselines on CIFAR-10-C
bash scripts/run_baselines.sh cifar10_c

# Or invoke a single method:
python test_time.py --cfg cfgs/cifar10_c/tent.yaml DATA_DIR ./data SAVE_DIR outputs/tent_c10
python test_time.py --cfg cfgs/cifar10_c/source.yaml DATA_DIR ./data SAVE_DIR outputs/zs_c10
```

`source.yaml` corresponds to the **ZS** (zero-shot CLIP) row in the paper.

The two baselines **WATT-P\*** and **WATT-S\*** were run from the WATT authors' original repository following BATCLIP's reproduction protocol: <https://github.com/Mehrdad-Noori/WATT>. Likewise, **StatA** was evaluated with the original code from <https://github.com/MaxZanella/StatA>. Both are reported with their published learning rate and prompt-template ensemble settings, not with this repository's defaults.

### 4.3 Output schema

For CAMA, every run writes:

| File | Contents |
|---|---|
| `outputs/<exp>/<corruption>.json` | per-corruption summary: online_acc, offline_acc, lambda_0, lambda_eff, d_eff, pi, ... |
| `outputs/<exp>/main_table/<dataset>_results.csv` | one row per corruption with the same fields, used for aggregation |
| `outputs/<exp>/analysis/lambda_table.csv` | per-corruption lambda statistics referenced in the appendix |
| `outputs/<exp>/log.txt` | full training log |

Aggregate across corruptions with:

```bash
python -c "import pandas as pd; df = pd.read_csv('outputs/cama_c10/main_table/cifar10_c_results.csv'); print(df['online_acc'].mean())"
```

---

## 5. Hyperparameters

| | CIFAR-10-C | CIFAR-100-C | ImageNet-C |
|---|:-:|:-:|:-:|
| Optimizer | AdamW | Adam | AdamW |
| Learning rate | 1e-3 | 5e-4 | 5e-4 |
| Weight decay | 0.01 | 0.0 | 0.01 |
| Batch size `B` | 200 | 200 | 64 |
| `N_cal` | 3 | 3 | 16 |
| `zeta` | 0.05 | 0.05 | 0.05 |
| `kappa` | 0.1 | 0.1 | 0.1 |
| Backbone | ViT-B/16 (OpenAI) | ViT-B/16 (OpenAI) | ViT-B/16 (OpenAI) |
| Adapted parameters | LayerNorm affine in both encoders | (same) | (same) |
| Adaptation | episodic per corruption, 1 step per batch | (same) | (same) |
| Mixed precision | autocast + scaler init=1000 | (same) | (same) |

---

## 6. Repository hygiene

- All paths are repository-relative; the only required absolute paths are `DATA_DIR` (CLI override or YAML field) and the output directory.
- Random seeds: data loaders use `seed=1` with `shuffle=False`; PyTorch / NumPy / Python seeds are set in `conf.py:set_random_seed`.
- The repository ships **no datasets and no checkpoints**; everything is downloaded on demand.
- No telemetry, network calls beyond model-weight download, or external services.

---

## 7. License

Released under the MIT License. See `LICENSE`.

The vendored `robustbench/` subdirectory and the framework files derived from BATCLIP retain their original MIT-compatible licenses.
