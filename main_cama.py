#!/usr/bin/env python3
"""
main_cama.py — CAMA test-time adaptation runner.

Phase 1 (calibration, once per corruption):
  pi      : harmonic-rank reference from N_cal pooled batches
  lambda  : 1 + (lambda_0 - 1) * B / (B + d_eff),
            lambda_0 = mean ||g_E|| / mean ||g_K|| over N_cal batches
            d_eff    = sum_k Var(p_k) / pi_k  (clamped to K-1)
  p_dag   : pi^(lambda/(lambda-1)), normalized

Phase 2 (online adaptation):
  loss    : -I_batch + (lambda - 1) * KL(p_bar || p_dag)
  update  : LayerNorm affine only (both image + text encoders)

Usage examples
--------------
# CIFAR-10-C main run
python main_cama.py --dataset cifar10_c \\
    --output-dir outputs/cama_c10 \\
    --cfg cfgs/cifar10_c/ours.yaml DATA_DIR ./data

# CIFAR-100-C with weight decay 0
python main_cama.py --dataset cifar100_c --wd 0.0 \\
    --output-dir outputs/cama_c100 \\
    --cfg cfgs/cifar100_c/ours.yaml DATA_DIR ./data

# ImageNet-C subset smoke test
python main_cama.py --dataset imagenet_c \\
    --corruptions gaussian_noise,defocus_blur \\
    --output-dir outputs/cama_inc_smoke \\
    --cfg cfgs/imagenet_c/ours.yaml DATA_DIR ./data

Run from the repository root so that `cfgs/*.yaml` and `./data` resolve.
"""

import copy
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

try:
    import scipy.stats
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


# ── CLI parsing (script-specific flags peeled off before YACS sees argv) ──────
def _pop_arg(argv, flag, default=None, cast=None):
    i = 0
    while i < len(argv):
        if argv[i] == flag and i + 1 < len(argv):
            val = argv.pop(i + 1)
            argv.pop(i)
            return cast(val) if cast else val
        i += 1
    return default


def _pop_flag(argv, flag):
    if flag in argv:
        argv.remove(flag)
        return True
    return False


# Required
DATASET     = _pop_arg(sys.argv, "--dataset")
OUTPUT_DIR  = _pop_arg(sys.argv, "--output-dir")

# Optional
SEED        = _pop_arg(sys.argv, "--seed",        default=1,    cast=int)
CORR_OVERRIDE = _pop_arg(sys.argv, "--corruptions")

# Per-setting overrides (default = preset per dataset; None here → use preset)
_K_OVR      = _pop_arg(sys.argv, "--K",           cast=int)
_BS_OVR     = _pop_arg(sys.argv, "--bs",          cast=int)
_LR_OVR     = _pop_arg(sys.argv, "--lr",          cast=float)
_WD_OVR     = _pop_arg(sys.argv, "--wd",          cast=float)
_OPT_OVR    = _pop_arg(sys.argv, "--optimizer")              # adam | adamw
_NTOTAL_OVR = _pop_arg(sys.argv, "--n-total",     cast=int)
_NCAL_OVR   = _pop_arg(sys.argv, "--n-cal",       cast=int)
_SEV_OVR    = _pop_arg(sys.argv, "--severity",    cast=int)
_KILL_OVR   = _pop_arg(sys.argv, "--kill-thresh", cast=float)
_DIAG_OVR   = _pop_arg(sys.argv, "--diag-interval", cast=int)
_STREAM_OVR = _pop_arg(sys.argv, "--streaming")              # auto | true | false

# CAMA hyperparameters
_ALPHA_OVR  = _pop_arg(sys.argv, "--alpha",       cast=float)
_BETA_OVR   = _pop_arg(sys.argv, "--beta",        cast=float)

# Optional outputs / features
SAVE_EMB         = _pop_flag(sys.argv, "--save-embeddings")
SAVE_LN          = not _pop_flag(sys.argv, "--no-save-ln")
STEP_LOG         = _pop_flag(sys.argv, "--step-log")
DRY_RUN          = _pop_flag(sys.argv, "--dry-run")

if DATASET is None:
    raise SystemExit("ERROR: --dataset required  (cifar10_c | cifar100_c | imagenet_c)")
if OUTPUT_DIR is None:
    raise SystemExit("ERROR: --output-dir required")
_VALID_DATASETS = ("cifar10_c", "cifar100_c", "imagenet_c")
if DATASET not in _VALID_DATASETS:
    raise SystemExit(f"ERROR: unknown dataset '{DATASET}'")
if _OPT_OVR is not None and _OPT_OVR not in ("adam", "adamw"):
    raise SystemExit(f"ERROR: --optimizer must be 'adam' or 'adamw', got '{_OPT_OVR}'")
if _STREAM_OVR is not None and _STREAM_OVR not in ("auto", "true", "false"):
    raise SystemExit(f"ERROR: --streaming must be 'auto' | 'true' | 'false'")


# ── repo path setup ───────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from conf import cfg, load_cfg_from_args                      # noqa: E402
from models.model import get_model                             # noqa: E402
from datasets.data_loading import get_test_loader              # noqa: E402

try:
    from status_writer import write_status, compute_eta        # noqa: E402
except ImportError:
    def write_status(**kw): pass
    def compute_eta(*a, **kw): return "—"


# ── logging ───────────────────────────────────────────────────────────────────
class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_root = logging.getLogger()
_root.setLevel(logging.INFO)
if not _root.handlers:
    _root.addHandler(_FlushHandler(sys.stderr))
logger = logging.getLogger(__name__)


# ── per-dataset defaults ──────────────────────────────────────────────────────
_CFGS = {
    "cifar10_c":  dict(K=10,   BS=200, LR=1e-3, WD=0.01, N_TOTAL=10000, N_CAL=3,
                       SEVERITY=5, OPT="adamw", STREAMING=False,
                       KILL_THRESH=0.20, DIAG_INTERVAL=5),
    "cifar100_c": dict(K=100,  BS=200, LR=5e-4, WD=0.01, N_TOTAL=10000, N_CAL=3,
                       SEVERITY=5, OPT="adam",  STREAMING=False,
                       KILL_THRESH=0.05, DIAG_INTERVAL=5),
    "imagenet_c": dict(K=1000, BS=64,  LR=5e-4, WD=0.01, N_TOTAL=50000, N_CAL=16,
                       SEVERITY=5, OPT="adamw", STREAMING=True,
                       KILL_THRESH=0.10, DIAG_INTERVAL=50),
    # Long-tail variants — same hyperparameters as base, dataset_name controls
    # the loader path (which subsamples to long-tail distribution via env vars
    # CAMA_LT_IMBALANCE_FACTOR (default 100) and CAMA_LT_SEED (default 0)).
    # N_TOTAL set lower since long-tail subsampling reduces total count.
    "cifar10_c_lt":  dict(K=10,   BS=200, LR=1e-3, WD=0.01, N_TOTAL=2000,  N_CAL=3,
                          SEVERITY=5, OPT="adamw", STREAMING=False,
                          KILL_THRESH=0.20, DIAG_INTERVAL=5),
    "cifar100_c_lt": dict(K=100,  BS=200, LR=5e-4, WD=0.01, N_TOTAL=2000,  N_CAL=3,
                          SEVERITY=5, OPT="adam",  STREAMING=False,
                          KILL_THRESH=0.05, DIAG_INTERVAL=5),
}
_dc            = _CFGS[DATASET]
K              = _K_OVR      if _K_OVR      is not None else _dc["K"]
BS             = _BS_OVR     if _BS_OVR     is not None else _dc["BS"]
LR             = _LR_OVR     if _LR_OVR     is not None else _dc["LR"]
WD             = _WD_OVR     if _WD_OVR     is not None else _dc["WD"]
N_TOTAL        = _NTOTAL_OVR if _NTOTAL_OVR is not None else _dc["N_TOTAL"]
N_CAL          = _NCAL_OVR   if _NCAL_OVR   is not None else _dc["N_CAL"]
SEVERITY       = _SEV_OVR    if _SEV_OVR    is not None else _dc["SEVERITY"]
OPT_TYPE       = _OPT_OVR    if _OPT_OVR    is not None else _dc["OPT"]
KILL_THRESH    = _KILL_OVR   if _KILL_OVR   is not None else _dc["KILL_THRESH"]
DIAG_INTERVAL  = _DIAG_OVR   if _DIAG_OVR   is not None else _dc["DIAG_INTERVAL"]
if _STREAM_OVR in (None, "auto"):
    STREAMING = _dc["STREAMING"]
else:
    STREAMING = (_STREAM_OVR == "true")

ALPHA       = _ALPHA_OVR if _ALPHA_OVR is not None else 0.1
BETA        = _BETA_OVR  if _BETA_OVR  is not None else 0.3
PRIOR_MAIN  = "harmonic"

_ALL_CORRUPTIONS_DEFAULT = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]
CORRUPTION_FAMILY = {
    "gaussian_noise": "Noise",   "shot_noise": "Noise",     "impulse_noise": "Noise",
    "defocus_blur":   "Blur",    "glass_blur": "Blur",      "motion_blur": "Blur",    "zoom_blur": "Blur",
    "snow":           "Weather", "frost":      "Weather",   "fog":         "Weather", "brightness": "Weather",
    "contrast":       "Digital", "elastic_transform": "Digital", "pixelate": "Digital", "jpeg_compression": "Digital",
}
ALL_CORRUPTIONS = (
    [c.strip() for c in CORR_OVERRIDE.split(",") if c.strip()]
    if CORR_OVERRIDE else _ALL_CORRUPTIONS_DEFAULT
)


# ── prior / scaling helpers ───────────────────────────────────────────────────
def harmonic_simplex(logits, alpha=ALPHA, beta=BETA):
    """π_k ∝ (s_k + α)^β, s_k = mean_i 1/rank(logit_ik) normalized row-wise."""
    ranks   = logits.detach().argsort(dim=1, descending=True).argsort(dim=1).float() + 1
    weights = 1.0 / ranks
    weights = weights / weights.sum(dim=1, keepdim=True)
    s  = weights.mean(dim=0)
    pi = (s + alpha).pow(beta)
    return (pi / pi.sum()).detach()


def p_dag(pi, lam):
    """p† = π^(λ/(λ-1)) normalized (log-space, max-shift for stability)."""
    alpha    = lam / (lam - 1.0)
    log_pdag = alpha * (pi + 1e-30).log()
    log_pdag = log_pdag - log_pdag.max()
    pdag     = log_pdag.exp()
    return (pdag / pdag.sum()).detach()


def configure_model(model):
    """LayerNorm-only adaptation."""
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.requires_grad_(True)


def make_optimizer(model):
    params = [p for p in model.parameters() if p.requires_grad]
    if OPT_TYPE == "adam":
        return torch.optim.Adam(params, lr=LR, betas=(0.9, 0.999), weight_decay=WD)
    return torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.999), weight_decay=WD)


def _collect_grad_vector(model):
    """Flatten all LN grads into a single 1-D vector."""
    parts = []
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            for p in m.parameters():
                if p.grad is not None:
                    parts.append(p.grad.data.flatten().clone())
    return torch.cat(parts) if parts else torch.zeros(1)


# ── data helpers ──────────────────────────────────────────────────────────────
def _make_loader(corruption, preprocess, n_samples):
    return get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name=DATASET,
        preprocess=preprocess, data_root_dir=cfg.DATA_DIR,
        domain_name=corruption, domain_names_all=ALL_CORRUPTIONS,
        severity=SEVERITY, num_examples=n_samples, rng_seed=SEED,
        use_clip=cfg.MODEL.USE_CLIP, n_views=1, delta_dirichlet=0.0,
        batch_size=BS, shuffle=False, workers=4,
    )


def load_tensor(corruption, preprocess, n=None):
    """Preload corrupted data into CPU RAM (CIFAR only)."""
    if n is None:
        n = N_TOTAL
    loader = _make_loader(corruption, preprocess, n)
    imgs, labels = [], []
    for batch in loader:
        imgs.append(batch[0])
        labels.append(batch[1])
    return torch.cat(imgs)[:n], torch.cat(labels)[:n]


def load_clean_tensor(preprocess, n=None):
    """Preload clean test set (CIFAR only, used for cos_clean in analysis)."""
    if n is None:
        n = N_TOTAL
    loader = get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name=DATASET,
        preprocess=preprocess, data_root_dir=cfg.DATA_DIR,
        domain_name="none", domain_names_all=ALL_CORRUPTIONS,
        severity=SEVERITY, num_examples=n, rng_seed=SEED,
        use_clip=cfg.MODEL.USE_CLIP, n_views=1, delta_dirichlet=0.0,
        batch_size=BS, shuffle=False, workers=4,
    )
    imgs, labels = [], []
    for batch in loader:
        imgs.append(batch[0])
        labels.append(batch[1])
    return torch.cat(imgs)[:n], torch.cat(labels)[:n]


# ── eval / metric helpers ─────────────────────────────────────────────────────
def pairwise_cosine_mean(feats, n_sub=2500):
    N = feats.shape[0]
    if N > n_sub:
        idx   = torch.randperm(N)[:n_sub]
        feats = feats[idx]
    sim  = feats @ feats.T
    mask = ~torch.eye(feats.shape[0], dtype=torch.bool)
    return float(sim[mask].mean().item())


def compute_ece(confidences, accuracies, n_bins=10):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            avg_conf = float(confidences[mask].mean())
            avg_acc  = float(accuracies[mask].mean())
            ece += float(mask.float().mean()) * abs(avg_conf - avg_acc)
    return ece


def offline_eval_tensor(model, imgs_all, labels_all, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(imgs_all), BS):
            logits   = model(imgs_all[i:i+BS].to(device), return_features=True)[0]
            labels_b = labels_all[i:i+BS].to(device)
            correct += (logits.argmax(1) == labels_b).sum().item()
            total   += len(labels_b)
    model.train()
    return correct / total


def offline_eval_detailed(model, imgs_all, labels_all, device):
    """offline_acc + overconf_wrong + ECE."""
    model.eval()
    all_probs  = []
    all_labels_list = []
    with torch.no_grad():
        for i in range(0, len(imgs_all), BS):
            logits = model(imgs_all[i:i+BS].to(device), return_features=True)[0]
            all_probs.append(F.softmax(logits, dim=1).float().cpu())
            all_labels_list.append(labels_all[i:i+BS])
    model.train()

    all_probs  = torch.cat(all_probs)
    all_labels = torch.cat(all_labels_list)
    preds      = all_probs.argmax(1)
    correct    = (preds == all_labels).float()
    max_conf   = all_probs.max(1).values
    wrong      = preds != all_labels
    return (
        float(correct.mean()),
        float(((max_conf > 0.9) & wrong).float().mean()),
        compute_ece(max_conf, correct),
    )


def offline_eval_streaming(model, corruption, preprocess, device):
    loader = _make_loader(corruption, preprocess, N_TOTAL)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            logits   = model(batch[0].to(device), return_features=True)[0]
            labels_b = batch[1].to(device)
            correct += (logits.argmax(1) == labels_b).sum().item()
            total   += len(labels_b)
    del loader
    torch.cuda.empty_cache()
    model.train()
    return correct / total


def collect_features_tensor(model, imgs_all, device):
    """Forward all images; return (logits, L2-normed feats) on CPU."""
    model.eval()
    all_logits = []
    all_feats  = []
    with torch.no_grad():
        for i in range(0, len(imgs_all), BS):
            out = model(imgs_all[i:i+BS].to(device), return_features=True)
            all_logits.append(out[0].float().cpu())
            all_feats.append(out[1].float().cpu())
    model.train()
    return torch.cat(all_logits), torch.cat(all_feats)


# ── calibration (Phase 1) ─────────────────────────────────────────────────────
def compute_pi_kl_diagnostics(pi, p_bar_c=None, true_label_dist=None):
    """KL diagnostics for the calibrated prior π. All on CPU/CUDA, returns floats.

    Returned keys:
      - kl_pi_uniform : KL(π̂ ‖ uniform). 0 means π̂ collapsed to uniform.
      - kl_pi_pbar_cal: KL(π̂ ‖ p̄_cal). How far our prior is from the model's
                       own predicted marginal during calibration.
      - kl_pi_true   : KL(π̂ ‖ π_true). Only if true_label_dist provided
                       (long-tail experiments).
    """
    K_local = pi.shape[0]
    pi_uniform = torch.ones_like(pi) / K_local
    out = {
        "kl_pi_uniform": float((pi * ((pi + 1e-30).log() - (pi_uniform + 1e-30).log())).sum()),
    }
    if p_bar_c is not None:
        p = p_bar_c.to(pi.device)
        out["kl_pi_pbar_cal"] = float((pi * ((pi + 1e-30).log() - (p + 1e-30).log())).sum())
    if true_label_dist is not None:
        t = true_label_dist.to(pi.device)
        out["kl_pi_true"] = float((pi * ((pi + 1e-30).log() - (t + 1e-30).log())).sum())
        out["kl_true_pi"] = float((t * ((t + 1e-30).log() - (pi + 1e-30).log())).sum())
        out["tv_pi_true"] = float(0.5 * (pi - t).abs().sum())
        pi_np = pi.detach().float().cpu().numpy()
        t_np = t.detach().float().cpu().numpy()
        if _HAVE_SCIPY and float(np.std(pi_np)) > 0.0 and float(np.std(t_np)) > 0.0:
            out["prior_true_spearman"] = float(scipy.stats.spearmanr(pi_np, t_np).statistic)
        else:
            out["prior_true_spearman"] = float("nan")
        head_idx = int(torch.argmax(t).item())
        tail_idx = int(torch.argmin(t).item())
        pi_head_tail = float((pi[head_idx] + 1e-30) / (pi[tail_idx] + 1e-30))
        true_head_tail = float((t[head_idx] + 1e-30) / (t[tail_idx] + 1e-30))
        out["prior_head_class"] = head_idx
        out["prior_tail_class"] = tail_idx
        out["pi_head_tail_ratio"] = pi_head_tail
        out["true_head_tail_ratio"] = true_head_tail
        out["head_tail_log_ratio_error"] = float(
            abs(np.log(pi_head_tail + 1e-30) - np.log(true_head_tail + 1e-30))
        )
    return out


def empirical_label_distribution(labels, K_total):
    """Compute label histogram normalized to a probability vector."""
    if isinstance(labels, list):
        labels = torch.tensor(labels)
    counts = torch.bincount(labels.long().cpu(), minlength=K_total).float()
    return counts / counts.sum().clamp(min=1.0)


def label_distribution_summary(labels, K_total):
    """Return normalized label prior plus a compact long-tail composition record."""
    if isinstance(labels, list):
        labels = torch.tensor(labels)
    counts = torch.bincount(labels.long().cpu(), minlength=K_total).long()
    total = int(counts.sum().item())
    positive = counts[counts > 0]
    min_count = int(positive.min().item()) if len(positive) else 0
    max_count = int(counts.max().item()) if total else 0
    head_idx = int(torch.argmax(counts).item()) if total else None
    tail_idx = int(torch.argmin(counts).item()) if total else None
    head_count = int(counts[head_idx].item()) if head_idx is not None else 0
    tail_count = int(counts[tail_idx].item()) if tail_idx is not None else 0
    dist = counts.float() / max(total, 1)
    summary = {
        "lt_total": total,
        "lt_nonzero_classes": int((counts > 0).sum().item()),
        "lt_head_class": head_idx,
        "lt_tail_class": tail_idx,
        "lt_head_count": head_count,
        "lt_tail_count": tail_count,
        "lt_min_count": min_count,
        "lt_max_count": max_count,
        "lt_empirical_if": round(max_count / max(min_count, 1), 5) if min_count else None,
        "lt_head_tail_ratio": round(head_count / max(tail_count, 1), 5) if tail_count else None,
        "lt_label_counts": ",".join(str(int(x)) for x in counts.tolist()),
    }
    return dist, summary


def write_long_tail_distribution(path, labels, K_total, extra=None):
    true_dist, summary = label_distribution_summary(labels, K_total)
    payload = dict(summary)
    payload["true_label_dist"] = [round(float(x), 8) for x in true_dist.tolist()]
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return true_dist, summary


def calibrate(model, cal_batches, device, prior_type="harmonic"):
    """
    π        = harmonic_simplex on N_cal·B pooled logits
    λ₀       = mean(‖g_E^(b)‖) / mean(‖g_K^(b)‖),  g_K uses batch-local p̄_b
    d_eff    = Σ_k var(p_k) / π_k  over pooled cal predictions (clamp K−1)
    λ_eff    = 1 + (λ₀ − 1) · B / (B + d_eff)

    Returns (pi, lambda_0, lambda_eff, d_eff, I_batch_cal, b_hat,
             corrupt_feats, g_E_norms, g_K_norms).
    """
    n_cal = len(cal_batches)

    # 1. pooled forward (no grad) → logits / feats / pi
    all_logits = []
    all_feats  = []
    model.eval()
    with torch.no_grad():
        for imgs_b in cal_batches:
            out = model(imgs_b.to(device), return_features=True)
            all_logits.append(out[0].float())
            all_feats.append(out[1].float().cpu())
    model.train()

    logits_cat = torch.cat(all_logits, dim=0)          # (n_cal·B, K) GPU
    b_hat      = logits_cat.mean(0).cpu().float()

    if prior_type == "harmonic":
        pi = harmonic_simplex(logits_cat)
    elif prior_type == "uniform":
        pi = torch.ones(K, device=device) / K
    elif prior_type == "softmax":
        pi = F.softmax(logits_cat.mean(0), dim=0).detach()
    else:
        raise ValueError(f"unknown prior_type: {prior_type}")

    q_all    = F.softmax(logits_cat, dim=1)
    p_bar_c  = q_all.mean(0).detach()
    H_pbar_c = float(-(p_bar_c * (p_bar_c + 1e-8).log()).sum())
    mean_H_c = float(-(q_all * (q_all + 1e-8).log()).sum(1).mean())
    I_batch_cal = H_pbar_c - mean_H_c

    cal_probs = q_all.float().cpu()                    # (n_cal·B, K) CPU

    corrupt_feats = torch.cat(all_feats, dim=0).cpu()
    del all_logits, logits_cat, q_all, all_feats
    torch.cuda.empty_cache()

    # 2. λ₀: gradient-norm ratio with batch-local marginal in KL term
    g_E_norms = []
    g_K_norms = []
    for imgs_b in cal_batches:
        # g_E^(b)
        model.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(imgs_b.to(device), return_features=True)[0]
            q      = F.softmax(logits, dim=1)
            l_ent  = -(q * (q + 1e-8).log()).sum(1).mean()
        l_ent.backward()
        g_E_norms.append(float(_collect_grad_vector(model).norm().item()))

        # g_K^(b) — direct KL(p̄_b ‖ π), batch-local p̄ (B samples)
        model.zero_grad()
        with torch.cuda.amp.autocast():
            logits  = model(imgs_b.to(device), return_features=True)[0]
            q       = F.softmax(logits, dim=1)
            p_bar_b = q.mean(0)
            kl      = (p_bar_b * ((p_bar_b + 1e-8).log() - (pi + 1e-8).log())).sum()
        kl.backward()
        g_K_norms.append(float(_collect_grad_vector(model).norm().item()))
    model.zero_grad()

    mean_gE  = sum(g_E_norms) / n_cal
    mean_gK  = sum(g_K_norms) / n_cal
    lambda_0 = mean_gE / (mean_gK + 1e-30)

    # 3. d_eff + λ_eff
    Sigma_diag = cal_probs.var(dim=0)                  # (K,) CPU
    d_eff_val  = float((Sigma_diag / pi.cpu()).sum().item())
    d_eff_val  = min(d_eff_val, K - 1)
    lambda_eff = 1.0 + (lambda_0 - 1.0) * (BS / (BS + d_eff_val))
    del cal_probs, Sigma_diag

    return (pi, lambda_0, lambda_eff, d_eff_val, I_batch_cal,
            b_hat, corrupt_feats, g_E_norms, g_K_norms)


# ── adaptation (Phase 2) ──────────────────────────────────────────────────────
def adapt_one(
    model, optimizer, scaler,
    data_iter, n_steps,
    pi, lambda_eff,
    device,
    corruption, corr_idx, corr_total,
    step_log=False,
    imgs_all=None, labels_all=None,   # for step-wise offline (CIFAR)
    preprocess=None,                  # for streaming offline (ImageNet)
):
    """CAMA Loss B per-batch update.  Episodic reset handled by caller."""
    kill_step   = n_steps // 2
    cum_corr    = cum_seen = 0
    pred_counts = torch.zeros(K, dtype=torch.long)
    H_pbar_last = 0.0
    killed      = False
    t0          = time.time()
    trajectory  = []
    _step_wall  = []

    for step, batch in enumerate(data_iter):
        imgs_b   = batch[0].to(device)
        labels_b = batch[1].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out     = model(imgs_b, return_features=True)
            logits  = out[0]
            feats_b = out[1]
            q       = F.softmax(logits, dim=1)
            mean_H  = -(q * (q + 1e-8).log()).sum(1).mean()
            p_bar   = q.mean(0)
            H_pbar  = -(p_bar * (p_bar + 1e-8).log()).sum()
            I_batch = H_pbar - mean_H
            pdag_b  = p_dag(pi, lambda_eff)
            kl_dag  = (p_bar * ((p_bar + 1e-8).log() - (pdag_b + 1e-8).log())).sum()
            loss    = -I_batch + (lambda_eff - 1.0) * kl_dag

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds        = logits.argmax(1)
            cum_corr    += (preds == labels_b).sum().item()
            cum_seen    += len(labels_b)
            pred_counts += preds.cpu().bincount(minlength=K)
            H_pbar_last  = float(H_pbar.detach())

        online_acc = cum_corr / cum_seen
        cat_pct    = pred_counts.max().item() / cum_seen
        _now       = time.time()
        elapsed    = _now - t0
        _step_wall.append(_now)
        if len(_step_wall) > 6:
            _step_wall.pop(0)
        if len(_step_wall) >= 2:
            s_ps = (_step_wall[-1] - _step_wall[0]) / (len(_step_wall) - 1)
        else:
            s_ps = elapsed

        if step_log:
            pcos  = pairwise_cosine_mean(feats_b.detach().float().cpu(),
                                         n_sub=min(BS, 200))
            I_val = float(I_batch.detach())
            row = {
                "step":        step,
                "online_acc":  round(online_acc, 5),
                "H_pbar":      round(H_pbar_last, 5),
                "I_batch":     round(I_val, 5),
                "pairwise_cos": round(pcos, 5),
            }
            if step % 10 == 0 or step == n_steps - 1:
                if imgs_all is not None:
                    off_acc, oc_wrong, ece = offline_eval_detailed(
                        model, imgs_all, labels_all, device)
                else:
                    off_acc = offline_eval_streaming(model, corruption, preprocess, device)
                    oc_wrong = ece = None
                row["offline_acc"]    = round(off_acc, 5)
                row["overconf_wrong"] = round(oc_wrong, 5) if oc_wrong is not None else None
                row["ece"]            = round(ece, 5) if ece is not None else None
            trajectory.append(row)

        if (step + 1) % DIAG_INTERVAL == 0 or (step + 1) == n_steps:
            logger.info(
                f"  [{corr_idx+1}/{corr_total}] {corruption:22s} "
                f"step={step+1:>4}/{n_steps} "
                f"online={online_acc:.4f} cat%={cat_pct:.3f} H(p̄)={H_pbar_last:.3f}"
            )
            write_status(
                script=os.path.basename(__file__),
                phase=1, phase_total=1,
                corruption=corruption, corr_idx=corr_idx, corr_total=corr_total,
                step=step + 1, n_steps=n_steps,
                online_acc=online_acc, s_per_step=s_ps,
                eta=compute_eta(step + 1, n_steps, corr_idx, corr_total, s_ps),
                cat_pct=cat_pct, lambda_val=lambda_eff,
            )

        if (step + 1) == kill_step and online_acc < KILL_THRESH:
            logger.info(f"  KILL step={step+1}: online={online_acc:.4f} < {KILL_THRESH}")
            killed = True
            break

    return {
        "online_acc": round(cum_corr / cum_seen, 5),
        "cat_pct":    round(cat_pct, 5),
        "H_pbar":     round(H_pbar_last, 5),
        "killed":     killed,
        "elapsed_s":  round(time.time() - t0, 1),
    }, trajectory


# ── analysis metrics (CIFAR only: requires full-tensor adapted eval) ──────────
def collect_analysis_metrics(
    model, corruption, pi,
    lambda_0, lambda_eff, d_eff_val,
    corrupt_feats_cal,
    imgs_corrupt, labels_corrupt,
    cos_clean,
    device,
):
    adapted_logits, adapted_feats = collect_features_tensor(model, imgs_corrupt, device)

    p_bar_adapted = F.softmax(adapted_logits, dim=1).mean(0)
    pi_cpu        = pi.cpu().float()

    if _HAVE_SCIPY:
        sr = float(scipy.stats.spearmanr(pi_cpu.numpy(), p_bar_adapted.numpy()).statistic)
        pr = float(scipy.stats.pearsonr(pi_cpu.numpy(), p_bar_adapted.numpy()).statistic)
    else:
        sr = pr = float("nan")

    cos_corrupt = pairwise_cosine_mean(corrupt_feats_cal.cpu().float())
    cos_adapted = pairwise_cosine_mean(adapted_feats.cpu().float())
    cone_opened = round(cos_corrupt - cos_adapted, 5)

    all_p     = F.softmax(adapted_logits, dim=1)
    max_probs = all_p.max(1).values
    u_gap     = float((1.0 - max_probs).mean())
    mean_ent  = float(-(all_p * (all_p + 1e-8).log()).sum(1).mean())
    cat_pct   = float(p_bar_adapted.max())

    return {
        "corruption":          corruption,
        "spearman_r":          round(sr, 5),
        "pearson_r":           round(pr, 5),
        "lambda_0":            round(lambda_0, 5),
        "d_eff":               round(d_eff_val, 5),
        "lambda_eff":          round(lambda_eff, 5),
        "cos_clean":           round(cos_clean, 5) if cos_clean is not None else None,
        "cos_corrupt":         round(cos_corrupt, 5),
        "cos_adapted":         round(cos_adapted, 5),
        "cone_opened":         cone_opened,
        "u_soft_hard_gap":     round(u_gap, 5),
        "mean_entropy_adapted": round(mean_ent, 5),
        "cat_pct":             round(cat_pct, 5),
    }


def write_trajectory_csv(trajectory, path, method="CAMA"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["method", "step", "online_acc", "offline_acc",
                  "H_pbar", "I_batch", "pairwise_cos", "overconf_wrong", "ece"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in trajectory:
            w.writerow({
                "method":        method,
                "step":          row["step"],
                "online_acc":    row.get("online_acc", ""),
                "offline_acc":   row.get("offline_acc", ""),
                "H_pbar":        row.get("H_pbar", ""),
                "I_batch":       row.get("I_batch", ""),
                "pairwise_cos":  row.get("pairwise_cos", ""),
                "overconf_wrong": row.get("overconf_wrong", ""),
                "ece":           row.get("ece", ""),
            })
    logger.info(f"  Saved trajectory: {path}")


# ── TensorIter helper (CIFAR in-memory) ───────────────────────────────────────
class _TensorIter:
    def __init__(self, imgs, labels, bs):
        self.imgs, self.labels, self.bs = imgs, labels, bs
    def __iter__(self):
        for i in range(0, len(self.imgs), self.bs):
            yield self.imgs[i:i+self.bs], self.labels[i:i+self.bs]
    def __len__(self):
        return (len(self.imgs) + self.bs - 1) // self.bs


# ── Phase: main ───────────────────────────────────────────────────────────────
def run_main(model, state_init, preprocess, device, out_dir):
    out_main     = os.path.join(out_dir, "main_table", f"k{K}")
    out_analysis = os.path.join(out_dir, "analysis",   f"k{K}")
    out_fig2     = os.path.join(out_dir, "figure2",    f"k{K}")
    for d in (out_main, out_analysis, out_fig2):
        os.makedirs(d, exist_ok=True)

    results_csv_path = os.path.join(out_dir, "main_table", f"k{K}_results.csv")

    # clean baseline features (CIFAR only, for cos_clean)
    cos_clean_val = None
    imgs_clean    = None
    if not STREAMING:
        logger.info("Loading clean test set...")
        imgs_clean, _ = load_clean_tensor(preprocess)
        model.eval()
        with torch.no_grad():
            _, clean_feats_all = collect_features_tensor(model, imgs_clean, device)
        model.train()
        cos_clean_val = pairwise_cosine_mean(clean_feats_all.cpu().float())
        logger.info(f"  cos_clean = {cos_clean_val:.5f}")
        del clean_feats_all
        torch.cuda.empty_cache()

    configure_model(model)
    all_results     = []
    analysis_rows   = []
    cama_trajectory = None

    for corr_idx, corruption in enumerate(ALL_CORRUPTIONS):
        out_json = os.path.join(out_main, f"{corruption}.json")
        ana_json = os.path.join(out_analysis, f"{corruption}_analysis.json")
        if os.path.exists(out_json):
            with open(out_json) as f:
                cached_row = json.load(f)
            required_lt_diag = (
                "kl_true_pi",
                "tv_pi_true",
                "head_tail_log_ratio_error",
                "pi_head_tail_ratio",
                "true_head_tail_ratio",
                "prior_head_class",
                "prior_tail_class",
            )
            missing_lt_diag = (
                DATASET.endswith("_lt")
                and any(cached_row.get(k) in (None, "", "nan") for k in required_lt_diag)
            )
            if missing_lt_diag:
                logger.info(f"[RERUN] {corruption}: cached JSON missing long-tail prior diagnostics")
            else:
                logger.info(f"[SKIP] {corruption}: found {out_json}")
                all_results.append(cached_row)
                if os.path.exists(ana_json):
                    with open(ana_json) as f:
                        analysis_rows.append(json.load(f))
                continue

        # episodic reset
        model.load_state_dict(copy.deepcopy(state_init))
        configure_model(model)
        optimizer = make_optimizer(model)
        scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

        logger.info(f"\n{'='*60}")
        logger.info(f"[{corr_idx+1}/{len(ALL_CORRUPTIONS)}] {corruption}  K={K}")
        logger.info(f"{'='*60}")

        # load data
        if STREAMING:
            cal_loader = _make_loader(corruption, preprocess, N_CAL * BS)
            cal_batches = [b[0].cpu() for b in cal_loader]
            del cal_loader
            torch.cuda.empty_cache()
            imgs_corrupt = labels_corrupt = None
        else:
            imgs_corrupt, labels_corrupt = load_tensor(corruption, preprocess)
            cal_batches = [imgs_corrupt[i*BS:(i+1)*BS] for i in range(N_CAL)]

        # optional pre-adaptation snapshot (gaussian_noise CIFAR only)
        corrupt_feats_full = None
        corrupt_logits_full = None
        want_embed_snap = (SAVE_EMB and corruption == "gaussian_noise" and not STREAMING)
        if want_embed_snap:
            logger.info("  Collecting pre-adaptation features for embeddings.pt...")
            model.eval()
            corrupt_logits_full, corrupt_feats_full = collect_features_tensor(
                model, imgs_corrupt, device)
            model.train()

        # Phase 1: calibration
        logger.info(f"  Calibrating ({N_CAL} batches, prior={PRIOR_MAIN})...")
        pi, lam0, lam_eff, d_eff_val, I_b0, b_hat, corrupt_feats_cal, gE_norms, gK_norms = calibrate(
            model, cal_batches, device, prior_type=PRIOR_MAIN)
        logger.info(f"  λ₀={lam0:.4f}  d_eff={d_eff_val:.4f}  λ_eff={lam_eff:.4f}  I_batch={I_b0:.4f}")
        logger.info(f"  g_E_norms={[round(x,5) for x in gE_norms]}  g_K_norms={[round(x,5) for x in gK_norms]}")

        # Prior diagnostics — KL(π̂ ‖ uniform) always; KL(π̂ ‖ π_true) when
        # the test loader exposes a label distribution (long-tail variants).
        true_dist = None
        lt_summary = {}
        if DATASET.endswith("_lt"):
            try:
                _probe_loader = _make_loader(corruption, preprocess, N_TOTAL)
                _labels = [int(s[1]) for s in _probe_loader.dataset.samples]
                true_dist, lt_summary = write_long_tail_distribution(
                    os.path.join(out_analysis, f"{corruption}_label_distribution.json"),
                    torch.tensor(_labels),
                    K,
                    extra={
                        "dataset": DATASET,
                        "corruption": corruption,
                        "imbalance_factor_env": os.environ.get("CAMA_LT_IMBALANCE_FACTOR", "100"),
                        "lt_seed_env": os.environ.get("CAMA_LT_SEED", "0"),
                        "lt_class_order_env": os.environ.get("CAMA_LT_CLASS_ORDER", "index"),
                        "lt_order_seed_env": os.environ.get("CAMA_LT_ORDER_SEED", "0"),
                        "run_seed": SEED,
                    },
                )
                lt_summary.update({
                    "lt_if_env": os.environ.get("CAMA_LT_IMBALANCE_FACTOR", "100"),
                    "lt_seed": os.environ.get("CAMA_LT_SEED", "0"),
                    "lt_class_order": os.environ.get("CAMA_LT_CLASS_ORDER", "index"),
                    "lt_order_seed": os.environ.get("CAMA_LT_ORDER_SEED", "0"),
                })
                del _probe_loader
            except Exception as _e:
                logger.warning(f"  could not compute true_label_dist: {_e}")
        cal_kl = compute_pi_kl_diagnostics(pi, true_label_dist=true_dist)
        logger.info(f"  KL(π̂‖uniform)={cal_kl['kl_pi_uniform']:.5f}"
                    + (f"  KL(π̂‖π_true)={cal_kl['kl_pi_true']:.5f}" if 'kl_pi_true' in cal_kl else ""))

        del cal_batches
        torch.cuda.empty_cache()

        # Phase 2: adaptation
        do_step_log = STEP_LOG and (corruption == "gaussian_noise")
        n_steps     = N_TOTAL // BS

        if STREAMING:
            adapt_loader = _make_loader(corruption, preprocess, N_TOTAL)
        else:
            adapt_loader = _TensorIter(imgs_corrupt, labels_corrupt, BS)

        loop_result, trajectory = adapt_one(
            model, optimizer, scaler,
            adapt_loader, n_steps,
            pi, lam_eff, device,
            corruption, corr_idx, len(ALL_CORRUPTIONS),
            step_log=do_step_log,
            imgs_all=imgs_corrupt if not STREAMING else None,
            labels_all=labels_corrupt if not STREAMING else None,
            preprocess=preprocess if STREAMING else None,
        )
        del adapt_loader

        # offline eval
        if STREAMING:
            offline_acc = offline_eval_streaming(model, corruption, preprocess, device)
        else:
            offline_acc = offline_eval_tensor(model, imgs_corrupt, labels_corrupt, device)
        logger.info(f"  offline_acc={offline_acc:.5f}")

        # analysis metrics (CIFAR only — needs full-tensor adapted features)
        if not STREAMING:
            anrow = collect_analysis_metrics(
                model, corruption, pi,
                lam0, lam_eff, d_eff_val,
                corrupt_feats_cal,
                imgs_corrupt, labels_corrupt,
                cos_clean_val,
                device,
            )
            analysis_rows.append(anrow)
            ana_json = os.path.join(out_analysis, f"{corruption}_analysis.json")
            with open(ana_json, "w") as f:
                json.dump(anrow, f, indent=2)
            logger.info(
                f"  spearman={anrow['spearman_r']}  "
                f"cos_corrupt={anrow['cos_corrupt']}  cos_adapted={anrow['cos_adapted']}  "
                f"(saved {os.path.basename(ana_json)})"
            )

        # gaussian_noise extras (trajectory + embeddings snapshot)
        if corruption == "gaussian_noise":
            if do_step_log and trajectory:
                cama_trajectory = trajectory
                traj_path = os.path.join(out_fig2, "trajectory_CAMA.csv")
                write_trajectory_csv(trajectory, traj_path, method="CAMA")
            if (not STREAMING) and want_embed_snap and imgs_clean is not None:
                logger.info("  Saving embeddings.pt (clean/corrupt/adapted)...")
                adapted_logits, adapted_feats = collect_features_tensor(
                    model, imgs_corrupt, device)
                clean_logits, clean_feats = collect_features_tensor(
                    model, imgs_clean, device)
                emb_path = os.path.join(out_analysis, "gaussian_noise_embeddings.pt")
                torch.save({
                    "clean_logits":     clean_logits,
                    "clean_features":   clean_feats,
                    "corrupt_logits":   corrupt_logits_full,
                    "corrupt_features": corrupt_feats_full,
                    "adapted_logits":   adapted_logits,
                    "adapted_features": adapted_feats,
                    "labels":           labels_corrupt,
                }, emb_path)
                logger.info(f"  Saved: {emb_path}")
                del adapted_logits, adapted_feats, clean_logits, clean_feats

        # result row
        result = {
            "corruption":  corruption,
            "lambda_0":    round(lam0, 5),
            "d_eff":       round(d_eff_val, 5),
            "lambda_eff":  round(lam_eff, 5),
            "online_acc":  loop_result["online_acc"],
            "offline_acc": round(offline_acc, 5),
            "cat_pct":     loop_result["cat_pct"],
            "H_pbar":      loop_result["H_pbar"],
            "killed":      loop_result["killed"],
            "elapsed_s":   loop_result["elapsed_s"],
            "kl_pi_uniform": round(cal_kl["kl_pi_uniform"], 5),
            "kl_pi_true":  round(cal_kl["kl_pi_true"], 5) if "kl_pi_true" in cal_kl else None,
            "kl_true_pi":  round(cal_kl["kl_true_pi"], 5) if "kl_true_pi" in cal_kl else None,
            "tv_pi_true": round(cal_kl["tv_pi_true"], 5) if "tv_pi_true" in cal_kl else None,
            "prior_true_spearman": round(cal_kl["prior_true_spearman"], 5) if "prior_true_spearman" in cal_kl else None,
            "prior_head_class": cal_kl.get("prior_head_class"),
            "prior_tail_class": cal_kl.get("prior_tail_class"),
            "pi_head_tail_ratio": round(cal_kl["pi_head_tail_ratio"], 5) if "pi_head_tail_ratio" in cal_kl else None,
            "true_head_tail_ratio": round(cal_kl["true_head_tail_ratio"], 5) if "true_head_tail_ratio" in cal_kl else None,
            "head_tail_log_ratio_error": round(cal_kl["head_tail_log_ratio_error"], 5) if "head_tail_log_ratio_error" in cal_kl else None,
            "g_E_norms":   ",".join(f"{x:.5f}" for x in gE_norms),
            "g_K_norms":   ",".join(f"{x:.5f}" for x in gK_norms),
            "timestamp":   datetime.now().isoformat(),
        }
        result.update(lt_summary)
        all_results.append(result)

        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"  Saved: {out_json}")

        # LN checkpoint (optional)
        if SAVE_LN:
            ckpt_dir  = os.path.join(out_dir, "checkpoints", f"k{K}")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"{corruption}_ln.pt")
            torch.save(
                {k: v.cpu() for k, v in model.named_parameters() if v.requires_grad},
                ckpt_path,
            )
            logger.info(f"  Saved: {ckpt_path}")

        # cleanup
        if not STREAMING:
            del imgs_corrupt, labels_corrupt
        del corrupt_feats_cal, pi, corrupt_logits_full, corrupt_feats_full
        torch.cuda.empty_cache()

    # main table CSV
    fields = ["corruption", "lambda_0", "d_eff", "lambda_eff",
              "online_acc", "offline_acc", "cat_pct", "H_pbar", "killed",
              "kl_pi_uniform", "kl_pi_true", "kl_true_pi", "tv_pi_true",
              "prior_true_spearman", "prior_head_class", "prior_tail_class",
              "pi_head_tail_ratio", "true_head_tail_ratio",
              "head_tail_log_ratio_error", "lt_if_env", "lt_seed",
              "lt_class_order", "lt_order_seed", "lt_total",
              "lt_nonzero_classes", "lt_head_class", "lt_tail_class",
              "lt_head_count", "lt_tail_count", "lt_min_count",
              "lt_max_count", "lt_empirical_if", "lt_head_tail_ratio",
              "lt_label_counts", "g_E_norms", "g_K_norms", "timestamp"]
    with open(results_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)
    logger.info(f"\nSaved: {results_csv_path}")

    accs = [r["offline_acc"] for r in all_results if not r.get("killed")]
    summary_txt = os.path.join(out_main, "summary.txt")
    with open(summary_txt, "w") as f:
        f.write(f"dataset: {DATASET}  K={K}  BS={BS}  LR={LR}  WD={WD}  opt={OPT_TYPE}  N_CAL={N_CAL}\n")
        f.write(f"prior: {PRIOR_MAIN}  alpha={ALPHA}  beta={BETA}  severity={SEVERITY}\n")
        f.write(f"n_corruptions: {len(all_results)}\n")
        if accs:
            f.write(f"mean_offline_acc: {sum(accs)/len(accs):.5f}\n")
        for r in all_results:
            f.write(f"  {r['corruption']:22s} offline={r.get('offline_acc','?'):.5f}"
                    f"  λ_eff={r.get('lambda_eff','?'):.4f}\n")
    logger.info(f"Saved: {summary_txt}")
    if accs:
        logger.info(f"\nMean offline acc ({len(accs)} corruptions): {sum(accs)/len(accs):.5f}")

    # analysis CSVs (CIFAR only)
    if analysis_rows:
        eq_path = os.path.join(out_analysis, "exp4_equilibrium.csv")
        fields_eq = [
            "corruption", "spearman_r", "pearson_r", "lambda_0", "d_eff", "lambda_eff",
            "cos_clean", "cos_corrupt", "cos_adapted", "cone_opened",
            "u_soft_hard_gap", "mean_entropy_adapted", "cat_pct",
        ]
        with open(eq_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields_eq)
            w.writeheader()
            w.writerows(analysis_rows)
        logger.info(f"Saved: {eq_path}")

        lam_path = os.path.join(out_analysis, "lambda_table.csv")
        with open(lam_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["corruption", "family", "lambda_0", "d_eff", "lambda_eff"])
            w.writeheader()
            for r in all_results:
                w.writerow({
                    "corruption": r["corruption"],
                    "family":     CORRUPTION_FAMILY.get(r["corruption"], ""),
                    "lambda_0":   r.get("lambda_0", ""),
                    "d_eff":      r.get("d_eff", ""),
                    "lambda_eff": r.get("lambda_eff", ""),
                })
        logger.info(f"Saved: {lam_path}")

    return all_results, cama_trajectory



# ── main ──────────────────────────────────────────────────────────────────────
def _banner():
    return (
        f"CAMA canonical runner\n"
        f"  dataset = {DATASET}\n"
        f"  K = {K}  BS = {BS}  LR = {LR}  WD = {WD}  opt = {OPT_TYPE}\n"
        f"  N_TOTAL = {N_TOTAL}  N_CAL = {N_CAL}  SEVERITY = {SEVERITY}\n"
        f"  STREAMING = {STREAMING}  KILL_THRESH = {KILL_THRESH}  DIAG_INTERVAL = {DIAG_INTERVAL}\n"
        f"  prior = {PRIOR_MAIN}  α = {ALPHA}  β = {BETA}\n"
        f"  corruptions = {ALL_CORRUPTIONS}\n"
        f"  output_dir = {OUTPUT_DIR}   seed = {SEED}\n"
        f"  save_embeddings = {SAVE_EMB}  save_ln = {SAVE_LN}  step_log = {STEP_LOG}"
    )


def main():
    desc = f"CAMA  dataset={DATASET}"
    load_cfg_from_args(desc)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("\n" + _banner())
    logger.info(f"  device = {device}")
    logger.info(f"  start  = {datetime.now().isoformat()}")

    if DRY_RUN:
        logger.info("[DRY-RUN] configuration printed, exiting without execution.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "run_config.txt"), "w") as f:
        f.write(_banner() + "\n")
        f.write(f"start: {datetime.now().isoformat()}\n")

    model, preprocess = get_model(cfg, K, device)
    model.eval()
    state_init = copy.deepcopy(model.state_dict())

    t_start = time.time()
    run_main(model, state_init, preprocess, device, OUTPUT_DIR)

    elapsed = time.time() - t_start
    logger.info(f"\nDone. Total: {elapsed/60:.1f} min  ({datetime.now().isoformat()})")


if __name__ == "__main__":
    main()
