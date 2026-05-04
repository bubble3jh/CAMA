"""Optional status-file writer for monitoring long runs.

`main_cama.py` calls `write_status(...)` after each adaptation step. The function
writes a JSON snapshot to `/tmp/exp_status.json` so an external monitor process
can render progress without scraping logs. Both functions are no-ops if the
write fails (e.g., the path is not writable). Importing this module is purely
optional; `main_cama.py` falls back to no-op stubs when the import is missing.
"""

import json
import os
from datetime import datetime, timedelta

STATUS_PATH = "/tmp/exp_status.json"
_STARTED_AT = datetime.now().isoformat()


def compute_eta(step: int, n_steps: int,
                corr_idx: int, corr_total: int,
                s_per_step: float) -> str:
    """Estimate completion time from the current step and corruption position.

    Args:
        step:       current step within the corruption (1-indexed).
        n_steps:    total steps per corruption.
        corr_idx:   current corruption index (1-indexed).
        corr_total: total number of corruptions.
        s_per_step: seconds per step (a recent moving average is recommended).

    Returns:
        Local-time string ``HH:MM`` or ``"-"`` when the rate is unknown.
    """
    if s_per_step <= 0:
        return "-"

    remaining_steps_this_corr = max(n_steps - step, 0)
    remaining_full_corr = max(corr_total - corr_idx, 0)
    # ~10 step-equivalent units of offline evaluation per corruption
    OFFLINE_EQUIV_STEPS = 10
    remaining_total = (
        remaining_steps_this_corr
        + OFFLINE_EQUIV_STEPS
        + remaining_full_corr * (n_steps + OFFLINE_EQUIV_STEPS)
    )
    eta_dt = datetime.now() + timedelta(seconds=remaining_total * s_per_step)
    return eta_dt.strftime("%H:%M")


def write_status(script: str = "",
                 phase: int = 1, phase_total: int = 1,
                 corruption: str = "", corr_idx: int = 0, corr_total: int = 0,
                 step: int = 0, n_steps: int = 50,
                 online_acc: float = 0.0,
                 s_per_step: float = 0.0,
                 eta: str = "-",
                 cat_pct: float | None = None,
                 h_pbar: float | None = None,
                 lambda_val: float | None = None,
                 extra: dict | None = None) -> None:
    """Atomically write a status snapshot to ``STATUS_PATH``."""
    data = {
        "script":      script,
        "phase":       phase,
        "phase_total": phase_total,
        "corruption":  corruption,
        "corr_idx":    corr_idx,
        "corr_total":  corr_total,
        "step":        step,
        "n_steps":     n_steps,
        "online_acc":  round(online_acc, 4),
        "s_per_step":  round(s_per_step, 1),
        "eta":         eta,
        "started_at":  _STARTED_AT,
        "updated_at":  datetime.now().strftime("%H:%M:%S"),
    }
    if cat_pct    is not None: data["cat_pct"]    = round(float(cat_pct),    4)
    if h_pbar     is not None: data["h_pbar"]     = round(float(h_pbar),     4)
    if lambda_val is not None: data["lambda_val"] = round(float(lambda_val), 4)
    if extra: data.update(extra)

    tmp_path = STATUS_PATH + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, STATUS_PATH)
    except Exception:
        pass
