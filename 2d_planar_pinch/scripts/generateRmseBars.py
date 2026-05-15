#!/usr/bin/env python3
"""Generate the 2x2 per-channel RMSE bar chart for a trained denoiser.

Layout: one figure per variant; 2 rows x 2 columns of subplots.

    col 0 (signal channels)       col 1 (noisy channels)
    +---------------------------+---------------------------+
    | RMSE start vs best        | RMSE start vs best        |
    | semilog, baseline = 1     | semilog, baseline = 1     |
    +---------------------------+---------------------------+
    | fraction of variance      | fraction of variance      |
    | reduced (1 - (best/start)^2)| ...                     |
    +---------------------------+---------------------------+

Channel split (planar pinch convention):

    signal-carrying:
        jz, num_<species>, pz_<species>, Pxx/Pyy/Pzz_<species>,
        Tx/Ty/Tz_<species>

    noisy / near-constant-mean baseline:
        jx, jy, px_<species>, py_<species>,
        enex/y/z_<species>

Both raw RMSE and derived (P, T) values come from the run checkpoint:

    first_val_rmse + best_val_rmse           -> 17 raw output channels
    first_derived_rmse + best_derived_rmse   -> 6 derived per species

Output filename: rmse_<variant_id>.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "/home/ghosh/Codes/particle-denoise")


def parsePerChannelRmse(s: str) -> list[float]:
    return [float(x) for x in s.split(";")] if s else []


def loadStartFinal(metrics_path: Path, ckpt_path: Path,
                   species_names: list[str]) -> tuple[list[str],
                                                      list[float],
                                                      list[float],
                                                      int | None]:
    """Return `(channel_names, rmse_start, rmse_final, best_epoch)`
    with the 17 raw output channels + 6 derived (P, T) per species
    in canonical channel order."""
    # Raw RMSE from the checkpoint (preferred: matches the chart that
    # the training loop would have written).
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    first_val = ckpt.get("first_val_rmse")
    best_val  = ckpt.get("best_val_rmse")
    best_epoch = ckpt.get("best_epoch_for_rmse")
    if first_val is None or best_val is None:
        # Fall back to CSV: read epoch-0 and best-epoch per-channel RMSE
        # rows.
        val_rows = []
        with metrics_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != "val":
                    continue
                val_rows.append(row)
        val_rows.sort(key=lambda r: int(r["epoch"]))
        first_val = parsePerChannelRmse(val_rows[0]["per_channel_rmse"])
        best_row = min(val_rows, key=lambda r: float(r["loss"]))
        best_val = parsePerChannelRmse(best_row["per_channel_rmse"])
        best_epoch = int(best_row["epoch"])
    first_val = list(first_val)
    best_val  = list(best_val)

    # Derived (P, T): also from the checkpoint.
    first_derived = ckpt.get("first_derived_rmse")
    best_derived  = ckpt.get("best_derived_rmse")
    derived_names: list[str] = []
    for sp in species_names:
        derived_names.extend([f"Pxx_{sp}", f"Pyy_{sp}", f"Pzz_{sp}",
                              f"Tx_{sp}",  f"Ty_{sp}",  f"Tz_{sp}"])

    # The training loop's first_derived / best_derived are 6 per species
    # concatenated in `species_names` order. May be missing on runs
    # where the closure_consistency loss was off (no derived eval).
    if first_derived is not None and best_derived is not None:
        first_derived = list(first_derived)
        best_derived  = list(best_derived)
    elif first_derived is not None:
        first_derived = list(first_derived)
        best_derived  = first_derived
    else:
        first_derived = [float("nan")] * len(derived_names)
        best_derived  = first_derived

    return ([*_RAW_CHANNELS, *derived_names],
            [*first_val, *first_derived],
            [*best_val,  *best_derived],
            best_epoch)


# Canonical raw-output channel order (must match the CSV / checkpoint
# per_channel_rmse arrays).
_RAW_CHANNELS = [
    "jx", "jy", "jz",
    "num_electrons",
    "px_electrons", "py_electrons", "pz_electrons",
    "enex_electrons", "eney_electrons", "enez_electrons",
    "num_deuterium",
    "px_deuterium", "py_deuterium", "pz_deuterium",
    "enex_deuterium", "eney_deuterium", "enez_deuterium",
]


def isSignal(name: str) -> bool:
    """Signal-carrying channels: jz, num_<sp>, pz_<sp>, Pxx/Pyy/Pzz_<sp>,
    Tx/Ty/Tz_<sp>. Everything else (jx, jy, px/py_<sp>, ene_<axis>_<sp>)
    is noisy."""
    if name == "jz":
        return True
    if name.startswith("num_"):
        return True
    if name.startswith("pz_"):
        return True
    if name.startswith(("Pxx_", "Pyy_", "Pzz_")):
        return True
    if name.startswith(("Tx_", "Ty_", "Tz_")):
        return True
    return False


def plotRmseBars(out_path: Path,
                 channel_names: Sequence[str],
                 rmse_start: Sequence[float],
                 rmse_final: Sequence[float],
                 best_epoch: int | None,
                 variant_label: str) -> None:
    """2x2 panel: rows = (RMSE semilog, FVR); cols = (signal, noisy)."""
    names = list(channel_names)
    start = np.asarray(rmse_start, dtype=float)
    final = np.asarray(rmse_final, dtype=float)
    sig_idx   = [i for i, n in enumerate(names) if isSignal(n)]
    noisy_idx = [i for i, n in enumerate(names) if not isSignal(n)]

    n_sig   = len(sig_idx)
    n_noisy = len(noisy_idx)
    width_per_bar = 0.40
    fig_w = max(10.0, width_per_bar * max(n_sig, n_noisy) + 2.0)
    fig, axes = plt.subplots(2, 2, sharey="row",
                             figsize=(fig_w, 7.0),
                             gridspec_kw={"height_ratios": [3, 2]})
    (ax_sig_rmse, ax_noi_rmse), (ax_sig_fvr, ax_noi_fvr) = axes

    label_final = (f"best (epoch {best_epoch})"
                   if best_epoch is not None else "final")
    bar_w = 0.4
    floor = 1.0e-3

    for idx_list, ax_rmse, ax_fvr, title in (
        (sig_idx,   ax_sig_rmse, ax_sig_fvr, "signal-carrying channels"),
        (noisy_idx, ax_noi_rmse, ax_noi_fvr, "noisy / baseline channels"),
    ):
        sub_names = [names[i]  for i in idx_list]
        sub_start = start[idx_list]
        sub_final = final[idx_list]
        n = len(sub_names)
        x = np.arange(n)

        # --- top row: RMSE semilog ---------------------------------------
        s_log = np.where(sub_start > 0, sub_start, floor)
        f_log = np.where(sub_final > 0, sub_final, floor)
        ax_rmse.bar(x - bar_w / 2, s_log, bar_w, label="epoch 0",
                    color="C0", alpha=0.85)
        ax_rmse.bar(x + bar_w / 2, f_log, bar_w, label=label_final,
                    color="C2", alpha=0.85)
        ax_rmse.axhline(1.0, color="gray", linestyle="--",
                        linewidth=0.8,
                        label="constant-mean baseline (RMSE = 1 sd)")
        ax_rmse.set_yscale("log")
        if ax_rmse is ax_sig_rmse:
            ax_rmse.set_ylabel("RMSE [channel stddev]")
        ax_rmse.set_title(title, fontsize=10)
        ax_rmse.legend(fontsize=8, loc="upper right")
        ax_rmse.grid(True, axis="y", which="both", alpha=0.3)
        ax_rmse.set_xticks(x)
        ax_rmse.set_xticklabels([])

        # --- bottom row: fraction of variance reduced --------------------
        # Clip the negative tail at FVR_FLOOR so the in-range bars stay
        # readable; channels beyond the floor (the closure-derived
        # temperatures can blow up by orders of magnitude when the
        # predicted density underflows in a few cells) get a downward
        # arrow + a numeric annotation.
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = 1.0 - (sub_final / np.maximum(sub_start, 1e-12)) ** 2
        frac = np.where(np.isfinite(frac), frac, 0.0)
        FVR_FLOOR = -2.0
        clipped = frac < FVR_FLOOR
        frac_disp = np.where(clipped, FVR_FLOOR, frac)
        bar_colors = np.where(frac_disp >= 0, "C2", "C3")
        ax_fvr.bar(x, frac_disp, color=bar_colors, alpha=0.85)
        ax_fvr.axhline(0.0, color="gray", linestyle="-", linewidth=0.6)
        for xi, (was_clipped, true_frac) in enumerate(zip(clipped, frac)):
            if was_clipped:
                ax_fvr.annotate(
                    f"{true_frac:.0e}↓",
                    xy=(xi, FVR_FLOOR),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=7, color="C3")
        if ax_fvr is ax_sig_fvr:
            ax_fvr.set_ylabel("fraction of variance reduced")
        ax_fvr.set_xticks(x)
        ax_fvr.set_xticklabels(sub_names, rotation=60, ha="right",
                               fontsize=8)
        ax_fvr.grid(True, axis="y", alpha=0.3)

    # Shared FVR y limits: clipped at FVR_FLOOR below, +1.05 above.
    for ax_fvr in (ax_sig_fvr, ax_noi_fvr):
        ax_fvr.set_ylim(bottom=FVR_FLOOR - 0.05, top=1.05)

    fig.suptitle(f"per-channel RMSE: {variant_label}", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=100,
                pil_kwargs={"optimize": True, "compress_level": 9})
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",     required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--metrics",    required=True)
    ap.add_argument("--variant-id", required=True,
                    help="short name used in output filename")
    ap.add_argument("--variant-label", default=None,
                    help="human-readable label for the chart title; "
                         "defaults to the variant-id")
    ap.add_argument("--out-dir",    required=True)
    args = ap.parse_args()

    from denoiseCore.config import load as loadConfig
    cfg = loadConfig(args.config)
    species_names = list(cfg.data.channels.species_names)

    names, start, final, best_epoch = loadStartFinal(
        Path(args.metrics).expanduser().resolve(),
        Path(args.checkpoint).expanduser().resolve(),
        species_names,
    )
    out_path = (Path(args.out_dir).expanduser().resolve()
                / f"rmse_{args.variant_id}.png")
    plotRmseBars(
        out_path,
        channel_names=names,
        rmse_start=start,
        rmse_final=final,
        best_epoch=best_epoch,
        variant_label=(args.variant_label or args.variant_id),
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
