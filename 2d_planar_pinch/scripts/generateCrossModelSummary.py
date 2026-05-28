"""Cross-model summary figures for the factor-conditioned denoiser
comparison.

Writes into the docs figures directory:

  - factor_models_losses.png  -- one panel per model: train + val
    loss vs epoch on log y. Independent y-axes (the diffusion
    objective is not directly comparable to the CNNs').
  - factor_models_perchannel_rmse_<N>_<M>.png  -- grouped bar chart
    of per-channel debiased validation RMSE at one representative
    (N, M) pair (default 36 -> 1024).
  - factor_models_heatmap_<ch>.png  -- side-by-side debiased (N, M)
    heatmap for one channel, three panels (one per model), shared
    diverging RdBu_r colourmap centred on RMSE = 1 (the same
    convention as generateErrorHeatmaps.py / generateSplitOverviews.py:
    blue = below the constant-mean baseline, red = above).

Reads per-model metrics.csv and evaluate_rmse.json from
results_factor_models/<model>/.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

ROOT = Path.home() / "Runs/warpx/2d_planar_pinch/results_factor_models"
FIG  = Path.home() / "Documents/Work_Docs/2026_04_ML_ImplicitPIC/figures"

MODELS = [
    ("globalcnn",  "Global U-Net",       "#1f77b4"),
    ("localcnn",   "Local CNN",          "#2ca02c"),
    ("diffusion",  "Diffusion U-Net",    "#d62728"),
]

# Representative pair for the bar chart: the hardest denoising
# (lowest input ppc) up to ground truth.
TARGET_PAIR = (36, 1024)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def loadCurves(metrics_csv: Path):
    """Return (epoch_list, train_loss_list, val_loss_list) from a
    metrics.csv with rows (epoch, split, loss, per_channel_rmse)."""
    import csv
    epochs_train, train = [], []
    epochs_val,   val   = [], []
    with metrics_csv.open() as f:
        r = csv.DictReader(f)
        for row in r:
            e = int(row["epoch"])
            v = float(row["loss"])
            if row["split"] == "train":
                epochs_train.append(e); train.append(v)
            elif row["split"] == "val":
                epochs_val.append(e);   val.append(v)
    return epochs_train, train, epochs_val, val


def loadDebiased(rmse_json: Path):
    """Return {(ppc_in, ppc_out): {channel: debiased_rmse}} from an
    evaluate_rmse.json, applying the same per-channel target-noise
    debiasing as generateErrorHeatmaps.py._debiasGrid: K_c =
    min(rmse^2 * M) across all cells in the channel's (N, M) grid,
    debiased = sqrt(max(0, rmse^2 - K_c / M))."""
    data = json.loads(rmse_json.read_text())
    rows = data["pairs"]
    channels = data["channels"]
    K = {}
    for ch_idx, ch in enumerate(channels):
        cal = [r["per_channel_rmse"][ch_idx] ** 2 * int(r["ppc_out"])
               for r in rows]
        K[ch] = float(np.nanmin(cal)) if cal else 0.0
    out = {}
    for r in rows:
        ppc_in  = int(r["ppc_in"])
        ppc_out = int(r["ppc_out"])
        M = ppc_out
        per_ch = {}
        for ch_idx, ch in enumerate(channels):
            rms2 = r["per_channel_rmse"][ch_idx] ** 2
            debiased = max(0.0, rms2 - K[ch] / M) ** 0.5
            per_ch[ch] = debiased
        out[(ppc_in, ppc_out)] = per_ch
    return out, channels


# ---------------------------------------------------------------------
# Figure 1: loss curves (3 panels)
# ---------------------------------------------------------------------

def plotLossCurves():
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.0), squeeze=False)
    for ax, (m, label, color) in zip(axes[0], MODELS):
        et, tr, ev, va = loadCurves(ROOT / m / "metrics.csv")
        if tr:
            ax.semilogy(et, tr, marker="o", ms=2.5, linewidth=1,
                        color=color, label="train")
        if va:
            ax.semilogy(ev, va, marker="s", ms=2.5, linewidth=1,
                        color=color, linestyle="--", label="val",
                        alpha=0.8)
        ax.set_title(f"{label}  (best val={min(va):.3e} @ epoch {ev[va.index(min(va))]})",
                     fontsize=11)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")
    fig.suptitle("Factor-conditioned denoiser  --  train + val loss vs epoch",
                 fontsize=12.5)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out = FIG / "factor_models_losses.png"
    fig.savefig(str(out), dpi=160, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------
# Figure 2: per-channel debiased-RMSE bar chart at a fixed (N, M)
# ---------------------------------------------------------------------

def plotPerChannelBars():
    debiased = {}
    channels = None
    for (m, _, _) in MODELS:
        d, ch = loadDebiased(ROOT / m / "evaluate_rmse.json")
        debiased[m] = d
        channels = ch          # same channel order across models

    if any(TARGET_PAIR not in debiased[m] for (m, _, _) in MODELS):
        raise RuntimeError(f"target pair {TARGET_PAIR} missing in one of the models")

    x = np.arange(len(channels))
    width = 0.27
    fig, ax = plt.subplots(figsize=(14.0, 4.6))
    for i, (m, label, color) in enumerate(MODELS):
        vals = [debiased[m][TARGET_PAIR][ch] for ch in channels]
        ax.bar(x + (i - 1) * width, vals, width, color=color, label=label,
               alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("debiased (systematic) RMSE  [normalised units]")
    n, mppc = TARGET_PAIR
    ax.set_title(f"Per-channel debiased validation RMSE at ppc {n} -> {mppc}  "
                 f"(target-noise subtracted)",
                 fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    out = FIG / "factor_models_perchannel_rmse_36_1024.png"
    fig.savefig(str(out), dpi=160, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------
# Figure 3: side-by-side (N, M) debiased heatmap for jz
# ---------------------------------------------------------------------

def _heatmapMatrix(d, channel, channels):
    pairs = sorted(d.keys())
    ns = sorted({p[0] for p in pairs})
    ms = sorted({p[1] for p in pairs})
    # √N, √M on the axes so the heatmap is more even.
    mat = np.full((len(ms), len(ns)), np.nan)
    for (n, m), per_ch in d.items():
        i = ms.index(m)
        j = ns.index(n)
        mat[i, j] = per_ch[channel]
    return mat, ns, ms


def plotSideBySideHeatmaps(channel):
    debiased = {}
    channels = None
    for (m, _, _) in MODELS:
        d, ch = loadDebiased(ROOT / m / "evaluate_rmse.json")
        debiased[m] = d
        channels = ch

    # Stack matrices to get a shared diverging color scale matching
    # the per-model heatmap convention (RdBu_r centred on 1, the
    # constant-mean baseline; blue = below baseline, red = above).
    mats = []
    for (m, _, _) in MODELS:
        mat, ns, ms = _heatmapMatrix(debiased[m], channel, channels)
        mats.append(mat)
    stacked = np.concatenate([m[np.isfinite(m)].ravel() for m in mats])
    lo = float(stacked.min()) if stacked.size else 0.0
    hi = float(stacked.max()) if stacked.size else 2.0
    shared_norm = TwoSlopeNorm(vmin=min(lo, 0.99), vcenter=1.0,
                               vmax=max(hi, 1.01))

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.05]},
                             squeeze=False)
    for ax, (m, label, _), mat in zip(axes[0, :3], MODELS, mats):
        im = ax.imshow(np.ma.masked_invalid(mat),
                       origin="lower", aspect="auto",
                       cmap="RdBu_r", norm=shared_norm)
        ax.set_title(label, fontsize=11.5)
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels([str(int(round(np.sqrt(n)))) for n in ns],
                           rotation=0, fontsize=8)
        ax.set_yticks(range(len(ms)))
        ax.set_yticklabels([str(int(round(np.sqrt(m)))) for m in ms],
                           fontsize=8)
        ax.set_xlabel(r"input ppc per dimension  $\sqrt{N}$")
        ax.set_ylabel(r"output ppc per dimension  $\sqrt{M}$")
    cax = axes[0, 3]
    fig.colorbar(im, cax=cax)
    cax.set_ylabel(
        "systematic RMSE  (target noise floor subtracted; "
        "1 = constant-mean baseline)",
        fontsize=9)
    fig.suptitle(f"Channel {channel}  --  debiased (N, M) RMSE, side-by-side",
                 fontsize=12.5)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out = FIG / f"factor_models_heatmap_{channel}.png"
    fig.savefig(str(out), dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------

def main():
    plotLossCurves()
    plotPerChannelBars()
    for ch in ("jz", "jx", "num_electrons", "px_electrons"):
        plotSideBySideHeatmaps(ch)

if __name__ == "__main__":
    sys.exit(main() or 0)
