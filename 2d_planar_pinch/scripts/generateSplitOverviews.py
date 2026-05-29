#!/usr/bin/env python3
"""Render two overview heatmaps (signal-carrying vs noise-dominated)
from the evaluate_rmse.json the factor model writes.

Channel split (physical / by-symmetry, not by post-hoc RMSE):

  SIGNAL-CARRYING -- channels that carry physical structure:
    jz                       axial Z-pinch current
    num_electrons, num_deuterium  plasma density slab
    px_electrons, px_deuterium    radial implosion momentum density
    enex_electrons, enex_deuterium  radial kinetic-energy moment

  NOISE-DOMINATED -- channels that are zero by symmetry or are
  almost-uniform thermal energies and so are deposition-noise
  limited:
    jx, jy                                in-plane currents perp to z
    py_electrons, pz_electrons            out-of-plane electron momenta
    py_deuterium, pz_deuterium            out-of-plane ion momenta
    eney_electrons, enez_electrons        out-of-plane electron KE moments
    eney_deuterium, enez_deuterium        out-of-plane ion KE moments
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm, Normalize

SIGNAL = ("jz", "num_electrons", "num_deuterium",
          "px_electrons", "px_deuterium",
          "enex_electrons", "enex_deuterium")
NOISE  = ("jx", "jy",
          "py_electrons", "pz_electrons",
          "py_deuterium", "pz_deuterium",
          "eney_electrons", "enez_electrons",
          "eney_deuterium", "enez_deuterium")


def _debiasGrid(g: np.ndarray, ppcs_out) -> np.ndarray:
    """Subtract per-M target-noise floor (`K_c / M`, calibrated from
    the channel's cleanest cell) from RMSE^2. See
    generateErrorHeatmaps._debiasGrid for the rationale."""
    ppcs_arr = np.asarray(ppcs_out, dtype=np.float64)
    rmsesq = g.astype(np.float64) ** 2
    cal = rmsesq * ppcs_arr[:, None]
    finite = cal[np.isfinite(cal)]
    if not finite.size:
        return g.copy()
    K_c = float(np.nanmin(cal))
    floor_sq = K_c / ppcs_arr[:, None]
    return np.sqrt(np.maximum(0.0, rmsesq - floor_sq))


# Per-metric rendering spec. `key` is the JSON field; `debias` applies
# the target-noise floor subtraction (RMSE only); `cmap` + `diverging`
# select the colour treatment (RMSE: diverging RdBu_r centred on the
# constant-mean baseline of 1, blue=better; the rest: sequential
# viridis with a direction-labelled colorbar); `higher_better` flips
# the colourbar sense for PSNR.
_METRIC_SPEC = {
    "rmse":  dict(key="per_channel_rmse",  diverging=True,
                  cmap="RdBu_r", higher_better=False,
                  label="systematic RMSE  (target noise floor subtracted; "
                        "1 = constant-mean baseline)",
                  label_raw="RMSE  (1 = constant-mean baseline)",
                  title="RMSE"),
    "l1":    dict(key="per_channel_l1",    diverging=False,
                  cmap="viridis", higher_better=False,
                  label="MAE  [channel-stddev units]  (lower = better)",
                  title="MAE (L1)"),
    "linf":  dict(key="per_channel_linf",  diverging=False,
                  cmap="viridis", higher_better=False,
                  label="max abs error  [channel-stddev units]  (lower = better)",
                  title="max abs error (L-inf)"),
    "psnr":  dict(key="per_channel_psnr",  diverging=False,
                  cmap="viridis", higher_better=True,
                  label="PSNR  [dB]  (higher = better)",
                  title="PSNR"),
}


def _renderOverview(pairs, channels, names, out_path, title, metric, raw):
    spec = _METRIC_SPEC[metric]
    key = spec["key"]
    debias = spec["diverging"] and metric == "rmse" and not raw
    ppcs_in  = sorted({p["ppc_in"]  for p in pairs})
    ppcs_out = sorted({p["ppc_out"] for p in pairs})
    name_to_idx = {n: i for i, n in enumerate(channels)}

    # Build every panel's grid up front so all panels share one norm /
    # colorbar covering the full value range across the figure.
    ic = {x: j for j, x in enumerate(ppcs_in)}
    io = {x: j for j, x in enumerate(ppcs_out)}
    grids = {}
    for name in names:
        cidx = name_to_idx[name]
        g = np.full((len(ppcs_out), len(ppcs_in)), np.nan)
        for p in pairs:
            g[io[p["ppc_out"]], ic[p["ppc_in"]]] = p[key][cidx]
        if debias:
            g = _debiasGrid(g, ppcs_out)
        grids[name] = g
    all_finite = np.concatenate(
        [g[np.isfinite(g)].ravel() for g in grids.values()])

    if spec["diverging"]:
        # Diverging around the constant-mean baseline of 1.
        vmax = max(1.01, float(all_finite.max())) if all_finite.size else 2.0
        vmin = min(0.0, float(all_finite.min())) if all_finite.size else 0.0
        norm = TwoSlopeNorm(vmin=min(vmin, 0.99), vcenter=1.0, vmax=vmax)
    else:
        # Sequential: clip to 2nd..98th percentile so a few extreme
        # cells do not wash out the scale.
        if all_finite.size:
            lo = float(np.percentile(all_finite, 2.0))
            hi = float(np.percentile(all_finite, 98.0))
        else:
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1.0
        norm = Normalize(vmin=lo, vmax=hi)

    n = len(names)
    ncols = 4 if n > 4 else n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.2 * ncols + 1.2, 3.0 * nrows),
                             squeeze=False)
    im_handle = None
    for i, name in enumerate(names):
        ax = axes[i // ncols][i % ncols]
        im_handle = ax.imshow(np.ma.masked_invalid(grids[name]),
                              origin="lower", aspect="auto",
                              cmap=spec["cmap"], norm=norm)
        ax.set_xticks(range(len(ppcs_in)))
        ax.set_xticklabels([f"{int(round(x ** 0.5))}" for x in ppcs_in],
                           fontsize=6.5)
        ax.set_yticks(range(len(ppcs_out)))
        ax.set_yticklabels([f"{int(round(x ** 0.5))}" for x in ppcs_out],
                           fontsize=6.5)
        ax.set_title(name, fontsize=9)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.text(0.5, 0.02, "input ppc per dimension  $\\sqrt{N}$",
             ha="center")
    fig.text(0.02, 0.5, "output ppc per dimension  $\\sqrt{M}$",
             va="center", rotation=90)
    fig.tight_layout(rect=(0.03, 0.04, 0.92, 0.96))
    cbar_ax = fig.add_axes([0.93, 0.10, 0.013, 0.80])
    label = spec.get("label_raw", spec["label"]) if (
        metric == "rmse" and raw) else spec["label"]
    fig.colorbar(im_handle, cax=cbar_ax, label=label)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rmse-json", required=True,
                    help="evaluate JSON (per_channel_rmse/l1/linf/psnr)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--variant-id", default="factor")
    ap.add_argument("--metric", default="rmse",
                    choices=sorted(_METRIC_SPEC),
                    help="which metric's heatmaps to render")
    ap.add_argument("--raw", action="store_true",
                    help="RMSE only: report raw RMSE against the noisy "
                         "target instead of the target-noise-debiased value.")
    args = ap.parse_args()
    data = json.loads(Path(args.rmse_json).read_text())
    pairs = data["pairs"]
    channels = data["channels"]
    out = Path(args.out_dir)
    spec = _METRIC_SPEC[args.metric]
    desc = spec["title"]
    if args.metric == "rmse":
        desc = "RMSE" if args.raw else "systematic RMSE (target-noise debiased)"
    base = f"heatmap_{args.variant_id}_{args.metric}_overview"
    _renderOverview(pairs, channels, SIGNAL,
                    out / f"{base}_signal.png",
                    f"Signal-carrying channels: per-(N,M) {desc}",
                    args.metric, args.raw)
    _renderOverview(pairs, channels, NOISE,
                    out / f"{base}_noise.png",
                    f"Noise-dominated channels: per-(N,M) {desc}",
                    args.metric, args.raw)

if __name__ == "__main__":
    main()
