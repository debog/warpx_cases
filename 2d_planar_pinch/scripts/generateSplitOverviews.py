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
from matplotlib.colors import TwoSlopeNorm

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


def _renderOverview(pairs, channels, names, out_path, title,
                    debias=True):
    ppcs_in  = sorted({p["ppc_in"]  for p in pairs})
    ppcs_out = sorted({p["ppc_out"] for p in pairs})
    name_to_idx = {n: i for i, n in enumerate(channels)}

    # Build every panel's RMSE grid up front so all panels can share
    # one TwoSlopeNorm (and hence one colorbar) covering the full
    # value range across the figure.
    ic = {x: j for j, x in enumerate(ppcs_in)}
    io = {x: j for j, x in enumerate(ppcs_out)}
    grids = {}
    for name in names:
        cidx = name_to_idx[name]
        g = np.full((len(ppcs_out), len(ppcs_in)), np.nan)
        for p in pairs:
            g[io[p["ppc_out"]], ic[p["ppc_in"]]] = \
                p["per_channel_rmse"][cidx]
        if debias:
            g = _debiasGrid(g, ppcs_out)
        grids[name] = g
    all_finite = np.concatenate(
        [g[np.isfinite(g)].ravel() for g in grids.values()])
    vmax = max(1.01, float(all_finite.max())) if all_finite.size else 2.0
    vmin = min(0.0, float(all_finite.min())) if all_finite.size else 0.0
    norm = TwoSlopeNorm(vmin=min(vmin, 0.99), vcenter=1.0, vmax=vmax)

    n = len(names)
    ncols = 4 if n > 4 else n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.2 * ncols + 1.2, 3.0 * nrows),
                             squeeze=False)
    im_handle = None
    for i, name in enumerate(names):
        ax = axes[i // ncols][i % ncols]
        im_handle = ax.imshow(grids[name], origin="lower",
                              aspect="auto", cmap="RdBu_r", norm=norm)
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
    # Layout first, then attach a single colorbar to the right that
    # spans the full panel grid.
    fig.tight_layout(rect=(0.03, 0.04, 0.92, 0.96))
    cbar_ax = fig.add_axes([0.93, 0.10, 0.013, 0.80])
    label = ("RMSE  (1 = constant-mean baseline)" if not debias else
             "systematic RMSE  (target noise floor subtracted; "
             "1 = constant-mean baseline)")
    fig.colorbar(im_handle, cax=cbar_ax, label=label)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rmse-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--variant-id", default="factor")
    ap.add_argument("--raw", action="store_true",
                    help="report raw RMSE against the noisy target; "
                         "default subtracts the per-M target-noise "
                         "floor.")
    args = ap.parse_args()
    data = json.loads(Path(args.rmse_json).read_text())
    pairs = data["pairs"]
    channels = data["channels"]
    out = Path(args.out_dir)
    metric = "validation RMSE" if args.raw else \
             "systematic validation RMSE (target-noise debiased)"
    _renderOverview(pairs, channels, SIGNAL,
                    out / f"heatmap_{args.variant_id}_overview_signal.png",
                    f"Signal-carrying channels: per-(N,M) {metric}",
                    debias=not args.raw)
    _renderOverview(pairs, channels, NOISE,
                    out / f"heatmap_{args.variant_id}_overview_noise.png",
                    f"Noise-dominated channels: per-(N,M) {metric}",
                    debias=not args.raw)

if __name__ == "__main__":
    main()
