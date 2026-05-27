#!/usr/bin/env python3
"""Per-channel (N, f) error heatmaps for the factor-conditioned denoiser.

Consumes the JSON written by `particle-denoise evaluate` and, for each
output channel, draws a 2-D heatmap:

    x-axis : input per-dimension ppc, sqrt(N)
    y-axis : denoising factor f = sqrt(ppc_out / ppc_in) (also per-dim)
    colour : validation RMSE for that channel at (N, f), in
             channel-stddev units

The (N, f) grid is sparse -- only tier pairs present in the training
data are evaluable -- so missing cells are left blank. Colour is a
diverging scale centred on RMSE = 1 (the constant-mean baseline):
blue beats the baseline, red is worse. Each cell is annotated with
its RMSE value.

Output: one PNG per channel, `heatmap_<variant>_<channel>.png`, plus
a `heatmap_<variant>_overview.png` contact sheet of all channels.

Usage:
  python generateErrorHeatmaps.py --rmse-json evaluate_rmse.json \
      --variant-id <name> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.colors import TwoSlopeNorm   # noqa: E402


def _buildGrid(pairs, channel_idx, ppcs_in, ppcs_out):
    """Return an (n_ppc_out, n_ppc_in) array of RMSE for one channel,
    NaN where the (N, M) tier pair does not exist. Plotted with N on
    the x-axis and M on the y-axis; the denoising factor f = sqrt(M/N)
    is the off-diagonal distance."""
    ic = {n: i for i, n in enumerate(ppcs_in)}
    io = {n: i for i, n in enumerate(ppcs_out)}
    grid = np.full((len(ppcs_out), len(ppcs_in)), np.nan)
    for p in pairs:
        if p["ppc_in"] not in ic or p["ppc_out"] not in io:
            continue
        grid[io[p["ppc_out"]], ic[p["ppc_in"]]] = \
            p["per_channel_rmse"][channel_idx]
    return grid


def _norm(grid):
    """Diverging norm centred on RMSE = 1, padded so vmin < 1 < vmax
    even when every cell is on one side of the baseline."""
    finite = grid[np.isfinite(grid)]
    lo = float(finite.min()) if finite.size else 0.0
    hi = float(finite.max()) if finite.size else 2.0
    return TwoSlopeNorm(vmin=min(lo, 0.99), vcenter=1.0,
                        vmax=max(hi, 1.01))


def _drawHeatmap(ax, grid, ppcs_in, ppcs_out, title, annotate=True):
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, origin="lower", aspect="auto",
                   cmap="RdBu_r", norm=_norm(grid))
    ax.set_xticks(range(len(ppcs_in)))
    ax.set_xticklabels([f"{int(round(n ** 0.5))}" for n in ppcs_in],
                       fontsize=8)
    ax.set_yticks(range(len(ppcs_out)))
    ax.set_yticklabels([f"{int(round(n ** 0.5))}" for n in ppcs_out],
                       fontsize=8)
    ax.set_xlabel("input ppc per dimension  $\\sqrt{N}$")
    ax.set_ylabel("output ppc per dimension  $\\sqrt{M}$")
    ax.set_title(title, fontsize=10)
    if annotate:
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                v = grid[r, c]
                if np.isfinite(v):
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                            fontsize=6,
                            color="white" if abs(v - 1.0) > 0.6 else "black")
    return im


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rmse-json", required=True,
                    help="JSON written by `particle-denoise evaluate`")
    ap.add_argument("--variant-id", required=True,
                    help="short name used in output filenames")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.rmse_json).read_text())
    channels = data["channels"]
    pairs = data["pairs"]
    if not pairs:
        print("no pairs in the RMSE JSON; nothing to plot")
        return 1

    ppcs_in  = sorted({p["ppc_in"]  for p in pairs})
    ppcs_out = sorted({p["ppc_out"] for p in pairs})
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-channel PNGs.
    for ci, ch in enumerate(channels):
        grid = _buildGrid(pairs, ci, ppcs_in, ppcs_out)
        fig, ax = plt.subplots(
            figsize=(0.55 * len(ppcs_in) + 2.5,
                     0.55 * len(ppcs_out) + 2.0))
        im = _drawHeatmap(ax, grid, ppcs_in, ppcs_out,
                          f"{ch}: validation RMSE (channel-stddev units)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="RMSE  (1 = constant-mean baseline)")
        fig.tight_layout()
        fig.savefig(str(out_dir / f"heatmap_{args.variant_id}_{ch}.png"),
                    dpi=100, pil_kwargs={"optimize": True,
                                         "compress_level": 9})
        plt.close(fig)

    # Contact-sheet overview.
    n = len(channels)
    ncol = 4
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(3.4 * ncol, 2.6 * nrow),
                             squeeze=False)
    for ci, ch in enumerate(channels):
        ax = axes[ci // ncol][ci % ncol]
        grid = _buildGrid(pairs, ci, ppcs_in, ppcs_out)
        _drawHeatmap(ax, grid, ppcs_in, ppcs_out, ch, annotate=False)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"{args.variant_id}: per-channel (N, M) validation RMSE",
                 fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    fig.savefig(str(out_dir / f"heatmap_{args.variant_id}_overview.png"),
                dpi=90, pil_kwargs={"optimize": True, "compress_level": 9})
    plt.close(fig)

    print(f"wrote {len(channels)} per-channel heatmaps + overview "
          f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
