#!/usr/bin/env python3
"""Per-(N, M) heatmaps of the physics loss-term residuals.

Consumes the JSON written by `particle-denoise evaluate` (which now
records, per tier pair, the `loss_terms` dict: the residual of each
TotalLoss term -- recon, conservation, spectral, gradient,
charge_continuity, closure_consistency -- on the deployment
prediction). Draws one (N, M) panel per active term, so you can see
where each physical constraint is satisfied or violated across the
denoising grid.

These are scalar-per-pair residuals (not per-channel), so each panel
is a single (sqrt(N), sqrt(M)) heatmap with its own colorbar (the
terms differ in scale by orders of magnitude). Lower = the constraint
is better satisfied. A term that is flat-high everywhere means the
model cannot satisfy it; flat-low means there is nothing to gain.

Usage:
  python generateLossTermHeatmaps.py --rmse-json evaluate_rmse.json \
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
from matplotlib.colors import Normalize  # noqa: E402

# Display order; terms absent from the JSON or all-zero are skipped.
_TERM_ORDER = ("charge_continuity", "closure_consistency", "spectral",
               "gradient", "conservation", "recon", "energy_conservation")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rmse-json", required=True,
                    help="evaluate JSON containing per-pair loss_terms")
    ap.add_argument("--variant-id", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.rmse_json).read_text())
    pairs = data["pairs"]
    if not pairs or "loss_terms" not in pairs[0]:
        print("no loss_terms in the JSON; re-run evaluate with the "
              "loss-term recording enabled. nothing to plot.")
        return 1

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ppcs_in  = sorted({p["ppc_in"]  for p in pairs})
    ppcs_out = sorted({p["ppc_out"] for p in pairs})
    ic = {x: j for j, x in enumerate(ppcs_in)}
    io = {x: j for j, x in enumerate(ppcs_out)}

    # Which terms are present and not identically zero.
    present = [t for t in _TERM_ORDER if t in pairs[0]["loss_terms"]]
    grids = {}
    for t in present:
        g = np.full((len(ppcs_out), len(ppcs_in)), np.nan)
        for p in pairs:
            g[io[p["ppc_out"]], ic[p["ppc_in"]]] = p["loss_terms"].get(t, np.nan)
        if np.nanmax(np.abs(g)) > 0:
            grids[t] = g
    terms = list(grids)
    if not terms:
        print("all loss terms are zero; nothing to plot.")
        return 1

    n = len(terms)
    ncols = 3 if n > 4 else (2 if n > 1 else 1)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.2 * ncols, 3.4 * nrows),
                             squeeze=False)
    for i, t in enumerate(terms):
        ax = axes[i // ncols][i % ncols]
        g = grids[t]
        finite = g[np.isfinite(g)]
        # Per-panel scale (terms differ by orders of magnitude); clip
        # the top to the 98th percentile so a few cells do not wash out.
        vmax = float(np.percentile(finite, 98)) if finite.size else 1.0
        vmin = float(finite.min()) if finite.size else 0.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        im = ax.imshow(np.ma.masked_invalid(g), origin="lower",
                       aspect="auto", cmap="viridis",
                       norm=Normalize(vmin=vmin, vmax=vmax))
        ax.set_xticks(range(len(ppcs_in)))
        ax.set_xticklabels([f"{int(round(x ** 0.5))}" for x in ppcs_in],
                           fontsize=6.5)
        ax.set_yticks(range(len(ppcs_out)))
        ax.set_yticklabels([f"{int(round(x ** 0.5))}" for x in ppcs_out],
                           fontsize=6.5)
        ax.set_xlabel(r"input ppc/dim  $\sqrt{N}$", fontsize=8)
        ax.set_ylabel(r"output ppc/dim  $\sqrt{M}$", fontsize=8)
        ax.set_title(t, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(f"{args.variant_id}: physics loss-term residuals per (N, M)  "
                 f"(lower = constraint better satisfied)", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out = out_dir / f"heatmap_{args.variant_id}_lossterms.png"
    fig.savefig(str(out), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
