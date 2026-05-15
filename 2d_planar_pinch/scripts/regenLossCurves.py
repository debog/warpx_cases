#!/usr/bin/env python3
"""Re-render loss_vs_epoch.png from a metrics.csv at the current
plotLossCurves DPI / figure-size.

Used to refresh existing run dirs after lowering plot DPI in
denoiseCore/plotting.py, and to relabel the flow-matching variant's
panel from 'recon' (its bookkeeping artifact in old metrics CSVs) to
'flow_matching'.

Usage:
  python regenLossCurves.py <metrics.csv> <out.png> [flowunet]

The third positional arg, if present, must be the literal string
`flowunet`; in that case the panel labelled `recon` in the metrics
CSV is relabelled to `flow_matching` (back-compat for runs that
predated the dedicated flow_matching column).
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/home/ghosh/Codes/particle-denoise")
from denoiseCore.plotting import plotLossCurves


def regen(metrics_path: Path, out_path: Path,
          flowunet: bool = False) -> None:
    epochs_train: list[int] = []
    epochs_val:   list[int] = []
    train_loss:   list[float] = []
    val_loss:     list[float] = []
    train_parts:  dict[str, list[float]] = defaultdict(list)
    val_parts:    dict[str, list[float]] = defaultdict(list)

    part_cols = (
        "recon", "conservation", "spectral", "gradient",
        "charge_continuity", "closure_consistency",
        "energy_conservation", "flow_matching",
    )
    with metrics_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = int(row["epoch"])
            l  = float(row["loss"])
            if row["split"] == "train":
                epochs_train.append(ep)
                train_loss.append(l)
                for c in part_cols:
                    if c in row and row[c] != "":
                        train_parts[c].append(float(row[c]))
            else:
                epochs_val.append(ep)
                val_loss.append(l)
                for c in part_cols:
                    if c in row and row[c] != "":
                        val_parts[c].append(float(row[c]))

    if flowunet:
        if "recon" in train_parts:
            train_parts["flow_matching"] = train_parts.pop("recon")
        if "recon" in val_parts:
            val_parts["flow_matching"] = val_parts.pop("recon")

    n_train = len(epochs_train)
    n_val   = len(epochs_val)
    if n_train != n_val:
        n = min(n_train, n_val)
        epochs_train = epochs_train[:n]
        train_loss   = train_loss[:n]
        val_loss     = val_loss[:n]
        for k in list(train_parts):
            train_parts[k] = train_parts[k][:n]
        for k in list(val_parts):
            val_parts[k] = val_parts[k][:n]

    plotLossCurves(
        out_path,
        epochs=epochs_train,
        train_loss=train_loss,
        val_loss=val_loss,
        train_parts=dict(train_parts),
        val_parts=dict(val_parts),
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    metrics  = Path(sys.argv[1]).resolve()
    out      = Path(sys.argv[2]).resolve()
    flow     = (len(sys.argv) > 3 and sys.argv[3] == "flowunet")
    regen(metrics, out, flowunet=flow)
