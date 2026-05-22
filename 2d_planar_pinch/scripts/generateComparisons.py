#!/usr/bin/env python3
"""Visual comparison panels for the factor-conditioned denoiser.

For each (input tier N, target tier M) pair and a set of simulation
steps, runs the trained denoiser and writes one PNG per channel:

    row r   = simulation step r
    col 0   = input snapshot at ppc N
    col 1   = denoiser prediction
    col 2   = target snapshot at ppc M   (M = f^2 * N)
    col 3   = relative error  =  (pred - target) / RMS(target)

The first three columns share a per-row color scale so input /
prediction / target read at a glance; the error column uses a
symmetric +/- RdBu_r scale in units of one RMS(target).

Channels:
    raw      the output channels the network emits (jx, jy, jz and the
             per-species num / p / ene moments);
    derived  Pxx/Pyy/Pzz and Tx/Ty/Tz per species, computed from the
             (num, p, ene) moments via the moment-closure formula.
             Temperature is reported in eV.

Output filenames are flat under <out-dir>:

    comparison_ppc<N>_to_ppc<M>_<channel>.png

Usage:
    generateComparisons.py --config denoise/cfg_charbonnier.yaml \\
        --checkpoint models/.run_pdn_charbonnier.<plat>/best.pt \\
        --out-dir comparisons
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

# Path injection: the local repo, not any installed package.
sys.path.insert(0, "/home/ghosh/Codes/particle-denoise")

EV_PER_JOULE = 1.0 / 1.602176634e-19


# ---------------------------------------------------------------------
# Derived pressure / temperature
# ---------------------------------------------------------------------

def deriveSpeciesPandT(moments: Dict[str, np.ndarray],
                       species_names: list[str],
                       warpx_input) -> Dict[str, Dict[str, np.ndarray]]:
    """Return {species: {Pxx, Pyy, Pzz, Tx, Ty, Tz}} from a flat dict
    of physical-unit moment arrays keyed `<moment>_<species>`
    (num, px, py, pz, enex, eney, enez).

    Pressure tensor diagonal: P_ii = (m/V) * num * (<v_i^2> - <v_i>^2)
    with <v_i>   = p_i / (m * num),  <v_i^2> = 2 * ene_i / (m * num).
    Temperature : T_i = P_i * V / num, converted to eV.
    """
    vcell = float(warpx_input.cell_volume())
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for sp in species_names:
        mass = float(warpx_input.species_mass(sp))
        num = np.asarray(moments[f"num_{sp}"], dtype=np.float64)
        # `num` may briefly be 0 in sparse cells; floor the divisor.
        num_safe = np.where(num > 0.0, num, 1.0)
        P, T = {}, {}
        for ax in ("x", "y", "z"):
            p_i = np.asarray(moments[f"p{ax}_{sp}"], dtype=np.float64)
            ene_i = np.asarray(moments[f"ene{ax}_{sp}"], dtype=np.float64)
            mean_v = p_i / (mass * num_safe)
            mean_v2 = 2.0 * ene_i / (mass * num_safe)
            var = np.clip(mean_v2 - mean_v * mean_v, 0.0, None)
            P_ii = (mass / vcell) * num * var
            T_ii = P_ii * vcell / np.where(num > 0.0, num, np.nan)
            T_ii_eV = np.where(np.isfinite(T_ii * EV_PER_JOULE),
                               T_ii * EV_PER_JOULE, 0.0)
            P[ax] = P_ii
            T[ax] = T_ii_eV
        out[sp] = {
            "Pxx": P["x"], "Pyy": P["y"], "Pzz": P["z"],
            "Tx":  T["x"], "Ty":  T["y"], "Tz":  T["z"],
        }
    return out


# ---------------------------------------------------------------------
# Grid plot
# ---------------------------------------------------------------------

def plotComparisonGrid(out_path: Path,
                       channel_name: str,
                       rows: list,
                       row_labels: list[str],
                       ppc_in: int,
                       ppc_out: int,
                       units: str = "",
                       percentile: float = 2.0) -> None:
    """One PNG: N rows (one per step) x 4 columns (input, prediction,
    target, relative error).

    `rows` is a list of dicts with keys 'input', 'pred', 'target'
    holding the per-step 2-D arrays. Relative error is computed per row
    as `(pred - target) / max(RMS(target), eps)`.
    """
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 4, figsize=(10.0, 2.2 * n_rows),
                             squeeze=False)

    col_titles = (f"input  (ppc {ppc_in})", "prediction",
                  f"target  (ppc {ppc_out})", "error / RMS(target)")
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=10)

    for ridx, row in enumerate(rows):
        inp = np.asarray(row["input"])
        prd = np.asarray(row["pred"])
        tgt = np.asarray(row["target"])

        stacked = np.concatenate([a.ravel() for a in (inp, prd, tgt)])
        stacked = stacked[np.isfinite(stacked)]
        if stacked.size:
            lo = float(np.percentile(stacked, percentile))
            hi = float(np.percentile(stacked, 100.0 - percentile))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(stacked.min()), float(stacked.max())
        else:
            lo, hi = -1.0, 1.0
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0

        tgt_finite = tgt.ravel()[np.isfinite(tgt.ravel())]
        tgt_rms = (float(np.sqrt(np.mean(tgt_finite ** 2)))
                   if tgt_finite.size else 1.0)
        eps = 1.0e-30 * (float(np.abs(tgt_finite).max())
                         if tgt_finite.size else 1.0)
        scale = max(tgt_rms, eps)
        rel_err = (prd - tgt) / scale

        re_finite = rel_err.ravel()[np.isfinite(rel_err.ravel())]
        if re_finite.size:
            rmax = float(np.percentile(np.abs(re_finite),
                                       100.0 - percentile))
            if not np.isfinite(rmax) or rmax == 0.0:
                rmax = float(np.abs(re_finite).max()) or 1.0
        else:
            rmax = 1.0

        for cidx, (arr, vmin, vmax, cmap) in enumerate([
            (inp,     lo,    hi,   "viridis"),
            (prd,     lo,    hi,   "viridis"),
            (tgt,     lo,    hi,   "viridis"),
            (rel_err, -rmax, rmax, "RdBu_r"),
        ]):
            ax = axes[ridx][cidx]
            disp = arr.T if arr.ndim == 2 else arr[None, :]
            im = ax.imshow(disp, origin="lower", aspect="auto",
                           vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            if cidx == 0:
                ax.set_ylabel(row_labels[ridx], fontsize=9)

    suffix = f"  [{units}]" if units else ""
    fig.suptitle(f"{channel_name}{suffix}   "
                 f"ppc {ppc_in} -> {ppc_out}", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=80,
                pil_kwargs={"optimize": True, "compress_level": 9})
    plt.close(fig)


# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------

def denormChannels(tensor: torch.Tensor, names: list[str],
                   norm) -> Dict[str, np.ndarray]:
    """Invert normalization for a (C, ...) tensor of output channels,
    returning a flat {canonical_name: physical-unit array} dict."""
    arr = tensor.detach().float().cpu().numpy()
    return {name: norm.invert(name, arr[c])
            for c, name in enumerate(names)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default="comparisons")
    ap.add_argument("--pairs", default=None,
                    help="comma-separated ppc_in:ppc_out pairs to render; "
                         "default = every tier denoised to the top tier")
    ap.add_argument("--steps", default=None,
                    help="comma-separated step list; default = up to "
                         "--max-steps evenly spaced across the pair")
    ap.add_argument("--max-steps", type=int, default=4)
    args = ap.parse_args()

    from denoiseCore.config import load as loadConfig
    from denoiseCore.data.pairing import enumerateFactorPairs
    from denoiseCore.data.readers import getReader
    from denoiseCore.data.dataset import (FactorPairDataset,
                                          outputChannelNames,
                                          outputChannelIndices)
    from denoiseCore.device import getDevice
    from particleDenoiser.models import buildModel
    from particleDenoiser.training import fitNormalizer

    cfg = loadConfig(args.config)
    dev = getDevice()
    ckpt = torch.load(args.checkpoint, map_location=dev.device,
                      weights_only=False)
    reader = getReader(cfg.data.reader)
    tiers = cfg.data.ppc_tiers
    if len(tiers) < 2:
        raise RuntimeError("comparisons need >= 2 ppc tiers")

    warpx_input = None
    if cfg.data.warpx_input is not None:
        from denoiseCore.warpxInput import load as loadWarpXInput
        warpx_input = loadWarpXInput(cfg.data.warpx_input)

    all_pairs = enumerateFactorPairs(reader, tiers, cfg.data.steps)
    norm = fitNormalizer(reader, all_pairs, cfg.data.channels,
                         cfg.data.normalization, warpx_input)

    out_names = outputChannelNames(cfg.data.channels)
    out_indices = outputChannelIndices(cfg.data.channels)
    probe = FactorPairDataset(all_pairs[:1], reader, cfg.data.channels,
                              norm, cache=False, prefill=False,
                              warpx_input=warpx_input)[0]
    in_channels = probe.input_tensor.shape[0]
    dim = probe.input_tensor.ndim - 1
    model = buildModel(cfg.model, in_channels=in_channels,
                       out_channels=len(out_names),
                       out_indices=out_indices, dim=dim).to(dev.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Group pairs by (ppc_in, ppc_out).
    by_tier_pair: Dict[tuple, list] = {}
    for p in all_pairs:
        by_tier_pair.setdefault((p.ppc_in, p.ppc_out), []).append(p)

    # Which tier pairs to render.
    if args.pairs:
        wanted = []
        for tok in args.pairs.split(","):
            a, b = tok.split(":")
            wanted.append((int(a), int(b)))
    else:
        top = max(t.ppc for t in tiers)
        wanted = sorted(k for k in by_tier_pair if k[1] == top)

    want_steps = ([int(s) for s in args.steps.split(",")]
                  if args.steps else None)

    derived_species = cfg.data.channels.species_names
    can_derive = warpx_input is not None
    out_dir = Path(args.out_dir)

    for (ppc_in, ppc_out) in wanted:
        group = sorted(by_tier_pair.get((ppc_in, ppc_out), []),
                       key=lambda p: p.step)
        if not group:
            print(f"  no pairs for ppc {ppc_in} -> {ppc_out}, skipping")
            continue
        if want_steps is not None:
            group = [p for p in group if p.step in want_steps]
        elif len(group) > args.max_steps:
            idx = np.linspace(0, len(group) - 1, args.max_steps)
            group = [group[int(round(i))] for i in idx]
        if not group:
            continue

        # Per-channel row stacks, plus per-step derived-moment dicts.
        raw_rows: Dict[str, list] = {n: [] for n in out_names}
        labels: list[str] = []
        derived_in: list = []
        derived_pred: list = []
        derived_tgt: list = []

        for pair in group:
            ds = FactorPairDataset([pair], reader, cfg.data.channels,
                                   norm, cache=False, prefill=False,
                                   warpx_input=warpx_input)
            s = ds[0]
            x = s.input_tensor.unsqueeze(0).to(dev.device)
            s2i = torch.tensor([s.sigma2_in], device=dev.device)
            s2o = torch.tensor([s.sigma2_out], device=dev.device)
            with torch.no_grad():
                pred = model(x, s2i, s2o).squeeze(0)

            inp_oc = s.input_tensor[out_indices]
            phys_in = denormChannels(inp_oc, out_names, norm)
            phys_pred = denormChannels(pred, out_names, norm)
            phys_tgt = denormChannels(s.target_tensor, out_names, norm)

            for n in out_names:
                raw_rows[n].append({"input": phys_in[n],
                                     "pred": phys_pred[n],
                                     "target": phys_tgt[n]})
            labels.append(f"step {pair.step}")
            if can_derive:
                derived_in.append(
                    deriveSpeciesPandT(phys_in, derived_species, warpx_input))
                derived_pred.append(
                    deriveSpeciesPandT(phys_pred, derived_species, warpx_input))
                derived_tgt.append(
                    deriveSpeciesPandT(phys_tgt, derived_species, warpx_input))

        # Raw output channels.
        for n in out_names:
            out_path = out_dir / f"comparison_ppc{ppc_in}_to_ppc{ppc_out}_{n}.png"
            plotComparisonGrid(out_path, n, raw_rows[n], labels,
                               ppc_in, ppc_out)

        # Derived pressure / temperature channels.
        if can_derive:
            for sp in derived_species:
                for moment in ("Pxx", "Pyy", "Pzz", "Tx", "Ty", "Tz"):
                    rows = [{"input": di[sp][moment],
                             "pred": dp[sp][moment],
                             "target": dt[sp][moment]}
                            for di, dp, dt in
                            zip(derived_in, derived_pred, derived_tgt)]
                    units = "eV" if moment.startswith("T") else "Pa"
                    name = f"{moment}_{sp}"
                    out_path = (out_dir /
                                f"comparison_ppc{ppc_in}_to_ppc{ppc_out}_{name}.png")
                    plotComparisonGrid(out_path, name, rows, labels,
                                       ppc_in, ppc_out, units=units)

        print(f"  ppc {ppc_in} -> {ppc_out}: "
              f"{len(out_names) + (12 if can_derive else 0)} channels, "
              f"{len(group)} steps")

    print(f"comparison panels written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
