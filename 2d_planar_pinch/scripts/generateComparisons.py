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
    generateComparisons.py --config denoise/cfg_globalcnn_charbonnier.yaml \\
        --checkpoint models/.run_pdn_globalcnn_charbonnier.<plat>/best.pt \\
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

# Canonical comparison frames -- same four (step, time, regime) tuples
# the baseline page (status_baseline.html) shows. Picked so each
# panel lands on a distinct physics regime of the implosion.
_DEFAULT_PHASES = [
    (11025,  22.05, "ramp-up"),
    (54900, 109.80, "mid-snowplow"),
    (76950, 153.90, "peak compression"),
    (105300, 210.60, "rebound"),
]
_PHASE_LABEL = {step: f"{name}\nstep {step}  (t = {t:g} ns)"
                for step, t, name in _DEFAULT_PHASES}


# ---------------------------------------------------------------------
# Derived pressure / temperature
# ---------------------------------------------------------------------

def deriveSpeciesPandT(moments: Dict[str, np.ndarray],
                       species_names: list[str],
                       warpx_input,
                       num_ref: Dict[str, np.ndarray] | None = None,
                       ) -> Dict[str, Dict[str, np.ndarray]]:
    """Return {species: {Pxx, Pyy, Pzz, Tx, Ty, Tz}} from a flat dict
    of physical-unit moment arrays keyed `<moment>_<species>`
    (num, px, py, pz, enex, eney, enez).

    Pressure tensor diagonal: P_ii = (m/V) * num * (<v_i^2> - <v_i>^2)
    with <v_i>   = p_i / (m * num),  <v_i^2> = 2 * ene_i / (m * num).
    Temperature : T_i = P_i * V / num, converted to eV.

    `num_ref` (optional) supplies the per-species `num` array used to
    decide which cells are plasma vs vacuum. The denoiser's
    prediction can carry noisy positive `num` in cells that are
    physically vacuum, which would unmask cells where T blows up.
    Pass the target's `num` arrays here so the input / prediction /
    target derivations share a consistent mask.
    """
    vcell = float(warpx_input.cell_volume())
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for sp in species_names:
        mass = float(warpx_input.species_mass(sp))
        num = np.asarray(moments[f"num_{sp}"], dtype=np.float64)
        ref = (np.asarray(num_ref[f"num_{sp}"], dtype=np.float64)
               if num_ref is not None else num)
        ref_max = float(np.nanmax(ref)) if np.any(np.isfinite(ref)) else 0.0
        # Mask threshold at 1% of the reference peak: tight enough
        # that a noisy prediction can't unmask vacuum cells, loose
        # enough that the slab interior stays unmasked at every step.
        floor = ref_max * 1.0e-2 if ref_max > 0 else 1.0
        plasma = ref > floor
        # Denominator floor avoids /0 in cells that happen to be masked
        # anyway.
        num_safe = np.where(num > 0.0, num,
                            floor if floor > 0 else 1.0)
        P, T = {}, {}
        for ax in ("x", "y", "z"):
            # The output channel `p_x = (m/V_cell) * Sum_p v_x` is
            # momentum density (mass kg / area m^2 / time s in 2D).
            # The output channel `enex = Sum_p (v_x^2 / 2 * gamma^-1)`
            # is a *velocity^2 * N* deposit (no mass factor in the
            # WarpX particle_fields formula). Combining:
            #   <v_x>   = p_x * V_cell / (m * num)        [m/s]
            #   <v_x^2> = 2 * enex / num                  [m^2/s^2]
            #   kT_x    = m * (<v_x^2> - <v_x>^2)         [J]
            #   P_ii    = (num / V_cell) * kT_x           [Pa]
            p_i   = np.asarray(moments[f"p{ax}_{sp}"],  dtype=np.float64)
            ene_i = np.asarray(moments[f"ene{ax}_{sp}"], dtype=np.float64)
            mean_v  = p_i * vcell / (mass * num_safe)
            mean_v2 = 2.0 * ene_i / num_safe
            var = np.clip(mean_v2 - mean_v * mean_v, 0.0, None)
            kT = mass * var
            T_ii_eV = kT * EV_PER_JOULE
            P_ii = (num_safe / vcell) * kT
            P[ax] = np.where(plasma, P_ii, 0.0)
            T[ax] = np.where(plasma, T_ii_eV, 0.0)
        out[sp] = {
            "Pxx": P["x"], "Pyy": P["y"], "Pzz": P["z"],
            "Tx":  T["x"], "Ty":  T["y"], "Tz":  T["z"],
        }
    return out


# ---------------------------------------------------------------------
# Grid plot
# ---------------------------------------------------------------------

def _zAverage(arr: np.ndarray) -> np.ndarray:
    """Average a per-step field over the z (axial) direction, leaving a
    1-D profile along x (the compression axis).

    The planar-pinch grid is (nx, nz) = (256, 32): x is the long
    256-cell compression axis, z is the short 32-cell axial axis along
    which the slab is statistically uniform. z is array axis 1, so the
    z-average is `mean(axis=1)` and the resulting profile runs along x
    -- the same horizontal axis the 2-D maps are displayed on (they use
    `arr.T`, so their horizontal axis is array axis 0 = x). Falls back
    to the flattened array for non-2-D inputs (other geometries)."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 2:
        with np.errstate(invalid="ignore"):
            return np.nanmean(a, axis=1)
    return a.ravel()


def plotComparisonGrid(out_path: Path,
                       channel_name: str,
                       rows: list,
                       row_labels: list[str],
                       ppc_in: int,
                       ppc_out: int,
                       units: str = "",
                       percentile: float = 2.0) -> None:
    """One PNG: N rows (one per step) x 8 columns.

    Left 4 columns are the 2-D maps (input, prediction, target,
    relative error). Right 4 columns mirror them as z-averaged 1-D
    profiles of the *same* quantities in the *same* units: all four
    profiles -- including the error -- share the row's 2-D colour-scale
    y-limits (lo, hi), so the error profile is the *absolute*
    (pred - target) plotted on the value axis and its magnitude is read
    directly against the signal. z is the short axial axis; the profile
    runs along x (see `_zAverage`).

    Each subpanel is ~2.5 x 2.2 in (8 columns -> 20 in wide), matching
    the per-panel size of the earlier 4-column layout; the page CSS
    widens the content column so the figure renders at native size.

    `rows` is a list of dicts with keys 'input', 'pred', 'target'
    holding the per-step 2-D arrays. The 2-D error column is the
    relative error `(pred - target) / max(RMS(target), eps)`; the 1-D
    error profile is the absolute error `pred - target`.
    """
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 8, figsize=(20.0, 2.2 * n_rows),
                             squeeze=False)

    col_titles = (f"input  (ppc {ppc_in})", "prediction",
                  f"target  (ppc {ppc_out})", "error / RMS(target)",
                  "input  (z-avg)", "prediction  (z-avg)",
                  "target  (z-avg)", "pred - target  (z-avg)")
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=9)

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

        # --- left 4 columns: 2-D maps ---
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

        # --- right 4 columns: z-averaged 1-D profiles ---
        # All four share the (lo, hi) value axis; the error profile is
        # the absolute (pred - target) so its size reads directly
        # against the signal, not a separate +/- scale.
        for jcol, (arr, color, is_err) in enumerate([
            (inp,       "C0", False),
            (prd,       "C0", False),
            (tgt,       "C0", False),
            (prd - tgt, "C3", True),
        ]):
            ax = axes[ridx][4 + jcol]
            prof = _zAverage(arr)
            xx = np.arange(prof.shape[0])
            ax.plot(xx, prof, color=color, linewidth=1.0)
            ax.set_xlim(0, max(1, prof.shape[0] - 1))
            ax.set_ylim(lo, hi)
            if is_err:
                ax.axhline(0.0, color="0.6", linewidth=0.6)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.25, linewidth=0.4)
            # All four profiles share the first's (lo, hi) scale, so
            # only the first carries y-ticks.
            if jcol in (1, 2, 3):
                ax.set_yticklabels([])
            if jcol == 0 and units:
                ax.set_ylabel(units, fontsize=7)
            # x-ticks only on the bottom row to cut clutter.
            if ridx == n_rows - 1:
                ax.set_xlabel("x cell", fontsize=7)
            else:
                ax.set_xticklabels([])

    suffix = f"  [{units}]" if units else ""
    fig.suptitle(f"{channel_name}{suffix}   "
                 f"ppc {ppc_in} -> {ppc_out}    "
                 f"(right half: z-averaged profiles along x)", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=90,
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
                    help="comma-separated step list; default = the four "
                         "canonical baseline-page phases "
                         "(ramp-up, mid-snowplow, peak compression, "
                         "rebound). Pass 'auto' for max-steps evenly "
                         "spaced across the pair instead.")
    ap.add_argument("--max-steps", type=int, default=4,
                    help="only used with --steps=auto")
    ap.add_argument("--variant-id", default="",
                    help="optional prefix for filenames: "
                         "comparison_<variant>_ppc<N>_to_ppc<M>_<ch>.png. "
                         "Defaults to empty -> comparison_ppc<N>_to_ppc<M>_<ch>.png.")
    args = ap.parse_args()

    from denoiseCore.config import load as loadConfig
    from denoiseCore.data.pairing import enumerateFactorPairs
    from denoiseCore.data.readers import getReader
    from denoiseCore.data.dataset import (FactorPairDataset,
                                          outputChannelNames,
                                          outputChannelIndices)
    from denoiseCore.device import getDevice
    from particleDenoiser.models import buildModel, isDiffusion
    from particleDenoiser.training import fitNormalizer
    from particleDenoiser.sampler import multiStepSample

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
    if ckpt.get("norm") is not None:
        norm = ckpt["norm"]
        print("[generateComparisons] using normalizer from checkpoint")
    else:
        print("[generateComparisons] normalizer not in checkpoint, "
              "refitting (top-ppc tier only)")
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

    # `--steps`: None or "" -> use the canonical baseline phases;
    # "auto" -> evenly-spaced fallback;
    # comma-list -> exact set.
    if args.steps is None or args.steps == "":
        want_steps = [s for s, _, _ in _DEFAULT_PHASES]
    elif args.steps.strip().lower() == "auto":
        want_steps = None
    else:
        want_steps = [int(s) for s in args.steps.split(",")]

    derived_species = cfg.data.channels.species_names
    can_derive = warpx_input is not None
    out_dir = Path(args.out_dir)
    # `comparison_<variant>_` if --variant-id set, else `comparison_`.
    prefix = f"comparison_{args.variant_id}_" if args.variant_id else "comparison_"

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
                if isDiffusion(cfg.model):
                    pred = multiStepSample(
                        model, x, s2i, s2o,
                        n_steps=cfg.model.diffusion_steps).squeeze(0)
                else:
                    pred = model(x, s2i, s2o).squeeze(0)

            inp_oc = s.input_tensor[out_indices]
            phys_in = denormChannels(inp_oc, out_names, norm)
            phys_pred = denormChannels(pred, out_names, norm)
            phys_tgt = denormChannels(s.target_tensor, out_names, norm)

            for n in out_names:
                raw_rows[n].append({"input": phys_in[n],
                                     "pred": phys_pred[n],
                                     "target": phys_tgt[n]})
            labels.append(_PHASE_LABEL.get(pair.step,
                                           f"step {pair.step}"))
            if can_derive:
                # Mask comes from the target's num arrays so all three
                # derivations (input / prediction / target) are masked
                # consistently and prediction artifacts in vacuum
                # cells cannot blow up T.
                derived_tgt.append(
                    deriveSpeciesPandT(phys_tgt, derived_species,
                                       warpx_input))
                derived_in.append(
                    deriveSpeciesPandT(phys_in, derived_species,
                                       warpx_input, num_ref=phys_tgt))
                derived_pred.append(
                    deriveSpeciesPandT(phys_pred, derived_species,
                                       warpx_input, num_ref=phys_tgt))

        # Raw output channels.
        for n in out_names:
            out_path = out_dir / f"{prefix}ppc{ppc_in}_to_ppc{ppc_out}_{n}.png"
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
                                f"{prefix}ppc{ppc_in}_to_ppc{ppc_out}_{name}.png")
                    plotComparisonGrid(out_path, name, rows, labels,
                                       ppc_in, ppc_out, units=units)

        print(f"  ppc {ppc_in} -> {ppc_out}: "
              f"{len(out_names) + (12 if can_derive else 0)} channels, "
              f"{len(group)} steps")

    print(f"comparison panels written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
