#!/usr/bin/env python3
"""Generate 4x4 comparison-grid figures for a trained denoiser.

Layout: one figure per (variant, ppc tier, output/derived channel).
Each figure is a 4-row x 4-column grid:

    row r   = canonical phase snapshot r in time
    col 0   = low-ppc input
    col 1   = model prediction
    col 2   = ppc1024 ground truth
    col 3   = relative error  =  (pred - GT) / RMS(GT)

The first three columns share a per-row (vmin, vmax) so input /
prediction / GT can be compared at a glance. The error column uses
a symmetric +/-rmax scale (RdBu_r) so blue/red intensity is read as
the per-cell error in units of one RMS(GT).

Channel set per (variant, tier):

    raw   (17)  : jx, jy, jz,
                  num/<p|ene><x|y|z>_<species>
    derived (12) : Pxx/Pyy/Pzz/Tx/Ty/Tz per species (computed from
                   predicted moments via the moment-closure formula)

Inference dispatch:

    --cli global   particleDenoiser.models.buildModel + model(x)
    --cli local    localParticleDenoiser models + patch-tile inference
    --cli flow     flowParticleDenoiser FlowMatchingUNet + ODE sampler

Output filenames are flat under <out-dir>:

    comparison_<variant>_<tier>_<channel>.png

Save once, regenerate by re-running with new --checkpoint values.
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

# Path injection: local repo, not the installed package.
sys.path.insert(0, "/home/ghosh/Codes/particle-denoise")


# Canonical phases / tiers for the planar pinch case (see §4.1 of the
# working doc).
PHASES = [
    ("011025", "Ramp-up"),
    ("054900", "Peak compression"),
    ("076950", "Post-peak"),
    ("105300", "Rebound"),
]
TIERS_DEFAULT = ["ppc0036", "ppc0064", "ppc0144", "ppc0256"]

# Derived-channel names used in the output PNG filenames.
DERIVED_CHANNELS = [
    "Pxx_electrons", "Pyy_electrons", "Pzz_electrons",
    "Tx_electrons",  "Ty_electrons",  "Tz_electrons",
    "Pxx_deuterium", "Pyy_deuterium", "Pzz_deuterium",
    "Tx_deuterium",  "Ty_deuterium",  "Tz_deuterium",
]

EV_PER_JOULE = 1.0 / 1.602176634e-19


# ---------------------------------------------------------------------
# Inference dispatch
# ---------------------------------------------------------------------

def buildModelAndForwarder(cli_kind: str, cfg, ckpt, dev,
                           in_channels: int, out_channels: int,
                           out_indices, dim: int, patch_dataset=None):
    """Build the right model and return `(model, forward_fn)` where
    `forward_fn(x_norm_full_tensor) -> pred_norm_full_tensor` runs
    one inference pass on a whole snapshot. For local denoisers the
    patch-tile loop is hidden inside `forward_fn`."""
    from denoiseCore.device import ampContext

    if cli_kind == "global":
        from particleDenoiser.models import buildModel as buildGlobal
        model = buildGlobal(
            cfg.model,
            in_channels=in_channels, out_channels=out_channels,
            out_indices=out_indices, dim=dim,
            boundary_spec=cfg.data.boundary_spec,
        ).to(dev.device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        def fwd(x):  # x: (C_in, ...)
            xb = x.unsqueeze(0).to(dev.device)
            with torch.no_grad(), ampContext(dev, cfg.train.amp,
                                             cfg.train.amp_dtype):
                p = model(xb)
            return p.squeeze(0).float().cpu()

        return model, fwd

    if cli_kind == "local":
        from localParticleDenoiser.models import buildModel as buildLocal
        from localParticleDenoiser.cli import _denoiseSnapshotByPatches
        model = buildLocal(
            cfg.model,
            in_channels=in_channels, out_channels=out_channels,
            out_indices=out_indices, dim=dim,
        ).to(dev.device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        assert patch_dataset is not None
        boundary_aux = (
            (lambda coords: patch_dataset._boundaryAuxChannels(
                coords, dtype=torch.float32, device=dev.device))
            if cfg.data.boundary_channels else None
        )

        def fwd(x):
            padded = patch_dataset._padInput(x)
            return _denoiseSnapshotByPatches(
                model, padded, x,
                patch_size=cfg.model.patch_size,
                dev=dev, amp_enabled=cfg.train.amp,
                amp_dtype=cfg.train.amp_dtype,
                boundary_aux=boundary_aux,
            )

        return model, fwd

    if cli_kind == "flow":
        from flowParticleDenoiser.models import FlowMatchingUNet
        from flowParticleDenoiser.sampler import eulerSample, heunSample
        model = FlowMatchingUNet(
            in_channels=in_channels, out_channels=out_channels,
            out_indices=out_indices,
            base_channels=cfg.model.base_channels,
            depth=cfg.model.depth, kernel=cfg.model.kernel,
            activation=cfg.model.activation,
            time_embed_dim=cfg.model.time_embed_dim, dim=dim,
        ).to(dev.device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        sampler = (eulerSample if cfg.model.sampler_kind == "euler"
                   else heunSample)
        steps = cfg.model.sampler_steps

        def fwd(x):
            xb = x.unsqueeze(0).to(dev.device)
            with torch.no_grad(), ampContext(dev, cfg.train.amp,
                                             cfg.train.amp_dtype):
                p_full = sampler(model, xb, steps=steps)
            # Subset to output channels.
            return (p_full[0].index_select(0, model.out_indices)
                              .float().cpu())

        return model, fwd

    raise ValueError(f"unknown --cli kind {cli_kind!r}; "
                     f"expected one of: global, local, flow")


# ---------------------------------------------------------------------
# Derived quantities (P, T) per cell, per species
# ---------------------------------------------------------------------

def deriveSpeciesPandT(snap, warpx_input) -> Dict[str, Dict[str, np.ndarray]]:
    """Return {species: {Pxx, Pyy, Pzz, Tx, Ty, Tz}} from a snapshot
    whose .species[sp] dict already carries the raw moments
    (num, px, py, pz, enex, eney, enez) in physical units.

    Pressure tensor diagonal: P_ii = (m/V) * num * (<v_i^2> - <v_i>^2)
    where <v_i>   = p_i / (m * num)
          <v_i^2> = 2 * ene_i / (m * num)
    Temperature : T_i = P_i * V / num   (then converted to eV)
    """
    vcell = float(warpx_input.cell_volume())
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for sp in snap.species:
        mass = float(warpx_input.species_mass(sp))
        d = snap.species[sp]
        num  = np.asarray(d["num"], dtype=np.float64)
        # `num` may briefly be 0 in sparse cells; floor to keep
        # divisions sane and let the resulting tiny weights die on
        # their own.
        num_safe = np.where(num > 0.0, num, 1.0)
        P = {}
        T = {}
        for ax in ("x", "y", "z"):
            p_i   = np.asarray(d[f"p{ax}"],  dtype=np.float64)
            ene_i = np.asarray(d[f"ene{ax}"], dtype=np.float64)
            mean_v   = p_i / (mass * num_safe)
            mean_v2  = 2.0 * ene_i / (mass * num_safe)
            var = np.clip(mean_v2 - mean_v * mean_v, 0.0, None)
            P_ii = (mass / vcell) * num * var
            T_ii = (P_ii * vcell / np.where(num > 0.0, num, np.nan))
            T_ii_eV = T_ii * EV_PER_JOULE
            T_ii_eV = np.where(np.isfinite(T_ii_eV), T_ii_eV, 0.0)
            P[ax] = P_ii
            T[ax] = T_ii_eV
        out[sp] = {
            "Pxx": P["x"], "Pyy": P["y"], "Pzz": P["z"],
            "Tx":  T["x"], "Ty":  T["y"], "Tz":  T["z"],
        }
    return out


# ---------------------------------------------------------------------
# 4x4 grid plot
# ---------------------------------------------------------------------

def plotComparisonGrid(out_path: Path,
                       channel_name: str,
                       rows: list,
                       phase_labels: list[str],
                       units: str = "",
                       percentile: float = 2.0) -> None:
    """One PNG: 4 rows (phase snapshots) x 4 columns (input, pred, GT,
    relative error).

    `rows` is a list of 4 dicts, each with keys 'input', 'pred', 'gt'
    holding the per-snapshot 2-D arrays (or 1-D arrays for the 1-D
    case). Relative error is computed per-row as
    `(pred - gt) / max(RMS(gt), eps)`.

    First three columns share a per-row [percentile, 100-percentile]
    color scale. The error column uses a symmetric +/-rmax computed
    over the relative-error array for that row.
    """
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 4, figsize=(10.0, 2.2 * n_rows),
                             squeeze=False)

    col_titles = ("low-ppc input", "prediction",
                  "ground truth", "error / RMS(GT)")
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=10)

    for ridx, row in enumerate(rows):
        inp = np.asarray(row["input"])
        prd = np.asarray(row["pred"])
        gt  = np.asarray(row["gt"])

        # Common [percentile, 100-percentile] scale across the three
        # value panels.
        stacked = np.concatenate([a.ravel() for a in (inp, prd, gt)])
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

        # Per-row RMS(GT); floor to avoid 0/0.
        gt_finite = gt.ravel()[np.isfinite(gt.ravel())]
        gt_rms = (float(np.sqrt(np.mean(gt_finite ** 2)))
                  if gt_finite.size else 1.0)
        eps = 1.0e-30 * (float(np.abs(gt_finite).max())
                         if gt_finite.size else 1.0)
        scale = max(gt_rms, eps)
        rel_err = (prd - gt) / scale

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
            (gt,      lo,    hi,   "viridis"),
            (rel_err, -rmax, rmax, "RdBu_r"),
        ]):
            ax = axes[ridx][cidx]
            disp = arr.T if arr.ndim == 2 else arr[None, :]
            im = ax.imshow(disp, origin="lower", aspect="auto",
                           vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            if cidx == 0:
                ax.set_ylabel(phase_labels[ridx], fontsize=9)

    suffix = f"  [{units}]" if units else ""
    fig.suptitle(f"{channel_name}{suffix}", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=80,
                pil_kwargs={"optimize": True, "compress_level": 9})
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",     required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--variant-id", required=True,
                    help="short name used in output filenames")
    ap.add_argument("--cli", required=True,
                    choices=("global", "local", "flow"),
                    help="which model factory to use (matches the "
                         "trained denoiser variant)")
    ap.add_argument("--out-dir",    required=True)
    ap.add_argument("--tiers",  default=",".join(TIERS_DEFAULT))
    ap.add_argument("--steps",  default=",".join(s for s, _ in PHASES))
    args = ap.parse_args()

    from denoiseCore.config       import load as loadConfig
    from denoiseCore.data.dataset import (
        outputChannelNames, outputChannelIndices,
        snapshotToInputTensor,
        _rawMomentsToLoad, _augmentSnapshotDerived,
    )
    from denoiseCore.data.readers import getReader
    from denoiseCore.data.pairing import enumeratePairs
    from denoiseCore.device       import getDevice
    from denoiseCore.warpxInput   import load as loadWarpXInput

    cfg = loadConfig(args.config)
    dev = getDevice()
    ckpt = torch.load(args.checkpoint, map_location=dev.device,
                      weights_only=False)
    reader = getReader(cfg.data.reader)
    groups = enumeratePairs(reader, cfg.data.ppc_tiers, cfg.data.steps)
    wxi = loadWarpXInput(cfg.data.warpx_input) if cfg.data.warpx_input else None
    raw_mom_required = _rawMomentsToLoad(cfg.data.channels.species)
    # Load enough on-disk moments to compute the derived (P, T) per
    # species. Raw on-disk moment names are `ux/uy/uz` (not px/py/pz);
    # the snapshot becomes px/py/pz after `_augmentSnapshotDerived`.
    full_mom = sorted(set(raw_mom_required) | {
        "num", "ux", "uy", "uz", "enex", "eney", "enez"})

    # Need the existing training-side helper to fit the normalizer
    # (paths diverge by variant; use the global one which has no
    # patch-specific extras).
    if args.cli == "global":
        from particleDenoiser.training import fitNormalizerFromGroups
    elif args.cli == "local":
        from denoiseCore.trainingLoop import fitNormalizerFromGroups
    else:  # flow
        from flowParticleDenoiser.training import fitNormalizerFromGroups
    norm = fitNormalizerFromGroups(reader, groups,
                                   cfg.data.channels,
                                   cfg.data.normalization,
                                   warpx_input=wxi)

    gt_name = next(t.name for t in cfg.data.ppc_tiers if t.ground_truth)
    tiers_wanted = [t.strip() for t in args.tiers.split(",") if t.strip()]
    steps_wanted = [int(s) for s in args.steps.split(",")]

    # Probe to size the model.
    probe_info = groups[0].tier_infos[gt_name]
    probe_snap = reader.load(probe_info, cfg.data.channels.fields,
                             raw_mom_required,
                             cfg.data.channels.species_names)
    _augmentSnapshotDerived(probe_snap, cfg.data.channels.species,
                            cfg.data.channels.species_names, wxi)
    probe_x = snapshotToInputTensor(probe_snap, cfg.data.channels, norm)
    in_base = probe_x.shape[0]
    out_names   = outputChannelNames(cfg.data.channels)
    out_indices = outputChannelIndices(cfg.data.channels)
    out_channels = len(out_names)
    dim = probe_x.ndim - 1

    patch_dataset = None
    aux_chans = 0
    if args.cli == "local":
        from denoiseCore.data.dataset import PairedPICDataset
        from localParticleDenoiser.data import PatchDataset
        aux_chans = (len(cfg.data.boundary_spec)
                     if cfg.data.boundary_channels else 0)
        base = PairedPICDataset(groups, reader, cfg.data.channels, norm,
                                cache=cfg.data.cache_snapshots,
                                cache_max_gib=cfg.data.cache_max_gib,
                                warpx_input=wxi)
        patch_dataset = PatchDataset(
            base, patch_size=cfg.model.patch_size,
            boundary_spec=cfg.data.boundary_spec,
            boundary_channels=cfg.data.boundary_channels)
    in_channels = in_base + aux_chans

    model, fwd = buildModelAndForwarder(
        args.cli, cfg, ckpt, dev,
        in_channels=in_channels, out_channels=out_channels,
        out_indices=out_indices,
        dim=dim, patch_dataset=patch_dataset,
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    step_to_group = {g.step: g for g in groups}
    field_names   = cfg.data.channels.output_fields
    moment_names  = cfg.data.channels.output_species_moments
    species_names = cfg.data.channels.species_names

    # Step 1: for each (tier, step) collect the input, prediction, and
    # GT *physical-units* arrays per (raw output channel) and recompute
    # the derived (P, T) per species on each side.
    # Cache by (tier, step) so we can stack across phases per channel.
    Snap = dict  # type alias for clarity
    cache: dict[str, dict[int, Snap]] = {t: {} for t in tiers_wanted}

    for step in steps_wanted:
        if step not in step_to_group:
            print(f"  skipping step {step}: not in any pair group")
            continue
        g = step_to_group[step]
        # Ground truth (load all moments needed for derived).
        gt_info = g.tier_infos[gt_name]
        gt_snap = reader.load(gt_info, cfg.data.channels.fields,
                              full_mom, cfg.data.channels.species_names)
        _augmentSnapshotDerived(gt_snap, cfg.data.channels.species,
                                cfg.data.channels.species_names, wxi)
        gt_derived = deriveSpeciesPandT(gt_snap, wxi)

        for tier in tiers_wanted:
            if tier not in g.tier_infos:
                print(f"  skipping tier {tier} step {step}: not present")
                continue
            lo_info = g.tier_infos[tier]
            lo_snap = reader.load(lo_info, cfg.data.channels.fields,
                                  full_mom,
                                  cfg.data.channels.species_names)
            _augmentSnapshotDerived(lo_snap, cfg.data.channels.species,
                                    cfg.data.channels.species_names,
                                    wxi)
            lo_derived = deriveSpeciesPandT(lo_snap, wxi)

            # Inference.
            x_norm = torch.from_numpy(
                snapshotToInputTensor(lo_snap, cfg.data.channels, norm))
            pred_norm = fwd(x_norm)
            pred_arr = pred_norm.numpy()

            # Build a prediction snapshot: deep copy lo_snap, then
            # overwrite output channels with denormalized predictions.
            pred_snap = deepcopy(lo_snap)
            n_field = len(field_names)
            for k, ch in enumerate(out_names):
                phys = norm.invert(ch, pred_arr[k])
                if k < n_field:
                    pred_snap.fields[field_names[k]] = phys
                else:
                    j = k - n_field
                    sp_idx, mom_idx = divmod(j, len(moment_names))
                    sp  = species_names[sp_idx]
                    mom = moment_names[mom_idx]
                    pred_snap.species[sp][mom] = phys
            pred_derived = deriveSpeciesPandT(pred_snap, wxi)

            # Stash per-channel physical arrays for this (tier, step).
            phys: dict[str, dict[str, np.ndarray]] = {
                "input": {}, "pred": {}, "gt": {},
            }
            # Raw output channels.
            for k, ch in enumerate(out_names):
                if k < n_field:
                    phys["input"][ch] = lo_snap.fields[field_names[k]]
                    phys["pred"][ch]  = pred_snap.fields[field_names[k]]
                    phys["gt"][ch]    = gt_snap.fields[field_names[k]]
                else:
                    j = k - n_field
                    sp_idx, mom_idx = divmod(j, len(moment_names))
                    sp  = species_names[sp_idx]
                    mom = moment_names[mom_idx]
                    phys["input"][ch] = lo_snap.species[sp][mom]
                    phys["pred"][ch]  = pred_snap.species[sp][mom]
                    phys["gt"][ch]    = gt_snap.species[sp][mom]
            # Derived channels (per species).
            for sp in species_names:
                for q in ("Pxx", "Pyy", "Pzz", "Tx", "Ty", "Tz"):
                    name = f"{q}_{sp}"
                    phys["input"][name] = lo_derived[sp][q]
                    phys["pred"][name]  = pred_derived[sp][q]
                    phys["gt"][name]    = gt_derived[sp][q]
            cache[tier][step] = phys
            print(f"  inferred {tier} step {step}")

    # Step 2: assemble per-(tier, channel) 4x4 plots.
    all_channels = list(out_names) + list(DERIVED_CHANNELS)
    phase_labels = [f"{lab} (step {st})" for st, lab in PHASES]

    n_written = 0
    for tier in tiers_wanted:
        for ch in all_channels:
            rows = []
            step_labels = []
            for st, lab in PHASES:
                step = int(st)
                if step not in cache[tier]:
                    continue
                rows.append({
                    "input": cache[tier][step]["input"][ch],
                    "pred":  cache[tier][step]["pred"][ch],
                    "gt":    cache[tier][step]["gt"][ch],
                })
                step_labels.append(f"{lab}\n(step {st})")
            if not rows:
                continue
            out_path = out_dir / (
                f"comparison_{args.variant_id}_{tier}_{ch}.png")
            plotComparisonGrid(out_path, channel_name=ch,
                               rows=rows, phase_labels=step_labels)
            n_written += 1
    print(f"wrote {n_written} comparison grids -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
