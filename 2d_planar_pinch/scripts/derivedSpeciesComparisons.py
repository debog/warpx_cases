"""Generate per-species derived-quantity (n, p, P, T) comparison PNGs
for the trained global denoisers across the four canonical phases.

Run from /home/ghosh/Runs/warpx/2d_planar_pinch/eval/ so the relative
warpx_input + run_dir paths in the YAML resolve.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch


# Phase steps and tiers documented in §4.1 of the working doc.
PHASES   = ["011025", "054900", "076950", "105300"]
TIERS    = ["ppc0036", "ppc0064", "ppc0144", "ppc0256"]
SPECIES  = ["electrons", "deuterium"]
COMPS    = ["x", "y", "z"]

# Output channels emitted by the model, in canonical order.
OUT_CHANNELS = [
    "jx", "jy", "jz",
    *(f"{m}_{sp}" for sp in SPECIES
        for m in ("num", "px", "py", "pz", "enex", "eney", "enez")),
]


def deriveQuantities(snap, warpx_input):
    """Compute n_alpha, p_alpha_i, T_alpha_i, P_alpha_i per cell.

    `snap` is a Snapshot with snap.species[sp] containing the raw
    deposited moments num (Σw), ux/uy/uz (Σ w v_i), enex/y/z (Σ w m v_i²/2),
    OR the dataset-derived px/py/pz = m·u_i/V_cell.
    """
    vcell = warpx_input.cell_volume()
    out: dict[str, dict[str, np.ndarray]] = {sp: {} for sp in SPECIES}
    for sp in SPECIES:
        mass = warpx_input.species_mass(sp)
        sd = snap.species[sp]
        num = np.asarray(sd["num"], dtype=np.float64)
        n_density = num / vcell                      # m^{-d}
        out[sp]["n"] = n_density
        # Sum-of-weights × v_i; reconstruct from either source.
        for i in COMPS:
            if f"u{i}" in sd:
                u_i = np.asarray(sd[f"u{i}"], dtype=np.float64)
            else:
                # p_i = m·u_i/V_cell => u_i = p_i·V_cell/m
                p_i = np.asarray(sd[f"p{i}"], dtype=np.float64)
                u_i = p_i * vcell / mass
            ene_i = np.asarray(sd[f"ene{i}"], dtype=np.float64)
            # p_density = m·u_i/V_cell
            out[sp][f"p{i}"] = mass * u_i / vcell
            # kT_i = (2 ene_i / num) - m (u_i / num)^2  [per particle, with finite-num guard]
            with np.errstate(invalid="ignore", divide="ignore"):
                meanV   = np.where(num > 0.0, u_i / num, 0.0)
                meanVsq = np.where(num > 0.0, 2.0 * ene_i / (mass * num), 0.0)
                varV = meanVsq - meanV * meanV
                varV = np.where(varV > 0.0, varV, 0.0)
                kT = mass * varV                        # J
            out[sp][f"T{i}"] = kT
            out[sp][f"P{i}"] = n_density * kT             # Pa (in 3D); Pa·m in 2D etc.
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",     required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir",    required=True)
    args = ap.parse_args()

    from denoiseCore.config            import load as loadConfig
    from denoiseCore.data.readers      import getReader
    from denoiseCore.data.pairing      import enumeratePairs
    from denoiseCore.data.dataset      import (snapshotToInputTensor,
                                               outputChannelNames,
                                               outputChannelIndices,
                                               _rawMomentsToLoad,
                                               _augmentSnapshotWithMomentum)
    from denoiseCore.device            import getDevice, ampContext
    from denoiseCore.plotting          import plotPredictionTriptych
    from denoiseCore.warpxInput        import load as loadWarpXInput
    from particleDenoiser.models       import buildModel
    from particleDenoiser.training     import fitNormalizerFromGroups
    from copy                          import deepcopy

    cfg     = loadConfig(args.config)
    dev     = getDevice()
    ckpt    = torch.load(args.checkpoint, map_location=dev.device, weights_only=False)
    reader  = getReader(cfg.data.reader)
    groups  = enumeratePairs(reader, cfg.data.ppc_tiers, cfg.data.steps)
    wxi     = loadWarpXInput(cfg.data.warpx_input)
    raw_mom = _rawMomentsToLoad(cfg.data.channels.species)
    # Always load enough to compute derived quantities even if the
    # channel list omits px/py/pz on the species side (it does not in
    # this case, but be defensive).
    full_mom = sorted(set(raw_mom) | {"num", "ux", "uy", "uz",
                                       "enex", "eney", "enez"})
    norm    = fitNormalizerFromGroups(reader, groups,
                                      cfg.data.channels,
                                      cfg.data.normalization,
                                      warpx_input=wxi)

    gt_name = next(t.name for t in cfg.data.ppc_tiers if t.ground_truth)
    probe = reader.load(groups[0].tier_infos[gt_name],
                        cfg.data.channels.fields,
                        raw_mom, cfg.data.channels.species_names)
    _augmentSnapshotWithMomentum(probe, cfg.data.channels.species,
                                 cfg.data.channels.species_names, wxi)
    probe_x = snapshotToInputTensor(probe, cfg.data.channels, norm)
    in_channels  = probe_x.shape[0]
    out_indices  = outputChannelIndices(cfg.data.channels)
    out_names    = outputChannelNames(cfg.data.channels)
    out_channels = len(out_names)
    dim          = probe_x.ndim - 1
    model = buildModel(cfg.model, in_channels=in_channels,
                       out_channels=out_channels,
                       out_indices=out_indices, dim=dim,
                       boundary_spec=cfg.data.boundary_spec).to(dev.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Output-channel denormalization keys.
    species_set = set(cfg.data.channels.species_names)
    out_norm_keys: list[str] = []
    for ch in out_names:
        if "_" in ch:
            head, tail = ch.rsplit("_", 1)
            out_norm_keys.append(head if tail in species_set else ch)
        else:
            out_norm_keys.append(ch)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    step_to_group = {g.step: g for g in groups}
    for tier in TIERS:
        for step_str in PHASES:
            step = int(step_str)
            if step not in step_to_group:
                continue
            g = step_to_group[step]
            if tier not in g.tier_infos:
                continue

            # Low-ppc input snapshot (raw + augmented).
            lo_info = g.tier_infos[tier]
            lo_snap = reader.load(lo_info, cfg.data.channels.fields,
                                  full_mom, cfg.data.channels.species_names)
            _augmentSnapshotWithMomentum(lo_snap, cfg.data.channels.species,
                                         cfg.data.channels.species_names, wxi)
            # Ground truth (full moment set for derived computations).
            gt_info = g.tier_infos[gt_name]
            gt_snap = reader.load(gt_info, cfg.data.channels.fields,
                                  full_mom, cfg.data.channels.species_names)
            _augmentSnapshotWithMomentum(gt_snap, cfg.data.channels.species,
                                         cfg.data.channels.species_names, wxi)

            # Build the input tensor from the channel set the network expects.
            x_norm = snapshotToInputTensor(lo_snap, cfg.data.channels, norm)
            x = torch.from_numpy(x_norm).unsqueeze(0).to(dev.device)
            with torch.no_grad(), ampContext(dev, cfg.train.amp, cfg.train.amp_dtype):
                pred_norm = model(x)
            pred_arr = pred_norm.squeeze(0).float().cpu().numpy()

            # Build a "prediction snapshot" (deep copy of lo_snap, then
            # replace output channels with the denormalized prediction).
            # Field-side prediction lives under snap.fields[<name>]; the
            # species-side prediction lives under snap.species[sp][<mom>].
            pred_snap = deepcopy(lo_snap)
            n_field = len(cfg.data.channels.output_fields)
            for k, ch in enumerate(out_names):
                arr = norm.invert(out_norm_keys[k], pred_arr[k])
                if k < n_field:
                    pred_snap.fields[cfg.data.channels.output_fields[k]] = arr
                else:
                    j = k - n_field
                    n_mom = len(cfg.data.channels.output_species_moments)
                    sp_idx, mom_idx = divmod(j, n_mom)
                    sp = cfg.data.channels.species_names[sp_idx]
                    mom = cfg.data.channels.output_species_moments[mom_idx]
                    pred_snap.species[sp][mom] = arr

            # Derived quantities from raw moments + (for pred) the
            # network's predicted px/py/pz overwriting the input's
            # dataset-derived px/py/pz. The pred enex/y/z and num come
            # from the model too. ux/uy/uz on the pred side are unchanged
            # from the input (they are not output channels), so the
            # temperature computation should be based on p_i, not u_i,
            # for the pred snapshot.
            d_lo = deriveQuantities(lo_snap, wxi)
            d_pred = deriveQuantities(pred_snap, wxi)
            d_gt = deriveQuantities(gt_snap, wxi)

            # Case-collision-safe filenames so PNGs round-trip cleanly
            # through case-insensitive filesystems (rsync to macOS etc.).
            FILE_NAME = {
                "n":  "ndensity",
                "px": "mom_x", "py": "mom_y", "pz": "mom_z",
                "Px": "press_x", "Py": "press_y", "Pz": "press_z",
                "Tx": "temp_x",  "Ty": "temp_y",  "Tz": "temp_z",
            }
            for sp in SPECIES:
                step_dir = out_dir / tier / f"step_{step:06d}" / sp
                step_dir.mkdir(parents=True, exist_ok=True)
                for qkey, fname in FILE_NAME.items():
                    plotPredictionTriptych(
                        step_dir / f"{fname}.png",
                        channel_name=f"{fname}_{sp}",
                        input_arr=d_lo[sp][qkey],
                        prediction_arr=d_pred[sp][qkey],
                        target_arr=d_gt[sp][qkey],
                        relative_error=True,
                    )
            print(f"  {tier} step {step_str}: derived PNGs written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
