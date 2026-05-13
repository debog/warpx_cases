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

# Module-level imports from denoiseCore (path injected before running).
import sys
sys.path.insert(0, "/home/ghosh/Codes/particle-denoise")
from denoiseCore.data.dataset import (
    _augmentSnapshotDerived,
    _rawMomentsToLoad,
)


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


# Conversion constant: 1 eV in Joules. Temperature reports in eV.
EV_PER_JOULE = 1.0 / 1.602176634e-19

_LOG_CHANNELS = {"num", "part_per_cell"}


def fitPerSpeciesStats(reader, groups, channels, gt_name, warpx_input):
    """Per-(species, moment) z-score / log-z-score stats from GT
    snapshots. Returns {(species, moment): (mean, std, kind)}.

    The denoiseCore Normalizer pools species under one channel-name
    key, so for two-species cases with very different per-species
    scales (electrons / deuterium in this planar pinch) the pooled
    statistics are dominated by one species and the inverse on the
    other species is offset by orders of magnitude. This helper fits
    the per-species stats needed for an interim post-hoc rescale of
    the model's predictions; a proper fix is per-(species, moment)
    stats in the dataset itself plus retraining.
    """
    accum: dict[tuple[str, str], list[np.ndarray]] = {}
    # Raw moments the reader has on disk are everything in channels.species
    # minus the derived names (px/py/pz, Pxx/Pyy/Pzz, Tx/Ty/Tz), plus
    # whatever raw deposits those derived names depend on.
    raw_mom = sorted(set(_rawMomentsToLoad(channels.species))
                     | {"num", "ux", "uy", "uz", "enex", "eney", "enez"})
    for g in groups:
        info = g.tier_infos[gt_name]
        snap = reader.load(info, channels.fields, raw_mom, channels.species_names)
        _augmentSnapshotDerived(snap, channels.species,
                                     channels.species_names, warpx_input)
        for sp in channels.species_names:
            for mom in channels.species:
                accum.setdefault((sp, mom), []).append(
                    np.asarray(snap.species[sp][mom]).ravel())
    out: dict[tuple[str, str], tuple[float, float, str]] = {}
    for (sp, mom), parts in accum.items():
        x = np.concatenate(parts)
        kind = "log_zscore" if mom in _LOG_CHANNELS else "zscore"
        if kind == "log_zscore":
            x = np.log10(np.maximum(x, 1.0))
        out[(sp, mom)] = (float(x.mean()), float(x.std() or 1.0), kind)
    return out


def perSpeciesInvert(arr, mean: float, std: float, kind: str) -> np.ndarray:
    """Invert a normalized prediction using per-(species, moment) stats."""
    x = arr * std + mean
    if kind == "log_zscore":
        x = np.power(10.0, x)
    return x


def deriveQuantities(snap, warpx_input):
    """Compute n_alpha, p_alpha_i, T_alpha_i, P_alpha_i per cell from
    snap.species[sp] keyed on (num, px/py/pz, enex/y/z).

    WarpX `particle_fields` deposit conventions for this case (see
    inputs/planar_pinch_2d.in):
      num      = Σ w                                   (particles)
      ux/y/z   = Σ w (γ v_i)                           (m/s, ≈ v_i non-relat.)
      enex/y/z = Σ w (½ v_i² × γ-factor)               (m²/s², NO m factor)

    The kinetic temperature uses
      ⟨v_i²⟩  = 2 ene_i / num
      ⟨v_i⟩  = u_i / num             (u_i = p_i · V_cell / m)
      kT_i    = m × (⟨v_i²⟩ − ⟨v_i⟩²)
    Reported in eV. Pressure P_i = n × kT_i in Pa (J/m³).

    Always reads momentum density p_i, never the raw u_i: a prediction
    snapshot has the model's p_i but the dataset-augmented u_i is a
    leftover from the low-ppc input. Reading u_i directly would mix
    input-noise u_i with prediction-smooth num and yield T/P that
    look like noise hallucinations. For lo/gt snapshots p_i = m·u_i/V
    carries identical information; only the pred path is affected.
    """
    vcell = warpx_input.cell_volume()
    out: dict[str, dict[str, np.ndarray]] = {sp: {} for sp in SPECIES}
    for sp in SPECIES:
        mass = warpx_input.species_mass(sp)
        sd = snap.species[sp]
        num = np.asarray(sd["num"], dtype=np.float64)
        n_density = num / vcell
        out[sp]["n"] = n_density
        for i in COMPS:
            p_i = np.asarray(sd[f"p{i}"], dtype=np.float64)
            u_i = p_i * vcell / mass                  # Σ w v_i, m/s × particles
            ene_i = np.asarray(sd[f"ene{i}"], dtype=np.float64)
            out[sp][f"p{i}"] = p_i
            out[sp][f"e{i}"] = ene_i                  # raw WarpX deposit (≈ ½ v_i² Σw)
            with np.errstate(invalid="ignore", divide="ignore"):
                meanV   = np.where(num > 0.0, u_i / num, 0.0)
                meanVsq = np.where(num > 0.0, 2.0 * ene_i / num, 0.0)
                varV = meanVsq - meanV * meanV
                varV = np.where(varV > 0.0, varV, 0.0)
                kT_J = mass * varV
            out[sp][f"T{i}"] = kT_J * EV_PER_JOULE        # eV
            out[sp][f"P{i}"] = n_density * kT_J           # Pa
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
                                               outputChannelIndices)
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
    # Interim: per-species stats fit on the GT tier for post-hoc rescale
    # of the model's species-side predictions. The denoiseCore Normalizer
    # pools species, which for this 2-species case (m_e ≪ m_D) makes the
    # absolute-units inverse meaningless for the smaller-scale species.
    gt_name = next(t.name for t in cfg.data.ppc_tiers if t.ground_truth)
    ps_stats = fitPerSpeciesStats(reader, groups, cfg.data.channels,
                                  gt_name, wxi)
    probe = reader.load(groups[0].tier_infos[gt_name],
                        cfg.data.channels.fields,
                        raw_mom, cfg.data.channels.species_names)
    _augmentSnapshotDerived(probe, cfg.data.channels.species,
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
            _augmentSnapshotDerived(lo_snap, cfg.data.channels.species,
                                         cfg.data.channels.species_names, wxi)
            # Ground truth (full moment set for derived computations).
            gt_info = g.tier_infos[gt_name]
            gt_snap = reader.load(gt_info, cfg.data.channels.fields,
                                  full_mom, cfg.data.channels.species_names)
            _augmentSnapshotDerived(gt_snap, cfg.data.channels.species,
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
                if k < n_field:
                    arr = norm.invert(out_norm_keys[k], pred_arr[k])
                    pred_snap.fields[cfg.data.channels.output_fields[k]] = arr
                else:
                    j = k - n_field
                    n_mom = len(cfg.data.channels.output_species_moments)
                    sp_idx, mom_idx = divmod(j, n_mom)
                    sp = cfg.data.channels.species_names[sp_idx]
                    mom = cfg.data.channels.output_species_moments[mom_idx]
                    # Interim per-species denormalization: take the model's
                    # output in normalized (pooled) space and apply per-
                    # species (mean, std) stats fit on the GT tier. The
                    # pooled inverse used by `norm.invert` puts a smaller-
                    # scale species at the dominant species' scale, which
                    # is the source of the wildly-off pressure/temperature.
                    mean_sp, std_sp, kind_sp = ps_stats[(sp, mom)]
                    pred_snap.species[sp][mom] = perSpeciesInvert(
                        pred_arr[k], mean_sp, std_sp, kind_sp)

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
                "px": "mom_x",   "py": "mom_y",   "pz": "mom_z",
                "ex": "ene_x",   "ey": "ene_y",   "ez": "ene_z",
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
