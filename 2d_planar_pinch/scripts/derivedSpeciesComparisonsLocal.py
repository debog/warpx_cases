"""Generate per-species derived-quantity (n, p, P, T) comparison PNGs
for the trained LOCAL denoiser across the four canonical phases.

Mirrors `derivedSpeciesComparisons.py` (which targets the global U-Net)
but builds the local model + patch-based reassembly via
`localParticleDenoiser._denoiseSnapshotByPatches`. Same derived
formulae and plot layout, so the resulting PNGs are interchangeable
with the global helper's output for downstream tooling.

Run from /home/ghosh/Runs/warpx/2d_planar_pinch/eval/ so the relative
warpx_input + run_dir paths in the YAML resolve.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, "/home/ghosh/Codes/particle-denoise")
sys.path.insert(0, "/home/ghosh/Runs/warpx/2d_planar_pinch/scripts")
# Reuse the global helper's derived-quantity formulae and per-species
# rescale logic; only the model + prediction path differs.
from derivedSpeciesComparisons import (
    PHASES, TIERS, SPECIES, COMPS,
    EV_PER_JOULE, _LOG_CHANNELS,
    fitPerSpeciesStats, perSpeciesInvert, deriveQuantities,
)
from denoiseCore.data.dataset import (_augmentSnapshotDerived,
                                      _rawMomentsToLoad)


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
                                               PairedPICDataset)
    from denoiseCore.device            import getDevice
    from denoiseCore.plotting          import plotPredictionTriptych
    from denoiseCore.warpxInput        import load as loadWarpXInput
    from localParticleDenoiser.models  import buildModel
    from localParticleDenoiser.data    import PatchDataset
    from localParticleDenoiser.cli     import _denoiseSnapshotByPatches
    from particleDenoiser.training     import fitNormalizerFromGroups

    cfg     = loadConfig(args.config)
    dev     = getDevice()
    ckpt    = torch.load(args.checkpoint, map_location=dev.device,
                         weights_only=False)
    reader  = getReader(cfg.data.reader)
    groups  = enumeratePairs(reader, cfg.data.ppc_tiers, cfg.data.steps)
    wxi     = loadWarpXInput(cfg.data.warpx_input)
    raw_mom = _rawMomentsToLoad(cfg.data.channels.species)
    full_mom = sorted(set(raw_mom) | {"num", "ux", "uy", "uz",
                                       "enex", "eney", "enez"})
    norm    = fitNormalizerFromGroups(reader, groups,
                                      cfg.data.channels,
                                      cfg.data.normalization,
                                      warpx_input=wxi)
    gt_name = next(t.name for t in cfg.data.ppc_tiers if t.ground_truth)
    ps_stats = fitPerSpeciesStats(reader, groups, cfg.data.channels,
                                  gt_name, wxi)

    # Probe one snapshot to size the model.
    probe_info = groups[0].tier_infos[gt_name]
    probe = reader.load(probe_info, cfg.data.channels.fields,
                        raw_mom, cfg.data.channels.species_names)
    _augmentSnapshotDerived(probe, cfg.data.channels.species,
                            cfg.data.channels.species_names, wxi)
    probe_x = snapshotToInputTensor(probe, cfg.data.channels, norm)
    out_indices = outputChannelIndices(cfg.data.channels)
    out_names   = outputChannelNames(cfg.data.channels)
    base_in = probe_x.shape[0]
    aux_chans = (len(cfg.data.boundary_spec)
                 if cfg.data.boundary_channels else 0)
    in_channels = base_in + aux_chans
    dim = probe_x.ndim - 1
    model = buildModel(cfg.model, in_channels=in_channels,
                       out_channels=len(out_names),
                       out_indices=out_indices, dim=dim).to(dev.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # The local CLI's predict path uses PatchDataset to provide the
    # padded inputs and boundary-aux callable. Build a single
    # PairedPICDataset + PatchDataset over the full group set so we
    # reuse the same padding / aux logic.
    base = PairedPICDataset(groups, reader, cfg.data.channels, norm,
                            cache=False, cache_max_gib=cfg.data.cache_max_gib,
                            warpx_input=wxi)
    pds = PatchDataset(base,
                       patch_size=cfg.model.patch_size,
                       boundary_spec=cfg.data.boundary_spec,
                       boundary_channels=cfg.data.boundary_channels)
    amp_enabled = cfg.train.amp and dev.amp_available
    amp_dtype   = cfg.train.amp_dtype if amp_enabled else "fp32"

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_step_to_idx = {g.step: i for i, g in enumerate(groups)}
    step_to_group = {g.step: g for g in groups}

    for tier in TIERS:
        for step_str in PHASES:
            step = int(step_str)
            if step not in step_to_group:
                continue
            g = step_to_group[step]
            if tier not in g.tier_infos:
                continue

            # Raw snapshots (with derived augmentation) for the
            # derive-formula step.
            lo_info = g.tier_infos[tier]
            lo_snap = reader.load(lo_info, cfg.data.channels.fields,
                                  full_mom, cfg.data.channels.species_names)
            _augmentSnapshotDerived(lo_snap, cfg.data.channels.species,
                                    cfg.data.channels.species_names, wxi)
            gt_info = g.tier_infos[gt_name]
            gt_snap = reader.load(gt_info, cfg.data.channels.fields,
                                  full_mom, cfg.data.channels.species_names)
            _augmentSnapshotDerived(gt_snap, cfg.data.channels.species,
                                    cfg.data.channels.species_names, wxi)

            # Same machinery the local predict CLI uses to assemble a
            # full-snapshot prediction from per-cell patches.
            full_input = torch.from_numpy(
                snapshotToInputTensor(lo_snap, cfg.data.channels, norm))
            padded = pds._padInput(full_input)
            boundary_aux = (
                (lambda coords:
                 pds._boundaryAuxChannels(coords,
                                          dtype=full_input.dtype,
                                          device=full_input.device))
                if cfg.data.boundary_channels else None
            )
            pred_norm = _denoiseSnapshotByPatches(
                model, padded, full_input,
                patch_size=cfg.model.patch_size,
                dev=dev, amp_enabled=amp_enabled,
                amp_dtype=amp_dtype, boundary_aux=boundary_aux,
            )
            pred_arr = pred_norm.float().cpu().numpy()

            # Build a "prediction snapshot": deep-copy the low-ppc
            # snapshot, then overwrite each output channel with the
            # denormalized prediction, using per-(species, moment) stats.
            pred_snap = deepcopy(lo_snap)
            n_field = len(cfg.data.channels.output_fields)
            for k, ch in enumerate(out_names):
                if k < n_field:
                    field_name = cfg.data.channels.output_fields[k]
                    arr = norm.invert(field_name, pred_arr[k])
                    pred_snap.fields[field_name] = arr
                else:
                    j = k - n_field
                    n_mom = len(cfg.data.channels.output_species_moments)
                    sp_idx, mom_idx = divmod(j, n_mom)
                    sp = cfg.data.channels.species_names[sp_idx]
                    mom = cfg.data.channels.output_species_moments[mom_idx]
                    mean_sp, std_sp, kind_sp = ps_stats[(sp, mom)]
                    pred_snap.species[sp][mom] = perSpeciesInvert(
                        pred_arr[k], mean_sp, std_sp, kind_sp)

            d_lo   = deriveQuantities(lo_snap, wxi)
            d_pred = deriveQuantities(pred_snap, wxi)
            d_gt   = deriveQuantities(gt_snap, wxi)

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
