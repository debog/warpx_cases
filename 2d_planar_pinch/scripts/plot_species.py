#!/usr/bin/env python3
# Plot mesh_data/species_data* AMReX plotfiles from a WarpX run directory.
#
# Usage:
#   plot_species.py <run_dir> [--outdir <plots_dir>] [--steps S1,S2,...]
#
# Auto-detects WarpX dimensionality from the plotfile. WarpX particle_fields
# diagnostic (do_average=0) sums:
#   num  = Σ w_p
#   ux   = Σ w_p v_x
#   enex = Σ w_p (1/(1+γ)) u_x² c²   (non-relativistic: ≈ Σ w_p v_x²/2)
# From these we build per-cell n, T, ⟨v⟩. Outputs land in
#   1D: <outdir>/species_profile/<step>.png   (only 1D lineouts in z)
#   2D: <outdir>/species_2d/<step>.png        (n, T, <v_x> per species, 2D)
#       <outdir>/species_profile/<step>.png   (z-averaged 1D profiles in x)
import argparse
import glob
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yt
yt.set_log_level(40)

Q_E = 1.602176634e-19
M_E = 9.1093837015e-31
M_U = 1.66053906660e-27
M_D = 2.01410177812 * M_U - M_E
SPECIES_MASS = {"electrons": M_E, "deuterium": M_D}


def read_level0(pf_path):
    ds = yt.load(pf_path)
    t = float(ds.current_time)
    dim = int(ds.dimensionality)
    dims = ds.domain_dimensions.copy()
    left = ds.domain_left_edge.to("m").d
    right = ds.domain_right_edge.to("m").d
    cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge,
                          dims=ds.domain_dimensions)
    fnames = [fn for (ft, fn) in ds.field_list if ft == "boxlib"]
    species = sorted({fn.split("_", 1)[1] for fn in fnames if fn.startswith("num_")})

    if dim == 1:
        nz = int(dims[0])
        dz = (right[0] - left[0]) / nz
        # In 1D, "cell volume" is just dz (per unit transverse area). Particle
        # deposition is per-unit-transverse-area, so num/dz gives n in m^-3.
        vcell = dz
        z = np.linspace(left[0], right[0], nz + 1)
        z = 0.5 * (z[:-1] + z[1:])
        coord = {"dim": 1, "z": z}
    else:
        nx, nz = int(dims[0]), int(dims[1])
        dx = (right[0] - left[0]) / nx
        dz = (right[1] - left[1]) / nz
        vcell = dx * dz * 1.0
        x = np.linspace(left[0], right[0], nx + 1)
        x = 0.5 * (x[:-1] + x[1:])
        z = np.linspace(left[1], right[1], nz + 1)
        z = 0.5 * (z[:-1] + z[1:])
        coord = {"dim": 2, "x": x, "z": z}

    data = {}
    for s in species:
        d = {}
        for k in ["num", "ux", "uy", "uz", "enex", "eney", "enez"]:
            key = ("boxlib", f"{k}_{s}")
            if key in ds.field_list:
                arr = np.array(cg[key]).squeeze()
                d[k] = arr
        data[s] = d
    return {"coord": coord, "data": data, "t": t, "vcell": vcell}


def moments(d, mass):
    num = d["num"]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(num > 0, 1.0 / num, 0.0)
        vx = d["ux"] * mean
        vy = d["uy"] * mean if "uy" in d else np.zeros_like(num)
        vz = d["uz"] * mean if "uz" in d else np.zeros_like(num)
        vx2 = 2.0 * d["enex"] * mean if "enex" in d else np.zeros_like(num)
        vy2 = 2.0 * d["eney"] * mean if "eney" in d else np.zeros_like(num)
        vz2 = 2.0 * d["enez"] * mean if "enez" in d else np.zeros_like(num)
        var = (vx2 - vx**2) + (vy2 - vy**2) + (vz2 - vz**2)
        var = np.clip(var, 0.0, None)
        T_eV = mass * var / 3.0 / Q_E
    return vx, vy, vz, T_eV


def plot_one_1d(meta, step, outdir):
    coord = meta["coord"]
    data = meta["data"]
    t = meta["t"]
    vcell = meta["vcell"]
    z = coord["z"]
    species = list(data.keys())
    if not species:
        return
    dir_prof = os.path.join(outdir, "species_profile")
    os.makedirs(dir_prof, exist_ok=True)

    profiles = {}
    for s in species:
        d = data[s]
        if "num" not in d:
            continue
        mass = SPECIES_MASS.get(s, M_U)
        n_cc = d["num"] / vcell
        vx, vy, vz, T_eV = moments(d, mass)
        # In 1D the bulk flow coordinate is v_z (not v_x).
        profiles[s] = (n_cc, vz, T_eV)

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for s, (n_cc, vz, T_eV) in profiles.items():
        axes[0].plot(z * 100, n_cc, label=s)
        axes[1].plot(z * 100, T_eV, label=s)
        axes[2].plot(z * 100, vz,   label=s)
    axes[0].set_ylabel("n (m^-3)")
    axes[1].set_ylabel("T (eV)")
    axes[2].set_ylabel("<v_z> (m/s)")
    axes[2].set_xlabel("z (cm)")
    axes[0].set_title(f"1D species profiles  (t={t*1e9:.2f} ns, step {step})")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(dir_prof, f"{step}.png"), dpi=150)
    plt.close(fig)


def plot_one_2d(meta, step, outdir):
    coord = meta["coord"]
    data = meta["data"]
    t = meta["t"]
    vcell = meta["vcell"]
    x = coord["x"]
    z = coord["z"]
    extent = [x[0] * 100, x[-1] * 100, z[0] * 100, z[-1] * 100]
    species = list(data.keys())
    if not species:
        return
    dir_2d = os.path.join(outdir, "species_2d")
    dir_prof = os.path.join(outdir, "species_profile")
    os.makedirs(dir_2d, exist_ok=True)
    os.makedirs(dir_prof, exist_ok=True)
    n_species = len(species)
    fig, axes = plt.subplots(n_species, 3, figsize=(14, 3.2 * n_species), squeeze=False)
    profiles = {}
    for i, s in enumerate(species):
        mass = SPECIES_MASS.get(s, M_U)
        d = data[s]
        if "num" not in d:
            continue
        n_cc = d["num"] / vcell
        vx, vy, vz, T_eV = moments(d, mass)
        profiles[s] = (n_cc, vx, T_eV)
        panels = [
            (n_cc, f"n_{s} (m^-3)", "viridis", None),
            (T_eV, f"T_{s} (eV)", "viridis", None),
            (vx, f"<v_x>_{s} (m/s)", "RdBu_r", True),
        ]
        for j, (arr, title, cmap, sym) in enumerate(panels):
            ax = axes[i, j]
            if sym:
                vmax = float(np.max(np.abs(arr))) or 1.0
                vmin = -vmax
            else:
                vmin = float(np.min(arr))
                vmax = float(np.max(arr))
                if vmax == vmin:
                    vmax = vmin + 1.0
            im = ax.imshow(arr.T, origin="lower", extent=extent, aspect="auto",
                           vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_title(f"{title}  (t={t*1e9:.2f} ns, step {step})")
            ax.set_ylabel("z (cm)")
            if i == n_species - 1:
                ax.set_xlabel("x (cm)")
            plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(dir_2d, f"{step}.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for s, (n_cc, vx, T_eV) in profiles.items():
        axes[0].plot(x * 100, n_cc.mean(axis=1), label=s)
        axes[1].plot(x * 100, T_eV.mean(axis=1), label=s)
        axes[2].plot(x * 100, vx.mean(axis=1), label=s)
    axes[0].set_ylabel("n (m^-3)")
    axes[1].set_ylabel("T (eV)")
    axes[2].set_ylabel("<v_x> (m/s)")
    axes[2].set_xlabel("x (cm)")
    axes[0].set_title(f"z-averaged species profiles  (t={t*1e9:.2f} ns, step {step})")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(dir_prof, f"{step}.png"), dpi=150)
    plt.close(fig)


def plot_one(pf_path, outdir):
    meta = read_level0(pf_path)
    step = re.search(r"(\d+)$", os.path.basename(pf_path)).group(1)
    if not meta["data"]:
        print(f"  no species in {pf_path}")
        return
    if meta["coord"]["dim"] == 1:
        plot_one_1d(meta, step, outdir)
    else:
        plot_one_2d(meta, step, outdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--steps", default=None)
    args = ap.parse_args()
    rdir = os.path.abspath(args.run_dir)
    outdir = args.outdir or os.path.join(rdir, "plots")
    os.makedirs(outdir, exist_ok=True)
    pfs = sorted(glob.glob(os.path.join(rdir, "mesh_data", "species_data*")))
    pfs = [p for p in pfs if os.path.isdir(p)]
    if args.steps is not None:
        wanted = set(str(int(s.strip())) for s in args.steps.split(",") if s.strip())
        pfs = [p for p in pfs
               if str(int(re.search(r"(\d+)$", p).group(1))) in wanted]
    if not pfs:
        print(f"no species_data* plotfiles in {rdir}/mesh_data")
        return 0
    for p in pfs:
        print(f"  plotting {os.path.basename(p)}")
        plot_one(p, outdir)
    print(f"wrote species plots -> {outdir}")


if __name__ == "__main__":
    sys.exit(main())
