#!/usr/bin/env python3
# Plot mesh_data/field_data* AMReX plotfiles from a WarpX run directory.
#
# Usage:
#   plot_fields.py <run_dir> [--outdir <plots_dir>] [--steps S1,S2,...] [--fields f1,f2,...]
#
# Default fields: rho, By, jz, Ex, part_per_cell. Produces 2D colormaps (x,z)
# and z-averaged 1D lineouts vs x for each plotfile, written to
# <outdir>/fields_2d/<step>.png and <outdir>/fields_profile/<step>.png.
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

DEFAULT_FIELDS = ["rho", "By", "jz", "Ex", "part_per_cell"]


def read_level0(pf_path, fields):
    ds = yt.load(pf_path)
    t = float(ds.current_time)
    dims = ds.domain_dimensions.copy()
    left = ds.domain_left_edge.to("m").d
    right = ds.domain_right_edge.to("m").d
    nx, nz = int(dims[0]), int(dims[1])
    cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge,
                          dims=ds.domain_dimensions)
    out = {}
    for f in fields:
        key = ("boxlib", f)
        if key not in ds.field_list:
            continue
        arr = np.array(cg[key])
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        out[f] = arr
    x = np.linspace(left[0], right[0], nx + 1)
    x = 0.5 * (x[:-1] + x[1:])
    z = np.linspace(left[1], right[1], nz + 1)
    z = 0.5 * (z[:-1] + z[1:])
    return x, z, out, t


def plot_one(pf_path, outdir, fields):
    x, z, data, t = read_level0(pf_path, fields)
    step_match = re.search(r"(\d+)$", os.path.basename(pf_path))
    step = step_match.group(1) if step_match else "xxx"
    if not data:
        print(f"  no requested fields in {pf_path}")
        return
    dir_2d = os.path.join(outdir, "fields_2d")
    dir_prof = os.path.join(outdir, "fields_profile")
    os.makedirs(dir_2d, exist_ok=True)
    os.makedirs(dir_prof, exist_ok=True)
    n = len(data)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.2 * n), squeeze=False)
    extent = [x[0] * 100, x[-1] * 100, z[0] * 100, z[-1] * 100]
    for ax, (fname, arr) in zip(axes[:, 0], data.items()):
        vmax = np.max(np.abs(arr)) if arr.size else 1.0
        sym = fname in ("jx", "jy", "jz", "Ex", "Ey", "Ez", "Bx", "By", "Bz", "divE", "divB")
        if sym:
            vmin = -vmax if vmax > 0 else -1.0
            cmap = "RdBu_r"
        else:
            vmin = float(np.min(arr))
            vmax = float(np.max(arr))
            cmap = "viridis"
        im = ax.imshow(arr.T, origin="lower", extent=extent, aspect="auto",
                       vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_ylabel("z (cm)")
        ax.set_title(f"{fname}  (t={t*1e9:.2f} ns, step {step})")
        plt.colorbar(im, ax=ax)
    axes[-1, 0].set_xlabel("x (cm)")
    fig.tight_layout()
    fig.savefig(os.path.join(dir_2d, f"{step}.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(n, 1, figsize=(8, 2.0 * n), squeeze=False, sharex=True)
    for ax, (fname, arr) in zip(axes[:, 0], data.items()):
        prof = arr.mean(axis=1)
        ax.plot(x * 100, prof)
        ax.set_ylabel(fname)
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_title(f"z-averaged profiles  (t={t*1e9:.2f} ns, step {step})")
    axes[-1, 0].set_xlabel("x (cm)")
    fig.tight_layout()
    fig.savefig(os.path.join(dir_prof, f"{step}.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--steps", default=None,
                    help="comma-separated list of step numbers to plot; default=all")
    ap.add_argument("--fields", default=",".join(DEFAULT_FIELDS))
    args = ap.parse_args()
    rdir = os.path.abspath(args.run_dir)
    outdir = args.outdir or os.path.join(rdir, "plots")
    os.makedirs(outdir, exist_ok=True)
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    pfs = sorted(glob.glob(os.path.join(rdir, "mesh_data", "field_data*")))
    pfs = [p for p in pfs if os.path.isdir(p)]
    if args.steps is not None:
        wanted = set(str(int(s.strip())) for s in args.steps.split(",") if s.strip())
        pfs = [p for p in pfs
               if str(int(re.search(r"(\d+)$", p).group(1))) in wanted]
    if not pfs:
        print(f"no field_data* plotfiles in {rdir}/mesh_data")
        return 0
    for p in pfs:
        print(f"  plotting {os.path.basename(p)}")
        plot_one(p, outdir, fields)
    print(f"wrote field plots -> {outdir}")


if __name__ == "__main__":
    sys.exit(main())
