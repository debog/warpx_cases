#!/usr/bin/env python3
# Plot reduced_files/*.txt time-series diagnostics from a WarpX run directory.
#
# Usage:
#   plot_reduced.py <run_dir> [--outdir <plots_dir>]
#
# Expects <run_dir>/diags/reduced_files/{particle_energy,field_energy,
# poynting_flux,newton_solver}.txt.
import argparse
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with open(path) as f:
        header = f.readline().strip()
    cols = re.findall(r"\[\d+\]([^ ]+)", header)
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return cols, data


def col(cols, data, name):
    for i, c in enumerate(cols):
        if c.split("(")[0] == name.split("(")[0]:
            if name == c or name in c:
                return data[:, i]
    for i, c in enumerate(cols):
        if c.startswith(name):
            return data[:, i]
    raise KeyError(f"{name} not in {cols}")


def plot_particle_energy(path, outdir):
    if not os.path.isfile(path):
        return
    cols, data = load(path)
    t_ns = data[:, 1] * 1e9
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for c in cols[2:]:
        if c.endswith("_mean(J)") or c == "total_mean(J)":
            continue
        y = col(cols, data, c)
        ax1.plot(t_ns, y, label=c.replace("(J)", ""))
    ax1.set_xlabel("t (ns)")
    ax1.set_ylabel("energy (J)")
    ax1.set_title("particle energy (total)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    for c in cols[2:]:
        if not (c.endswith("_mean(J)") or c == "total_mean(J)"):
            continue
        y = col(cols, data, c)
        y_eV = y / 1.602176634e-19
        ax2.plot(t_ns, y_eV, label=c.replace("_mean(J)", ""))
    ax2.set_xlabel("t (ns)")
    ax2.set_ylabel("mean energy / particle (eV)")
    ax2.set_title("mean particle energy")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "particle_energy.png"), dpi=150)
    plt.close(fig)


def plot_field_energy(path, outdir):
    if not os.path.isfile(path):
        return
    cols, data = load(path)
    t_ns = data[:, 1] * 1e9
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in cols[2:]:
        y = col(cols, data, c)
        ax.plot(t_ns, y, label=c.replace("(J)", "").replace("_lev0", ""))
    ax.set_xlabel("t (ns)")
    ax.set_ylabel("energy (J)")
    ax.set_title("field energy (level 0)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "field_energy.png"), dpi=150)
    plt.close(fig)


def plot_poynting(path, outdir):
    if not os.path.isfile(path):
        return
    cols, data = load(path)
    t_ns = data[:, 1] * 1e9
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for c in cols[2:]:
        y = col(cols, data, c)
        if c.startswith("outward_power"):
            ax1.plot(t_ns, y, label=c.replace("outward_power_", "").replace("(W)", ""))
        elif c.startswith("integrated_energy_loss"):
            ax2.plot(t_ns, y, label=c.replace("integrated_energy_loss_", "").replace("(J)", ""))
    ax1.set_xlabel("t (ns)")
    ax1.set_ylabel("power (W)")
    ax1.set_title("outward Poynting power through each face")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("t (ns)")
    ax2.set_ylabel("integrated energy loss (J)")
    ax2.set_title("integrated outward energy loss")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "poynting_flux.png"), dpi=150)
    plt.close(fig)


def plot_newton(path, outdir):
    if not os.path.isfile(path):
        return
    cols, data = load(path)
    t_ns = data[:, 1] * 1e9
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax = axes[0, 0]
    ax.plot(t_ns, col(cols, data, "iters"), label="Newton iters / step")
    ax.set_xlabel("t (ns)")
    ax.set_ylabel("iterations")
    ax.set_title("Newton iterations per step")
    ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    ax.semilogy(t_ns, col(cols, data, "norm_abs"), label="|r|")
    ax.semilogy(t_ns, col(cols, data, "norm_rel"), label="|r|/|r0|")
    ax.set_xlabel("t (ns)")
    ax.set_ylabel("residual")
    ax.set_title("Newton residual (end of step)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax = axes[1, 0]
    ax.plot(t_ns, col(cols, data, "gmres_iters"), label="GMRES iters (last Newton)")
    ax.set_xlabel("t (ns)")
    ax.set_ylabel("iterations")
    ax.set_title("Linear-solve iterations")
    ax.grid(True, alpha=0.3)
    ax = axes[1, 1]
    ax.semilogy(t_ns, col(cols, data, "gmres_last_res"))
    ax.set_xlabel("t (ns)")
    ax.set_ylabel("|r|")
    ax.set_title("GMRES last residual")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "newton_solver.png"), dpi=150)
    plt.close(fig)


def plot_energy_conservation(pe_path, fe_path, py_path, outdir):
    if not (os.path.isfile(pe_path) and os.path.isfile(fe_path)):
        return
    pe_cols, pe = load(pe_path)
    fe_cols, fe = load(fe_path)
    t_ns = pe[:, 1] * 1e9
    Ep = col(pe_cols, pe, "total(J)")
    if pe.shape[0] != fe.shape[0]:
        n = min(pe.shape[0], fe.shape[0])
        pe, fe, t_ns, Ep = pe[:n], fe[:n], t_ns[:n], Ep[:n]
    Ef = col(fe_cols, fe, "total_lev0(J)")
    E_loss = np.zeros_like(Ep)
    if os.path.isfile(py_path):
        py_cols, py = load(py_path)
        if py.shape[0] >= len(Ep):
            py = py[: len(Ep)]
        for c in py_cols:
            if c.startswith("integrated_energy_loss"):
                E_loss += col(py_cols, py, c)
    Etot = Ep + Ef + E_loss
    E0 = Etot[0] if Etot[0] != 0 else 1.0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(t_ns, Ep, label="particle")
    ax1.plot(t_ns, Ef, label="field")
    ax1.plot(t_ns, E_loss, label="boundary loss (integrated)")
    ax1.plot(t_ns, Etot, "k--", label="sum")
    ax1.set_xlabel("t (ns)")
    ax1.set_ylabel("energy (J)")
    ax1.set_title("energy budget")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax2.plot(t_ns, (Etot - E0) / abs(E0))
    ax2.set_xlabel("t (ns)")
    ax2.set_ylabel("(E(t) - E(0)) / |E(0)|")
    ax2.set_title("relative energy conservation error")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "energy_conservation.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    rdir = os.path.abspath(args.run_dir)
    outdir = args.outdir or os.path.join(rdir, "plots")
    os.makedirs(outdir, exist_ok=True)
    rf = os.path.join(rdir, "diags", "reduced_files")
    pe = os.path.join(rf, "particle_energy.txt")
    fe = os.path.join(rf, "field_energy.txt")
    py = os.path.join(rf, "poynting_flux.txt")
    nw = os.path.join(rf, "newton_solver.txt")
    plot_particle_energy(pe, outdir)
    plot_field_energy(fe, outdir)
    plot_poynting(py, outdir)
    plot_newton(nw, outdir)
    plot_energy_conservation(pe, fe, py, outdir)
    print(f"wrote reduced plots -> {outdir}")


if __name__ == "__main__":
    sys.exit(main())
