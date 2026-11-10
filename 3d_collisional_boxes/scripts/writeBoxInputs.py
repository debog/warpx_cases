"""! @file writeBoxInputs.py
@brief Writes the WarpX input deck of the collisional relaxation boxes (LEANPIC step 0).

Each cell is a closed box: the particles are not pushed (do_not_push), there is no field
solve, and only deuterium-deuterium Coulomb collisions act. Cells differ in their initial
temperature anisotropy, set by the cell indices:
- x index ix (0..7): a = 3^((ix - 3)/4), from 0.44 to 3 (a = 1 at ix = 3);
- y index iy (0..7): b = 2^((iy - 3)/4), from 0.59 to 2 (b = 1 at iy = 3);
- temperatures T_i = 3 T0 w_i / (w_x + w_y + w_z) with w = (a, sqrt(b), 1/sqrt(b)), so every
  cell has the same mean temperature T0. Cell (3, 3) is isotropic; cells with iy = 3 are
  bi-Maxwellian with T_y = T_z (analytic isotropization rate).
The 8 cells along z are replicas: same settings, independent samples.

Output: the raw velocity moments up to order 4 (35 monomials of the normalised momentum
u = gamma v / c) summed with the particle weights over each half of each cell's particles.
The halves are set by the parity of a fine digit of u_x, fmod(floor(abs(ux)*1e12),2):
stripes of 3e-4 m/s in velocity, far below any thermal speed, and abs(ux)*1e12 < 2^53.

Random seed: WarpX seeds MPI rank r with (r + 1) n for warpx.random_seed = n, so two runs with
the same n are identical (test runs 1 and 2 matched particle for particle), and two different
values can still share a stream (n = 2 on rank 0 equals n = 1 on rank 1). Every repeat or extra
run gets its own prime n larger than the number of ranks; n = 1 is WarpX's default.

Usage: python3 writeBoxInputs.py {output file} [{particles per cell} {steps} {moment interval} [{seed}]]
"""

import itertools
import sys

SCALE = "1e12"
# Filters without spaces and without "=": WarpX reads a particle_fields filter with a plain
# ParmParse query, which keeps only the first whitespace-separated token
# (Source/Diagnostics/Diagnostics.cpp), and ParmParse takes an "=" inside an unspaced value
# (as in ">=") as the start of a new definition. A particle is kept where the filter is
# nonzero: half A has an even digit, half B an odd one.
FILTERS = {"A": f"1-fmod(floor(abs(ux)*{SCALE}),2)", "B": f"fmod(floor(abs(ux)*{SCALE}),2)"}


def monomials(order=4):
    """! @brief Exponents (a, b, c) of ux^a uy^b uz^c with a + b + c <= order (35 for order 4)."""
    return [e for n in range(order + 1) for e in itertools.product(range(n + 1), repeat=3) if sum(e) == n]


def expression(exponents):
    """! @brief Parser expression of one monomial; "1" for the zeroth order."""
    terms = [f"{v}^{p}" if p > 1 else v for v, p in zip(("ux", "uy", "uz"), exponents) if p > 0]
    return "*".join(terms) if terms else "1"


def stdFunction(axis):
    """! @brief Parser expression of the thermal spread of u along one axis, constant in each cell."""
    a = "pow(3,(floor(x/dx)-3)/4)"
    sb = "pow(2,(floor(y/dx)-3)/8)"
    weight = {"x": a, "y": sb, "z": f"(1/{sb})"}[axis]
    return f"sqrt(3*T0*{weight}/({a}+{sb}+1/{sb})*q_e/mD)/clight"


def deck(ppc, steps, interval, seed=1):
    """! @brief The input deck as a string."""
    names = []
    lines = []
    for half, filt in FILTERS.items():
        for e in monomials():
            name = f"{half}{e[0]}{e[1]}{e[2]}"
            names.append(name)
            lines += [
                f"moments.particle_fields.{name}(x,y,z,ux,uy,uz) = {expression(e)}",
                f"moments.particle_fields.{name}.filter(x,y,z,ux,uy,uz) = {filt}",
                f"moments.particle_fields.{name}.do_average = 0",
            ]
    return f"""#################################
# Collisional relaxation boxes, LEANPIC step 0: test run
# Written by scripts/writeBoxInputs.py; see its docstring for the design.
#################################
my_constants.n0 = 1.e23          # deuterium density [1/m^3] (planar pinch)
my_constants.T0 = 1.0            # mean temperature [eV] (planar pinch)
my_constants.AD = 2.01410
my_constants.mD = AD*m_u - m_e   # deuterium mass [kg]
my_constants.dx = 1.0e-5         # cell size [m]; irrelevant without pushes
my_constants.dt = 0.5e-12        # about 1/60 of the deuterium collision time

max_step = {steps}
amr.n_cell = 8 8 8
amr.max_grid_size = 2
amr.blocking_factor = 2
amr.max_level = 0
geometry.dims = 3
geometry.prob_lo = 0. 0. 0.
geometry.prob_hi = 8*dx 8*dx 8*dx
boundary.field_lo = periodic periodic periodic
boundary.field_hi = periodic periodic periodic

warpx.verbose = 1
warpx.random_seed = {seed}
warpx.const_dt = dt
algo.maxwell_solver = none
algo.particle_shape = 1

particles.species_names = deuterium
deuterium.charge = q_e
deuterium.mass = mD
deuterium.injection_style = NRandomPerCell
deuterium.num_particles_per_cell = {ppc}
deuterium.profile = constant
deuterium.density = n0
deuterium.momentum_distribution_type = maxwellian
deuterium.maxwellian_u_std_distribution_type = parser
deuterium.ux_std_function(x,y,z) = {stdFunction("x")}
deuterium.uy_std_function(x,y,z) = {stdFunction("y")}
deuterium.uz_std_function(x,y,z) = {stdFunction("z")}
deuterium.do_not_push = 1
deuterium.do_not_deposit = 1

collisions.collision_names = dd
dd.species = deuterium deuterium
dd.CoulombLog = 10.0

diagnostics.diags_names = moments checkpoint
moments.intervals = 0:{steps}:{interval}
moments.diag_type = Full
moments.format = plotfile
moments.write_species = 0
moments.fields_to_plot = none
moments.particle_fields_species = deuterium
moments.particle_fields_to_plot = {" ".join(names)}
""" + "\n".join(lines) + f"""

checkpoint.intervals = 0:{steps}:{steps // 2}
checkpoint.diag_type = Full
checkpoint.format = checkpoint
"""


if __name__ == "__main__":
    out = sys.argv[1]
    ppc, steps, interval = (int(v) for v in sys.argv[2:5]) if len(sys.argv) > 4 else (10000, 400, 4)
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    with open(out, "w") as f:
        f.write(deck(ppc, steps, interval, seed))
