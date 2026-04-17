#!/bin/bash

clear

NNODE=1
export OMP_NUM_THREADS=1
export LCHOST=dane
export WARPX_BUILD=/g/g92/ghosh5/Codes/WarpX-builds/dane

DIM=rcylinder
rootdir=$PWD
INP_FILE=$rootdir/common/dynamic_pinch_1d.in
EXEC=$(ls $WARPX_BUILD/build/bin/warpx.${DIM})
echo "Executable file is ${EXEC}."

ntasks=56
runcmd="srun -n $ntasks -p pdebug"

run_case() {
    local casename=$1
    shift
    local extra_args="$@"

    local dirname=".run_${casename}_${LCHOST}"
    echo ""
    echo "======================================"
    echo "  Case: $casename"
    echo "  Extra args: $extra_args"
    echo "======================================"

    if [ -d "$dirname" ]; then
        echo "  deleting existing directory $dirname"
        rm -rf "$dirname"
    fi
    echo "  creating directory $dirname"
    mkdir "$dirname"
    cd "$dirname"
    cp "$INP_FILE" .
    local INP=$(ls *.in)
    local outfile="out.${LCHOST}.log"

    echo "  running WarpX..."
    $runcmd $EXEC $INP \
        implicit_evolve.use_mass_matrices_jacobian=true \
        implicit_evolve.particle_suborbits=true \
        max_step=110870 \
        newton.require_convergence=false \
        $extra_args \
        2>&1 | tee "$outfile"

    echo "  Done with case $casename (exit code: $?)"
    cd "$rootdir"
}

echo "============================================"
echo "  PART 1: JFNK Epsilon sweep (Newton solver)"
echo "============================================"

run_case "eps_1e-3"  "newton.jfnk_epsilon=1.0e-3"
run_case "eps_1e-4"  "newton.jfnk_epsilon=1.0e-4"
run_case "eps_1e-5"  "newton.jfnk_epsilon=1.0e-5"
run_case "eps_1e-6"  "newton.jfnk_epsilon=1.0e-6"
run_case "eps_1e-7"  "newton.jfnk_epsilon=1.0e-7"
run_case "eps_1e-8"  "newton.jfnk_epsilon=1.0e-8"
run_case "eps_1e-10" "newton.jfnk_epsilon=1.0e-10"

echo ""
echo "============================================"
echo "  PART 2: PETSc SNES solver"
echo "============================================"

run_case "snes" "implicit_evolve.nonlinear_solver=petsc_snes"

echo ""
echo "======================================"
echo "  ALL CASES COMPLETE"
echo "======================================"
