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
        newton.max_iterations=100 \
        $extra_args \
        2>&1 | tee "$outfile"

    echo "  Done with case $casename (exit code: $?)"
    cd "$rootdir"
}

echo "================================================="
echo "  SNES with bt line search, restart_length=100"
echo "================================================="
run_case "snes_bt_r100" \
    "implicit_evolve.nonlinear_solver=petsc_snes" \
    "-snes_linesearch_type bt" \
    "-ksp_gmres_restart 100"

echo "================================================="
echo "  SNES basic (no line search), restart_length=100"
echo "================================================="
run_case "snes_basic_r100" \
    "implicit_evolve.nonlinear_solver=petsc_snes" \
    "-ksp_gmres_restart 100"

echo ""
echo "======================================"
echo "  ALL CASES COMPLETE"
echo "======================================"
