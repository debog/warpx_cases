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

casename="native_newton_gmres"
dirname=".run_${casename}_${LCHOST}"

echo ""
echo "======================================"
echo "  Case: $casename"
echo "  Native Newton + AMReX GMRES"
echo "======================================"

if [ -d "$dirname" ]; then
    echo "  deleting existing directory $dirname"
    rm -rf "$dirname"
fi
echo "  creating directory $dirname"
mkdir "$dirname"
cd "$dirname"
cp "$INP_FILE" .
INP=$(ls *.in)
outfile="out.${LCHOST}.log"

echo "  running WarpX..."
$runcmd $EXEC $INP \
    implicit_evolve.use_mass_matrices_jacobian=true \
    implicit_evolve.particle_suborbits=true \
    max_step=110870 \
    newton.linear_solver=amrex_gmres \
    2>&1 | tee "$outfile"

echo "  Done (exit code: $?)"
cd "$rootdir"
