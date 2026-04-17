#!/bin/bash

clear

NNODE=1
export OMP_NUM_THREADS=1

DIM=rcylinder
rootdir=$PWD
INP_FILE=$rootdir/common/dynamic_pinch_1d.in
EXEC=$(ls $WARPX_BUILD/build/bin/warpx.${DIM})
echo "Executable file is ${EXEC}."

ntasks=56
runcmd="srun -n $ntasks -p pdebug"

run_case() {
    local casename=$1
    local mm_jac=$2
    local suborbits=$3

    local dirname=".run_${casename}_${LCHOST}"
    echo ""
    echo "======================================"
    echo "  Case: $casename"
    echo "  MM Jacobian: $mm_jac"
    echo "  Suborbits:   $suborbits"
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
        implicit_evolve.use_mass_matrices_jacobian=$mm_jac \
        implicit_evolve.particle_suborbits=$suborbits \
        max_step=111000 \
        2>&1 | tee "$outfile"

    echo "  Done with case $casename (exit code: $?)"
    cd "$rootdir"
}

# Case 1: MM Jacobian ON + Suborbits OFF
run_case "mm_on_sub_off" true false

# Case 2: MM Jacobian ON + Suborbits ON
run_case "mm_on_sub_on" true true

# Case 3: MM Jacobian OFF (JFNK) + Suborbits OFF
run_case "mm_off_sub_off" false false

# Case 4: MM Jacobian OFF (JFNK) + Suborbits ON
run_case "mm_off_sub_on" false true

echo ""
echo "======================================"
echo "  ALL CASES COMPLETE"
echo "======================================"
