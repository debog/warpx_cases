#!/bin/bash
#
# Usage: bash common/run.sh <DIM> <PC_TYPE> [MAX_STEP] [EXTRA_ARGS...]
#
#   DIM:      1d, 2d, 3d
#   PC_TYPE:  jacobi, jacobi_w1, jacobi_w2, ccmlmg, petsc, nopc
#   MAX_STEP: number of time steps (default: 10)
#
# Examples:
#   bash common/run.sh 1d jacobi 100
#   bash common/run.sh 2d ccmlmg 10
#   bash common/run.sh 1d jacobi_w1 100   # Jacobi with mass_matrices_pc_width=1
#   bash common/run.sh 2d petsc 10        # PETSc ASM+LU preconditioner
#

set -e

DIM=${1:?  "Usage: $0 <DIM> <PC_TYPE> [MAX_STEP]"}
PC_TYPE=${2:?  "Usage: $0 <DIM> <PC_TYPE> [MAX_STEP]"}
MAX_STEP=${3:-10}
shift 3 2>/dev/null || true

# Grid / particle settings per dimension
case $DIM in
    1d)
        NX_ARGS="amr.n_cell = 224"
        NP_ARGS="my_constants.Nppc = 100"
        NTASKS=1
        ;;
    2d)
        NX_ARGS="amr.n_cell = 128 128"
        NP_ARGS="my_constants.Nppc_x = 16  my_constants.Nppc_z = 16"
        NTASKS=4
        ;;
    3d)
        NX_ARGS="amr.n_cell = 32 32 32"
        NP_ARGS="my_constants.Nppc_x = 4  my_constants.Nppc_y = 4  my_constants.Nppc_z = 4"
        NTASKS=4
        ;;
    *)
        echo "ERROR: Unknown DIM=$DIM (expected 1d, 2d, 3d)"
        exit 1
        ;;
esac

# PC-specific arguments
MM_WIDTH_ARGS=""
LINEAR_SOLVER="amrex_gmres"
PETSC_FLAGS=""
case $PC_TYPE in
    jacobi)
        PC_ARGS="jacobian.pc_type = pc_jacobi
                 pc_jacobi.verbose = true
                 pc_jacobi.max_iter = 200
                 pc_jacobi.omega = 0.667
                 pc_jacobi.relative_tolerance = 1e-6"
        ;;
    jacobi_w1)
        PC_ARGS="jacobian.pc_type = pc_jacobi
                 pc_jacobi.verbose = true
                 pc_jacobi.max_iter = 200
                 pc_jacobi.omega = 0.667
                 pc_jacobi.relative_tolerance = 1e-6"
        MM_WIDTH_ARGS="implicit_evolve.mass_matrices_pc_width = 1"
        ;;
    jacobi_w2)
        PC_ARGS="jacobian.pc_type = pc_jacobi
                 pc_jacobi.verbose = true
                 pc_jacobi.max_iter = 200
                 pc_jacobi.omega = 0.667
                 pc_jacobi.relative_tolerance = 1e-6"
        MM_WIDTH_ARGS="implicit_evolve.mass_matrices_pc_width = 2"
        ;;
    ccmlmg)
        PC_ARGS="jacobian.pc_type = pc_curl_curl_mlmg
                 pc_curl_curl_mlmg.verbose = false
                 pc_curl_curl_mlmg.max_iter = 10
                 pc_curl_curl_mlmg.relative_tolerance = 1e-4"
        ;;
    petsc)
        LINEAR_SOLVER="petsc_ksp"
        if [[ "$LCHOST" == "matrix" ]]; then
            PETSC_FLAGS="-pc_type asm -pc_asm_overlap 32 -sub_pc_type lu -use_gpu_aware_mpi 0"
        elif [[ "$LCHOST" == "tuolumne" ]]; then
            PETSC_FLAGS="-pc_type asm -pc_asm_overlap 32 -sub_pc_type lu"
        else
            PETSC_FLAGS="-pc_type lu"
        fi
        PC_ARGS="jacobian.pc_type = pc_petsc"
        ;;
    nopc)
        PC_ARGS="jacobian.pc_type = none"
        ;;
    *)
        echo "ERROR: Unknown PC_TYPE=$PC_TYPE (expected jacobi, jacobi_w1, jacobi_w2, ccmlmg, petsc, nopc)"
        exit 1
        ;;
esac

# Paths
rootdir=$PWD
INP_FILE=$rootdir/common/planar_pinch_${DIM}.in
EXEC=${WARPX_BUILD}/build/bin/warpx.${DIM}*
EXEC=$(ls $EXEC 2>/dev/null | head -1)

if [ ! -f "$EXEC" ]; then
    echo "ERROR: Executable not found at ${WARPX_BUILD}/build/bin/warpx.${DIM}*"
    exit 1
fi
echo "Executable: $EXEC"

# Create run directory
dirname=".run_${DIM}_${PC_TYPE}.${LCHOST}"
if [ -d "$dirname" ]; then
    echo "Removing existing directory $dirname"
    rm -rf "$dirname"
fi
mkdir -p "$dirname"
cd "$dirname"

cp "$INP_FILE" .
INP=$(ls *.in)

# Platform-specific run command
NNODE=1
export OMP_NUM_THREADS=1
runcmd=""
if [[ "$LCHOST" == "dane" ]]; then
    runcmd="srun -n $NTASKS -p pdebug"
elif [[ "$LCHOST" == "matrix" ]]; then
    NGPU=$NTASKS
    runcmd="srun -n $NTASKS -G $NGPU -N $NNODE -p pdebug"
elif [[ "$LCHOST" == "tuolumne" ]]; then
    runcmd="flux run --exclusive --nodes=$NNODE --ntasks $NTASKS -q=pdebug"
fi

echo "Running: $DIM, pc_type=$PC_TYPE, max_step=$MAX_STEP"
echo "Command: $runcmd $EXEC ..."

$runcmd $EXEC $INP \
    $NX_ARGS \
    $NP_ARGS \
    max_step = $MAX_STEP \
    implicit_evolve.nonlinear_solver = newton \
    newton.linear_solver = $LINEAR_SOLVER \
    $PC_ARGS \
    $MM_WIDTH_ARGS \
    $PETSC_FLAGS \
    "$@" \
    2>&1 > out.${LCHOST}.log

cd "$rootdir"
echo "Done. Output in $dirname/out.${LCHOST}.log"
