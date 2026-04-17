#!/bin/bash
#
# Run all PC comparison cases for a given dimension.
# Usage: bash run_all.sh <DIM> [MAX_STEP]
#
# Example:
#   bash run_all.sh 1d 100
#   bash run_all.sh 2d 10
#

DIM=${1:?  "Usage: $0 <DIM> [MAX_STEP]"}
MAX_STEP=${2:-10}

echo "================================================"
echo " Running all PC types for DIM=$DIM, steps=$MAX_STEP"
echo "================================================"

for PC in nopc jacobi jacobi_w1 jacobi_w2 ccmlmg petsc; do
    echo ""
    echo "================================================"
    echo " PC type: $PC"
    echo "================================================"
    bash common/run.sh $DIM $PC $MAX_STEP
done

echo ""
echo "================================================"
echo " All runs complete. Compare GMRES iteration counts:"
echo "================================================"
for PC in nopc jacobi jacobi_w1 jacobi_w2 ccmlmg petsc; do
    dir=".run_${DIM}_${PC}.${LCHOST}"
    if [ -f "$dir/out.${LCHOST}.log" ]; then
        total_gmres=$(grep -c "GMRES:" "$dir/out.${LCHOST}.log" 2>/dev/null || echo 0)
        echo "  $PC: $total_gmres total GMRES iterations"
    fi
done
