#!/bin/bash

# analyze.sh - Parse WarpX simulation logs and generate statistics report

# Detect platform
platform="${LCHOST}"
if [[ -z "$platform" ]]; then
    echo "Error: LCHOST environment variable not set"
    exit 1
fi

echo "Analyzing logs for platform: $platform"

# Find all run directories for this platform in both SemiImpl and ThetaImpl parent directories
run_dirs_semi=($(find .run_SemiImpl -name "*.${platform}.*" -type d 2>/dev/null | sort))
run_dirs_theta=($(find .run_ThetaImpl -name "*.${platform}.*" -type d 2>/dev/null | sort))

total_dirs=$((${#run_dirs_semi[@]} + ${#run_dirs_theta[@]}))

if [[ $total_dirs -eq 0 ]]; then
    echo "Error: No run directories found for platform '$platform'"
    exit 1
fi

echo "Found ${#run_dirs_semi[@]} SemiImpl run directories"
echo "Found ${#run_dirs_theta[@]} ThetaImpl run directories"
echo "Total: $total_dirs run directories"

# Output file
report_file="report_${platform}.md"

# Function to parse case directory and extract stats
parse_case_stats() {
    local run_dir=$1
    local log_file="${run_dir}/out.${platform}.log"

    if [[ ! -f "$log_file" ]]; then
        echo "N/A N/A N/A N/A N/A"
        return
    fi

    # Count GMRES iterations
    local gmres_count=$(grep -E "(GMRES: iter =|GMRES \(PETSc KSP\): iter =)" "$log_file" | wc -l)

    # Count Newton iterations (excluding iteration=0)
    local newton_native=$(grep "Newton: iteration =" "$log_file" | grep -v "iteration =   0" | wc -l)
    local newton_petsc=$(grep "Newton (PETSc SNES): iter =" "$log_file" | grep -v "iter = 0" | wc -l)
    local newton_count=$((newton_native + newton_petsc))

    # Count timesteps
    local timestep_count=$(grep -c "STEP .* ends\." "$log_file")

    # Calculate averages (rounded to nearest integer)
    local avg_gmres_per_step avg_newton_per_step avg_gmres_per_newton
    if [[ $timestep_count -gt 0 ]]; then
        avg_gmres_per_step=$(printf %.0f $(echo "scale=4; $gmres_count / $timestep_count" | bc))
        avg_newton_per_step=$(printf %.0f $(echo "scale=4; $newton_count / $timestep_count" | bc))
    else
        avg_gmres_per_step="N/A"
        avg_newton_per_step="N/A"
    fi

    if [[ $newton_count -gt 0 ]]; then
        avg_gmres_per_newton=$(printf %.0f $(echo "scale=4; $gmres_count / $newton_count" | bc))
    else
        avg_gmres_per_newton="N/A"
    fi

    echo "$gmres_count $newton_count $avg_gmres_per_step $avg_newton_per_step $avg_gmres_per_newton"
}

# Function to find case directory
find_case_dir() {
    local integrator=$1  # SemiImpl or ThetaImpl
    local pc=$2          # noPC, JacobiPC, etc.
    local mmw=$3         # 0, 1, 2, or "none"
    local solver=$4      # native_jfnk, petsc_ksp, petsc_snes

    local parent_dir=".run_${integrator}"
    local case_pattern

    if [[ "$mmw" == "none" ]]; then
        case_pattern="${pc}.${solver}.${platform}.*"
    else
        case_pattern="${pc}_mmw${mmw}.${solver}.${platform}.*"
    fi

    local dir=$(find "$parent_dir" -name "$case_pattern" -type d 2>/dev/null | head -1)
    echo "$dir"
}

# Initialize report
cat > "$report_file" <<EOF
# WarpX Simulation Analysis Report - ${platform}

Analysis generated on: $(date)

## Semi-Implicit
---------------

| Preconditioner | Solver      | MMW | Total GMRES | Total Newton | Avg GMRES/Step | Avg Newton/Step | Avg GMRES/Newton |
|----------------|-------------|-----|-------------|--------------|----------------|-----------------|------------------|
EOF

# Process SemiImpl cases
# Order: noPC, JacobiPC (mmw 0,1,2), PETScPCASMwLU (mmw 0,1,2), PETScPCLU (mmw 0,1,2), PETScPCSOR (mmw 0,1,2)

for solver in native_jfnk petsc_ksp petsc_snes; do
    case_dir=$(find_case_dir "SemiImpl" "noPC" "none" "$solver")
    if [[ -n "$case_dir" ]]; then
        echo "Processing: SemiImpl noPC $solver"
        stats=($(parse_case_stats "$case_dir"))
        printf "| %-14s | %-11s | %-3s | %-11s | %-12s | %-14s | %-15s | %-16s |\n" \
            "None" "$solver" "--" "${stats[0]}" "${stats[1]}" "${stats[2]}" "${stats[3]}" "${stats[4]}" >> "$report_file"
    fi
done

for mmw in 0 1 2; do
    for solver in native_jfnk petsc_ksp petsc_snes; do
        case_dir=$(find_case_dir "SemiImpl" "JacobiPC" "$mmw" "$solver")
        if [[ -n "$case_dir" ]]; then
            echo "Processing: SemiImpl JacobiPC mmw$mmw $solver"
            stats=($(parse_case_stats "$case_dir"))
            printf "| %-14s | %-11s | %-3s | %-11s | %-12s | %-14s | %-15s | %-16s |\n" \
                "Jacobi" "$solver" "$mmw" "${stats[0]}" "${stats[1]}" "${stats[2]}" "${stats[3]}" "${stats[4]}" >> "$report_file"
        fi
    done
done

for mmw in 0 1 2; do
    for solver in petsc_ksp petsc_snes; do
        case_dir=$(find_case_dir "SemiImpl" "PETScPCASMwLU" "$mmw" "$solver")
        if [[ -n "$case_dir" ]]; then
            echo "Processing: SemiImpl PETScPCASMwLU mmw$mmw $solver"
            stats=($(parse_case_stats "$case_dir"))
            printf "| %-14s | %-11s | %-3s | %-11s | %-12s | %-14s | %-15s | %-16s |\n" \
                "ASM-LU" "$solver" "$mmw" "${stats[0]}" "${stats[1]}" "${stats[2]}" "${stats[3]}" "${stats[4]}" >> "$report_file"
        fi
    done
done

for mmw in 0 1 2; do
    for solver in petsc_ksp petsc_snes; do
        case_dir=$(find_case_dir "SemiImpl" "PETScPCLU" "$mmw" "$solver")
        if [[ -n "$case_dir" ]]; then
            echo "Processing: SemiImpl PETScPCLU mmw$mmw $solver"
            stats=($(parse_case_stats "$case_dir"))
            printf "| %-14s | %-11s | %-3s | %-11s | %-12s | %-14s | %-15s | %-16s |\n" \
                "LU" "$solver" "$mmw" "${stats[0]}" "${stats[1]}" "${stats[2]}" "${stats[3]}" "${stats[4]}" >> "$report_file"
        fi
    done
done

for mmw in 0 1 2; do
    for solver in petsc_ksp petsc_snes; do
        case_dir=$(find_case_dir "SemiImpl" "PETScPCSOR" "$mmw" "$solver")
        if [[ -n "$case_dir" ]]; then
            echo "Processing: SemiImpl PETScPCSOR mmw$mmw $solver"
            stats=($(parse_case_stats "$case_dir"))
            printf "| %-14s | %-11s | %-3s | %-11s | %-12s | %-14s | %-15s | %-16s |\n" \
                "SOR" "$solver" "$mmw" "${stats[0]}" "${stats[1]}" "${stats[2]}" "${stats[3]}" "${stats[4]}" >> "$report_file"
        fi
    done
done

# Add ThetaImpl section
cat >> "$report_file" <<EOF

## Theta-Implicit
----------------

| Preconditioner | Solver      | MMW | Total GMRES | Total Newton | Avg GMRES/Step | Avg Newton/Step | Avg GMRES/Newton |
|----------------|-------------|-----|-------------|--------------|----------------|-----------------|------------------|
EOF

# Process ThetaImpl cases
# Order: noPC, JacobiPC (mmw 0,1,2), PETScPCASMwLU (mmw 0,1,2), PETScPCLU (mmw 0,1,2)

for solver in native_jfnk petsc_ksp petsc_snes; do
    case_dir=$(find_case_dir "ThetaImpl" "noPC" "none" "$solver")
    if [[ -n "$case_dir" ]]; then
        echo "Processing: ThetaImpl noPC $solver"
        stats=($(parse_case_stats "$case_dir"))
        printf "| %-14s | %-11s | %-3s | %-11s | %-12s | %-14s | %-15s | %-16s |\n" \
            "None" "$solver" "--" "${stats[0]}" "${stats[1]}" "${stats[2]}" "${stats[3]}" "${stats[4]}" >> "$report_file"
    fi
done

for mmw in 0 1 2; do
    for solver in native_jfnk petsc_ksp petsc_snes; do
        case_dir=$(find_case_dir "ThetaImpl" "JacobiPC" "$mmw" "$solver")
        if [[ -n "$case_dir" ]]; then
            echo "Processing: ThetaImpl JacobiPC mmw$mmw $solver"
            stats=($(parse_case_stats "$case_dir"))
            printf "| %-14s | %-11s | %-3s | %-11s | %-12s | %-14s | %-15s | %-16s |\n" \
                "Jacobi" "$solver" "$mmw" "${stats[0]}" "${stats[1]}" "${stats[2]}" "${stats[3]}" "${stats[4]}" >> "$report_file"
        fi
    done
done

for mmw in 0 1 2; do
    for solver in petsc_ksp petsc_snes; do
        case_dir=$(find_case_dir "ThetaImpl" "PETScPCASMwLU" "$mmw" "$solver")
        if [[ -n "$case_dir" ]]; then
            echo "Processing: ThetaImpl PETScPCASMwLU mmw$mmw $solver"
            stats=($(parse_case_stats "$case_dir"))
            printf "| %-14s | %-11s | %-3s | %-11s | %-12s | %-14s | %-15s | %-16s |\n" \
                "ASM-LU" "$solver" "$mmw" "${stats[0]}" "${stats[1]}" "${stats[2]}" "${stats[3]}" "${stats[4]}" >> "$report_file"
        fi
    done
done

for mmw in 0 1 2; do
    for solver in petsc_ksp petsc_snes; do
        case_dir=$(find_case_dir "ThetaImpl" "PETScPCLU" "$mmw" "$solver")
        if [[ -n "$case_dir" ]]; then
            echo "Processing: ThetaImpl PETScPCLU mmw$mmw $solver"
            stats=($(parse_case_stats "$case_dir"))
            printf "| %-14s | %-11s | %-3s | %-11s | %-12s | %-14s | %-15s | %-16s |\n" \
                "LU" "$solver" "$mmw" "${stats[0]}" "${stats[1]}" "${stats[2]}" "${stats[3]}" "${stats[4]}" >> "$report_file"
        fi
    done
done

# Add footer
cat >> "$report_file" <<EOF

## Notes

- **Preconditioner**: Preconditioning method (None, Jacobi, ASM-LU, LU, SOR)
- **Solver**: Nonlinear/linear solver (native_jfnk, petsc_ksp, petsc_snes)
- **MMW**: Mass matrix width for preconditioner (-- = not applicable)
- **Total GMRES**: Total GMRES iterations across all timesteps
- **Total Newton**: Total Newton iterations across all timesteps (excluding iteration 0)
- **Avg GMRES/Step**: Average GMRES iterations per timestep
- **Avg Newton/Step**: Average Newton iterations per timestep
- **Avg GMRES/Newton**: Average GMRES iterations per Newton iteration

EOF

echo ""
echo "Analysis complete. Report written to: $report_file"
