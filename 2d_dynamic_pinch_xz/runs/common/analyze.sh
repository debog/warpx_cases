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
        echo "N/A N/A N/A N/A N/A N/A"
        return
    fi

    # Count linear iterations (GMRES, weighted_jacobi, chebyshev, etc.)
    # Look for GMRES patterns
    local gmres_count=$(grep -E "(GMRES: iter =|GMRES \(PETSc KSP\): iter =)" "$log_file" | wc -l)
    # Look for generic linear solver iterations (for weighted_jacobi, chebyshev, etc.)
    local linear_count=$(grep -E "Linear solver: iter =" "$log_file" | wc -l)
    # Total linear iterations
    local total_linear=$((gmres_count + linear_count))

    # Count Newton iterations (excluding iteration=0)
    local newton_native=$(grep "Newton: iteration =" "$log_file" | grep -v "iteration =   0" | wc -l)
    local newton_petsc=$(grep "Newton (PETSc SNES): iter =" "$log_file" | grep -v "iter = 0" | wc -l)
    local newton_count=$((newton_native + newton_petsc))

    # Count timesteps
    local timestep_count=$(grep -c "STEP .* ends\." "$log_file")

    # Extract walltime (look for "TotalTime" or "Total Time" at end of log)
    local walltime=$(grep -E "TotalTime|Total Time" "$log_file" | tail -1 | awk '{print $NF}')
    if [[ -z "$walltime" ]]; then
        walltime="N/A"
    fi

    # Calculate averages (rounded to nearest integer)
    local avg_linear_per_step avg_newton_per_step avg_linear_per_newton
    if [[ $timestep_count -gt 0 ]]; then
        avg_linear_per_step=$(printf %.0f $(echo "scale=4; $total_linear / $timestep_count" | bc))
        avg_newton_per_step=$(printf %.0f $(echo "scale=4; $newton_count / $timestep_count" | bc))
    else
        avg_linear_per_step="N/A"
        avg_newton_per_step="N/A"
    fi

    if [[ $newton_count -gt 0 ]]; then
        avg_linear_per_newton=$(printf %.0f $(echo "scale=4; $total_linear / $newton_count" | bc))
    else
        avg_linear_per_newton="N/A"
    fi

    echo "$total_linear $newton_count $avg_linear_per_step $avg_newton_per_step $avg_linear_per_newton $walltime"
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

## Semi-Implicit (4 MPI ranks on Dane)
---------------

| PC     | Solver          | MMW | Linear | Newton | Linear/Step | Newton/Step | Linear/Newton | Walltime (s) |
|--------|-----------------|-----|--------|--------|-------------|-------------|---------------|--------------|
EOF

# Process SemiImpl cases
# Order: noPC, JacobiPC (mmw 0,1,2), ChebyshevPC (mmw 0,1,2), PETScPCASMwLU (mmw 0,1,2), PETScPCLU (mmw 0,1,2), PETScPCSOR (mmw 0,1,2)

for solver in native_jfnk petsc_ksp weighted_jacobi chebyshev; do
    case_dir=$(find_case_dir "SemiImpl" "noPC" "none" "$solver")
    if [[ -n "$case_dir" ]]; then
        echo "Processing: SemiImpl noPC $solver"
        stats=($(parse_case_stats "$case_dir"))
        printf "| %-6s | %-15s | %-3s | %-6s | %-6s | %-11s | %-11s | %-13s | %-12s |\n" \
            "None" "$solver" "--" "${stats[0]}" "${stats[1]}" "${stats[2]}" "${stats[3]}" "${stats[4]}" "${stats[5]}" >> "$report_file"
    fi
done

for mmw in 0 1 2; do
    for solver in native_jfnk petsc_ksp; do
        case_dir=$(find_case_dir "SemiImpl" "JacobiPC" "$mmw" "$solver")
        if [[ -n "$case_dir" ]]; then
            echo "Processing: SemiImpl JacobiPC mmw$mmw $solver"
            stats=($(parse_case_stats "$case_dir"))
            printf "| %-6s | %-15s | %-3s | %-6s | %-6s | %-11s | %-11s | %-13s | %-12s |\n" \
                "Jacobi" "$solver" "$mmw" "${stats[0]}" "${stats[1]}" "${stats[2]}" "${stats[3]}" "${stats[4]}" "${stats[5]}" >> "$report_file"
        fi
    done
done

for mmw in 0 1 2; do
    for solver in native_jfnk petsc_ksp; do
        case_dir=$(find_case_dir "SemiImpl" "ChebyshevPC" "$mmw" "$solver")
        if [[ -n "$case_dir" ]]; then
            echo "Processing: SemiImpl ChebyshevPC mmw$mmw $solver"
            stats=($(parse_case_stats "$case_dir"))
            printf "| %-6s | %-15s | %-3s | %-6s | %-6s | %-11s | %-11s | %-13s | %-12s |\n" \
                "Cheby" "$solver" "$mmw" "${stats[0]}" "${stats[1]}" "${stats[2]}" "${stats[3]}" "${stats[4]}" "${stats[5]}" >> "$report_file"
        fi
    done
done

for mmw in 0 1 2; do
    case_dir=$(find_case_dir "SemiImpl" "PETScPCASMwLU" "$mmw" "petsc_ksp")
    if [[ -n "$case_dir" ]]; then
        echo "Processing: SemiImpl PETScPCASMwLU mmw$mmw petsc_ksp"
        stats=($(parse_case_stats "$case_dir"))
        printf "| %-6s | %-15s | %-3s | %-6s | %-6s | %-11s | %-11s | %-13s | %-12s |\n" \
            "ASM-LU" "petsc_ksp" "$mmw" "${stats[0]}" "${stats[1]}" "${stats[2]}" "${stats[3]}" "${stats[4]}" "${stats[5]}" >> "$report_file"
    fi
done

# PETScPCLU only available on Dane (excluded from Matrix and Tuolumne)
if [[ "x$platform" != "xmatrix" && "x$platform" != "xtuolumne" ]]; then
    for mmw in 0 1 2; do
        case_dir=$(find_case_dir "SemiImpl" "PETScPCLU" "$mmw" "petsc_ksp")
        if [[ -n "$case_dir" ]]; then
            echo "Processing: SemiImpl PETScPCLU mmw$mmw petsc_ksp"
            stats=($(parse_case_stats "$case_dir"))
            printf "| %-6s | %-15s | %-3s | %-6s | %-6s | %-11s | %-11s | %-13s | %-12s |\n" \
                "LU" "petsc_ksp" "$mmw" "${stats[0]}" "${stats[1]}" "${stats[2]}" "${stats[3]}" "${stats[4]}" "${stats[5]}" >> "$report_file"
        fi
    done
fi

for mmw in 0 1 2; do
    case_dir=$(find_case_dir "SemiImpl" "PETScPCSOR" "$mmw" "petsc_ksp")
    if [[ -n "$case_dir" ]]; then
        echo "Processing: SemiImpl PETScPCSOR mmw$mmw petsc_ksp"
        stats=($(parse_case_stats "$case_dir"))
        printf "| %-6s | %-15s | %-3s | %-6s | %-6s | %-11s | %-11s | %-13s | %-12s |\n" \
            "SOR" "petsc_ksp" "$mmw" "${stats[0]}" "${stats[1]}" "${stats[2]}" "${stats[3]}" "${stats[4]}" "${stats[5]}" >> "$report_file"
    fi
done

# Add ThetaImpl section
cat >> "$report_file" <<EOF

## Theta-Implicit (4 MPI ranks on Dane)
----------------

| PC     | Solver          | MMW | Linear | Newton | Linear/Step | Newton/Step | Linear/Newton | Walltime (s) |
|--------|-----------------|-----|--------|--------|-------------|-------------|---------------|--------------|
EOF

# Process ThetaImpl cases
# Order: noPC, JacobiPC (mmw 0,1,2), ChebyshevPC (mmw 0,1,2), PETScPCASMwLU (mmw 0,1,2), PETScPCLU (mmw 0,1,2)

for solver in native_jfnk petsc_ksp weighted_jacobi chebyshev; do
    case_dir=$(find_case_dir "ThetaImpl" "noPC" "none" "$solver")
    if [[ -n "$case_dir" ]]; then
        echo "Processing: ThetaImpl noPC $solver"
        stats=($(parse_case_stats "$case_dir"))
        printf "| %-6s | %-15s | %-3s | %-6s | %-6s | %-11s | %-11s | %-13s | %-12s |\n" \
            "None" "$solver" "--" "${stats[0]}" "${stats[1]}" "${stats[2]}" "${stats[3]}" "${stats[4]}" "${stats[5]}" >> "$report_file"
    fi
done

for mmw in 0 1 2; do
    for solver in native_jfnk petsc_ksp; do
        case_dir=$(find_case_dir "ThetaImpl" "JacobiPC" "$mmw" "$solver")
        if [[ -n "$case_dir" ]]; then
            echo "Processing: ThetaImpl JacobiPC mmw$mmw $solver"
            stats=($(parse_case_stats "$case_dir"))
            printf "| %-6s | %-15s | %-3s | %-6s | %-6s | %-11s | %-11s | %-13s | %-12s |\n" \
                "Jacobi" "$solver" "$mmw" "${stats[0]}" "${stats[1]}" "${stats[2]}" "${stats[3]}" "${stats[4]}" "${stats[5]}" >> "$report_file"
        fi
    done
done

for mmw in 0 1 2; do
    for solver in native_jfnk petsc_ksp; do
        case_dir=$(find_case_dir "ThetaImpl" "ChebyshevPC" "$mmw" "$solver")
        if [[ -n "$case_dir" ]]; then
            echo "Processing: ThetaImpl ChebyshevPC mmw$mmw $solver"
            stats=($(parse_case_stats "$case_dir"))
            printf "| %-6s | %-15s | %-3s | %-6s | %-6s | %-11s | %-11s | %-13s | %-12s |\n" \
                "Cheby" "$solver" "$mmw" "${stats[0]}" "${stats[1]}" "${stats[2]}" "${stats[3]}" "${stats[4]}" "${stats[5]}" >> "$report_file"
        fi
    done
done

for mmw in 0 1 2; do
    case_dir=$(find_case_dir "ThetaImpl" "PETScPCASMwLU" "$mmw" "petsc_ksp")
    if [[ -n "$case_dir" ]]; then
        echo "Processing: ThetaImpl PETScPCASMwLU mmw$mmw petsc_ksp"
        stats=($(parse_case_stats "$case_dir"))
        printf "| %-6s | %-15s | %-3s | %-6s | %-6s | %-11s | %-11s | %-13s | %-12s |\n" \
            "ASM-LU" "petsc_ksp" "$mmw" "${stats[0]}" "${stats[1]}" "${stats[2]}" "${stats[3]}" "${stats[4]}" "${stats[5]}" >> "$report_file"
    fi
done

# PETScPCLU only available on Dane (excluded from Matrix and Tuolumne)
if [[ "x$platform" != "xmatrix" && "x$platform" != "xtuolumne" ]]; then
    for mmw in 0 1 2; do
        case_dir=$(find_case_dir "ThetaImpl" "PETScPCLU" "$mmw" "petsc_ksp")
        if [[ -n "$case_dir" ]]; then
            echo "Processing: ThetaImpl PETScPCLU mmw$mmw petsc_ksp"
            stats=($(parse_case_stats "$case_dir"))
            printf "| %-6s | %-15s | %-3s | %-6s | %-6s | %-11s | %-11s | %-13s | %-12s |\n" \
                "LU" "petsc_ksp" "$mmw" "${stats[0]}" "${stats[1]}" "${stats[2]}" "${stats[3]}" "${stats[4]}" "${stats[5]}" >> "$report_file"
        fi
    done
fi

# Add footer
cat >> "$report_file" <<EOF

## Notes

- **PC**: Preconditioner (None, Jacobi, Cheby=Chebyshev, ASM-LU, LU, SOR)
- **Solver**: Nonlinear/linear solver (native_jfnk, petsc_ksp, weighted_jacobi, chebyshev)
- **MMW**: Mass matrix width (-- = not applicable)
- **Linear**: Total linear solver iterations across all timesteps (GMRES, weighted Jacobi, Chebyshev, etc.)
- **Newton**: Total Newton iterations across all timesteps (excluding iteration 0)
- **Linear/Step**: Average linear iterations per timestep
- **Newton/Step**: Average Newton iterations per timestep
- **Linear/Newton**: Average linear iterations per Newton iteration

EOF

echo ""
echo "Analysis complete. Report written to: $report_file"
