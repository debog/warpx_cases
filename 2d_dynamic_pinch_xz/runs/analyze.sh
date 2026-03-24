#!/bin/bash

# analyze.sh - Parse WarpX simulation logs and generate statistics report

# Detect platform
platform="${LCHOST}"
if [[ -z "$platform" ]]; then
    echo "Error: LCHOST environment variable not set"
    exit 1
fi

echo "Analyzing logs for platform: $platform"

# Find all run directories for this platform
run_dirs=($(ls -d .run_*.${platform}.* 2>/dev/null | sort))

if [[ ${#run_dirs[@]} -eq 0 ]]; then
    echo "Error: No run directories found for platform '$platform'"
    exit 1
fi

echo "Found ${#run_dirs[@]} run directories"

# Output file
report_file="report_${platform}.md"

# Initialize report
cat > "$report_file" <<EOF
# WarpX Simulation Analysis Report - ${platform}

Analysis generated on: $(date)

## Iteration Statistics

| Case Name                        | Total GMRES | Total Newton | Avg GMRES/Step | Avg Newton/Step | Avg GMRES/Newton |
|----------------------------------|-------------|--------------|----------------|-----------------|------------------|
EOF

# Process each run directory
for run_dir in "${run_dirs[@]}"; do
    # Extract case name from directory name
    # Directory format: .run_{case_name}.{platform}.nx{X}nz{X}npx{X}npz{X}
    case_name=$(echo "$run_dir" | sed -E "s/^\.run_(.*)\.${platform}\..*$/\1/")

    # Log file path
    log_file="${run_dir}/out.${platform}.log"

    if [[ ! -f "$log_file" ]]; then
        echo "Warning: Log file not found: $log_file"
        continue
    fi

    echo "Processing: $case_name"

    # Count iterations based on solver type
    # For Newton: Count non-zero iterations
    # Native format: "Newton: iteration =   X,"
    # PETSc format: "Newton (PETSc SNES): iter = X,"

    # Count GMRES iterations
    # Native format: "GMRES: iter = X,"
    # PETSc format: "GMRES (PETSc KSP): iter = X,"
    gmres_count=$(grep -E "(GMRES: iter =|GMRES \(PETSc KSP\): iter =)" "$log_file" | wc -l)

    # Count Newton iterations (excluding iteration=0)
    # For native solver
    newton_native=$(grep "Newton: iteration =" "$log_file" | grep -v "iteration =   0" | wc -l)
    # For PETSc solver
    newton_petsc=$(grep "Newton (PETSc SNES): iter =" "$log_file" | grep -v "iter = 0" | wc -l)
    newton_count=$((newton_native + newton_petsc))

    # Count timesteps
    timestep_count=$(grep -c "STEP .* ends\." "$log_file")

    # Calculate averages (rounded to nearest integer)
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

    # Write to report with proper padding
    printf "| %-32s | %-11s | %-12s | %-14s | %-15s | %-16s |\n" \
        "$case_name" "$gmres_count" "$newton_count" "$avg_gmres_per_step" "$avg_newton_per_step" "$avg_gmres_per_newton" >> "$report_file"
done

# Add footer
cat >> "$report_file" <<EOF

## Notes

- **Total GMRES**: Total number of GMRES iterations across all timesteps
- **Total Newton**: Total number of Newton iterations across all timesteps (excluding iteration 0)
- **Avg GMRES/Step**: Average GMRES iterations per timestep
- **Avg Newton/Step**: Average Newton iterations per timestep
- **Avg GMRES/Newton**: Average GMRES iterations per Newton iteration

EOF

echo ""
echo "Analysis complete. Report written to: $report_file"
