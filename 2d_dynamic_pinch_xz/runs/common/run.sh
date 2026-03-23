#!/bin/bash

# Configuration
max_step=10
nx="224"
nz="16"
npx="20"
npz="10"
NNODE=1
export OMP_NUM_THREADS=1
DIM=2d

# Define PC options
declare -A PC_OPTIONS
PC_OPTIONS["noPC"]="none"
PC_OPTIONS["JacobiPC"]="pc_jacobi"
PC_OPTIONS["CurlCurlMLMGPC"]="pc_curl_curl_mlmg"
PC_OPTIONS["PETScPCASMwLU"]="pc_petsc"
PC_OPTIONS["PETScPCLU"]="pc_petsc"

# Define solver options
declare -A SOLVER_OPTIONS
SOLVER_OPTIONS["native_jfnk"]="newton"
SOLVER_OPTIONS["petsc_ksp"]="newton|petsc_ksp"
SOLVER_OPTIONS["petsc_snes"]="petsc_snes"

# Mass matrix width options (for PCs other than noPC)
MMW_OPTIONS=(0 1 2)

# Function to print help
print_help() {
    cat << EOF
Usage: ./run.sh [OPTIONS]

Options:
  -l              List all available cases
  -c <case>...    Create and run one or more cases (supports wildcards)
  -a              Run all cases
  -h              Print this help message

Case Format: <pc_option>.<solver_option>
  or: <pc_option>_mmw<N>.<solver_option> (for PCs with mass matrix width)

PC Options:
  noPC                 - No preconditioner (all solvers)
  JacobiPC_mmw<N>      - Jacobi preconditioner (mmw: 0, 1, 2; all solvers)
  CurlCurlMLMGPC       - CurlCurl MLMG preconditioner (mmw fixed at 0; all solvers)
  PETScPCASMwLU_mmw<N> - PETSc ASM+LU preconditioner (mmw: 0, 1, 2; petsc_ksp/petsc_snes only)
  PETScPCLU_mmw<N>     - PETSc LU preconditioner (mmw: 0, 1, 2; petsc_ksp/petsc_snes only)

Solver Options:
  native_jfnk  - Native JFNK (Newton with GMRES)
  petsc_ksp    - PETSc KSP linear solver
  petsc_snes   - PETSc SNES nonlinear solver

Platform Detection: Automatically detects Dane, Matrix, or Tuolumne

Examples:
  ./run.sh -l                                         # List all cases
  ./run.sh -c noPC.native_jfnk                        # Run single case
  ./run.sh -c noPC.petsc_ksp JacobiPC_mmw1.petsc_ksp  # Run multiple cases
  ./run.sh -c '*.petsc_ksp'                           # Run all petsc_ksp cases
  ./run.sh -c 'JacobiPC*'                             # Run all JacobiPC cases
  ./run.sh -c 'CurlCurlMLMGPC.*'                      # Run all CurlCurlMLMGPC cases
  ./run.sh -c 'PETScPCASMwLU*'                        # Run all PETScPCASMwLU cases
  ./run.sh -c 'PETScPCLU*'                            # Run all PETScPCLU cases
  ./run.sh -c '*.petsc_ksp' '*.petsc_snes'            # Run multiple patterns
  ./run.sh -c 'JacobiPC_mmw?.native_jfnk'             # Use ? for single char
  ./run.sh -a                                         # Run all cases

Note: When using wildcards, quote the pattern to prevent shell expansion

EOF
}

# Function to generate all possible case names
generate_all_cases() {
    local cases=()

    # noPC cases (no mmw)
    for solver in "${!SOLVER_OPTIONS[@]}"; do
        cases+=("noPC.${solver}")
    done

    # CurlCurlMLMGPC cases (mmw fixed at 0, no suffix)
    for solver in "${!SOLVER_OPTIONS[@]}"; do
        cases+=("CurlCurlMLMGPC.${solver}")
    done

    # JacobiPC cases (with mmw options, all solvers)
    for mmw in "${MMW_OPTIONS[@]}"; do
        for solver in "${!SOLVER_OPTIONS[@]}"; do
            cases+=("JacobiPC_mmw${mmw}.${solver}")
        done
    done

    # PETScPCASMwLU cases (with mmw options, only petsc_ksp and petsc_snes)
    for mmw in "${MMW_OPTIONS[@]}"; do
        for solver in "petsc_ksp" "petsc_snes"; do
            cases+=("PETScPCASMwLU_mmw${mmw}.${solver}")
        done
    done

    # PETScPCLU cases (with mmw options, only petsc_ksp and petsc_snes)
    for mmw in "${MMW_OPTIONS[@]}"; do
        for solver in "petsc_ksp" "petsc_snes"; do
            cases+=("PETScPCLU_mmw${mmw}.${solver}")
        done
    done

    echo "${cases[@]}"
}

# Function to list all cases
list_cases() {
    echo "Available cases:"
    echo ""

    local all_cases=($(generate_all_cases))
    for case in "${all_cases[@]}"; do
        echo "  $case"
    done

    echo ""
    echo "Total cases: ${#all_cases[@]}"
}

# Function to match pattern against all cases
match_cases() {
    local pattern=$1
    local all_cases=($(generate_all_cases))
    local matched_cases=()

    # Convert shell wildcard pattern to regex for matching
    for case in "${all_cases[@]}"; do
        # Use bash pattern matching
        if [[ "$case" == $pattern ]]; then
            matched_cases+=("$case")
        fi
    done

    echo "${matched_cases[@]}"
}

# Function to parse case name and get PC, MMW, and solver
parse_case() {
    local case_name=$1
    local pc_part=""
    local solver_part=""
    local mmw_val="1"

    # Split by last dot to separate PC and solver
    if [[ $case_name =~ ^(.+)\.([^.]+)$ ]]; then
        pc_part="${BASH_REMATCH[1]}"
        solver_part="${BASH_REMATCH[2]}"
    else
        echo "Error: Invalid case format. Expected <pc_option>.<solver_option>"
        return 1
    fi

    # Check if solver is valid
    if [[ ! -v SOLVER_OPTIONS[$solver_part] ]]; then
        echo "Error: Invalid solver option '$solver_part'"
        echo "Valid solvers: ${!SOLVER_OPTIONS[@]}"
        return 1
    fi

    # Parse PC and MMW
    if [[ $pc_part == "noPC" ]]; then
        PC_TYPE="noPC"
        MMW="1"
    elif [[ $pc_part == "CurlCurlMLMGPC" ]]; then
        PC_TYPE="CurlCurlMLMGPC"
        MMW="0"  # CurlCurlMLMGPC only supports mmw=0
    elif [[ $pc_part =~ ^(JacobiPC|PETScPCASMwLU|PETScPCLU)_mmw([0-2])$ ]]; then
        PC_TYPE="${BASH_REMATCH[1]}"
        MMW="${BASH_REMATCH[2]}"
    else
        echo "Error: Invalid PC option '$pc_part'"
        echo "Expected format: noPC, CurlCurlMLMGPC, or JacobiPC_mmw<0|1|2>/PETScPCASMwLU_mmw<0|1|2>/PETScPCLU_mmw<0|1|2>"
        return 1
    fi

    # Validate PC and solver combinations
    if [[ ("$PC_TYPE" == "PETScPCASMwLU" || "$PC_TYPE" == "PETScPCLU") && "$solver_part" == "native_jfnk" ]]; then
        echo "Error: $PC_TYPE is not compatible with native_jfnk solver"
        echo "Valid solvers for $PC_TYPE: petsc_ksp, petsc_snes"
        return 1
    fi

    SOLVER_TYPE="$solver_part"
    return 0
}

# Function to create run_case.sh script inside the run directory
create_run_case_script() {
    local dirname=$1
    local case_name=$2
    local warpx_params="$3"

    cat > "${dirname}/run_case.sh" << 'EOFSCRIPT'
#!/bin/bash

# This script runs the case from within the run directory
# Generated automatically by run.sh

# Get platform from environment variable
platform="$LCHOST"
if [ -z "$platform" ]; then
    echo "Error: LCHOST environment variable not set"
    exit 1
fi

# Read parameters from the current directory name and input file
EXEC=$(ls $WARPX_BUILD/build/bin/warpx.2d 2>/dev/null)
if [ -z "$EXEC" ]; then
    echo "Error: WarpX executable not found. Check WARPX_BUILD environment variable."
    exit 1
fi

INP=$(ls *.in 2>/dev/null | head -1)
if [ -z "$INP" ]; then
    echo "Error: No input file found in current directory"
    exit 1
fi

outfile="out.${platform}.log"
NNODE=1

# Determine run command based on platform
runcmd=""
if [[ "x$platform" == "xdane" ]]; then
    ntasks=64
    runcmd="srun -n $ntasks -p pdebug --export=ALL"
elif [[ "x$platform" == "xmatrix" ]]; then
    ntasks=4
    export OMP_NUM_THREADS=1
    runcmd="srun -n $ntasks -G $ntasks -N $NNODE -p pdebug --export=ALL"
elif [[ "x$platform" == "xtuolumne" ]]; then
    ntasks=4
    export OMP_NUM_THREADS=1
    runcmd="flux run --exclusive --nodes=$NNODE --ntasks $ntasks --verbose --setopt=mpibind=verbose:1 -q=pdebug --env OMP_NUM_THREADS=1"
else
    echo "Error: Unknown platform '$platform'"
    exit 1
fi

echo "Running WarpX from directory: $PWD"
echo "Platform: $platform"
echo "Input file: $INP"
echo "Output file: $outfile"
EOFSCRIPT

    # Append case-specific information and command
    cat >> "${dirname}/run_case.sh" << EOFSCRIPT
echo "Case: $case_name"

# Run WarpX with case-specific parameters
echo "Output writing to \$outfile (and displaying on screen)"
\$runcmd \$EXEC \$INP \\
$warpx_params \\
    2>&1 | tee \$outfile
exit_code=\${PIPESTATUS[0]}

if [ \$exit_code -eq 0 ]; then
    echo "Run completed successfully"
else
    echo "ERROR: Run failed with exit code \$exit_code"
    backtrace_files=\$(ls Backtrace.* 2>/dev/null | wc -l)
    if [ \$backtrace_files -gt 0 ]; then
        echo "Found \$backtrace_files backtrace file(s)"
    fi
    exit \$exit_code
fi
EOFSCRIPT

    chmod +x "${dirname}/run_case.sh"
}

# Function to run a specific case
run_case() {
    local case_name=$1

    # Parse case name
    if ! parse_case "$case_name"; then
        return 1
    fi

    echo "============================================"
    echo "Running case: $case_name"
    echo "PC Type: $PC_TYPE, MMW: $MMW, Solver: $SOLVER_TYPE"
    echo "============================================"

    # Build directory name
    if [ "$PC_TYPE" == "noPC" ]; then
        dir_prefix=".run_noPC.${SOLVER_TYPE}.${LCHOST}."
    elif [ "$PC_TYPE" == "CurlCurlMLMGPC" ]; then
        dir_prefix=".run_CurlCurlMLMGPC.${SOLVER_TYPE}.${LCHOST}."
    else
        dir_prefix=".run_${PC_TYPE}_mmw${MMW}.${SOLVER_TYPE}.${LCHOST}."
    fi

    rootdir=$PWD
    INP_FILE=$(ls $rootdir/common/*.in 2>/dev/null | head -1)
    if [ -z "$INP_FILE" ]; then
        echo "Error: No input file found in $rootdir/common/"
        return 1
    fi

    outfile="out.${LCHOST}.log"
    EXEC=$(ls $WARPX_BUILD/build/bin/warpx.${DIM} 2>/dev/null)
    if [ -z "$EXEC" ]; then
        echo "Error: WarpX executable not found. Check WARPX_BUILD environment variable."
        return 1
    fi
    echo "Executable file is ${EXEC}."

    # Create directory
    echo "Creating directory for nx=$nx, nz=$nz, npx=$npx, npz=$npz"
    dirname=$dir_prefix$(printf "nx%05dnz%05d" $nx $nz)$(printf "npx%03dnpz%03d" $npx $npz)

    if [ -d "$dirname" ]; then
        echo "  Deleting existing directory $dirname"
        rm -rf $dirname
    fi
    echo "  Creating directory $dirname"
    mkdir $dirname

    cd $dirname
    echo "  Copying input file"
    cp $INP_FILE .
    INP=$(ls *.in)

    # Determine run command based on platform
    runcmd=""
    addflags=""
    if [[ "x$LCHOST" == "xdane" ]]; then
        ntasks=64
        runcmd="srun -n $ntasks -p pdebug --export=ALL"
    elif [[ "x$LCHOST" == "xmatrix" ]]; then
        ntasks=4
        export OMP_NUM_THREADS=1  # Explicitly set for Matrix
        runcmd="srun -n $ntasks -G $ntasks -N $NNODE -p pdebug --export=ALL"
    elif [[ "x$LCHOST" == "xtuolumne" ]]; then
        ntasks=4
        export OMP_NUM_THREADS=1  # Explicitly set for Tuolumne
        runcmd="flux run --exclusive --nodes=$NNODE --ntasks $ntasks --verbose --setopt=mpibind=verbose:1 -q=pdebug --env OMP_NUM_THREADS=1"
    else
        echo "Error: Unknown platform. Set LCHOST environment variable."
        cd $rootdir
        return 1
    fi

    # Build WarpX parameters (without run command and redirection)
    warpx_params="amr.n_cell = $nx $nz \\
        my_constants.Nppc_x = $npx \\
        my_constants.Nppc_z = $npz \\
        max_step = $max_step"

    # Add solver-specific options
    case $SOLVER_TYPE in
        native_jfnk)
            warpx_params="$warpx_params \\
        implicit_evolve.nonlinear_solver = newton"
            ;;
        petsc_ksp)
            warpx_params="$warpx_params \\
        implicit_evolve.nonlinear_solver = newton \\
        newton.linear_solver = petsc_ksp"
            ;;
        petsc_snes)
            warpx_params="$warpx_params \\
        implicit_evolve.nonlinear_solver = petsc_snes"
            ;;
    esac

    # Add PC-specific options
    case $PC_TYPE in
        noPC)
            warpx_params="$warpx_params \\
        jacobian.pc_type = none \\
        implicit_evolve.mass_matrices_pc_width = $MMW"
            ;;
        JacobiPC)
            warpx_params="$warpx_params \\
        jacobian.pc_type = pc_jacobi \\
        implicit_evolve.mass_matrices_pc_width = $MMW"
            ;;
        CurlCurlMLMGPC)
            warpx_params="$warpx_params \\
        jacobian.pc_type = pc_curl_curl_mlmg \\
        pc_curl_curl_mlmg.verbose = false \\
        pc_curl_curl_mlmg.max_iter = 10 \\
        pc_curl_curl_mlmg.relative_tolerance = 1e-4 \\
        implicit_evolve.mass_matrices_pc_width = $MMW"
            ;;
        PETScPCASMwLU)
            # PETSc ASM+LU PC requires platform-specific options
            pctype="asm -pc_asm_overlap 32 -sub_pc_type lu"
            if [[ "x$LCHOST" == "xdane" ]]; then
                addflags="-mat_view ::ascii_info"
            elif [[ "x$LCHOST" == "xmatrix" ]]; then
                pctype="$pctype -log_view_gpu_time"
                addflags="-use_gpu_aware_mpi 0 -mat_view ::ascii_info"
            elif [[ "x$LCHOST" == "xtuolumne" ]]; then
                pctype="$pctype -log_view_gpu_time"
                addflags="-mat_view ::ascii_info"
            fi
            warpx_params="$warpx_params \\
        jacobian.pc_type = pc_petsc \\
        implicit_evolve.mass_matrices_pc_width = $MMW \\
        -pc_type $pctype \\
        -log_view \\
        ${addflags}"
            ;;
        PETScPCLU)
            # PETSc LU PC requires platform-specific options
            pctype="lu"
            if [[ "x$LCHOST" == "xdane" ]]; then
                addflags="-mat_view ::ascii_info"
            elif [[ "x$LCHOST" == "xmatrix" ]]; then
                addflags="-use_gpu_aware_mpi 0 -mat_view ::ascii_info -log_view_gpu_time"
            elif [[ "x$LCHOST" == "xtuolumne" ]]; then
                addflags="-mat_view ::ascii_info -log_view_gpu_time"
            fi
            warpx_params="$warpx_params \\
        jacobian.pc_type = pc_petsc \\
        implicit_evolve.mass_matrices_pc_width = $MMW \\
        -pc_type $pctype \\
        -log_view \\
        ${addflags}"
            ;;
    esac

    # Create the run_case.sh script with parameters
    create_run_case_script "$PWD" "$case_name" "$warpx_params"

    # Build full command with run command and output redirection
    warpx_cmd="$runcmd $EXEC $INP \\
        $warpx_params \\
        > $outfile 2>&1"

    echo "  Running WarpX with input file $INP"
    echo "  Output writing to $outfile"
    eval $warpx_cmd
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "  Run completed successfully"
        cd $rootdir
        echo ""
        echo "Case $case_name completed. Results in $dirname"
        echo "============================================"
        return 0
    else
        echo "  ERROR: Run failed with exit code $exit_code"
        # Check for backtrace files
        backtrace_files=$(ls Backtrace.* 2>/dev/null | wc -l)
        if [ $backtrace_files -gt 0 ]; then
            echo "  Found $backtrace_files backtrace file(s)"
        fi
        cd $rootdir
        echo ""
        echo "Case $case_name FAILED. Check logs in $dirname"
        echo "============================================"
        return 1
    fi
}

# Function to run all cases
run_all_cases() {
    echo "Running all cases..."
    echo ""

    local failed_cases=()
    local total=0
    local succeeded=0

    # Run noPC cases
    for solver in "${!SOLVER_OPTIONS[@]}"; do
        total=$((total + 1))
        if run_case "noPC.${solver}"; then
            succeeded=$((succeeded + 1))
        else
            failed_cases+=("noPC.${solver}")
        fi
    done

    # Run CurlCurlMLMGPC cases (no mmw suffix)
    for solver in "${!SOLVER_OPTIONS[@]}"; do
        total=$((total + 1))
        if run_case "CurlCurlMLMGPC.${solver}"; then
            succeeded=$((succeeded + 1))
        else
            failed_cases+=("CurlCurlMLMGPC.${solver}")
        fi
    done

    # Run JacobiPC cases with mmw options (all solvers)
    for mmw in "${MMW_OPTIONS[@]}"; do
        for solver in "${!SOLVER_OPTIONS[@]}"; do
            total=$((total + 1))
            if run_case "JacobiPC_mmw${mmw}.${solver}"; then
                succeeded=$((succeeded + 1))
            else
                failed_cases+=("JacobiPC_mmw${mmw}.${solver}")
            fi
        done
    done

    # Run PETScPCASMwLU cases with mmw options (only petsc_ksp and petsc_snes)
    for mmw in "${MMW_OPTIONS[@]}"; do
        for solver in "petsc_ksp" "petsc_snes"; do
            total=$((total + 1))
            if run_case "PETScPCASMwLU_mmw${mmw}.${solver}"; then
                succeeded=$((succeeded + 1))
            else
                failed_cases+=("PETScPCASMwLU_mmw${mmw}.${solver}")
            fi
        done
    done

    # Run PETScPCLU cases with mmw options (only petsc_ksp and petsc_snes)
    for mmw in "${MMW_OPTIONS[@]}"; do
        for solver in "petsc_ksp" "petsc_snes"; do
            total=$((total + 1))
            if run_case "PETScPCLU_mmw${mmw}.${solver}"; then
                succeeded=$((succeeded + 1))
            else
                failed_cases+=("PETScPCLU_mmw${mmw}.${solver}")
            fi
        done
    done

    # Summary
    echo ""
    echo "============================================"
    echo "All cases completed"
    echo "Total: $total, Succeeded: $succeeded, Failed: ${#failed_cases[@]}"
    if [ ${#failed_cases[@]} -gt 0 ]; then
        echo ""
        echo "Failed cases:"
        for case in "${failed_cases[@]}"; do
            echo "  - $case"
        done
    fi
    echo "============================================"
}

# Function to run multiple cases
run_multiple_cases() {
    local patterns=("$@")
    local cases_to_run=()
    local failed_cases=()
    local succeeded=0

    # Collect all unique cases from patterns
    for pattern in "${patterns[@]}"; do
        local matched=($(match_cases "$pattern"))
        if [ ${#matched[@]} -eq 0 ]; then
            echo "Warning: No cases matched pattern '$pattern'"
        else
            for case in "${matched[@]}"; do
                # Add to cases_to_run if not already present
                if [[ ! " ${cases_to_run[@]} " =~ " ${case} " ]]; then
                    cases_to_run+=("$case")
                fi
            done
        fi
    done

    if [ ${#cases_to_run[@]} -eq 0 ]; then
        echo "Error: No cases matched any of the provided patterns"
        return 1
    fi

    echo "============================================"
    echo "Matched ${#cases_to_run[@]} case(s) to run:"
    for case in "${cases_to_run[@]}"; do
        echo "  - $case"
    done
    echo "============================================"
    echo ""

    # Run all matched cases
    for case in "${cases_to_run[@]}"; do
        if run_case "$case"; then
            succeeded=$((succeeded + 1))
        else
            failed_cases+=("$case")
        fi
    done

    # Summary
    echo ""
    echo "============================================"
    echo "Multiple cases completed"
    echo "Total: ${#cases_to_run[@]}, Succeeded: $succeeded, Failed: ${#failed_cases[@]}"
    if [ ${#failed_cases[@]} -gt 0 ]; then
        echo ""
        echo "Failed cases:"
        for case in "${failed_cases[@]}"; do
            echo "  - $case"
        done
    fi
    echo "============================================"

    [ ${#failed_cases[@]} -eq 0 ] && return 0 || return 1
}

# Main script logic
if [ $# -eq 0 ]; then
    print_help
    exit 0
fi

# Parse options
while getopts "lc:ah" opt; do
    case $opt in
        l)
            list_cases
            exit 0
            ;;
        c)
            # Collect all arguments after -c
            cases_args=("$OPTARG")
            # Shift to get additional arguments after -c
            shift $((OPTIND - 1))
            while [ $# -gt 0 ] && [[ ! "$1" =~ ^- ]]; do
                cases_args+=("$1")
                shift
            done

            run_multiple_cases "${cases_args[@]}"
            exit $?
            ;;
        a)
            run_all_cases
            exit 0
            ;;
        h)
            print_help
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            print_help
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            print_help
            exit 1
            ;;
    esac
done
