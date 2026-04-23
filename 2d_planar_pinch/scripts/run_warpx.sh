#!/bin/bash
#
# Unified WarpX launcher script
# Supports multiple HPC platforms (LC and NERSC) and local desktop execution
#
# Usage:
#   ./run_warpx.sh [OPTIONS]
#
# Options:
#   -c, --case=NAME       Input case name(s) (space-separated list, supports wildcards)
#                         Default: planar_pinch_2d
#                         Examples: -c planar_pinch_2d
#                                   -c case1 case2
#                                   -c planar_pinch*
#   -m, --mode=MODE       Execution mode: interactive (default) or batch
#   -n, --ntasks=N        Override number of MPI tasks
#   -N, --nnodes=N        Override number of nodes
#   -q, --queue=NAME      Override queue/partition name
#   -t, --walltime=TIME   Override walltime (e.g., 1:00:00 or 1h)
#   -s, --max-steps=N     Override number of timesteps (uses input file default if unset)
#   -d, --dry-run         Show what would be executed without running
#   -l, --list-cases      List available input cases
#   -P, --list-platforms  List supported platforms
#   -v, --verbose         Enable verbose output
#   -h, --help            Show this help message
#
# Additional WarpX parameters can be passed as non-option arguments:
#   Example: ./run_warpx.sh -c case1 jacobian.pc_type=none
#   Example: ./run_warpx.sh -c case1 max_step=100 amrex.verbose=2
#
# Environment:
#   LCHOST            LC platform identifier (auto-detected for LC machines)
#   NERSC_HOST        NERSC platform identifier (auto-detected for NERSC machines)
#   WARPX_BUILD       Path to WarpX build directory (required)
#   WARPX_DIR         Path to WarpX source directory (optional)
#   CASE              Alternative way to specify case name
#

set -e

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$SCRIPT_DIR/platforms.conf"
INPUTS_DIR="$ROOT_DIR/inputs"
DEFAULT_CASE="planar_pinch_2d"
# DIM is auto-derived from the case-name suffix (_1d, _2d, _3d, _rz) in the
# per-case loop below; default if no matching suffix is found is "2d".
DEFAULT_DIM="2d"
DIM="$DEFAULT_DIM"

# =============================================================================
# Color output (disabled if not a terminal)
# =============================================================================
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
else
    RED='' GREEN='' YELLOW='' BLUE='' NC=''
fi

# =============================================================================
# Helper functions
# =============================================================================
error() { echo -e "${RED}ERROR:${NC} $*" >&2; exit 1; }
warn()  { echo -e "${YELLOW}WARNING:${NC} $*" >&2; }
info()  { echo -e "${GREEN}==>${NC} $*"; }
debug() { [[ -n "$VERBOSE" ]] && echo -e "${BLUE}DEBUG:${NC} $*" >&2 || true; }

usage() {
    sed -n '/^# Usage:/,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \?//'
    exit 0
}

# Parse configuration file for a given platform
# Usage: get_config PLATFORM KEY [DEFAULT]
get_config() {
    local platform="$1" key="$2" default="${3:-}"
    local in_section=false value=""

    while IFS= read -r line; do
        # Skip comments and empty lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue

        # Check for section header
        if [[ "$line" =~ ^\[([a-zA-Z0-9_-]+)\] ]]; then
            if [[ "${BASH_REMATCH[1]}" == "$platform" ]]; then
                in_section=true
            else
                in_section=false
            fi
            continue
        fi

        # Parse key=value if in correct section
        if $in_section && [[ "$line" =~ ^[[:space:]]*([a-zA-Z0-9_]+)[[:space:]]*=[[:space:]]*(.*) ]]; then
            if [[ "${BASH_REMATCH[1]}" == "$key" ]]; then
                value="${BASH_REMATCH[2]}"
                # Trim trailing whitespace and comments
                value="${value%%#*}"
                value="${value%"${value##*[![:space:]]}"}"
                echo "$value"
                return 0
            fi
        fi
    done < "$CONFIG_FILE"

    echo "$default"
}

# Check if platform exists in config
platform_exists() {
    local platform="$1"
    grep -q "^\[$platform\]" "$CONFIG_FILE" 2>/dev/null
}

# List available platforms
list_platforms() {
    echo "Available platforms:"
    grep '^\[' "$CONFIG_FILE" | tr -d '[]' | while read -r p; do
        local scheduler=$(get_config "$p" "scheduler")
        local gpu=$(get_config "$p" "gpu_support" "false")
        printf "  %-12s scheduler=%-6s gpu=%s\n" "$p" "$scheduler" "$gpu"
    done
}

# List available input cases
list_cases() {
    echo "Available cases in $INPUTS_DIR:"
    for f in "$INPUTS_DIR"/*.in; do
        [[ -f "$f" ]] || continue
        local name=$(basename "$f" .in)
        echo "  $name"
    done
}

# Validate environment and inputs
validate() {
    # Check WARPX_BUILD
    if [[ -z "$WARPX_BUILD" ]]; then
        error "WARPX_BUILD environment variable is not set.
       Please set it to your WarpX build directory, e.g.:
       export WARPX_BUILD=/path/to/WarpX/build"
    fi

    if [[ ! -d "$WARPX_BUILD" ]]; then
        error "WARPX_BUILD directory does not exist: $WARPX_BUILD"
    fi

    # Source WarpX profile if it exists (sets up ADIOS2, etc.)
    # Try to find platform-specific profile
    local profiles=(
        "$WARPX_BUILD/${PLATFORM}_*.profile"
        "$WARPX_BUILD/*.profile"
    )
    local profile_sourced=false
    for pattern in "${profiles[@]}"; do
        for WARPX_PROFILE in $pattern; do
            if [[ -f "$WARPX_PROFILE" ]]; then
                info "Sourcing WarpX profile: $WARPX_PROFILE"
                source "$WARPX_PROFILE"
                profile_sourced=true
                break 2
            fi
        done
    done
    if [[ "$profile_sourced" == "false" ]]; then
        warn "No WarpX profile found in $WARPX_BUILD"
        warn "Diagnostics may fail without ADIOS2/HDF5 libraries"
    fi

    # Find WarpX executable (different paths for LC vs NERSC)
    if [[ -n "$NERSC_HOST" ]]; then
        # NERSC path structure
        EXEC="$WARPX_BUILD/bin/warpx.${DIM}"
    else
        # LC path structure
        EXEC="$WARPX_BUILD/build/bin/warpx.${DIM}"
    fi

    if [[ ! -x "$EXEC" ]]; then
        error "WarpX executable not found or not executable: $EXEC
       Check that WARPX_BUILD is set correctly and the executable exists."
    fi

    # Check input file
    INPUT_FILE="$INPUTS_DIR/${CASE}.in"
    if [[ ! -f "$INPUT_FILE" ]]; then
        error "Input file not found: $INPUT_FILE
       Use --list-cases to see available cases."
    fi

    # Check config file
    if [[ ! -f "$CONFIG_FILE" ]]; then
        error "Platform configuration file not found: $CONFIG_FILE"
    fi

    # Validate platform
    if ! platform_exists "$PLATFORM"; then
        error "Unknown platform: $PLATFORM
       Use --list-platforms to see available platforms."
    fi
}

# =============================================================================
# Job generation functions
# =============================================================================

generate_slurm_batch() {
    local jobfile="$1"
    cat > "$jobfile" << EOF
#!/bin/bash

#SBATCH -J warpx_${CASE}
#SBATCH -N ${NNODES}
#SBATCH -n ${NTASKS}
#SBATCH -t ${WALLTIME}
#SBATCH --exclusive
#SBATCH --export=ALL
EOF

    [[ -n "$ACCOUNT" ]] && echo "#SBATCH -A ${ACCOUNT}" >> "$jobfile"
    [[ -n "$QUEUE" ]] && echo "#SBATCH -p ${QUEUE}" >> "$jobfile"

    if [[ "$GPU_SUPPORT" == "true" ]]; then
        echo "#SBATCH --gpus-per-task=${GPUS_PER_TASK}" >> "$jobfile"
    fi

    cat >> "$jobfile" << EOF

# Clean up previous run outputs
echo "Cleaning up previous run outputs..."
rm -f out.*.log Backtrace.* *core* warpx_used_inputs
rm -rf diags

export OMP_NUM_THREADS=1

EOF

    # Add environment variables if needed
    local env_vars=$(get_config "$PLATFORM" "env_vars")
    if [[ -n "$env_vars" ]]; then
        for var in $env_vars; do
            echo "export $var" >> "$jobfile"
        done
        echo "" >> "$jobfile"
    fi

    # Build srun command with GPU support if needed
    if [[ "$GPU_SUPPORT" == "true" ]]; then
        local total_gpus=$((NTASKS * GPUS_PER_TASK))
        if [[ "$PLATFORM" == "perlmutter" ]]; then
            # Perlmutter-specific GPU handling
            cat >> "$jobfile" << EOF
# CUDA visible devices are ordered inverse to local task IDs for Perlmutter
srun --cpu-bind=cores -n ${NTASKS} bash -c "
    export CUDA_VISIBLE_DEVICES=\\\$((3-SLURM_LOCALID));
    ${EXEC} ${INP} ${WARPX_EXTRA_ARGS}" 2>&1 | tee out.${PLATFORM}.log
EOF
        else
            echo "srun --exclusive -N ${NNODES} -G ${total_gpus} -n ${NTASKS} ${EXEC} ${INP} ${WARPX_EXTRA_ARGS} 2>&1 | tee out.${PLATFORM}.log" >> "$jobfile"
        fi
    else
        echo "srun -N ${NNODES} -n ${NTASKS} ${EXEC} ${INP} ${WARPX_EXTRA_ARGS} 2>&1 | tee out.${PLATFORM}.log" >> "$jobfile"
    fi
}

generate_flux_batch() {
    local jobfile="$1"
    cat > "$jobfile" << EOF
#!/bin/bash

#flux: --job-name=warpx_${CASE}
#flux: --output={{name}}-{{id}}.out
#flux: --nodes=${NNODES}
#flux: --time=${WALLTIME}
#flux: --exclusive
EOF

    [[ -n "$ACCOUNT" ]] && echo "#flux: --bank=${ACCOUNT}" >> "$jobfile"
    [[ -n "$QUEUE" ]] && echo "#flux: --queue=${QUEUE}" >> "$jobfile"

    # Add environment variables for GPU support
    local env_vars=$(get_config "$PLATFORM" "env_vars")
    if [[ -n "$env_vars" ]]; then
        for var in $env_vars; do
            echo "export $var" >> "$jobfile"
        done
    fi

    cat >> "$jobfile" << EOF

# Clean up previous run outputs
echo "Cleaning up previous run outputs..."
rm -f out.*.log Backtrace.* *core* warpx_used_inputs
rm -rf diags

export OMP_NUM_THREADS=1

flux run --exclusive --nodes=${NNODES} --ntasks ${NTASKS} ${EXEC} ${INP} ${WARPX_EXTRA_ARGS} 2>&1 | tee out.${PLATFORM}.log
EOF
}

generate_run_script() {
    local runfile="run.sh"
    local scheduler=$(get_config "$PLATFORM" "scheduler")
    local runcmd=""

    case "$scheduler" in
        slurm)
            local debug_queue=$(get_config "$PLATFORM" "debug_queue" "pdebug")
            runcmd="srun --exclusive -N $NNODES -n $NTASKS -p $debug_queue --export=ALL"
            if [[ "$GPU_SUPPORT" == "true" ]]; then
                local total_gpus=$((NTASKS * GPUS_PER_TASK))
                runcmd="$runcmd -G ${total_gpus} --gpus-per-task=${GPUS_PER_TASK}"
            fi
            ;;
        flux)
            local debug_queue=$(get_config "$PLATFORM" "debug_queue" "pdebug")
            runcmd="flux run --exclusive --nodes=$NNODES --ntasks $NTASKS --verbose --setopt=mpibind=verbose:1 -q=$debug_queue"
            ;;
        direct)
            local mpi_launcher=$(get_config "$PLATFORM" "mpi_launcher" "mpirun")
            if command -v "$mpi_launcher" &>/dev/null; then
                runcmd="$mpi_launcher -n $NTASKS"
            else
                runcmd=""
            fi
            ;;
    esac

    cat > "$runfile" << EOF
#!/bin/bash
#
# Interactive run script for WarpX case: $CASE
# Generated by run_warpx.sh on $(date)
#
# Platform: $PLATFORM
# Tasks:    $NTASKS
# Nodes:    $NNODES
#

set -e

# Clean up previous run outputs
echo "Cleaning up previous run outputs..."
rm -f out.*.log Backtrace.* *core* warpx_used_inputs
rm -rf diags

export OMP_NUM_THREADS=1
EOF

    # Add environment variables for GPU support if needed
    local env_vars=$(get_config "$PLATFORM" "env_vars")
    if [[ -n "$env_vars" ]]; then
        for var in $env_vars; do
            echo "export $var" >> "$runfile"
        done
    fi

    cat >> "$runfile" << EOF

# Run WarpX
EOF
    if [[ -n "$runcmd" ]]; then
        echo "$runcmd $EXEC $INP $WARPX_EXTRA_ARGS 2>&1 | tee out.${PLATFORM}.log" >> "$runfile"
    else
        echo "$EXEC $INP $WARPX_EXTRA_ARGS 2>&1 | tee out.${PLATFORM}.log" >> "$runfile"
    fi

    chmod +x "$runfile"
}

generate_standalone_job_script() {
    local jobfile="warpx.job"
    local scheduler=$(get_config "$PLATFORM" "scheduler")

    if [[ "$scheduler" == "direct" ]]; then
        # For desktop, just create a simple run script
        cat > "$jobfile" << EOF
#!/bin/bash
#
# Job script for WarpX case: $CASE
# Platform '$PLATFORM' does not support batch mode
# Use ./run.sh to run interactively
#

echo "Platform '$PLATFORM' does not support batch submission."
echo "Please use ./run.sh to run interactively."
exit 1
EOF
        chmod +x "$jobfile"
        return
    fi

    case "$scheduler" in
        slurm)
            generate_slurm_batch "$jobfile"
            ;;
        flux)
            generate_flux_batch "$jobfile"
            ;;
        *)
            error "Unknown scheduler: $scheduler"
            ;;
    esac

    chmod +x "$jobfile"
}

# =============================================================================
# Execution functions
# =============================================================================

run_interactive() {
    local scheduler=$(get_config "$PLATFORM" "scheduler")
    local runcmd=""

    case "$scheduler" in
        slurm)
            local debug_queue=$(get_config "$PLATFORM" "debug_queue" "pdebug")
            runcmd="srun --exclusive -N $NNODES -n $NTASKS -p $debug_queue --export=ALL"
            if [[ "$GPU_SUPPORT" == "true" ]]; then
                local total_gpus=$((NTASKS * GPUS_PER_TASK))
                runcmd="$runcmd -G ${total_gpus} --gpus-per-task=${GPUS_PER_TASK}"
            fi
            ;;
        flux)
            local debug_queue=$(get_config "$PLATFORM" "debug_queue" "pdebug")
            local tasks_per_node=$((NTASKS / NNODES))
            runcmd="flux run --exclusive --nodes=$NNODES --tasks-per-node=$tasks_per_node --verbose --setopt=mpibind=verbose:1 -q=$debug_queue"
            # Set environment for GPU
            local env_vars=$(get_config "$PLATFORM" "env_vars")
            for var in $env_vars; do
                export "${var?}"
            done
            ;;
        direct)
            local mpi_launcher=$(get_config "$PLATFORM" "mpi_launcher" "mpirun")
            if command -v "$mpi_launcher" &>/dev/null; then
                runcmd="$mpi_launcher -n $NTASKS"
            else
                warn "MPI launcher '$mpi_launcher' not found, running without MPI"
                runcmd=""
            fi
            ;;
        *)
            error "Unknown scheduler: $scheduler"
            ;;
    esac

    info "Running WarpX interactively"
    info "  Platform:   $PLATFORM"
    info "  Case:       $CASE"
    info "  Tasks:      $NTASKS"
    info "  Nodes:      $NNODES"
    info "  Executable: $EXEC"
    info "  Input:      $INP"
    [[ -n "$MAX_STEPS" ]] && info "  Max steps:  $MAX_STEPS"
    echo

    if [[ -n "$DRY_RUN" ]]; then
        echo "Would execute:"
        echo "  cd $WORKDIR"
        echo "  export OMP_NUM_THREADS=$OMP_THREADS"
        echo "  $runcmd $EXEC $INP $WARPX_EXTRA_ARGS"
        return 0
    fi

    export OMP_NUM_THREADS=$OMP_THREADS

    if [[ -n "$runcmd" ]]; then
        $runcmd $EXEC $INP $WARPX_EXTRA_ARGS 2>&1 | tee "out.${PLATFORM}.log"
    else
        $EXEC $INP $WARPX_EXTRA_ARGS 2>&1 | tee "out.${PLATFORM}.log"
    fi
}

run_batch() {
    local scheduler=$(get_config "$PLATFORM" "scheduler")
    local jobfile="warpx.job"

    if [[ "$scheduler" == "direct" ]]; then
        warn "Platform '$PLATFORM' does not support batch mode, falling back to interactive"
        run_interactive
        return
    fi

    info "Submitting WarpX batch job"
    info "  Platform:   $PLATFORM"
    info "  Case:       $CASE"
    info "  Tasks:      $NTASKS"
    info "  Nodes:      $NNODES"
    info "  Queue:      ${QUEUE:-default}"
    info "  Walltime:   $WALLTIME"
    info "  Executable: $EXEC"
    info "  Input:      $INP"
    [[ -n "$MAX_STEPS" ]] && info "  Max steps:  $MAX_STEPS"
    echo

    case "$scheduler" in
        slurm)
            generate_slurm_batch "$jobfile"
            if [[ -n "$DRY_RUN" ]]; then
                echo "Would submit job script:"
                cat "$jobfile"
                return 0
            fi
            sbatch "$jobfile"
            ;;
        flux)
            generate_flux_batch "$jobfile"
            if [[ -n "$DRY_RUN" ]]; then
                echo "Would submit job script:"
                cat "$jobfile"
                return 0
            fi
            flux batch "$jobfile"
            ;;
        *)
            error "Unknown scheduler for batch mode: $scheduler"
            ;;
    esac

    info "Job submitted from directory: $WORKDIR"
}

# =============================================================================
# Main
# =============================================================================

# Show help if no arguments provided
if [[ $# -eq 0 ]]; then
    usage
fi

# Parse command line arguments
CASES=()
MODE="interactive"
OVERRIDE_NTASKS=""
OVERRIDE_NNODES=""
OVERRIDE_QUEUE=""
OVERRIDE_WALLTIME=""
MAX_STEPS=""
DRY_RUN=""
VERBOSE=""
EXTRA_WARPX_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--case=*)
            if [[ "$1" == -c ]]; then
                shift
                # Collect all non-option arguments as case names (but not key=value pairs)
                while [[ $# -gt 0 && "$1" != -* && "$1" != *"="* ]]; do
                    # Expand wildcards by checking if pattern matches any input files
                    if [[ "$1" == *"*"* ]] || [[ "$1" == *"?"* ]]; then
                        # This is a glob pattern - expand it
                        shopt -s nullglob
                        matched_files=("$INPUTS_DIR"/$1.in)
                        shopt -u nullglob
                        for matched in "${matched_files[@]}"; do
                            [[ -f "$matched" ]] && CASES+=("$(basename "$matched" .in)")
                        done
                    else
                        CASES+=("$1")
                    fi
                    shift
                done
                # Already shifted, so continue without shift at end
                continue
            else
                CASES+=("${1#*=}")
            fi
            ;;
        -m|--mode=*)
            [[ "$1" == -m ]] && { shift; MODE="$1"; } || MODE="${1#*=}" ;;
        -n|--ntasks=*)
            [[ "$1" == -n ]] && { shift; OVERRIDE_NTASKS="$1"; } || OVERRIDE_NTASKS="${1#*=}" ;;
        -N|--nnodes=*)
            [[ "$1" == -N ]] && { shift; OVERRIDE_NNODES="$1"; } || OVERRIDE_NNODES="${1#*=}" ;;
        -q|--queue=*)
            [[ "$1" == -q ]] && { shift; OVERRIDE_QUEUE="$1"; } || OVERRIDE_QUEUE="${1#*=}" ;;
        -t|--walltime=*)
            [[ "$1" == -t ]] && { shift; OVERRIDE_WALLTIME="$1"; } || OVERRIDE_WALLTIME="${1#*=}" ;;
        -s|--max-steps=*)
            [[ "$1" == -s ]] && { shift; MAX_STEPS="$1"; } || MAX_STEPS="${1#*=}" ;;
        -d|--dry-run)   DRY_RUN=1 ;;
        -v|--verbose)   VERBOSE=1 ;;
        -l|--list-cases) list_cases; exit 0 ;;
        -P|--list-platforms) list_platforms; exit 0 ;;
        -h|--help)      usage ;;
        -*)
            error "Unknown option: $1" ;;
        *)
            # Non-option arguments are WarpX parameters (e.g., jacobian.pc_type=none)
            EXTRA_WARPX_ARGS+=("$1") ;;
    esac
    shift
done

# Set defaults
if [[ ${#CASES[@]} -eq 0 ]]; then
    CASES=("$DEFAULT_CASE")
fi

# Auto-detect platform
if [[ -n "$NERSC_HOST" ]]; then
    # NERSC platform (Perlmutter, etc.)
    PLATFORM="$NERSC_HOST"
elif [[ -n "$LCHOST" ]]; then
    # LC platform (dane, matrix, tuolumne)
    PLATFORM="$LCHOST"
else
    # Try to detect LC machines from hostname
    HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"
    case "$HOSTNAME_SHORT" in
        dane*)      PLATFORM="dane" ;;
        matrix*)    PLATFORM="matrix" ;;
        tuolumne*)  PLATFORM="tuolumne" ;;
        *)          PLATFORM="desktop" ;;
    esac

    if [[ "$PLATFORM" == "desktop" ]]; then
        info "No HPC environment detected, using local desktop ($(nproc 2>/dev/null || echo '?') cores available)"
    else
        info "Auto-detected LC platform: $PLATFORM (from hostname $HOSTNAME_SHORT)"
    fi
fi

# Process each case
for CASE in "${CASES[@]}"; do
    if [[ ${#CASES[@]} -gt 1 ]]; then
        info ""
        info "=========================================="
        info "Processing case: $CASE"
        info "=========================================="
        info ""
    fi

# Derive dimensionality from case-name suffix (_1d, _2d, _3d, _rz)
case "$CASE" in
    *_1d|*_1D) DIM="1d" ;;
    *_2d|*_2D) DIM="2d" ;;
    *_3d|*_3D) DIM="3d" ;;
    *_rz|*_RZ) DIM="rz" ;;
    *)         DIM="$DEFAULT_DIM" ;;
esac
debug "Derived DIM=$DIM from CASE=$CASE"

# Validate inputs and environment
validate

# Load platform configuration
SCHEDULER=$(get_config "$PLATFORM" "scheduler")
NTASKS="${OVERRIDE_NTASKS:-$(get_config "$PLATFORM" "ntasks" "4")}"
NNODES="${OVERRIDE_NNODES:-$(get_config "$PLATFORM" "nnodes" "1")}"
QUEUE="${OVERRIDE_QUEUE:-$(get_config "$PLATFORM" "queue")}"
WALLTIME="${OVERRIDE_WALLTIME:-$(get_config "$PLATFORM" "walltime" "4:00:00")}"
GPU_SUPPORT=$(get_config "$PLATFORM" "gpu_support" "false")
GPUS_PER_TASK=$(get_config "$PLATFORM" "gpus_per_task" "1")
CORES_PER_NODE=$(get_config "$PLATFORM" "cores_per_node" "4")
ACCOUNT=$(get_config "$PLATFORM" "account")
OMP_THREADS=$(get_config "$PLATFORM" "omp_threads" "1")

debug "Platform config loaded:"
debug "  scheduler=$SCHEDULER ntasks=$NTASKS nnodes=$NNODES"
debug "  queue=$QUEUE walltime=$WALLTIME gpu=$GPU_SUPPORT"

# Create working directory
WORKDIR="$ROOT_DIR/.run_${CASE}.${PLATFORM}.$(printf "nproc%05d" "$NTASKS")"
if [[ -d "$WORKDIR" ]]; then
    info "Removing existing directory: $WORKDIR"
    rm -rf "$WORKDIR"
fi
info "Creating working directory: $WORKDIR"
mkdir -p "$WORKDIR"

cd "$WORKDIR"

# Copy input file to working directory
cp "$INPUT_FILE" .
INP="$(basename "$INPUT_FILE")"

# Copy .petscrc file if it exists
if [[ -f "$INPUTS_DIR/.petscrc" ]]; then
    cp "$INPUTS_DIR/.petscrc" .
fi

# Build extra arguments for WarpX
WARPX_EXTRA_ARGS=""
[[ -n "$MAX_STEPS" ]] && WARPX_EXTRA_ARGS="max_step=$MAX_STEPS"
# Add GPU-aware MPI flag for GPU platforms
[[ "$GPU_SUPPORT" == "true" ]] && WARPX_EXTRA_ARGS="$WARPX_EXTRA_ARGS amrex.use_gpu_aware_mpi=1"
# Add user-specified WarpX arguments
for arg in "${EXTRA_WARPX_ARGS[@]}"; do
    WARPX_EXTRA_ARGS="$WARPX_EXTRA_ARGS $arg"
done

# Generate run.sh and warpx.job scripts in the run directory
info "Generating run scripts in $WORKDIR"
generate_run_script
generate_standalone_job_script
info "  Created run.sh for interactive execution"
info "  Created warpx.job for batch submission"

# Execute based on mode
case "$MODE" in
    interactive|i)
        run_interactive
        ;;
    batch|b)
        run_batch
        ;;
    *)
        error "Unknown mode: $MODE (use 'interactive' or 'batch')"
        ;;
esac

done  # End of case loop

info "Done."
