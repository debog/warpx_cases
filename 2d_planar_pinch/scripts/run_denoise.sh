#!/bin/bash
#
# Unified particle-denoise launcher script
# Supports multiple HPC platforms and local execution
#
# Usage:
#   ./run_denoise.sh [OPTIONS] ACTION
#
# Actions (mutually exclusive):
#   inspect                Run particle-denoise inspect
#   train                  Run particle-denoise train
#   evaluate               Run particle-denoise evaluate
#
# Options:
#   -c, --config=FILE      Configuration YAML file (required)
#   -m, --mode=MODE        Execution mode: interactive (default) or batch
#   -o, --out-dir=DIR      Output directory for training (default: runs/<timestamp>)
#   -k, --checkpoint=FILE  Checkpoint file for evaluation (required for evaluate)
#   -n, --ntasks=N         Override number of MPI tasks (for multi-GPU)
#   -N, --nnodes=N         Override number of nodes
#   -q, --queue=NAME       Override queue/partition name
#   -t, --walltime=TIME    Override walltime (e.g., 4:00:00)
#   -d, --dry-run          Show what would be executed without running
#   -P, --list-platforms   List supported platforms
#   -v, --verbose          Enable verbose output
#   -h, --help             Show this help message
#
# Additional particle-denoise parameters can be passed as non-option arguments:
#   Example: ./run_denoise.sh -c config.yaml train train.lr=2e-4
#   Example: ./run_denoise.sh -c config.yaml train model.base_channels=64
#

set -e

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$SCRIPT_DIR/platforms.conf"
DENOISE_EXEC="particle-denoise"

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

# Validate environment and inputs
validate() {
    # Check particle-denoise is available
    if ! command -v "$DENOISE_EXEC" &>/dev/null && ! command -v python &>/dev/null; then
        error "Neither '$DENOISE_EXEC' nor 'python' found in PATH.
       Please ensure particle-denoise is installed."
    fi

    # If particle-denoise command exists, use it; otherwise use python -m
    if command -v "$DENOISE_EXEC" &>/dev/null; then
        DENOISE_CMD="$DENOISE_EXEC"
    else
        DENOISE_CMD="python -m particleDenoiser"
    fi

    # Check config file
    if [[ -z "$CONFIG_YAML" ]]; then
        error "Configuration file not specified. Use -c/--config to specify a YAML config file."
    fi

    if [[ ! -f "$CONFIG_YAML" ]]; then
        error "Configuration file not found: $CONFIG_YAML"
    fi

    # Check platform config file
    if [[ ! -f "$CONFIG_FILE" ]]; then
        error "Platform configuration file not found: $CONFIG_FILE"
    fi

    # Validate platform
    if ! platform_exists "$PLATFORM"; then
        error "Unknown platform: $PLATFORM
       Use --list-platforms to see available platforms."
    fi

    # Validate action
    if [[ -z "$ACTION" ]]; then
        error "No action specified. Use one of: inspect, train, evaluate"
    fi

    case "$ACTION" in
        inspect|train|evaluate) ;;
        *) error "Unknown action: $ACTION (use inspect, train, or evaluate)" ;;
    esac

    # Validate action-specific requirements
    if [[ "$ACTION" == "evaluate" && -z "$CHECKPOINT" ]]; then
        error "Checkpoint file required for evaluate action. Use -k/--checkpoint to specify."
    fi
}

# =============================================================================
# Job generation functions
# =============================================================================

generate_slurm_batch() {
    local jobfile="$1"
    cat > "$jobfile" << EOF
#!/bin/bash

#SBATCH -J denoise_${ACTION}
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

# Change to the directory where the script was invoked
cd $ROOT_DIR

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

    # Build command
    local cmd="$DENOISE_CMD $ACTION --config $CONFIG_YAML"
    [[ "$ACTION" == "train" && -n "$OUT_DIR" ]] && cmd="$cmd --out-dir $OUT_DIR"
    [[ "$ACTION" == "evaluate" && -n "$CHECKPOINT" ]] && cmd="$cmd --checkpoint $CHECKPOINT"
    [[ -n "$EXTRA_DENOISE_ARGS" ]] && cmd="$cmd $EXTRA_DENOISE_ARGS"

    echo "" >> "$jobfile"

    # For multi-GPU, use srun
    if [[ "$GPU_SUPPORT" == "true" && "$NTASKS" -gt 1 ]]; then
        local total_gpus=$((NTASKS * GPUS_PER_TASK))
        echo "srun --exclusive -N ${NNODES} -G ${total_gpus} -n ${NTASKS} $cmd 2>&1 | tee $WORKDIR/denoise_${ACTION}.${PLATFORM}.log" >> "$jobfile"
    else
        echo "$cmd 2>&1 | tee $WORKDIR/denoise_${ACTION}.${PLATFORM}.log" >> "$jobfile"
    fi
}

generate_flux_batch() {
    local jobfile="$1"
    cat > "$jobfile" << EOF
#!/bin/bash

#flux: --job-name=denoise_${ACTION}
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

# Change to the directory where the script was invoked
cd $ROOT_DIR

export OMP_NUM_THREADS=1

EOF

    echo "" >> "$jobfile"

    # Build command
    local cmd="$DENOISE_CMD $ACTION --config $CONFIG_YAML"
    [[ "$ACTION" == "train" && -n "$OUT_DIR" ]] && cmd="$cmd --out-dir $OUT_DIR"
    [[ "$ACTION" == "evaluate" && -n "$CHECKPOINT" ]] && cmd="$cmd --checkpoint $CHECKPOINT"
    [[ -n "$EXTRA_DENOISE_ARGS" ]] && cmd="$cmd $EXTRA_DENOISE_ARGS"

    if [[ "$GPU_SUPPORT" == "true" && "$NTASKS" -gt 1 ]]; then
        echo "flux run --exclusive --nodes=${NNODES} --ntasks ${NTASKS} $cmd 2>&1 | tee $WORKDIR/denoise_${ACTION}.${PLATFORM}.log" >> "$jobfile"
    else
        echo "$cmd 2>&1 | tee $WORKDIR/denoise_${ACTION}.${PLATFORM}.log" >> "$jobfile"
    fi
}

generate_run_script() {
    local runfile="$WORKDIR/run_${ACTION}.sh"
    local scheduler=$(get_config "$PLATFORM" "scheduler")
    local runcmd=""

    case "$scheduler" in
        slurm)
            local debug_queue=$(get_config "$PLATFORM" "debug_queue" "pdebug")
            if [[ "$GPU_SUPPORT" == "true" && "$NTASKS" -gt 1 ]]; then
                local total_gpus=$((NTASKS * GPUS_PER_TASK))
                runcmd="srun --exclusive -N $NNODES -n $NTASKS -G $total_gpus -p $debug_queue"
            else
                runcmd="srun --exclusive -N $NNODES -n $NTASKS -p $debug_queue"
            fi
            ;;
        flux)
            local debug_queue=$(get_config "$PLATFORM" "debug_queue" "pdebug")
            runcmd="flux run --exclusive --nodes=$NNODES --ntasks=$NTASKS -q=$debug_queue --setattr=thp=always"
            ;;
        direct)
            runcmd=""
            ;;
    esac

    # Build command
    local cmd="$DENOISE_CMD $ACTION --config $CONFIG_YAML"
    [[ "$ACTION" == "train" && -n "$OUT_DIR" ]] && cmd="$cmd --out-dir $OUT_DIR"
    [[ "$ACTION" == "evaluate" && -n "$CHECKPOINT" ]] && cmd="$cmd --checkpoint $CHECKPOINT"
    [[ -n "$EXTRA_DENOISE_ARGS" ]] && cmd="$cmd $EXTRA_DENOISE_ARGS"

    cat > "$runfile" << EOF
#!/bin/bash
#
# Interactive run script for particle-denoise: $ACTION
# Generated by run_denoise.sh on $(date)
#
# Platform: $PLATFORM
# Nodes:    $NNODES
# Tasks:    $NTASKS
#

set -e

# Change to the directory where the script was invoked
cd $ROOT_DIR

export OMP_NUM_THREADS=1

# Export platform-specific environment variables
EOF

    local env_vars=$(get_config "$PLATFORM" "env_vars")
    if [[ -n "$env_vars" ]]; then
        for var in $env_vars; do
            echo "export $var" >> "$runfile"
        done
        echo "" >> "$runfile"
    fi

    cat >> "$runfile" << EOF

# Run particle-denoise
EOF

    if [[ -n "$runcmd" ]]; then
        echo "$runcmd $cmd 2>&1 | tee $WORKDIR/denoise_${ACTION}.${PLATFORM}.log" >> "$runfile"
    else
        echo "$cmd 2>&1 | tee $WORKDIR/denoise_${ACTION}.${PLATFORM}.log" >> "$runfile"
    fi

    chmod +x "$runfile"
}

generate_standalone_job_script() {
    local jobfile="$WORKDIR/denoise_${ACTION}.job"
    local scheduler=$(get_config "$PLATFORM" "scheduler")

    if [[ "$scheduler" == "direct" ]]; then
        # For desktop, just create a simple run script
        cat > "$jobfile" << EOF
#!/bin/bash
#
# Job script for particle-denoise: $ACTION
# Platform '$PLATFORM' does not support batch mode
# Use ./run_${ACTION}.sh to run interactively
#

echo "Platform '$PLATFORM' does not support batch submission."
echo "Please use ./run_${ACTION}.sh to run interactively."
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

    info "Running particle-denoise interactively"
    info "  Platform:   $PLATFORM"
    info "  Action:     $ACTION"
    info "  Config:     $CONFIG_YAML"
    info "  Nodes:      $NNODES"
    info "  Tasks:      $NTASKS"
    if [[ "$ACTION" == "train" ]]; then
        if [[ -n "$OUT_DIR" ]]; then
            info "  Output dir: $OUT_DIR"
        else
            info "  Output dir: (auto-detected, see particle-denoise output)"
        fi
    fi
    [[ "$ACTION" == "evaluate" ]] && info "  Checkpoint: $CHECKPOINT"
    echo

    # Build command
    local cmd="$DENOISE_CMD $ACTION --config $CONFIG_YAML"
    [[ "$ACTION" == "train" && -n "$OUT_DIR" ]] && cmd="$cmd --out-dir $OUT_DIR"
    [[ "$ACTION" == "evaluate" && -n "$CHECKPOINT" ]] && cmd="$cmd --checkpoint $CHECKPOINT"
    [[ -n "$EXTRA_DENOISE_ARGS" ]] && cmd="$cmd $EXTRA_DENOISE_ARGS"

    # For interactive mode, use --exclusive to get all resources on allocated node
    local logfile="$WORKDIR/denoise_${ACTION}.${PLATFORM}.log"
    local debug_queue=$(get_config "$PLATFORM" "debug_queue" "pdebug")

    # Export platform-specific environment variables
    local env_vars=$(get_config "$PLATFORM" "env_vars")
    if [[ -n "$env_vars" ]]; then
        for var in $env_vars; do
            export "${var?}"
        done
    fi

    case "$scheduler" in
        slurm)
            if [[ "$GPU_SUPPORT" == "true" && "$NTASKS" -gt 1 ]]; then
                local total_gpus=$((NTASKS * GPUS_PER_TASK))
                local runcmd="srun --exclusive -N $NNODES -n $NTASKS -G $total_gpus -p $debug_queue"
            else
                local runcmd="srun --exclusive -N $NNODES -n $NTASKS -p $debug_queue"
            fi
            local full_cmd="$runcmd $cmd"
            ;;
        flux)
            local runcmd="flux run --exclusive --nodes=$NNODES --ntasks=$NTASKS -q=$debug_queue --setattr=thp=always"
            local full_cmd="$runcmd $cmd"
            ;;
        direct)
            local full_cmd="$cmd"
            ;;
        *)
            error "Unknown scheduler: $scheduler"
            ;;
    esac

    if [[ -n "$DRY_RUN" ]]; then
        echo "Would execute:"
        echo "  $full_cmd"
        echo ""
        echo "Log would be saved to: $logfile"
        return 0
    fi

    # Execute the command
    case "$scheduler" in
        slurm|flux)
            eval "$full_cmd 2>&1 | tee \"$logfile\""
            ;;
        direct)
            eval "$cmd 2>&1 | tee \"$logfile\""
            ;;
    esac

    if [[ -z "$DRY_RUN" ]]; then
        info "Log saved to: $logfile"
    fi
}

run_batch() {
    local scheduler=$(get_config "$PLATFORM" "scheduler")
    local jobfile="$WORKDIR/denoise_${ACTION}.job"

    if [[ "$scheduler" == "direct" ]]; then
        warn "Platform '$PLATFORM' does not support batch mode, falling back to interactive"
        run_interactive
        return
    fi

    info "Submitting particle-denoise batch job"
    info "  Platform:   $PLATFORM"
    info "  Action:     $ACTION"
    info "  Config:     $CONFIG_YAML"
    info "  Nodes:      $NNODES"
    info "  Tasks:      $NTASKS"
    info "  Queue:      ${QUEUE:-default}"
    info "  Walltime:   $WALLTIME"
    if [[ "$ACTION" == "train" ]]; then
        if [[ -n "$OUT_DIR" ]]; then
            info "  Output dir: $OUT_DIR"
        else
            info "  Output dir: (auto-detected, see particle-denoise output)"
        fi
    fi
    [[ "$ACTION" == "evaluate" ]] && info "  Checkpoint: $CHECKPOINT"
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

    info "Job submitted from directory: $ROOT_DIR"
    info "Job script: $jobfile"
    info "Log will be saved to: $WORKDIR/denoise_${ACTION}.${PLATFORM}.log"
}

# =============================================================================
# Main
# =============================================================================

# Show help if no arguments provided
if [[ $# -eq 0 ]]; then
    usage
fi

# Parse command line arguments
CONFIG_YAML=""
MODE="interactive"
OUT_DIR=""
CHECKPOINT=""
OVERRIDE_NTASKS=""
OVERRIDE_NNODES=""
OVERRIDE_QUEUE=""
OVERRIDE_WALLTIME=""
DRY_RUN=""
VERBOSE=""
ACTION=""
EXTRA_DENOISE_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config=*)
            [[ "$1" == -c ]] && { shift; CONFIG_YAML="$1"; } || CONFIG_YAML="${1#*=}" ;;
        -m|--mode=*)
            [[ "$1" == -m ]] && { shift; MODE="$1"; } || MODE="${1#*=}" ;;
        -o|--out-dir=*)
            [[ "$1" == -o ]] && { shift; OUT_DIR="$1"; } || OUT_DIR="${1#*=}" ;;
        -k|--checkpoint=*)
            [[ "$1" == -k ]] && { shift; CHECKPOINT="$1"; } || CHECKPOINT="${1#*=}" ;;
        -n|--ntasks=*)
            [[ "$1" == -n ]] && { shift; OVERRIDE_NTASKS="$1"; } || OVERRIDE_NTASKS="${1#*=}" ;;
        -N|--nnodes=*)
            [[ "$1" == -N ]] && { shift; OVERRIDE_NNODES="$1"; } || OVERRIDE_NNODES="${1#*=}" ;;
        -q|--queue=*)
            [[ "$1" == -q ]] && { shift; OVERRIDE_QUEUE="$1"; } || OVERRIDE_QUEUE="${1#*=}" ;;
        -t|--walltime=*)
            [[ "$1" == -t ]] && { shift; OVERRIDE_WALLTIME="$1"; } || OVERRIDE_WALLTIME="${1#*=}" ;;
        -d|--dry-run)   DRY_RUN=1 ;;
        -v|--verbose)   VERBOSE=1 ;;
        -P|--list-platforms) list_platforms; exit 0 ;;
        -h|--help)      usage ;;
        inspect|train|evaluate)
            [[ -n "$ACTION" ]] && error "Multiple actions specified: $ACTION and $1"
            ACTION="$1" ;;
        -*)
            error "Unknown option: $1" ;;
        *)
            # Non-option arguments are particle-denoise parameters (e.g., train.lr=2e-4)
            EXTRA_DENOISE_ARGS="$EXTRA_DENOISE_ARGS $1" ;;
    esac
    shift
done

# Make CONFIG_YAML absolute if it's relative
if [[ -n "$CONFIG_YAML" && "${CONFIG_YAML:0:1}" != "/" ]]; then
    CONFIG_YAML="$(pwd)/$CONFIG_YAML"
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
        info "No HPC environment detected, using local desktop"
    else
        info "Auto-detected LC platform: $PLATFORM (from hostname $HOSTNAME_SHORT)"
    fi
fi

# Validate inputs and environment
validate

# Load platform configuration
SCHEDULER=$(get_config "$PLATFORM" "scheduler")
NTASKS="${OVERRIDE_NTASKS:-$(get_config "$PLATFORM" "ntasks" "1")}"
NNODES="${OVERRIDE_NNODES:-$(get_config "$PLATFORM" "nnodes" "1")}"
QUEUE="${OVERRIDE_QUEUE:-$(get_config "$PLATFORM" "queue")}"
WALLTIME="${OVERRIDE_WALLTIME:-$(get_config "$PLATFORM" "walltime" "4:00:00")}"
GPU_SUPPORT=$(get_config "$PLATFORM" "gpu_support" "false")
GPUS_PER_TASK=$(get_config "$PLATFORM" "gpus_per_task" "1")
ACCOUNT=$(get_config "$PLATFORM" "account")

debug "Platform config loaded:"
debug "  scheduler=$SCHEDULER ntasks=$NTASKS nnodes=$NNODES"
debug "  queue=$QUEUE walltime=$WALLTIME gpu=$GPU_SUPPORT"

# Create working directory for logs and job scripts
WORKDIR="$ROOT_DIR/.run_denoise.${PLATFORM}"
if [[ -d "$WORKDIR" ]]; then
    info "Using existing directory: $WORKDIR"
else
    info "Creating working directory: $WORKDIR"
    mkdir -p "$WORKDIR"
fi

# Generate run.sh and denoise.job scripts in the working directory
info "Generating run scripts in $WORKDIR"
generate_run_script
generate_standalone_job_script
info "  Created run_${ACTION}.sh for interactive execution"
info "  Created denoise_${ACTION}.job for batch submission"

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

info "Done."
