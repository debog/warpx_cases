#!/bin/bash
#
# Paired ppc-sweep driver for WarpX 2D planar pinch ML-denoiser training data.
#
# Mirrors run_warpx.sh but loops over a list of (Nppc_x, Nppc_z) pairs, each
# landing in its own working directory and overriding only
# my_constants.Nppc_x / Nppc_z on the WarpX command line. Physics driver,
# Delta t, stop_time, grid, solver, boundary conditions are all untouched.
#
# Default sweep: 2x2=4,4x4=16, 8x8=64, 16x16=256, 32x32=1024 (factor-4 noise spacing).
#
# Usage:
#   ./run_ppc_sweep.sh [OPTIONS]
#
# Options:
#   -c, --case=NAME       Input case name. Default: planar_pinch_2d
#   -p, --ppc=LIST        Comma-separated list of NXxNZ specs.
#                         Default: 2x2,4x4,8x8,16x16,32x32
#                         Example: -p 10x10,20x20,40x40
#   -m, --mode=MODE       interactive | batch (default: batch)
#   -n, --ntasks=N        Override MPI tasks
#   -N, --nnodes=N        Override nodes
#   -q, --queue=NAME      Override queue/partition
#   -t, --walltime=TIME   Override walltime
#   -s, --max-steps=N     Override max_step (otherwise uses input file)
#       --skip-existing   Skip any ppc whose WORKDIR already exists
#   -d, --dry-run         Show what would be done without submitting
#   -v, --verbose         Enable verbose output
#   -h, --help            Show this help
#
# Extra WarpX key=value arguments can be passed as non-option arguments; they
# are appended to every job in the sweep.
#   Example: ./run_ppc_sweep.sh -p 10x10,20x20 amrex.verbose=2
#
# Environment: same as run_warpx.sh (WARPX_BUILD required, LCHOST/NERSC_HOST
# autodetected).
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
DEFAULT_PPC_LIST="2x2,4x4,8x8,16x16,32x32"
DIM="2d"

# =============================================================================
# Color output
# =============================================================================
if [[ -t 1 ]]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; BLUE='\033[0;34m'; NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' NC=''
fi

error() { echo -e "${RED}ERROR:${NC} $*" >&2; exit 1; }
warn()  { echo -e "${YELLOW}WARNING:${NC} $*" >&2; }
info()  { echo -e "${GREEN}==>${NC} $*"; }
debug() { [[ -n "$VERBOSE" ]] && echo -e "${BLUE}DEBUG:${NC} $*" >&2 || true; }

usage() {
    sed -n '/^# Usage:/,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \?//'
    exit 0
}

# -----------------------------------------------------------------------------
# platforms.conf reader (same format as run_warpx.sh)
# -----------------------------------------------------------------------------
get_config() {
    local platform="$1" key="$2" default="${3:-}"
    local in_section=false value=""
    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue
        if [[ "$line" =~ ^\[([a-zA-Z0-9_-]+)\] ]]; then
            if [[ "${BASH_REMATCH[1]}" == "$platform" ]]; then in_section=true
            else in_section=false; fi
            continue
        fi
        if $in_section && [[ "$line" =~ ^[[:space:]]*([a-zA-Z0-9_]+)[[:space:]]*=[[:space:]]*(.*) ]]; then
            if [[ "${BASH_REMATCH[1]}" == "$key" ]]; then
                value="${BASH_REMATCH[2]}"
                value="${value%%#*}"
                value="${value%"${value##*[![:space:]]}"}"
                echo "$value"
                return 0
            fi
        fi
    done < "$CONFIG_FILE"
    echo "$default"
}

platform_exists() { grep -q "^\[$1\]" "$CONFIG_FILE" 2>/dev/null; }

# -----------------------------------------------------------------------------
# Environment validation (shared between all ppc in the sweep)
# -----------------------------------------------------------------------------
validate_env() {
    [[ -z "$WARPX_BUILD" ]] && error "WARPX_BUILD is not set."
    [[ ! -d "$WARPX_BUILD" ]] && error "WARPX_BUILD does not exist: $WARPX_BUILD"

    # Source platform profile if available
    local profiles=("$WARPX_BUILD/${PLATFORM}_*.profile" "$WARPX_BUILD/*.profile")
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
    [[ "$profile_sourced" == "false" ]] && warn "No WarpX profile found in $WARPX_BUILD"

    # Executable
    if [[ -n "$NERSC_HOST" ]]; then
        EXEC="$WARPX_BUILD/bin/warpx.${DIM}"
    else
        EXEC="$WARPX_BUILD/build/bin/warpx.${DIM}"
    fi
    [[ ! -x "$EXEC" ]] && error "WarpX executable not found or not executable: $EXEC"

    INPUT_FILE="$INPUTS_DIR/${CASE}.in"
    [[ ! -f "$INPUT_FILE" ]] && error "Input file not found: $INPUT_FILE"
    [[ ! -f "$CONFIG_FILE" ]] && error "Platform config not found: $CONFIG_FILE"
    platform_exists "$PLATFORM" || error "Unknown platform: $PLATFORM"
}

# -----------------------------------------------------------------------------
# Per-ppc batch script generators (slurm / flux variants)
# -----------------------------------------------------------------------------
generate_slurm_batch() {
    local jobfile="$1" job_name="$2" extra_args="$3"
    cat > "$jobfile" << EOF
#!/bin/bash

#SBATCH -J ${job_name}
#SBATCH -N ${NNODES}
#SBATCH -n ${NTASKS}
#SBATCH -t ${WALLTIME}
#SBATCH --exclusive
#SBATCH --export=ALL
EOF
    [[ -n "$ACCOUNT" ]] && echo "#SBATCH -A ${ACCOUNT}" >> "$jobfile"
    [[ -n "$QUEUE" ]]   && echo "#SBATCH -p ${QUEUE}"   >> "$jobfile"
    [[ "$GPU_SUPPORT" == "true" ]] && echo "#SBATCH --gpus-per-task=${GPUS_PER_TASK}" >> "$jobfile"

    cat >> "$jobfile" << EOF

echo "Cleaning up previous run outputs..."
rm -f out.*.log Backtrace.* *core* warpx_used_inputs
rm -rf diags

export OMP_NUM_THREADS=${OMP_THREADS}

EOF

    local env_vars
    env_vars=$(get_config "$PLATFORM" "env_vars")
    if [[ -n "$env_vars" ]]; then
        for var in $env_vars; do echo "export $var" >> "$jobfile"; done
        echo "" >> "$jobfile"
    fi

    if [[ "$GPU_SUPPORT" == "true" ]]; then
        local total_gpus=$((NTASKS * GPUS_PER_TASK))
        if [[ "$PLATFORM" == "perlmutter" ]]; then
            cat >> "$jobfile" << EOF
# CUDA device ordering is inverse to local task ID on Perlmutter
srun --cpu-bind=cores -n ${NTASKS} bash -c "
    export CUDA_VISIBLE_DEVICES=\\\$((3-SLURM_LOCALID));
    ${EXEC} ${INP} ${extra_args}" 2>&1 | tee out.${PLATFORM}.log
EOF
        else
            echo "srun --exclusive -N ${NNODES} -G ${total_gpus} -n ${NTASKS} ${EXEC} ${INP} ${extra_args} 2>&1 | tee out.${PLATFORM}.log" >> "$jobfile"
        fi
    else
        echo "srun -N ${NNODES} -n ${NTASKS} ${EXEC} ${INP} ${extra_args} 2>&1 | tee out.${PLATFORM}.log" >> "$jobfile"
    fi
}

generate_flux_batch() {
    local jobfile="$1" job_name="$2" extra_args="$3"
    cat > "$jobfile" << EOF
#!/bin/bash

#flux: --job-name=${job_name}
#flux: --output={{name}}-{{id}}.out
#flux: --nodes=${NNODES}
#flux: --time=${WALLTIME}
#flux: --exclusive
EOF
    [[ -n "$ACCOUNT" ]] && echo "#flux: --bank=${ACCOUNT}" >> "$jobfile"
    [[ -n "$QUEUE" ]]   && echo "#flux: --queue=${QUEUE}"  >> "$jobfile"

    local env_vars
    env_vars=$(get_config "$PLATFORM" "env_vars")
    if [[ -n "$env_vars" ]]; then
        for var in $env_vars; do echo "export $var" >> "$jobfile"; done
    fi

    cat >> "$jobfile" << EOF

echo "Cleaning up previous run outputs..."
rm -f out.*.log Backtrace.* *core* warpx_used_inputs
rm -rf diags

export OMP_NUM_THREADS=${OMP_THREADS}

flux run --exclusive --nodes=${NNODES} --ntasks ${NTASKS} ${EXEC} ${INP} ${extra_args} 2>&1 | tee out.${PLATFORM}.log
EOF
}

# Simple interactive run wrapper (rarely useful for a sweep, but supported)
run_interactive_one() {
    local extra_args="$1"
    local scheduler runcmd
    scheduler=$(get_config "$PLATFORM" "scheduler")

    case "$scheduler" in
        slurm)
            local debug_queue
            debug_queue=$(get_config "$PLATFORM" "debug_queue" "pdebug")
            runcmd="srun --exclusive -N $NNODES -n $NTASKS -p $debug_queue --export=ALL"
            if [[ "$GPU_SUPPORT" == "true" ]]; then
                local total_gpus=$((NTASKS * GPUS_PER_TASK))
                runcmd="$runcmd -G ${total_gpus} --gpus-per-task=${GPUS_PER_TASK}"
            fi
            ;;
        flux)
            local debug_queue tasks_per_node env_vars
            debug_queue=$(get_config "$PLATFORM" "debug_queue" "pdebug")
            tasks_per_node=$((NTASKS / NNODES))
            runcmd="flux run --exclusive --nodes=$NNODES --tasks-per-node=$tasks_per_node --verbose --setopt=mpibind=verbose:1 -q=$debug_queue"
            env_vars=$(get_config "$PLATFORM" "env_vars")
            for var in $env_vars; do export "${var?}"; done
            ;;
        direct)
            local mpi_launcher
            mpi_launcher=$(get_config "$PLATFORM" "mpi_launcher" "mpirun")
            if command -v "$mpi_launcher" &>/dev/null; then
                runcmd="$mpi_launcher -n $NTASKS"
            else
                warn "MPI launcher '$mpi_launcher' not found, running serially"
                runcmd=""
            fi
            ;;
        *) error "Unknown scheduler: $scheduler" ;;
    esac

    export OMP_NUM_THREADS=$OMP_THREADS
    if [[ -n "$DRY_RUN" ]]; then
        echo "Would execute in $(pwd):"
        echo "  $runcmd $EXEC $INP $extra_args"
        return 0
    fi
    if [[ -n "$runcmd" ]]; then
        $runcmd $EXEC $INP $extra_args 2>&1 | tee "out.${PLATFORM}.log"
    else
        $EXEC $INP $extra_args 2>&1 | tee "out.${PLATFORM}.log"
    fi
}

# -----------------------------------------------------------------------------
# Single-ppc submission
# -----------------------------------------------------------------------------
submit_one_ppc() {
    local nx="$1" nz="$2"
    local total=$((nx * nz))
    local tag=$(printf "ppc%04d" "$total")
    local job_name="warpx_${CASE}_${tag}"
    local workdir="$ROOT_DIR/.run_${CASE}.${PLATFORM}.${tag}.$(printf "nproc%05d" "$NTASKS")"

    info ""
    info "=========================================="
    info "ppc $total  (Nppc_x=$nx, Nppc_z=$nz)"
    info "=========================================="

    if [[ -d "$workdir" ]]; then
        if [[ -n "$SKIP_EXISTING" ]]; then
            info "Skipping (exists): $workdir"
            return 0
        fi
        info "Removing existing directory: $workdir"
        rm -rf "$workdir"
    fi
    info "Creating working directory: $workdir"
    mkdir -p "$workdir"
    cd "$workdir"

    cp "$INPUT_FILE" .
    INP="$(basename "$INPUT_FILE")"
    [[ -f "$INPUTS_DIR/.petscrc" ]] && cp "$INPUTS_DIR/.petscrc" .

    # Build WarpX extra-args string:
    # - ppc overrides always come from the sweep loop
    # - optional max_step
    # - GPU-aware MPI flag on GPU platforms
    # - any user-supplied extras pass through to every run
    local extra_args="my_constants.Nppc_x=${nx} my_constants.Nppc_z=${nz}"
    [[ -n "$MAX_STEPS" ]] && extra_args="$extra_args max_step=${MAX_STEPS}"
    [[ "$GPU_SUPPORT" == "true" ]] && extra_args="$extra_args amrex.use_gpu_aware_mpi=1"
    for arg in "${EXTRA_WARPX_ARGS[@]}"; do
        extra_args="$extra_args $arg"
    done

    local scheduler
    scheduler=$(get_config "$PLATFORM" "scheduler")

    case "$MODE" in
        interactive|i)
            info "Running interactively in $workdir"
            run_interactive_one "$extra_args"
            ;;
        batch|b)
            case "$scheduler" in
                slurm)
                    generate_slurm_batch "warpx.job" "$job_name" "$extra_args"
                    chmod +x "warpx.job"
                    if [[ -n "$DRY_RUN" ]]; then
                        info "Dry-run; warpx.job contents:"
                        cat "warpx.job"
                    else
                        info "Submitting: sbatch warpx.job"
                        sbatch "warpx.job"
                    fi
                    ;;
                flux)
                    generate_flux_batch "warpx.job" "$job_name" "$extra_args"
                    chmod +x "warpx.job"
                    if [[ -n "$DRY_RUN" ]]; then
                        info "Dry-run; warpx.job contents:"
                        cat "warpx.job"
                    else
                        info "Submitting: flux batch warpx.job"
                        flux batch "warpx.job"
                    fi
                    ;;
                direct)
                    warn "Platform '$PLATFORM' has no batch scheduler; running interactively"
                    run_interactive_one "$extra_args"
                    ;;
                *) error "Unknown scheduler: $scheduler" ;;
            esac
            ;;
        *) error "Unknown mode: $MODE (interactive|batch)" ;;
    esac

    cd "$ROOT_DIR"
}

# =============================================================================
# Main
# =============================================================================
if [[ $# -eq 0 ]]; then :; fi   # help not auto-shown; default sweep is useful

CASE="$DEFAULT_CASE"
PPC_LIST="$DEFAULT_PPC_LIST"
MODE="batch"
OVERRIDE_NTASKS=""
OVERRIDE_NNODES=""
OVERRIDE_QUEUE=""
OVERRIDE_WALLTIME=""
MAX_STEPS=""
DRY_RUN=""
VERBOSE=""
SKIP_EXISTING=""
EXTRA_WARPX_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--case=*)       [[ "$1" == -c ]] && { shift; CASE="$1"; } || CASE="${1#*=}" ;;
        -p|--ppc=*)        [[ "$1" == -p ]] && { shift; PPC_LIST="$1"; } || PPC_LIST="${1#*=}" ;;
        -m|--mode=*)       [[ "$1" == -m ]] && { shift; MODE="$1"; } || MODE="${1#*=}" ;;
        -n|--ntasks=*)     [[ "$1" == -n ]] && { shift; OVERRIDE_NTASKS="$1"; } || OVERRIDE_NTASKS="${1#*=}" ;;
        -N|--nnodes=*)     [[ "$1" == -N ]] && { shift; OVERRIDE_NNODES="$1"; } || OVERRIDE_NNODES="${1#*=}" ;;
        -q|--queue=*)      [[ "$1" == -q ]] && { shift; OVERRIDE_QUEUE="$1"; } || OVERRIDE_QUEUE="${1#*=}" ;;
        -t|--walltime=*)   [[ "$1" == -t ]] && { shift; OVERRIDE_WALLTIME="$1"; } || OVERRIDE_WALLTIME="${1#*=}" ;;
        -s|--max-steps=*)  [[ "$1" == -s ]] && { shift; MAX_STEPS="$1"; } || MAX_STEPS="${1#*=}" ;;
        --skip-existing)   SKIP_EXISTING=1 ;;
        -d|--dry-run)      DRY_RUN=1 ;;
        -v|--verbose)      VERBOSE=1 ;;
        -h|--help)         usage ;;
        -*)                error "Unknown option: $1" ;;
        *)                 EXTRA_WARPX_ARGS+=("$1") ;;
    esac
    shift
done

# -------- platform autodetect (same logic as run_warpx.sh) --------
if [[ -n "$NERSC_HOST" ]]; then
    PLATFORM="$NERSC_HOST"
elif [[ -n "$LCHOST" ]]; then
    PLATFORM="$LCHOST"
else
    HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"
    case "$HOSTNAME_SHORT" in
        dane*)     PLATFORM="dane" ;;
        matrix*)   PLATFORM="matrix" ;;
        tuolumne*) PLATFORM="tuolumne" ;;
        *)         PLATFORM="desktop" ;;
    esac
    if [[ "$PLATFORM" == "desktop" ]]; then
        info "No HPC environment detected, using local desktop"
    else
        info "Auto-detected LC platform: $PLATFORM (from hostname $HOSTNAME_SHORT)"
    fi
fi

validate_env

# -------- load platform defaults, apply overrides --------
SCHEDULER=$(get_config "$PLATFORM" "scheduler")
NTASKS="${OVERRIDE_NTASKS:-$(get_config "$PLATFORM" "ntasks" "4")}"
NNODES="${OVERRIDE_NNODES:-$(get_config "$PLATFORM" "nnodes" "1")}"
QUEUE="${OVERRIDE_QUEUE:-$(get_config "$PLATFORM" "queue")}"
WALLTIME="${OVERRIDE_WALLTIME:-$(get_config "$PLATFORM" "walltime" "4:00:00")}"
GPU_SUPPORT=$(get_config "$PLATFORM" "gpu_support" "false")
GPUS_PER_TASK=$(get_config "$PLATFORM" "gpus_per_task" "1")
ACCOUNT=$(get_config "$PLATFORM" "account")
OMP_THREADS=$(get_config "$PLATFORM" "omp_threads" "1")

debug "Platform config: scheduler=$SCHEDULER ntasks=$NTASKS nnodes=$NNODES queue=$QUEUE walltime=$WALLTIME gpu=$GPU_SUPPORT"

info "ppc sweep"
info "  Case:       $CASE"
info "  Platform:   $PLATFORM"
info "  Scheduler:  $SCHEDULER"
info "  Tasks:      $NTASKS"
info "  Nodes:      $NNODES"
info "  Queue:      ${QUEUE:-default}"
info "  Walltime:   $WALLTIME"
info "  Mode:       $MODE"
info "  ppc list:   $PPC_LIST"
[[ -n "$MAX_STEPS" ]] && info "  max_step:   $MAX_STEPS"
[[ ${#EXTRA_WARPX_ARGS[@]} -gt 0 ]] && info "  extra warpx: ${EXTRA_WARPX_ARGS[*]}"
[[ -n "$DRY_RUN" ]] && info "  DRY RUN (no submission)"
echo

# -------- parse ppc list and submit --------
IFS=',' read -r -a PPC_ENTRIES <<< "$PPC_LIST"
for entry in "${PPC_ENTRIES[@]}"; do
    entry="${entry// /}"
    [[ -z "$entry" ]] && continue
    if [[ ! "$entry" =~ ^([0-9]+)x([0-9]+)$ ]]; then
        error "ppc entry '$entry' is not in NXxNZ form (e.g., 20x20)"
    fi
    nx="${BASH_REMATCH[1]}"
    nz="${BASH_REMATCH[2]}"
    submit_one_ppc "$nx" "$nz"
done

info ""
info "Done. Sweep submissions complete."
