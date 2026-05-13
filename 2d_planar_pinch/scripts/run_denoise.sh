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
#   predict                Run particle-denoise predict (generate triptychs)
#   diagnose               Run particle-denoise diagnose (RMSE bar charts)
#   all                    Run inspect, train, and evaluate in sequence
#
# Options:
#   -c, --config=CASES     One or more configuration cases (required)
#                          Can specify multiple cases or use wildcards
#                          Example: -c unet_l2 (single case)
#                          Example: -c unet_l2 unet_l1 (multiple cases)
#                          Example: -c "unet_*" (wildcard - all U-Net cases)
#                          Example: -c "unet_l*" (wildcard - unet_l1, unet_l2, etc.)
#
#   -l, --list             List available cases and exit
#                          Example: ./run_denoise.sh -l
#
#   -m, --mode=MODE        Execution mode: interactive (default) or batch
#                          Example: -m batch (submit SLURM/Flux job)
#                          Example: -m interactive (run directly)
#
#   -o, --out-dir=DIR      Output directory for training/predict (default: WORKDIR)
#                          Example: -o /path/to/custom/output
#
#   -k, --checkpoint=FILE  Checkpoint file for evaluate/predict (required for these actions)
#                          Example: -k .run_pdn_unet_l2.matrix/best.pt
#                          Example: -k /path/to/checkpoint.pt
#
#   --metrics=FILE         Metrics CSV file for diagnose action (required for diagnose)
#                          Example: --metrics=.run_pdn_unet_l2.matrix/metrics.csv
#
#   --tier=LIST            Comma-separated tier list for predict (default: all)
#                          Example: --tier=0,1,2
#                          Example: --tier=1
#
#   --steps=LIST           Comma-separated step list for predict (default: 4 evenly-spaced)
#                          Example: --steps=0,100,200,300
#                          Example: --steps=50,150,250
#
#   -n, --ntasks=N         Override number of MPI tasks (for multi-GPU)
#                          Example: -n 8 (use 8 GPUs)
#
#   -N, --nnodes=N         Override number of nodes
#                          Example: -N 2 (use 2 nodes)
#
#   -q, --queue=NAME       Override queue/partition name
#                          Example: -q pbatch
#                          Example: -q debug
#
#   -t, --walltime=TIME    Override walltime (format: HH:MM:SS or H:MM:SS)
#                          Example: -t 4:00:00 (4 hours)
#                          Example: -t 0:30:00 (30 minutes)
#
#   -r, --resume[=PATH]    Resume training from a checkpoint instead of
#                          starting fresh. Bare flag uses
#                          <WORKDIR>/last.pt; with =PATH, that ckpt.
#                          Implies the WORKDIR is preserved (not deleted)
#                          for train/all actions.
#                          Example: -r                    (resume from last.pt)
#                          Example: --resume=/path/last.pt
#
#   -d, --dry-run          Show what would be executed without running
#                          Example: ./run_denoise.sh -d -c config.yaml train
#
#   -P, --list-platforms   List supported platforms
#                          Example: ./run_denoise.sh -P
#
#   -v, --verbose          Enable verbose output
#                          Example: ./run_denoise.sh -v -c config.yaml inspect
#
#   -h, --help             Show this help message
#
# Additional particle-denoise parameters can be passed as non-option arguments:
#   Example: ./run_denoise.sh -c config.yaml train train.lr=2e-4
#   Example: ./run_denoise.sh -c config.yaml train model.base_channels=64
#
# Common Usage Examples:
#
#   1. List available cases:
#      ./run_denoise.sh -l
#
#   2. Run all steps (inspect, train, evaluate) in interactive mode:
#      ./run_denoise.sh -c unet_l2 all
#
#   3. Submit batch job to run all steps:
#      ./run_denoise.sh -m batch -c unet_l2 all
#
#   4. Train with custom learning rate (interactive):
#      ./run_denoise.sh -c unet_l2 train train.lr=1e-4
#
#   5. Train in batch mode with 8 GPUs and 2-hour walltime:
#      ./run_denoise.sh -m batch -n 8 -t 2:00:00 -c localcnn_kpcn train
#
#   6. Evaluate with specific checkpoint:
#      ./run_denoise.sh -c unet_l2 -k .run_pdn_unet_l2.matrix/best.pt evaluate
#
#   7. Generate prediction triptychs for specific tiers and steps:
#      ./run_denoise.sh -c unet_l2 -k best.pt --tier=0,1 --steps=50,100,150 predict
#
#   8. Run diagnose with metrics file:
#      ./run_denoise.sh -c unet_l2 --metrics=metrics.csv diagnose
#
#   9. Dry-run to see commands without executing:
#      ./run_denoise.sh -d -c unet_charbonnier train
#
#   10. Submit batch job to debug queue with verbose output:
#       ./run_denoise.sh -v -m batch -q debug -c unet_l2 inspect
#
#   11. Train multiple cases sequentially:
#       ./run_denoise.sh -c unet_l2 unet_l1 unet_charbonnier train
#
#   12. Train all U-Net cases with wildcard:
#       ./run_denoise.sh -c "unet_*" train
#
#   13. Run all local-CNN cases in batch mode:
#       ./run_denoise.sh -m batch -c "localcnn_*" all
#

set -e
set -o pipefail   # so `cmd | tee log` propagates cmd's exit status

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$SCRIPT_DIR/platforms_denoise.conf"
# DENOISE_EXEC is set dynamically based on case name (see below)

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

# List available cases
list_cases() {
    echo "Available cases:"
    local denoise_dir="${ROOT_DIR}/denoise"
    if [[ ! -d "$denoise_dir" ]]; then
        echo "  No denoise directory found at: $denoise_dir"
        return
    fi

    for yaml in "$denoise_dir"/planar_pinch_2d_*.yaml; do
        [[ ! -f "$yaml" ]] && continue
        local basename=$(basename "$yaml")
        # Skip the base file
        [[ "$basename" == *"_base.yaml" ]] && continue
        # Extract case name: planar_pinch_2d_<case>.yaml -> <case>
        local case_name="${basename#planar_pinch_2d_}"
        case_name="${case_name%.yaml}"
        echo "  $case_name"
    done
}

# Expand short case name to full path if needed
# Usage: expand_config_path <case_name_or_path>
# Returns: full path to YAML file
expand_config_path() {
    local input="$1"

    # If it's already an absolute path, use it as-is
    if [[ "${input:0:1}" == "/" ]]; then
        echo "$input"
        return
    fi

    # If it's a relative path with directory separators, resolve it
    if [[ "$input" == */* ]]; then
        echo "$(pwd)/$input"
        return
    fi

    # Otherwise, treat it as a short case name
    # Try to find planar_pinch_2d_<case>.yaml in denoise directory
    local denoise_dir="${ROOT_DIR}/denoise"
    local full_path="${denoise_dir}/planar_pinch_2d_${input}.yaml"

    if [[ -f "$full_path" ]]; then
        echo "$full_path"
    else
        # Not found as short name, return original input (will fail validation later)
        echo "$input"
    fi
}

# Validate environment and inputs
validate() {
    # Check particle-denoise is available
    if ! command -v "$DENOISE_EXEC" &>/dev/null && ! command -v python &>/dev/null; then
        error "Neither '$DENOISE_EXEC' nor 'python' found in PATH.
       Please ensure particle-denoise is installed."
    fi

    # If the case-appropriate denoise command is on PATH, use it;
    # otherwise fall back to the matching Python module.
    if command -v "$DENOISE_EXEC" &>/dev/null; then
        DENOISE_CMD="$DENOISE_EXEC"
    elif [[ "$DENOISE_EXEC" == "local-particle-denoise" ]]; then
        DENOISE_CMD="python -m localParticleDenoiser"
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
        error "No action specified. Use one of: inspect, train, evaluate, predict, diagnose, all"
    fi

    case "$ACTION" in
        inspect|train|all) ;;
        evaluate|predict)
            if [[ -z "$CHECKPOINT" ]]; then
                error "Checkpoint file required for $ACTION. Use -k/--checkpoint to specify."
            fi
            if [[ ! -f "$CHECKPOINT" ]]; then
                error "Checkpoint file not found: $CHECKPOINT"
            fi
            ;;
        diagnose)
            if [[ -z "$METRICS_CSV" ]]; then
                error "Metrics CSV file required for diagnose. Use --metrics to specify."
            fi
            if [[ ! -f "$METRICS_CSV" ]]; then
                error "Metrics CSV file not found: $METRICS_CSV"
            fi
            ;;
        *) error "Unknown action: $ACTION (use inspect, train, evaluate, predict, diagnose, or all)" ;;
    esac
}

# Run all actions in sequence (inspect, train, evaluate)
run_all_actions() {
    info "Running all actions in sequence: inspect -> train -> evaluate"
    echo

    # Save original ACTION and OUT_DIR
    local saved_action="$ACTION"
    local saved_out_dir="$OUT_DIR"

    # Run inspect
    ACTION="inspect"
    info "Step 1/3: Running inspect"
    run_interactive
    local inspect_status=$?
    if [[ $inspect_status -ne 0 && -z "$DRY_RUN" ]]; then
        error "Inspect failed with exit code $inspect_status. Stopping."
    fi
    echo

    # Run train
    ACTION="train"
    # Set OUT_DIR for training if not already set
    if [[ -z "$OUT_DIR" ]]; then
        OUT_DIR="$WORKDIR"
    fi
    info "Step 2/3: Running train"
    run_interactive
    local train_status=$?
    if [[ $train_status -ne 0 && -z "$DRY_RUN" ]]; then
        error "Train failed with exit code $train_status. Stopping."
    fi
    echo

    # Find the best checkpoint (skip check in dry-run mode)
    if [[ -z "$DRY_RUN" && ! -f "$WORKDIR/best.pt" ]]; then
        error "Training did not produce best.pt checkpoint. Cannot proceed to evaluate."
    fi
    CHECKPOINT="$WORKDIR/best.pt"

    # Run evaluate
    ACTION="evaluate"
    info "Step 3/3: Running evaluate with checkpoint: $CHECKPOINT"
    run_interactive
    local eval_status=$?
    if [[ $eval_status -ne 0 && -z "$DRY_RUN" ]]; then
        error "Evaluate failed with exit code $eval_status."
    fi

    # Restore original ACTION and OUT_DIR
    ACTION="$saved_action"
    OUT_DIR="$saved_out_dir"

    if [[ -z "$DRY_RUN" ]]; then
        info "All actions completed successfully!"
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
    # --cpus-per-task drives the per-task cgroup memory under SLURM's
    # MaxMemPerCPU enforcement; without it tasks default to ~1 CPU
    # of memory and can be OOM-killed long before the node is full.
    [[ -n "$CPUS_PER_TASK" ]] && echo "#SBATCH --cpus-per-task=${CPUS_PER_TASK}" >> "$jobfile"

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

    # Build srun prefix command with GPU support if needed.
    # --cpu-bind=none: by default srun on Matrix (and many SLURM
    # clusters) pins each task to a single core via --cpu-bind=cores,
    # which makes os.sched_getaffinity() return a 1-CPU mask even
    # when --cpus-per-task allocates many. PyTorch's DataLoader
    # then clamps num_workers to 1, regardless of how many cores
    # the cgroup grants. --cpu-bind=none lifts the kernel-level
    # binding so the task can use all cores in its cgroup.
    local srun_prefix=""
    local cpu_arg=""
    [[ -n "$CPUS_PER_TASK" ]] && cpu_arg=" -c ${CPUS_PER_TASK}"
    if [[ "$GPU_SUPPORT" == "true" ]]; then
        local total_gpus=$((NTASKS * GPUS_PER_TASK))
        srun_prefix="srun --exclusive --cpu-bind=none -N ${NNODES} -G ${total_gpus} -n ${NTASKS}${cpu_arg}"
    else
        srun_prefix="srun --cpu-bind=none -N ${NNODES} -n ${NTASKS}${cpu_arg}"
    fi

    echo "" >> "$jobfile"

    # Handle "all" action - run inspect, train, evaluate sequentially in same job
    # Translate the script-level RESUME into a flag we can append to
    # the denoise CLI for the train (and 'all') subcommand.
    local resume_flag=""
    case "$RESUME" in
        "")    resume_flag="" ;;
        auto)  resume_flag=" --resume" ;;
        *)     resume_flag=" --resume=${RESUME}" ;;
    esac

    if [[ "$ACTION" == "all" ]]; then
        cat >> "$jobfile" << 'EOFSCRIPT'
# Step 1/3: Run inspect
echo "==> Step 1/3: Running inspect"
EOFSCRIPT
        local cmd_inspect="$DENOISE_CMD inspect --config $CONFIG_YAML"
        [[ -n "$EXTRA_DENOISE_ARGS" ]] && cmd_inspect="$cmd_inspect $EXTRA_DENOISE_ARGS"
        echo "$srun_prefix $cmd_inspect 2>&1 | tee $WORKDIR/denoise_inspect.${PLATFORM}.log" >> "$jobfile"
        cat >> "$jobfile" << 'EOFSCRIPT'
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Inspect failed. Stopping."
    exit 1
fi
echo ""

# Step 2/3: Run train
echo "==> Step 2/3: Running train"
EOFSCRIPT
        local cmd_train="$DENOISE_CMD train --config $CONFIG_YAML"
        [[ -n "$OUT_DIR" ]] && cmd_train="$cmd_train --out-dir $OUT_DIR"
        cmd_train="$cmd_train$resume_flag"
        [[ -n "$EXTRA_DENOISE_ARGS" ]] && cmd_train="$cmd_train $EXTRA_DENOISE_ARGS"
        echo "$srun_prefix $cmd_train 2>&1 | tee $WORKDIR/denoise_train.${PLATFORM}.log" >> "$jobfile"
        cat >> "$jobfile" << EOFSCRIPT
if [ \${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Train failed. Stopping."
    exit 1
fi
echo ""

# Step 3/3: Run evaluate with best checkpoint
echo "==> Step 3/3: Running evaluate with checkpoint: $WORKDIR/best.pt"
if [ ! -f "$WORKDIR/best.pt" ]; then
    echo "ERROR: Training did not produce best.pt checkpoint. Stopping."
    exit 1
fi
EOFSCRIPT
        local cmd_evaluate="$DENOISE_CMD evaluate --config $CONFIG_YAML --checkpoint $WORKDIR/best.pt"
        [[ -n "$EXTRA_DENOISE_ARGS" ]] && cmd_evaluate="$cmd_evaluate $EXTRA_DENOISE_ARGS"
        echo "$srun_prefix $cmd_evaluate 2>&1 | tee $WORKDIR/denoise_evaluate.${PLATFORM}.log" >> "$jobfile"
        cat >> "$jobfile" << 'EOFSCRIPT'
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Evaluate failed."
    exit 1
fi
echo ""
echo "==> All actions completed successfully!"
EOFSCRIPT
    else
        # Single action - original behavior
        local cmd="$DENOISE_CMD $ACTION --config $CONFIG_YAML"
        [[ "$ACTION" == "train" && -n "$OUT_DIR" ]] && cmd="$cmd --out-dir $OUT_DIR"
        [[ "$ACTION" == "train" ]] && cmd="$cmd$resume_flag"
        [[ "$ACTION" == "evaluate" && -n "$CHECKPOINT" ]] && cmd="$cmd --checkpoint $CHECKPOINT"
        [[ "$ACTION" == "predict" && -n "$CHECKPOINT" ]] && cmd="$cmd --checkpoint $CHECKPOINT"
        [[ "$ACTION" == "predict" && -n "$OUT_DIR" ]] && cmd="$cmd --out-dir $OUT_DIR"
        [[ "$ACTION" == "predict" && -n "$TIER_LIST" ]] && cmd="$cmd --tier $TIER_LIST"
        [[ "$ACTION" == "predict" && -n "$STEPS_LIST" ]] && cmd="$cmd --steps $STEPS_LIST"
        [[ "$ACTION" == "diagnose" && -n "$METRICS_CSV" ]] && cmd="$cmd --metrics $METRICS_CSV"
        [[ "$ACTION" == "diagnose" && -n "$OUT_DIR" ]] && cmd="$cmd --out-path $OUT_DIR/per_channel_rmse.png"
        [[ -n "$EXTRA_DENOISE_ARGS" ]] && cmd="$cmd $EXTRA_DENOISE_ARGS"

        echo "$srun_prefix $cmd 2>&1 | tee $WORKDIR/denoise_${ACTION}.${PLATFORM}.log" >> "$jobfile"
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

    # Build flux run prefix
    local cpu_arg=""
    [[ -n "$CPUS_PER_TASK" ]] && cpu_arg=" --cores-per-task=${CPUS_PER_TASK}"
    local flux_prefix="flux run --exclusive --nodes=${NNODES} --ntasks ${NTASKS}${cpu_arg}"

    # See generate_slurm_batch for resume_flag rationale.
    local resume_flag=""
    case "$RESUME" in
        "")    resume_flag="" ;;
        auto)  resume_flag=" --resume" ;;
        *)     resume_flag=" --resume=${RESUME}" ;;
    esac

    # Handle "all" action - run inspect, train, evaluate sequentially in same job
    if [[ "$ACTION" == "all" ]]; then
        cat >> "$jobfile" << 'EOFSCRIPT'
# Step 1/3: Run inspect
echo "==> Step 1/3: Running inspect"
EOFSCRIPT
        local cmd_inspect="$DENOISE_CMD inspect --config $CONFIG_YAML"
        [[ -n "$EXTRA_DENOISE_ARGS" ]] && cmd_inspect="$cmd_inspect $EXTRA_DENOISE_ARGS"
        echo "$flux_prefix $cmd_inspect 2>&1 | tee $WORKDIR/denoise_inspect.${PLATFORM}.log" >> "$jobfile"
        cat >> "$jobfile" << 'EOFSCRIPT'
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Inspect failed. Stopping."
    exit 1
fi
echo ""

# Step 2/3: Run train
echo "==> Step 2/3: Running train"
EOFSCRIPT
        local cmd_train="$DENOISE_CMD train --config $CONFIG_YAML"
        [[ -n "$OUT_DIR" ]] && cmd_train="$cmd_train --out-dir $OUT_DIR"
        cmd_train="$cmd_train$resume_flag"
        [[ -n "$EXTRA_DENOISE_ARGS" ]] && cmd_train="$cmd_train $EXTRA_DENOISE_ARGS"
        echo "$flux_prefix $cmd_train 2>&1 | tee $WORKDIR/denoise_train.${PLATFORM}.log" >> "$jobfile"
        cat >> "$jobfile" << EOFSCRIPT
if [ \${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Train failed. Stopping."
    exit 1
fi
echo ""

# Step 3/3: Run evaluate with best checkpoint
echo "==> Step 3/3: Running evaluate with checkpoint: $WORKDIR/best.pt"
if [ ! -f "$WORKDIR/best.pt" ]; then
    echo "ERROR: Training did not produce best.pt checkpoint. Stopping."
    exit 1
fi
EOFSCRIPT
        local cmd_evaluate="$DENOISE_CMD evaluate --config $CONFIG_YAML --checkpoint $WORKDIR/best.pt"
        [[ -n "$EXTRA_DENOISE_ARGS" ]] && cmd_evaluate="$cmd_evaluate $EXTRA_DENOISE_ARGS"
        echo "$flux_prefix $cmd_evaluate 2>&1 | tee $WORKDIR/denoise_evaluate.${PLATFORM}.log" >> "$jobfile"
        cat >> "$jobfile" << 'EOFSCRIPT'
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Evaluate failed."
    exit 1
fi
echo ""
echo "==> All actions completed successfully!"
EOFSCRIPT
    else
        # Single action - original behavior
        local cmd="$DENOISE_CMD $ACTION --config $CONFIG_YAML"
        [[ "$ACTION" == "train" && -n "$OUT_DIR" ]] && cmd="$cmd --out-dir $OUT_DIR"
        [[ "$ACTION" == "train" ]] && cmd="$cmd$resume_flag"
        [[ "$ACTION" == "evaluate" && -n "$CHECKPOINT" ]] && cmd="$cmd --checkpoint $CHECKPOINT"
        [[ "$ACTION" == "predict" && -n "$CHECKPOINT" ]] && cmd="$cmd --checkpoint $CHECKPOINT"
        [[ "$ACTION" == "predict" && -n "$OUT_DIR" ]] && cmd="$cmd --out-dir $OUT_DIR"
        [[ "$ACTION" == "predict" && -n "$TIER_LIST" ]] && cmd="$cmd --tier $TIER_LIST"
        [[ "$ACTION" == "predict" && -n "$STEPS_LIST" ]] && cmd="$cmd --steps $STEPS_LIST"
        [[ "$ACTION" == "diagnose" && -n "$METRICS_CSV" ]] && cmd="$cmd --metrics $METRICS_CSV"
        [[ "$ACTION" == "diagnose" && -n "$OUT_DIR" ]] && cmd="$cmd --out-path $OUT_DIR/per_channel_rmse.png"
        [[ -n "$EXTRA_DENOISE_ARGS" ]] && cmd="$cmd $EXTRA_DENOISE_ARGS"

        echo "$flux_prefix $cmd 2>&1 | tee $WORKDIR/denoise_${ACTION}.${PLATFORM}.log" >> "$jobfile"
    fi
}

generate_run_script() {
    local runfile="$WORKDIR/run_${ACTION}.sh"
    local scheduler=$(get_config "$PLATFORM" "scheduler")
    local runcmd=""

    case "$scheduler" in
        slurm)
            local debug_queue=$(get_config "$PLATFORM" "debug_queue" "pdebug")
            runcmd="srun --exclusive --cpu-bind=none -N $NNODES -n $NTASKS -p $debug_queue --export=ALL"
            [[ -n "$CPUS_PER_TASK" ]] && runcmd="$runcmd -c $CPUS_PER_TASK"
            if [[ "$GPU_SUPPORT" == "true" ]]; then
                local total_gpus=$((NTASKS * GPUS_PER_TASK))
                runcmd="$runcmd -G ${total_gpus}"
            fi
            ;;
        flux)
            local debug_queue=$(get_config "$PLATFORM" "debug_queue" "pdebug")
            runcmd="flux run --exclusive --nodes=$NNODES --ntasks=$NTASKS --verbose --setopt=mpibind=verbose:1 --queue=$debug_queue"
            [[ -n "$CPUS_PER_TASK" ]] && runcmd="$runcmd --cores-per-task=$CPUS_PER_TASK"
            ;;
        direct)
            runcmd=""
            ;;
    esac

    # See generate_slurm_batch for resume_flag rationale.
    local resume_flag=""
    case "$RESUME" in
        "")    resume_flag="" ;;
        auto)  resume_flag=" --resume" ;;
        *)     resume_flag=" --resume=${RESUME}" ;;
    esac

    # Build command
    local cmd="$DENOISE_CMD $ACTION --config $CONFIG_YAML"
    [[ ("$ACTION" == "train" || "$ACTION" == "all") && -n "$OUT_DIR" ]] && cmd="$cmd --out-dir $OUT_DIR"
    [[ "$ACTION" == "train" ]] && cmd="$cmd$resume_flag"
    [[ "$ACTION" == "evaluate" && -n "$CHECKPOINT" ]] && cmd="$cmd --checkpoint $CHECKPOINT"
    [[ "$ACTION" == "predict" && -n "$CHECKPOINT" ]] && cmd="$cmd --checkpoint $CHECKPOINT"
    [[ "$ACTION" == "predict" && -n "$OUT_DIR" ]] && cmd="$cmd --out-dir $OUT_DIR"
    [[ "$ACTION" == "predict" && -n "$TIER_LIST" ]] && cmd="$cmd --tier $TIER_LIST"
    [[ "$ACTION" == "predict" && -n "$STEPS_LIST" ]] && cmd="$cmd --steps $STEPS_LIST"
    [[ "$ACTION" == "diagnose" && -n "$METRICS_CSV" ]] && cmd="$cmd --metrics $METRICS_CSV"
    [[ "$ACTION" == "diagnose" && -n "$OUT_DIR" ]] && cmd="$cmd --out-path $OUT_DIR/per_channel_rmse.png"
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

    # Handle "all" action - run inspect, train, evaluate sequentially
    if [[ "$ACTION" == "all" ]]; then
        cat >> "$runfile" << 'EOFSCRIPT'
# Step 1/3: Run inspect
echo "==> Step 1/3: Running inspect"
EOFSCRIPT
        local cmd_inspect="$DENOISE_CMD inspect --config $CONFIG_YAML"
        [[ -n "$EXTRA_DENOISE_ARGS" ]] && cmd_inspect="$cmd_inspect $EXTRA_DENOISE_ARGS"
        if [[ -n "$runcmd" ]]; then
            echo "$runcmd $cmd_inspect 2>&1 | tee $WORKDIR/denoise_inspect.${PLATFORM}.log" >> "$runfile"
        else
            echo "$cmd_inspect 2>&1 | tee $WORKDIR/denoise_inspect.${PLATFORM}.log" >> "$runfile"
        fi
        cat >> "$runfile" << 'EOFSCRIPT'
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Inspect failed. Stopping."
    exit 1
fi
echo ""

# Step 2/3: Run train
echo "==> Step 2/3: Running train"
EOFSCRIPT
        local cmd_train="$DENOISE_CMD train --config $CONFIG_YAML"
        [[ -n "$OUT_DIR" ]] && cmd_train="$cmd_train --out-dir $OUT_DIR"
        cmd_train="$cmd_train$resume_flag"
        [[ -n "$EXTRA_DENOISE_ARGS" ]] && cmd_train="$cmd_train $EXTRA_DENOISE_ARGS"
        if [[ -n "$runcmd" ]]; then
            echo "$runcmd $cmd_train 2>&1 | tee $WORKDIR/denoise_train.${PLATFORM}.log" >> "$runfile"
        else
            echo "$cmd_train 2>&1 | tee $WORKDIR/denoise_train.${PLATFORM}.log" >> "$runfile"
        fi
        cat >> "$runfile" << EOFSCRIPT
if [ \${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Train failed. Stopping."
    exit 1
fi
echo ""

# Step 3/3: Run evaluate with best checkpoint
echo "==> Step 3/3: Running evaluate with checkpoint: $WORKDIR/best.pt"
if [ ! -f "$WORKDIR/best.pt" ]; then
    echo "ERROR: Training did not produce best.pt checkpoint. Stopping."
    exit 1
fi
EOFSCRIPT
        local cmd_evaluate="$DENOISE_CMD evaluate --config $CONFIG_YAML --checkpoint $WORKDIR/best.pt"
        [[ -n "$EXTRA_DENOISE_ARGS" ]] && cmd_evaluate="$cmd_evaluate $EXTRA_DENOISE_ARGS"
        if [[ -n "$runcmd" ]]; then
            echo "$runcmd $cmd_evaluate 2>&1 | tee $WORKDIR/denoise_evaluate.${PLATFORM}.log" >> "$runfile"
        else
            echo "$cmd_evaluate 2>&1 | tee $WORKDIR/denoise_evaluate.${PLATFORM}.log" >> "$runfile"
        fi
        cat >> "$runfile" << 'EOFSCRIPT'
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Evaluate failed."
    exit 1
fi
echo ""
echo "==> All actions completed successfully!"
EOFSCRIPT
    else
        # Single action
        if [[ -n "$runcmd" ]]; then
            echo "$runcmd $cmd 2>&1 | tee $WORKDIR/denoise_${ACTION}.${PLATFORM}.log" >> "$runfile"
        else
            echo "$cmd 2>&1 | tee $WORKDIR/denoise_${ACTION}.${PLATFORM}.log" >> "$runfile"
        fi
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

    case "$ACTION" in
        train)
            if [[ -n "$OUT_DIR" ]]; then
                info "  Output dir: $OUT_DIR"
            else
                info "  Output dir: (auto-detected, see particle-denoise output)"
            fi
            ;;
        evaluate|predict)
            info "  Checkpoint: $CHECKPOINT"
            [[ "$ACTION" == "predict" && -n "$OUT_DIR" ]] && info "  Output dir: $OUT_DIR"
            [[ "$ACTION" == "predict" && -n "$TIER_LIST" ]] && info "  Tiers:      $TIER_LIST"
            [[ "$ACTION" == "predict" && -n "$STEPS_LIST" ]] && info "  Steps:      $STEPS_LIST"
            ;;
        diagnose)
            info "  Metrics:    $METRICS_CSV"
            [[ -n "$OUT_DIR" ]] && info "  Output:     $OUT_DIR/per_channel_rmse.png"
            ;;
    esac
    echo

    # See generate_slurm_batch for resume_flag rationale.
    local resume_flag=""
    case "$RESUME" in
        "")    resume_flag="" ;;
        auto)  resume_flag=" --resume" ;;
        *)     resume_flag=" --resume=${RESUME}" ;;
    esac

    # Build command
    local cmd="$DENOISE_CMD $ACTION --config $CONFIG_YAML"
    [[ ("$ACTION" == "train" || "$ACTION" == "all") && -n "$OUT_DIR" ]] && cmd="$cmd --out-dir $OUT_DIR"
    [[ "$ACTION" == "train" ]] && cmd="$cmd$resume_flag"
    [[ "$ACTION" == "evaluate" && -n "$CHECKPOINT" ]] && cmd="$cmd --checkpoint $CHECKPOINT"
    [[ "$ACTION" == "predict" && -n "$CHECKPOINT" ]] && cmd="$cmd --checkpoint $CHECKPOINT"
    [[ "$ACTION" == "predict" && -n "$OUT_DIR" ]] && cmd="$cmd --out-dir $OUT_DIR"
    [[ "$ACTION" == "predict" && -n "$TIER_LIST" ]] && cmd="$cmd --tier $TIER_LIST"
    [[ "$ACTION" == "predict" && -n "$STEPS_LIST" ]] && cmd="$cmd --steps $STEPS_LIST"
    [[ "$ACTION" == "diagnose" && -n "$METRICS_CSV" ]] && cmd="$cmd --metrics $METRICS_CSV"
    [[ "$ACTION" == "diagnose" && -n "$OUT_DIR" ]] && cmd="$cmd --out-path $OUT_DIR/per_channel_rmse.png"
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
            local runcmd="srun --exclusive --cpu-bind=none -N $NNODES -n $NTASKS -p $debug_queue --export=ALL"
            [[ -n "$CPUS_PER_TASK" ]] && runcmd="$runcmd -c $CPUS_PER_TASK"
            if [[ "$GPU_SUPPORT" == "true" ]]; then
                local total_gpus=$((NTASKS * GPUS_PER_TASK))
                runcmd="$runcmd -G ${total_gpus}"
            fi
            local full_cmd="$runcmd $cmd"
            ;;
        flux)
            local runcmd="flux run --exclusive --nodes=$NNODES --ntasks=$NTASKS --verbose --setopt=mpibind=verbose:1 --queue=$debug_queue"
            [[ -n "$CPUS_PER_TASK" ]] && runcmd="$runcmd --cores-per-task=$CPUS_PER_TASK"
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

    case "$ACTION" in
        train)
            if [[ -n "$OUT_DIR" ]]; then
                info "  Output dir: $OUT_DIR"
            else
                info "  Output dir: (auto-detected, see particle-denoise output)"
            fi
            ;;
        evaluate|predict)
            info "  Checkpoint: $CHECKPOINT"
            [[ "$ACTION" == "predict" && -n "$OUT_DIR" ]] && info "  Output dir: $OUT_DIR"
            [[ "$ACTION" == "predict" && -n "$TIER_LIST" ]] && info "  Tiers:      $TIER_LIST"
            [[ "$ACTION" == "predict" && -n "$STEPS_LIST" ]] && info "  Steps:      $STEPS_LIST"
            ;;
        diagnose)
            info "  Metrics:    $METRICS_CSV"
            [[ -n "$OUT_DIR" ]] && info "  Output:     $OUT_DIR/per_channel_rmse.png"
            ;;
    esac
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
CASES=()
MODE="interactive"
OUT_DIR=""
CHECKPOINT=""
METRICS_CSV=""
TIER_LIST=""
STEPS_LIST=""
OVERRIDE_NTASKS=""
OVERRIDE_NNODES=""
OVERRIDE_QUEUE=""
OVERRIDE_WALLTIME=""
DRY_RUN=""
VERBOSE=""
ACTION=""
EXTRA_DENOISE_ARGS=""
RESUME=""        # "" = no resume; "auto" = use <WORKDIR>/last.pt; PATH = use that file

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config|--config=*)
            # Normalise --config=value -> --config value so the slurp
            # loop below handles all three input forms uniformly.
            if [[ "$1" == --config=* ]]; then
                set -- "--config" "${1#*=}" "${@:2}"
            fi
            shift   # consume the -c / --config flag itself
            # Collect non-option args as case names until we hit a flag,
            # a key=value override, or an action keyword.
            while [[ $# -gt 0 && "$1" != -* && "$1" != *"="* ]]; do
                case "$1" in
                    inspect|train|evaluate|predict|diagnose|all)
                        break
                        ;;
                esac

                if [[ "$1" == *"*"* || "$1" == *"?"* ]]; then
                    # Wildcard - expand against the denoise dir.
                    shopt -s nullglob
                    _denoise_dir="${ROOT_DIR}/denoise"
                    _matched_files=("$_denoise_dir"/planar_pinch_2d_${1}.yaml)
                    shopt -u nullglob
                    for _matched in "${_matched_files[@]}"; do
                        [[ -f "$_matched" ]] || continue
                        _basename=$(basename "$_matched")
                        [[ "$_basename" == *"_base.yaml" ]] && continue
                        _case_name="${_basename#planar_pinch_2d_}"
                        _case_name="${_case_name%.yaml}"
                        CASES+=("$_case_name")
                    done
                else
                    CASES+=("$1")
                fi
                shift
            done
            continue   # inner loop already consumed the value(s)
            ;;
        -m|--mode|--mode=*)
            case "$1" in
                --mode=*) MODE="${1#*=}" ;;
                *)        shift; MODE="$1" ;;
            esac ;;
        -o|--out-dir|--out-dir=*)
            case "$1" in
                --out-dir=*) OUT_DIR="${1#*=}" ;;
                *)           shift; OUT_DIR="$1" ;;
            esac ;;
        -k|--checkpoint|--checkpoint=*)
            case "$1" in
                --checkpoint=*) CHECKPOINT="${1#*=}" ;;
                *)              shift; CHECKPOINT="$1" ;;
            esac ;;
        --metrics|--metrics=*)
            case "$1" in
                --metrics=*) METRICS_CSV="${1#*=}" ;;
                *)           shift; METRICS_CSV="$1" ;;
            esac ;;
        --tier|--tier=*)
            case "$1" in
                --tier=*) TIER_LIST="${1#*=}" ;;
                *)        shift; TIER_LIST="$1" ;;
            esac ;;
        --steps|--steps=*)
            case "$1" in
                --steps=*) STEPS_LIST="${1#*=}" ;;
                *)         shift; STEPS_LIST="$1" ;;
            esac ;;
        -n|--ntasks|--ntasks=*)
            case "$1" in
                --ntasks=*) OVERRIDE_NTASKS="${1#*=}" ;;
                *)          shift; OVERRIDE_NTASKS="$1" ;;
            esac ;;
        -N|--nnodes|--nnodes=*)
            case "$1" in
                --nnodes=*) OVERRIDE_NNODES="${1#*=}" ;;
                *)          shift; OVERRIDE_NNODES="$1" ;;
            esac ;;
        -q|--queue|--queue=*)
            case "$1" in
                --queue=*) OVERRIDE_QUEUE="${1#*=}" ;;
                *)         shift; OVERRIDE_QUEUE="$1" ;;
            esac ;;
        -t|--walltime|--walltime=*)
            case "$1" in
                --walltime=*) OVERRIDE_WALLTIME="${1#*=}" ;;
                *)            shift; OVERRIDE_WALLTIME="$1" ;;
            esac ;;
        -r|--resume|--resume=*)
            case "$1" in
                --resume=*) RESUME="${1#*=}" ;;
                *)          RESUME="auto" ;;
            esac ;;
        -d|--dry-run)   DRY_RUN=1 ;;
        -v|--verbose)   VERBOSE=1 ;;
        -l|--list)      list_cases; exit 0 ;;
        -P|--list-platforms) list_platforms; exit 0 ;;
        -h|--help)      usage ;;
        inspect|train|evaluate|predict|diagnose|all)
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

# Set default case if none specified
if [[ ${#CASES[@]} -eq 0 ]]; then
    error "No case specified. Use -c to specify one or more cases."
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

# Snapshot the user-supplied OUT_DIR / CHECKPOINT so each case
# starts from the original CLI state instead of inheriting the
# previous case's WORKDIR-derived defaults.
USER_OUT_DIR="$OUT_DIR"
USER_CHECKPOINT="$CHECKPOINT"

# Process each case
for CASE in "${CASES[@]}"; do
    OUT_DIR="$USER_OUT_DIR"
    CHECKPOINT="$USER_CHECKPOINT"

    if [[ ${#CASES[@]} -gt 1 ]]; then
        info ""
        info "=========================================="
        info "Processing case: $CASE"
        info "=========================================="
        info ""
    fi

# Expand case name to full config path
CONFIG_YAML=$(expand_config_path "$CASE")

# Extract case name from config file (needed to determine which executable to use)
# Expected format: planar_pinch_2d_<case>.yaml
CONFIG_BASENAME=$(basename "$CONFIG_YAML")
if [[ "$CONFIG_BASENAME" =~ planar_pinch_2d_(.+)\.yaml ]]; then
    CASE_NAME="${BASH_REMATCH[1]}"
else
    # Fallback: use full basename without extension
    CASE_NAME="${CONFIG_BASENAME%.yaml}"
fi

# Select the appropriate denoise executable based on case name.
# Architecture prefix in the YAML name picks the CLI:
#   localcnn_*   -> local-particle-denoise   (LocalCNN / LocalKPCN / LocalHierarchicalCNN)
#   unet_*       -> particle-denoise         (global U-Net)
#   flowunet_*   -> flow-particle-denoise    (flow-matching denoiser; see task #90)
if [[ "$CASE_NAME" == localcnn_* ]]; then
    DENOISE_EXEC="local-particle-denoise"
elif [[ "$CASE_NAME" == flowunet_* ]]; then
    DENOISE_EXEC="flow-particle-denoise"
else
    DENOISE_EXEC="particle-denoise"
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
CPUS_PER_TASK=$(get_config "$PLATFORM" "cpus_per_task" "")
ACCOUNT=$(get_config "$PLATFORM" "account")

debug "Platform config loaded:"
debug "  scheduler=$SCHEDULER ntasks=$NTASKS nnodes=$NNODES"
debug "  queue=$QUEUE walltime=$WALLTIME gpu=$GPU_SUPPORT"
debug "Using $DENOISE_EXEC for case: $CASE_NAME"

# Create working directory for logs and job scripts
WORKDIR="$ROOT_DIR/.run_pdn_${CASE_NAME}.${PLATFORM}"

# Only delete existing directory for actions that create fresh output.
# Actions like evaluate, diagnose, predict need existing files (checkpoints,
# metrics), and --resume is exactly the same situation: we must keep the
# previous training artefacts (last.pt, metrics.csv, ...) intact.
case "$ACTION" in
    train|inspect|all)
        if [[ -d "$WORKDIR" && -z "$RESUME" ]]; then
            info "Removing existing directory: $WORKDIR"
            rm -rf "$WORKDIR"
        fi
        if [[ -n "$RESUME" && -d "$WORKDIR" ]]; then
            info "Resuming in existing directory: $WORKDIR"
        else
            info "Creating working directory: $WORKDIR"
        fi
        mkdir -p "$WORKDIR"
        ;;
    evaluate|diagnose|predict)
        if [[ -d "$WORKDIR" ]]; then
            info "Using existing directory: $WORKDIR"
        else
            info "Creating working directory: $WORKDIR"
            mkdir -p "$WORKDIR"
        fi
        ;;
    *)
        # Default: create if doesn't exist
        if [[ ! -d "$WORKDIR" ]]; then
            info "Creating working directory: $WORKDIR"
            mkdir -p "$WORKDIR"
        fi
        ;;
esac

# Set default output directory if not specified by the user.
# OUT_DIR was reset to $USER_OUT_DIR at the top of this loop, so
# this only fills in a per-case default when the user did not pass -o.
if [[ -z "$OUT_DIR" ]]; then
    case "$ACTION" in
        train|predict|all)
            OUT_DIR="$WORKDIR"
            ;;
        diagnose)
            # OUT_DIR controls where per_channel_rmse.png is written;
            # default to the metrics CSV's directory.
            if [[ -n "$METRICS_CSV" ]]; then
                OUT_DIR="$(dirname "$METRICS_CSV")"
            fi
            ;;
    esac
fi

# Generate run.sh and denoise.job scripts in the working directory
info "Generating run scripts in $WORKDIR"
generate_run_script
generate_standalone_job_script
info "  Created run_${ACTION}.sh for interactive execution"
info "  Created denoise_${ACTION}.job for batch submission"

# Execute based on mode and action
if [[ "$ACTION" == "all" ]]; then
    # Special handling for "all" action - run all three steps
    case "$MODE" in
        interactive|i)
            run_all_actions
            ;;
        batch|b)
            run_batch
            ;;
        *)
            error "Unknown mode: $MODE (use 'interactive' or 'batch')"
            ;;
    esac
else
    # Normal single-action execution
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
fi

done  # End of case loop

info "Done."
