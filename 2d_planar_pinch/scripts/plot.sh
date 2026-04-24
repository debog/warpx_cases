#!/bin/bash
#
# Plotting driver for planar_pinch results (1D and 2D siblings).
# Intended to run on the HPC machine after a simulation finishes.
# Loops over .run_planar_pinch_{1,2}d.*/ directories under the parent case dir
# and invokes plot_reduced.py, plot_fields.py, plot_species.py on each.
# The Python plotters auto-detect dimensionality from the plotfiles.
#
# Usage:
#   ./run_plots.sh [OPTIONS]
#
# Options:
#   -R, --root-dir=DIR    Parent dir containing .run_* subdirs.
#                         Default: the parent of this script's directory.
#   -r, --run-dir=PATH    Plot only this run dir. May be an absolute path, a
#                         path relative to ROOT_DIR, or a glob pattern (e.g.
#                         '.run_planar_pinch_*d.tuolumne.*' — quote it so the
#                         shell doesn't expand in CWD). May be given multiple
#                         times. Default: all .run_planar_pinch_*d.* under ROOT.
#   -p, --platform=NAME   Shortcut for --run-dir: matches .run_planar_pinch_*d.NAME.*.
#                         May be given multiple times (dane/matrix/tuolumne).
#   -D, --dim=1d|2d|all   Restrict to a given dim (1d / 2d / all). Default: all.
#   -s, --steps=LIST      Comma-separated plotfile steps for field/species plots.
#                         Default: all plotfiles in mesh_data/.
#   -o, --outdir=DIR      Output dir name relative to each run dir. Default: plots.
#       --reduced-only    Only run plot_reduced.py (skip field/species).
#       --fields-only     Only run plot_fields.py.
#       --species-only    Only run plot_species.py.
#       --skip-reduced    Skip plot_reduced.py.
#       --skip-fields     Skip plot_fields.py.
#       --skip-species    Skip plot_species.py.
#   -f, --force           Regenerate plots even if they are newer than inputs.
#                         Default: per-output mtime check, skip when up-to-date.
#   -d, --dry-run         Show commands without executing.
#   -v, --verbose         Verbose Python output.
#   -h, --help            Show this message.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(dirname "$SCRIPT_DIR")"

PY=${PYTHON:-python3}
PLOT_REDUCED="$SCRIPT_DIR/plot_reduced.py"
PLOT_FIELDS="$SCRIPT_DIR/plot_fields.py"
PLOT_SPECIES="$SCRIPT_DIR/plot_species.py"

if [[ -t 1 ]]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; BLUE='\033[0;34m'; NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; NC=''
fi

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERR]${NC}  $*" >&2; }
ok()    { echo -e "${GREEN}[OK]${NC}   $*"; }

usage() { sed -n '2,31p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

ROOT_DIR="$DEFAULT_ROOT"
RUN_DIRS=()
PLATFORMS=()
STEPS=""
OUTDIR_REL="plots"
DIM_FILTER="all"
DO_REDUCED=1
DO_FIELDS=1
DO_SPECIES=1
FORCE=0
DRYRUN=0
VERBOSE=0

USER_SELECTED=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -R|--root-dir) ROOT_DIR="$2"; shift 2 ;;
        --root-dir=*) ROOT_DIR="${1#*=}"; shift ;;
        -r|--run-dir) RUN_DIRS+=("$2"); USER_SELECTED=1; shift 2 ;;
        --run-dir=*) RUN_DIRS+=("${1#*=}"); USER_SELECTED=1; shift ;;
        -p|--platform) PLATFORMS+=("$2"); USER_SELECTED=1; shift 2 ;;
        --platform=*) PLATFORMS+=("${1#*=}"); USER_SELECTED=1; shift ;;
        -D|--dim) DIM_FILTER="$2"; shift 2 ;;
        --dim=*) DIM_FILTER="${1#*=}"; shift ;;
        -s|--steps) STEPS="$2"; shift 2 ;;
        --steps=*) STEPS="${1#*=}"; shift ;;
        -o|--outdir) OUTDIR_REL="$2"; shift 2 ;;
        --outdir=*) OUTDIR_REL="${1#*=}"; shift ;;
        --reduced-only) DO_FIELDS=0; DO_SPECIES=0; shift ;;
        --fields-only)  DO_REDUCED=0; DO_SPECIES=0; shift ;;
        --species-only) DO_REDUCED=0; DO_FIELDS=0; shift ;;
        --skip-reduced) DO_REDUCED=0; shift ;;
        --skip-fields)  DO_FIELDS=0;  shift ;;
        --skip-species) DO_SPECIES=0; shift ;;
        -f|--force) FORCE=1; shift ;;
        -d|--dry-run) DRYRUN=1; shift ;;
        -v|--verbose) VERBOSE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) err "unknown option: $1"; usage; exit 1 ;;
    esac
done

# Choose the run-dir glob pattern from the dim filter.
case "$DIM_FILTER" in
    1d|1D)  DIM_GLOB="1d" ;;
    2d|2D)  DIM_GLOB="2d" ;;
    all|"") DIM_GLOB="?d" ;;
    *) err "unknown --dim value: $DIM_FILTER (expected 1d|2d|all)"; exit 1 ;;
esac

# Expand any glob patterns in RUN_DIRS entries. Each entry is tried first
# as-given (absolute or relative to CWD) and then relative to ROOT_DIR.
# Entries without glob metacharacters are passed through unchanged; entries
# with no matches in either context are flagged with a warning.
expand_run_dirs() {
    local -a expanded=()
    local entry matches
    for entry in "${RUN_DIRS[@]}"; do
        if [[ "$entry" == *"*"* || "$entry" == *"?"* || "$entry" == *"["* ]]; then
            shopt -s nullglob
            # shellcheck disable=SC2206
            matches=( $entry )
            if [[ ${#matches[@]} -eq 0 ]]; then
                # shellcheck disable=SC2206
                matches=( "$ROOT_DIR"/$entry )
            fi
            shopt -u nullglob
            if [[ ${#matches[@]} -eq 0 ]]; then
                warn "no matches for pattern: $entry"
            else
                expanded+=( "${matches[@]}" )
            fi
        else
            expanded+=( "$entry" )
        fi
    done
    RUN_DIRS=( "${expanded[@]}" )
}
expand_run_dirs

# Expand --platform entries into run-dir candidates.
for plat in "${PLATFORMS[@]}"; do
    for d in "$ROOT_DIR"/.run_planar_pinch_${DIM_GLOB}."$plat".*; do
        [[ -d "$d" ]] && RUN_DIRS+=("$d")
    done
done

# Default: all .run_* under ROOT matching the dim filter — but only if the
# user did not specify any -r / -p. If they did and got zero matches, treat
# that as "plot nothing" (not "plot everything"), which is the sensible intent.
if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
    if [[ $USER_SELECTED -eq 1 ]]; then
        err "no run directories matched any of your -r/-p selections"
        exit 1
    fi
    for d in "$ROOT_DIR"/.run_planar_pinch_${DIM_GLOB}.*; do
        [[ -d "$d" ]] && RUN_DIRS+=("$d")
    done
fi

if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
    err "no run directories found under $ROOT_DIR"
    exit 1
fi

run_cmd() {
    if [[ $VERBOSE -eq 1 ]]; then info "  \$ $*"; fi
    if [[ $DRYRUN -eq 1 ]]; then
        echo "DRYRUN: $*"
    else
        "$@"
    fi
}

plot_one_dir() {
    local rdir="$1"
    if [[ ! -d "$rdir" ]]; then
        warn "skip (not a directory): $rdir"
        return
    fi
    local outdir="$rdir/$OUTDIR_REL"
    info "=== $(basename "$rdir") ==="
    info "  outdir: $outdir"

    local has_reduced=0
    [[ -d "$rdir/diags/reduced_files" ]] && has_reduced=1
    local has_fields=0
    compgen -G "$rdir/mesh_data/field_data*" > /dev/null && has_fields=1
    local has_species=0
    compgen -G "$rdir/mesh_data/species_data*" > /dev/null && has_species=1

    local force_flag=()
    [[ $FORCE -eq 1 ]] && force_flag=(--force)

    if [[ $DO_REDUCED -eq 1 && $has_reduced -eq 1 ]]; then
        run_cmd "$PY" "$PLOT_REDUCED" "$rdir" --outdir "$outdir" "${force_flag[@]}"
    elif [[ $DO_REDUCED -eq 1 ]]; then
        warn "  no diags/reduced_files/ — skipping plot_reduced.py"
    fi

    if [[ $DO_FIELDS -eq 1 && $has_fields -eq 1 ]]; then
        if [[ -n "$STEPS" ]]; then
            run_cmd "$PY" "$PLOT_FIELDS" "$rdir" --outdir "$outdir" --steps "$STEPS" "${force_flag[@]}"
        else
            run_cmd "$PY" "$PLOT_FIELDS" "$rdir" --outdir "$outdir" "${force_flag[@]}"
        fi
    elif [[ $DO_FIELDS -eq 1 ]]; then
        warn "  no mesh_data/field_data* — skipping plot_fields.py"
    fi

    if [[ $DO_SPECIES -eq 1 && $has_species -eq 1 ]]; then
        if [[ -n "$STEPS" ]]; then
            run_cmd "$PY" "$PLOT_SPECIES" "$rdir" --outdir "$outdir" --steps "$STEPS" "${force_flag[@]}"
        else
            run_cmd "$PY" "$PLOT_SPECIES" "$rdir" --outdir "$outdir" "${force_flag[@]}"
        fi
    elif [[ $DO_SPECIES -eq 1 ]]; then
        warn "  no mesh_data/species_data* — skipping plot_species.py"
    fi

    ok "done: $rdir"
}

info "ROOT_DIR = $ROOT_DIR"
info "run dirs:"
for d in "${RUN_DIRS[@]}"; do info "  $d"; done

for d in "${RUN_DIRS[@]}"; do
    if [[ -d "$d" ]]; then
        plot_one_dir "$d"
    elif [[ -d "$ROOT_DIR/$d" ]]; then
        plot_one_dir "$ROOT_DIR/$d"
    else
        warn "not found: $d"
    fi
done
