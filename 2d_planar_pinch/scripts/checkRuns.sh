#!/bin/bash
#
# checkRuns.sh -- completeness / health check for WarpX ppc-sweep runs.
#
# Examines every run directory
#
#     .run_<case>.<platform>.ppc<NNNN>.nproc<NNNNN>
#
# in the current directory (or the directories passed as arguments) and
# reports, per run:
#
#   * did it finish cleanly  (AMReX "finalized" in the screen log)?
#   * if yes:
#       - are all expected plotfiles present?  The expected count is
#         derived from warpx_used_inputs (stop_time, const_dt, max_step,
#         the per-diagnostic intervals);
#       - any NaNs / errors / non-convergence / warnings in the log?
#       - are the plotfiles structurally intact (Header + Level_0 with
#         non-empty FAB data)?
#       - does warpx_used_inputs match the directory name, ppc included?
#   * if no:
#       - allocation timeout, or a genuine crash?
#
# Screen output may live in more than one file (the run tees stdout to
# out.*.log and the scheduler captures warpx_<case>-<jobid>.out /
# slurm*.out); all are located, the most recent is scanned.
#
# Side effect: WarpX renames pre-existing plotfile directories to
# <name>.old.<timestamp> when a restart would otherwise overwrite them.
# Those leftovers are removed by default (they confuse the file-count
# check and consume disk). Pass -n / --dry-run to skip the removal.
#
# Usage:
#   cd <dir with .run_* subdirectories>
#   /path/to/checkRuns.sh [-n|--dry-run] [run_dir ...]
#
# Note: `set -u` is intentionally off. Older bash (< 4.4, as on
# Tuolumne) treats an empty array reference as "unbound", which trips
# the arg-parsing and the "no run dirs found" branches.

# --------------------------------------------------------------------
# Presentation helpers
# --------------------------------------------------------------------
if [[ -t 1 ]]; then
    RED=$'\033[0;31m'; GRN=$'\033[0;32m'; YEL=$'\033[0;33m'
    BLU=$'\033[0;34m'; BLD=$'\033[1m'; NC=$'\033[0m'
else
    RED='' GRN='' YEL='' BLU='' BLD='' NC=''
fi
pass() { echo "    ${GRN}OK${NC}    $*"; }
warnln() { echo "    ${YEL}WARN${NC}  $*"; }
failln() { echo "    ${RED}FAIL${NC}  $*"; }
info()  { echo "    ----  $*"; }

# --------------------------------------------------------------------
# warpx_used_inputs parsing
#
# Each line is `key = <raw value>` and, when the value contained
# my_constants expressions, a trailing `# <resolved value>`. The
# resolved form is authoritative.
# --------------------------------------------------------------------

# resolved_value KEYFILE KEY  -> the resolved value (after '#'), else
# the raw value (after '=').
resolved_value() {
    local file="$1" key="$2" line
    line=$(grep -E "^[[:space:]]*${key}[[:space:]]*=" "$file" 2>/dev/null | head -1)
    [[ -z "$line" ]] && return 1
    if [[ "$line" == *"#"* ]]; then
        echo "${line##*#}" | xargs
    else
        echo "${line#*=}" | xargs
    fi
}

# last_int STRING -> the last integer token (handles `225`, `225 `,
# `0::225` interval-slice forms).
last_int() { grep -oE '[0-9]+' <<<"$1" | tail -1; }

# --------------------------------------------------------------------
# Plotfile structural validation
#
# A valid AMReX plotfile directory has a non-empty `Header` whose
# second line is the component count, and a `Level_0/` holding at
# least one non-empty `Cell_D_*` data file.
# --------------------------------------------------------------------
validate_plotfile() {
    local pf="$1"
    [[ -d "$pf" ]]                 || { echo "not a directory"; return 1; }
    [[ -s "$pf/Header" ]]          || { echo "missing/empty Header"; return 1; }
    local ncomp
    ncomp=$(sed -n '2p' "$pf/Header")
    [[ "$ncomp" =~ ^[0-9]+$ ]]     || { echo "unparseable Header"; return 1; }
    [[ -d "$pf/Level_0" ]]         || { echo "missing Level_0/"; return 1; }
    local data
    data=$(find "$pf/Level_0" -maxdepth 1 -name 'Cell_D_*' -size +0c 2>/dev/null | head -1)
    [[ -n "$data" ]]               || { echo "no non-empty Cell_D_* data"; return 1; }
    echo "ok"; return 0
}

# Validate the first, the last two, and up to 8 evenly-spaced
# plotfiles for a given prefix. Echoes a one-line summary; returns
# non-zero if any sampled plotfile is bad. WarpX restart leftovers
# (<name>.old.<timestamp>) are excluded.
validate_plotfile_set() {
    local prefix="$1" label="$2"
    local -a all
    mapfile -t all < <(ls -d ${prefix}[0-9]* 2>/dev/null | grep -v '\.old\.' | sort)
    local n=${#all[@]}
    if (( n == 0 )); then echo "MISSING"; return 1; fi
    local -a idx=(0 $((n-1)) $((n-2)))
    local k
    for k in 1 2 3 4 5 6 7 8; do idx+=( $(( k * (n-1) / 9 )) ); done
    local bad=0 checked=0 seen=" "
    for i in "${idx[@]}"; do
        (( i < 0 || i >= n )) && continue
        [[ "$seen" == *" $i "* ]] && continue
        seen+="$i "
        local reason
        reason=$(validate_plotfile "${all[$i]}")
        checked=$((checked+1))
        if [[ "$reason" != "ok" ]]; then
            bad=$((bad+1))
            failln "$label: ${all[$i]} -- $reason"
        fi
    done
    if (( bad == 0 )); then
        echo "intact ($checked/$n sampled)"
        return 0
    fi
    echo "$bad of $checked sampled plotfiles bad"
    return 1
}

# --------------------------------------------------------------------
# Per-run check
# --------------------------------------------------------------------
TOTAL=0; CLEAN=0; PROBLEM=0
declare -a SUMMARY

check_run() {
    local dir="${1%/}"
    local base; base=$(basename "$dir")
    TOTAL=$((TOTAL+1))

    echo
    echo "${BLD}=== $base ===${NC}"

    # ---- decode the directory name ---------------------------------
    local case plat ppc_dir nproc
    if [[ "$base" =~ ^\.run_(.+)\.([^.]+)\.ppc([0-9]+)\.nproc([0-9]+)$ ]]; then
        case="${BASH_REMATCH[1]}"; plat="${BASH_REMATCH[2]}"
        ppc_dir=$((10#${BASH_REMATCH[3]})); nproc=$((10#${BASH_REMATCH[4]}))
    else
        warnln "directory name does not match .run_<case>.<plat>.ppc<N>.nproc<N>"
        SUMMARY+=("$base : ${YEL}name not recognised${NC}")
        return
    fi
    info "case=$case  platform=$plat  ppc(dir)=$ppc_dir  nproc=$nproc"

    # ---- prune WarpX restart leftovers -----------------------------
    # On restart WarpX renames an existing plotfile dir to
    # <name>.old.<timestamp>. Those inflate the file-count check and
    # waste disk; remove them unless --dry-run is set.
    local -a olds
    mapfile -t olds < <(find "$dir" -maxdepth 3 -name '*.old.*' 2>/dev/null)
    if (( ${#olds[@]} > 0 )); then
        if (( DRY_RUN )); then
            info "would remove ${#olds[@]} .old.* entry/ies (re-run without -n to delete)"
        else
            rm -rf -- "${olds[@]}"
            info "removed ${#olds[@]} .old.* entry/ies (WarpX restart leftovers)"
        fi
    fi

    # ---- locate screen logs ----------------------------------------
    local -a logs
    mapfile -t logs < <(find "$dir" -maxdepth 1 -type f \
        \( -name 'out.*.log' -o -name 'warpx_*.out' -o -name 'slurm*.out' \
           -o -name '*.o[0-9]*' -o -name '*.err' \) 2>/dev/null \
        | xargs -r ls -1t 2>/dev/null)
    if (( ${#logs[@]} == 0 )); then
        failln "no screen-output file found (out.*.log / warpx_*.out / slurm*.out)"
        SUMMARY+=("$base : ${RED}no log${NC}")
        PROBLEM=$((PROBLEM+1))
        return
    fi
    local log="${logs[0]}"
    info "screen log: $(basename "$log")  ($(du -h "$log" | cut -f1))"
    (( ${#logs[@]} > 1 )) && info "(${#logs[@]} log files present; scanning the most recent)"

    # ---- crash / timeout / clean markers ---------------------------
    local tail_big tail_small
    tail_big=$(tail -c 3000000 "$log" 2>/dev/null)
    tail_small=$(tail -c 300000 "$log" 2>/dev/null)

    local finished=0 crashed=0 timedout=0
    grep -qE 'AMReX .* finalized' <<<"$tail_small" && finished=1

    # crash evidence: backtrace / core files written by this run, or
    # fatal signals in the log.
    local -a bt core
    mapfile -t bt   < <(find "$dir" -maxdepth 1 -name 'Backtrace.*' -size +0c 2>/dev/null)
    mapfile -t core < <(find "$dir" -maxdepth 1 -name '*core*' 2>/dev/null)
    if (( ${#bt[@]} > 0 )) || (( ${#core[@]} > 0 )) || \
       grep -qE 'SIGSEGV|SIGABRT|SIGBUS|MPICH ERROR|amrex::Abort|Assertion failed|corrupted size|caught signal|core dumped|type=exec' <<<"$tail_big"; then
        crashed=1
    fi
    # timeout evidence (Flux / SLURM).
    if grep -qE 'type=timeout|DUE TO TIME LIMIT|TIME LIMIT|wall.?clock.*(limit|exceeded)|exceeded.*time limit' <<<"$tail_big"; then
        timedout=1
    fi

    # ---- last simulation step reached ------------------------------
    local laststep
    laststep=$(tail -c 30000000 "$log" 2>/dev/null \
               | grep -aoE 'STEP [0-9]+' | tail -1 | grep -oE '[0-9]+')
    [[ -n "$laststep" ]] && info "last STEP in log: $laststep"

    # ================================================================
    # Branch: did not finish cleanly
    # ================================================================
    if (( finished == 0 )); then
        if (( crashed == 1 )) && (( timedout == 0 )); then
            failln "did NOT finish -- genuine crash"
            (( ${#bt[@]}   > 0 )) && info "backtrace files: ${#bt[@]}"
            (( ${#core[@]} > 0 )) && info "core files: $(basename "${core[0]}")"
            grep -aE 'SIGSEGV|SIGABRT|MPICH ERROR|PETSC ERROR.*(SEGV|signal)|Assertion failed|corrupted size|job.exception' <<<"$tail_big" \
                | sed 's/^/      | /' | tail -6
            SUMMARY+=("$base : ${RED}CRASH${NC} (last step ${laststep:-?})")
        elif (( timedout == 1 )); then
            failln "did NOT finish -- allocation timeout"
            grep -aE 'type=timeout|TIME LIMIT|wall.?clock' <<<"$tail_big" \
                | sed 's/^/      | /' | tail -4
            SUMMARY+=("$base : ${YEL}TIMEOUT${NC} (last step ${laststep:-?})")
        elif (( crashed == 1 )); then
            failln "did NOT finish -- crash during a timed-out allocation"
            SUMMARY+=("$base : ${RED}CRASH+TIMEOUT${NC} (last step ${laststep:-?})")
        else
            local age="?"
            age=$(( ( $(date +%s) - $(stat -c %Y "$log") ) / 60 ))
            warnln "no finish marker, no crash/timeout signature"
            info "log last modified ${age} min ago -- may still be running, or killed without a trace"
            SUMMARY+=("$base : ${YEL}INCOMPLETE${NC} (last step ${laststep:-?})")
        fi
        PROBLEM=$((PROBLEM+1))
        return
    fi

    # ================================================================
    # Branch: finished cleanly -- full health check
    # ================================================================
    pass "finished cleanly (AMReX finalized)"
    local problems=0

    local used="$dir/warpx_used_inputs"
    if [[ ! -s "$used" ]]; then
        warnln "warpx_used_inputs missing -- cannot verify inputs / file counts"
        SUMMARY+=("$base : ${YEL}CLEAN, unverified${NC}")
        CLEAN=$((CLEAN+1))
        return
    fi

    # ---- verify inputs, ppc included -------------------------------
    local ppc_line ppc_run=""
    for sp_key in $(grep -oE '^[a-zA-Z0-9_]+\.num_particles_per_cell_each_dim' "$used" | sort -u); do
        ppc_line=$(resolved_value "$used" "${sp_key//./\\.}")
        local a b
        a=$(awk '{print $1}' <<<"$ppc_line"); b=$(awk '{print $2}' <<<"$ppc_line")
        if [[ "$a" =~ ^[0-9]+$ && "$b" =~ ^[0-9]+$ ]]; then
            local prod=$((a*b))
            if [[ -z "$ppc_run" ]]; then ppc_run=$prod; fi
            if (( prod != ppc_dir )); then
                failln "ppc mismatch: ${sp_key%%.*} has ${a}x${b}=${prod}, directory says ${ppc_dir}"
                problems=$((problems+1))
            fi
        fi
    done
    if [[ -n "$ppc_run" ]] && (( ppc_run == ppc_dir )); then
        pass "ppc matches: warpx_used_inputs = ${ppc_run}, directory = ${ppc_dir}"
    fi

    # ---- expected step count ---------------------------------------
    local stop_time const_dt max_step
    stop_time=$(resolved_value "$used" 'stop_time')
    const_dt=$(resolved_value "$used" 'warpx\.const_dt')
    max_step=$(resolved_value "$used" 'max_step')
    local nsteps=""
    nsteps=$(awk -v t="$stop_time" -v dt="$const_dt" -v ms="$max_step" 'BEGIN{
        n = -1
        if (t+0 > 0 && dt+0 > 0) n = int(t/dt + 0.5)
        if (ms+0 > 0 && (n < 0 || ms+0 < n)) n = ms+0
        print n }')
    if [[ -z "$nsteps" || "$nsteps" -le 0 ]]; then
        warnln "could not derive the step count from warpx_used_inputs"
    else
        info "expected final step: $nsteps  (stop_time=$stop_time, const_dt=$const_dt)"
        if [[ -n "$laststep" ]]; then
            local short=$(( nsteps - laststep ))
            (( short > nsteps/100 )) && \
                warnln "last logged STEP ($laststep) is well short of $nsteps"
        fi
    fi

    # ---- per-diagnostic plotfile counts ----------------------------
    local diags
    diags=$(resolved_value "$used" 'diagnostics\.diags_names')
    for d in $diags; do
        local fmt prefix interval
        fmt=$(resolved_value "$used" "${d//./\\.}\.format")
        [[ "$fmt" == "checkpoint" ]] && continue
        prefix=$(resolved_value "$used" "${d//./\\.}\.file_prefix")
        [[ -z "$prefix" ]] && prefix="diags/$d"
        interval=$(last_int "$(resolved_value "$used" "${d//./\\.}\.intervals")")
        [[ -z "$interval" || "$interval" -le 0 ]] && continue

        local actual
        actual=$(ls -d "$dir/${prefix}"[0-9]* 2>/dev/null \
                 | grep -vc '\.old\.')

        if (( actual == 0 )); then
            if [[ "$prefix" == diags/* ]]; then
                info "$d: 0 plotfiles under ${prefix}* (diags/ is cleared post-run; skipped)"
            else
                failln "$d: 0 plotfiles under ${prefix}*  (expected output is missing)"
                problems=$((problems+1))
            fi
            continue
        fi

        if [[ -n "$nsteps" && "$nsteps" -gt 0 ]]; then
            local exp
            exp=$(awk -v n="$nsteps" -v iv="$interval" 'BEGIN{
                e = int(n/iv) + 1; if (n % iv != 0) e++; print e }')
            if (( actual >= exp-1 && actual <= exp+1 )); then
                pass "$d: $actual plotfiles (expected ~$exp at interval $interval)"
            else
                failln "$d: $actual plotfiles, expected ~$exp at interval $interval"
                problems=$((problems+1))
            fi
        else
            info "$d: $actual plotfiles at interval $interval (no expected count)"
        fi

        # structural integrity of the plotfiles themselves
        local vsummary
        vsummary=$(cd "$dir" && validate_plotfile_set "${prefix}" "$d")
        if [[ "$vsummary" == intact* ]]; then
            pass "$d: plotfiles structurally $vsummary"
        else
            failln "$d: $vsummary"
            problems=$((problems+1))
        fi
    done

    # ---- scan the log for NaN / errors / non-convergence -----------
    local hits
    hits=$(grep -anE \
        'NaN|[^a-zA-Z]nan[^a-zA-Z]|SIGSEGV|SIGABRT|amrex::Abort|PETSC ERROR|did not converge|failed to converge|DIVERGED|not converged|out of memory' \
        "$log" 2>/dev/null)
    local n_nan n_err n_conv
    n_nan=$(grep -ciE 'NaN|[^a-zA-Z]nan[^a-zA-Z]'                 <<<"$hits")
    n_err=$(grep -ciE 'SIGSEGV|SIGABRT|amrex::Abort|PETSC ERROR|out of memory' <<<"$hits")
    # "Picard solver for <n> particles failed to converge" is an
    # expected, tolerated condition for this case -- excluded here.
    n_conv=$(grep -iE 'did not converge|failed to converge|DIVERGED|not converged' <<<"$hits" \
             | grep -vci 'Picard solver for')
    [[ -z "$hits" ]] && { n_nan=0; n_err=0; n_conv=0; }

    if (( n_nan > 0 )); then
        failln "log contains $n_nan line(s) mentioning NaN/nan"
        grep -iE 'NaN|[^a-zA-Z]nan[^a-zA-Z]' <<<"$hits" | head -3 | sed 's/^/      | /'
        problems=$((problems+1))
    else
        pass "no NaN reported in the log"
    fi
    if (( n_err > 0 )); then
        failln "log contains $n_err fatal-error line(s)"
        grep -iE 'SIGSEGV|SIGABRT|amrex::Abort|PETSC ERROR|out of memory' <<<"$hits" | head -3 | sed 's/^/      | /'
        problems=$((problems+1))
    else
        pass "no fatal-error lines in the log"
    fi
    if (( n_conv > 0 )); then
        warnln "$n_conv solver non-convergence line(s) in the log (tolerated; newton.require_convergence=false)"
    else
        pass "no solver non-convergence reported (Picard non-convergence ignored)"
    fi

    # WarpX prints a boxed warning summary at several points:
    #   **** WARNINGS *********...
    #   * GLOBAL warning list after [ ... ]
    #   * --> [priority] ... <individual warning entries> ...
    #   ***********************...        (closing all-asterisks line)
    # Capture the last box that actually carries entries ("-->"); if no
    # box has any, all are "No recorded warnings".
    local warnbox
    # The box closes on a long all-asterisks line; a lone "*" is an
    # in-box separator and must not be mistaken for the close.
    warnbox=$(awk '
        /\*\*\*\* WARNINGS/      {cap=1; buf=$0 ORS; hw=0; next}
        cap && /^\*\*\*\*\*+$/   {buf=buf $0 ORS; cap=0; last=buf;
                                  if (hw) lastwarn=buf; next}
        cap                      {buf=buf $0 ORS; if ($0 ~ /-->/) hw=1}
        END                      {printf "%s", (lastwarn != "" ? lastwarn : last)}' "$log")
    if [[ -z "$warnbox" ]]; then
        info "no WarpX warning summary block in the log"
    else
        # Each "-->" is one warning entry. Picard-solver non-convergence
        # entries are expected for this case and are not counted.
        local n_arrows n_picard n_other
        n_arrows=$(grep -c -- '-->' <<<"$warnbox")
        n_picard=$(grep -c 'Picard solver for' <<<"$warnbox")
        n_other=$(( n_arrows - n_picard ))
        if (( n_arrows == 0 )); then
            pass "WarpX warning summary: no recorded warnings"
        elif (( n_other <= 0 )); then
            pass "WarpX warning summary: only Picard non-convergence warnings (ignored)"
        else
            warnln "WarpX raised $n_other non-Picard warning(s):"
            # One line per warning topic: "<count> x [topic]". The
            # topic is the second bracketed field on a "-->" line;
            # Picard-solver entries are skipped.
            awk '
                /-->/ {
                    if (h != "" && blk !~ /Picard solver for/) {
                        split(h, p, /[][]/); cnt[p[4]]++ }
                    h = $0; blk = $0 ORS; next
                }
                h != "" { blk = blk $0 ORS }
                END {
                    if (h != "" && blk !~ /Picard solver for/) {
                        split(h, p, /[][]/); cnt[p[4]]++ }
                    for (t in cnt) printf "%d x [%s]\n", cnt[t], t
                }
            ' <<<"$warnbox" | sort -rn | sed 's/^/      | /'
        fi
    fi

    # ---- verdict ---------------------------------------------------
    if (( problems == 0 )); then
        echo "  ${GRN}${BLD}VERDICT: complete and healthy${NC}"
        SUMMARY+=("$base : ${GRN}CLEAN${NC}")
        CLEAN=$((CLEAN+1))
    else
        echo "  ${RED}${BLD}VERDICT: finished but $problems problem(s) found${NC}"
        SUMMARY+=("$base : ${RED}CLEAN-but-$problems-issue(s)${NC}")
        PROBLEM=$((PROBLEM+1))
    fi
}

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
DRY_RUN=0
declare -a positional
while (( $# > 0 )); do
    case "$1" in
        -n|--dry-run) DRY_RUN=1; shift ;;
        -h|--help)
            sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        --)           shift; positional+=("$@"); break ;;
        -*)           echo "unknown option: $1" >&2; exit 2 ;;
        *)            positional+=("$1"); shift ;;
    esac
done
declare -a targets
if (( ${#positional[@]} > 0 )); then
    targets=("${positional[@]}")
else
    mapfile -t targets < <(ls -d .run_*.ppc*.nproc* 2>/dev/null | sort)
fi

if (( ${#targets[@]} == 0 )); then
    echo "No run directories (.run_<case>.<plat>.ppc<N>.nproc<N>) found in $(pwd)" >&2
    exit 1
fi

echo "${BLD}WarpX run check -- $(pwd)${NC}"
echo "${#targets[@]} run director$([[ ${#targets[@]} == 1 ]] && echo y || echo ies) to examine"

for t in "${targets[@]}"; do
    [[ -d "$t" ]] || { echo; echo "${BLD}=== $t ===${NC}"; failln "not a directory"; continue; }
    check_run "$t"
done

# --------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------
echo
echo "${BLD}========================= SUMMARY =========================${NC}"
for s in "${SUMMARY[@]}"; do echo "  $s"; done
echo "${BLD}-----------------------------------------------------------${NC}"
echo "  $TOTAL examined   ${GRN}$CLEAN healthy${NC}   ${RED}$PROBLEM with problems${NC}"

(( PROBLEM > 0 )) && exit 1
exit 0
