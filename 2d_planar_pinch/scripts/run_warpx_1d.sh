#!/bin/bash
#
# Sibling 1D-Cartesian planar-pinch driver.
#
# Thin wrapper over run_warpx.sh that defaults the case to planar_pinch_1d.
# run_warpx.sh auto-derives DIM="1d" from the _1d case-name suffix, so the 1D
# build of WarpX (warpx.1d) is invoked with the 1D inputs file.
#
# Same option surface as run_warpx.sh; anything passed on the command line is
# forwarded verbatim. If the user does not supply -c / --case, this script
# injects "-c planar_pinch_1d" before delegating.
#
# Usage:
#   ./run_warpx_1d.sh                   # submit the default 1D case
#   ./run_warpx_1d.sh -m interactive    # interactive instead of batch
#   ./run_warpx_1d.sh my_constants.Nppc=1600    # override ppc on CLI
#   ./run_warpx_1d.sh -h                # help from run_warpx.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_WARPX="$SCRIPT_DIR/run_warpx.sh"

if [[ ! -x "$RUN_WARPX" ]]; then
    echo "ERROR: run_warpx.sh not found or not executable: $RUN_WARPX" >&2
    exit 1
fi

# Detect whether the user already supplied a case; if not, inject the default.
has_case=false
for arg in "$@"; do
    case "$arg" in
        -c|--case|--case=*) has_case=true; break ;;
    esac
done

if $has_case; then
    exec "$RUN_WARPX" "$@"
else
    exec "$RUN_WARPX" -c planar_pinch_1d "$@"
fi
