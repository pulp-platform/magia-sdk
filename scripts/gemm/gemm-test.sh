#!/usr/bin/env bash
# Full local GEMM chain pipeline: generate golden data, build, (build GVSoC), run.
#
# Usage:
#   bash scripts/gemm/gemm-test.sh
#   TILES=2 bash scripts/gemm/gemm-test.sh
#   TILES=4 GEMM_PLATFORM=rtl bash scripts/gemm/gemm-test.sh
#
# Environment variables (forwarded to the underlying scripts):
#   TILES, COMPILER, EVAL, GEMM_PLATFORM, DIM_A..DIM_F, SEED, MAKE

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TILES="${TILES:-8}"
GEMM_PLATFORM="${GEMM_PLATFORM:-gvsoc}"
MAKE="${MAKE:-make}"

export TILES GEMM_PLATFORM MAKE

bash "$SCRIPT_DIR/gemm-gen.sh"
bash "$SCRIPT_DIR/gemm-build.sh"

if [[ "$GEMM_PLATFORM" == "gvsoc" ]]; then
    "$MAKE" gvsoc tiles="$TILES"
fi

bash "$SCRIPT_DIR/gemm-run.sh"
