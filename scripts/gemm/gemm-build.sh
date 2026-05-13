#!/usr/bin/env bash
# Clean and build the GEMM chain test for an NxN tile mesh.
#
# Usage:
#   bash scripts/gemm/gemm-build.sh
#   TILES=2 COMPILER=GCC_PULP EVAL=1 bash scripts/gemm/gemm-build.sh
#
# Environment variables (all optional):
#   TILES      mesh dimension N (default: 8)
#   COMPILER   GCC_PULP|GCC_MULTILIB (default: GCC_PULP)
#   EVAL       0|1 (default: 0)
#   MAKE       make binary to invoke (default: make)

set -euo pipefail

TILES="${TILES:-8}"
COMPILER="${COMPILER:-GCC_PULP}"
EVAL="${EVAL:-0}"
MAKE="${MAKE:-make}"

"$MAKE" clean
"$MAKE" build target_platform=magia_v2 tiles="$TILES" compiler="$COMPILER" eval="$EVAL"
