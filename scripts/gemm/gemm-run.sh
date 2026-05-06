#!/usr/bin/env bash
# Run the GEMM chain test on the configured platform.
#
# Usage:
#   bash scripts/gemm/gemm-run.sh
#   TILES=2 GEMM_PLATFORM=rtl bash scripts/gemm/gemm-run.sh
#
# Environment variables (all optional):
#   TILES          mesh dimension N      (default: 8)
#   GEMM_PLATFORM  gvsoc|rtl             (default: gvsoc)
#   MAKE           make binary           (default: make)

set -euo pipefail

TILES="${TILES:-8}"
GEMM_PLATFORM="${GEMM_PLATFORM:-gvsoc}"
MAKE="${MAKE:-make}"

"$MAKE" run test=test_gemm platform="$GEMM_PLATFORM" tiles="$TILES"
