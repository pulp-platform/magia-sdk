#!/usr/bin/env bash
# Run the GEMM chain test on the configured platform.
#
# Usage:
#   bash scripts/gemm/gemm-run.sh
#   TILES=2 GEMM_PLATFORM=rtl bash scripts/gemm/gemm-run.sh
#
# Environment variables (all optional):
#   TILES          mesh dimension N         (default: 8)
#   GEMM_PLATFORM  gvsoc|rtl                (default: gvsoc)
#   TEST_NAME      GEMM variant binary name (default: test_gemm)
#                  e.g. test_gemm, test_gemm_interlaced,
#                       test_gemm_l1_naive, test_gemm_l1_interlaced,
#                       test_gemm_fifo
#   MAKE           make binary              (default: make)

set -euo pipefail

TILES="${TILES:-8}"
GEMM_PLATFORM="${GEMM_PLATFORM:-gvsoc}"
TEST_NAME="${TEST_NAME:-test_gemm}"
MAKE="${MAKE:-make}"

"$MAKE" run test="$TEST_NAME" platform="$GEMM_PLATFORM" tiles="$TILES"
