#!/usr/bin/env bash
# GEMM chain CI orchestration.
#
# Runs the GEMM chain test on an 8x8 mesh, driving the individual Makefile
# startpoints (gvsoc, gemm-gen, gemm-build, gemm-run) and emitting a final
# PASS/FAIL banner.
#
# Usage:
#   bash scripts/gemm-ci.sh
#   GEMM_PLATFORM=rtl bash scripts/gemm-ci.sh
#
# Environment variables (all optional, defaults match the Makefile):
#   GEMM_PLATFORM  gvsoc|rtl   (default: gvsoc)
#   COMPILER       GCC_PULP|GCC_MULTILIB (default: GCC_PULP)
#   EVAL           0|1         (default: 0)
#   DIM_A..DIM_F               (default: 4 8 16 32 64 128)
#   SEED                       (default: 42)
#   MAKE                       (default: make)

set -euo pipefail

GEMM_PLATFORM="${GEMM_PLATFORM:-gvsoc}"
COMPILER="${COMPILER:-GCC_PULP}"
EVAL="${EVAL:-0}"
DIM_A="${DIM_A:-4}"
DIM_B="${DIM_B:-8}"
DIM_C="${DIM_C:-16}"
DIM_D="${DIM_D:-32}"
DIM_E="${DIM_E:-64}"
DIM_F="${DIM_F:-128}"
SEED="${SEED:-42}"
MAKE="${MAKE:-make}"

if [[ "$GEMM_PLATFORM" == "gvsoc" ]]; then
    echo "====== Building GVSoC for tiles=8 ======"
    "$MAKE" gvsoc tiles=8 || { echo "GVSOC BUILD FAILED"; exit 1; }
    echo ""
fi

echo "====== Generating golden data ======"
"$MAKE" gemm-gen \
    dim_a="$DIM_A" dim_b="$DIM_B" dim_c="$DIM_C" \
    dim_d="$DIM_D" dim_e="$DIM_E" dim_f="$DIM_F" \
    seed="$SEED" || { echo "GEMM-GEN FAILED"; exit 1; }
echo ""

echo "====== Building test (tiles=8) ======"
"$MAKE" gemm-build tiles=8 compiler="$COMPILER" eval="$EVAL" \
    || { echo "GEMM-BUILD FAILED"; exit 1; }
echo ""

echo "====== Running GEMM chain test ======"
if "$MAKE" gemm-run tiles=8 gemm_platform="$GEMM_PLATFORM"; then
    echo ""
    echo "====== PASS ======"
else
    echo ""
    echo "====== FAIL ======"
    exit 1
fi
