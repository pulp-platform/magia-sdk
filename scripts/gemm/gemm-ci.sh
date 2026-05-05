#!/usr/bin/env bash
# GEMM chain CI orchestration.
#
# Runs the GEMM chain test on an 8x8 mesh, driving the individual gemm-*
# scripts and emitting a final PASS/FAIL banner.
#
# Usage:
#   bash scripts/gemm/gemm-ci.sh
#   GEMM_PLATFORM=rtl bash scripts/gemm/gemm-ci.sh
#
# Environment variables (all optional, defaults match the previous Makefile):
#   GEMM_PLATFORM  gvsoc|rtl   (default: gvsoc)
#   COMPILER       GCC_PULP|GCC_MULTILIB (default: GCC_PULP)
#   EVAL           0|1         (default: 0)
#   DIM_A..DIM_F               (default: 4 8 16 32 64 128)
#   SEED                       (default: 42)
#   MAKE                       (default: make)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

export GEMM_PLATFORM COMPILER EVAL DIM_A DIM_B DIM_C DIM_D DIM_E DIM_F SEED MAKE
export TILES=8

if [[ "$GEMM_PLATFORM" == "gvsoc" ]]; then
    echo "====== Building GVSoC for tiles=8 ======"
    "$MAKE" gvsoc tiles=8 || { echo "GVSOC BUILD FAILED"; exit 1; }
    echo ""
fi

echo "====== Generating golden data ======"
bash "$SCRIPT_DIR/gemm-gen.sh" || { echo "GEMM-GEN FAILED"; exit 1; }
echo ""

echo "====== Building test (tiles=8) ======"
bash "$SCRIPT_DIR/gemm-build.sh" || { echo "GEMM-BUILD FAILED"; exit 1; }
echo ""

echo "====== Running GEMM chain test ======"
if bash "$SCRIPT_DIR/gemm-run.sh"; then
    echo ""
    echo "====== PASS ======"
else
    echo ""
    echo "====== FAIL ======"
    exit 1
fi
