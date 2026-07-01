#!/usr/bin/env bash
# Generate GEMM chain golden data (test.h) via the PyTorch reference model.
#
# Usage:
#   bash scripts/gemm/gemm-gen.sh
#   DIM_A=16 DIM_B=8 SEED=7 bash scripts/gemm/gemm-gen.sh
#
# Environment variables (all optional, defaults match the previous Makefile):
#   DIM_A..DIM_F   GEMM chain dimensions (default: 4 8 16 32 64 128)
#   SEED           PRNG seed             (default: 42)

set -euo pipefail

DIM_A="${DIM_A:-4}"
DIM_B="${DIM_B:-8}"
DIM_C="${DIM_C:-16}"
DIM_D="${DIM_D:-32}"
DIM_E="${DIM_E:-64}"
DIM_F="${DIM_F:-128}"
SEED="${SEED:-42}"

python3 tests/magia/mesh/gemm_comm/gen_golden.py \
    --dim-a "$DIM_A" --dim-b "$DIM_B" --dim-c "$DIM_C" \
    --dim-d "$DIM_D" --dim-e "$DIM_E" --dim-f "$DIM_F" \
    --seed "$SEED"
