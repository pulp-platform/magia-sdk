#!/usr/bin/env bash

# Copyright 2025 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE.SHL for details.
# SPDX-License-Identifier: SHL-0.51

# Author: Alessandro Nadalini <alessandro.nadalini3@unibo.it>
#         Alberto Dequino <alberto.dequino@unibo.it>

LOGFILE="$1"

if [[ ! -f "$LOGFILE" ]]; then
  testlist="test_helloworld test_fsync_levels test_fsync_rc test_fsync_diag test_mm_is test_mm_ws test_mm_os test_idma_2d test_idma_1d test_cemm_global test_mm_is_2 test_mm_os_2 test_mm_ws_2 test_fsync_lr test_gemv"
  testlist_spatz="fft_fs hello_spatz onnx_add onnx_averagepool onnx_batchnorm onnx_ceil onnx_clip onnx_div onnx_exp onnx_gemm onnx_globalaveragepool onnx_globalmaxpool onnx_groupnorm onnx_hardsigmoid onnx_hardswish onnx_instancenorm onnx_layernorm onnx_maxpool onnx_relu onnx_sigmoid onnx_softmax onnx_sub onnx_swish"
  nlist="1 2 4 8 16"
  for n in ${nlist}; do
    for test in ${testlist}; do
      LOGFILE=scripts/regression_output_${n}_tiles/${test}.txt
      # Extract the number from the last occurrence of "Errors: N"
      errors=$(grep -oP 'Error 1' "$LOGFILE" | tail -n 1)

      if [[ -z "$errors" ]]; then
        echo "No error found in test $LOGFILE"
      else
        echo "ERROR FOUND IN TEST $LOGFILE"
      fi
    done
  done
  for n in ${nlist}; do
    for test in ${testlist_spatz}; do
      LOGFILE=scripts/regression_output_spatz_${n}_tiles/${test}.txt
      # Extract the number from the last occurrence of "Errors: N"
      errors=$(grep -oP 'Test FAILED' "$LOGFILE")

      if [[ -z "$errors" ]]; then
        echo "No error found in test $LOGFILE"
      else
        echo "ERROR FOUND IN TEST $LOGFILE"
      fi
    done
  done
else
  # Extract the number from the last occurrence of "Errors: N"
  errors=$(grep -oP 'Error 1' "$LOGFILE" | tail -n 1)

  if [[ -z "$errors" ]]; then
    exit 0
  else
    exit 1
  fi
fi
