#!/usr/bin/env bash

# Copyright 2025 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE.SHL for details.
# SPDX-License-Identifier: SHL-0.51

# Author: Alessandro Nadalini <alessandro.nadalini3@unibo.it>
#         Alberto Dequino <alberto.dequino@unibo.it>

testlist="test_helloworld test_fsync_levels test_fsync_rc test_fsync_diag test_mm_is test_mm_ws test_mm_os test_idma_2d test_idma_1d test_cemm_global test_mm_is_2 test_mm_os_2 test_mm_ws_2 test_fsync_lr test_gemv"
nlist="1 2 4 8 16"
for n in ${nlist}; do
  for test in ${testlist}; do
    LOGFILE=scripts/regression_output_${n}_tiles/${test}.txt
    # Extract the number from the last occurrence of "Errors: N"
    errors=$(grep -oP 'Error 1' "$LOGFILE" | tail -n 1)

    if [[ -z "$errors" ]]; then
      echo "No error found in test $LOGFILE"
    else
      exit "ERROR FOUND IN TEST $LOGFILE"
    fi
  done
done
