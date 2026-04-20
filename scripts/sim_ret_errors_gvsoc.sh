#!/usr/bin/env bash

# Copyright 2025 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE.SHL for details.
# SPDX-License-Identifier: SHL-0.51

# Author: Alessandro Nadalini <alessandro.nadalini3@unibo.it>
#         Alberto Dequino <alberto.dequino@unibo.it>

LOGFILE="$1"

if [[ ! -f "$LOGFILE" ]]; then
  echo "Error: File not found!"
  exit 2
fi

# Extract the number from the last occurrence of "Errors: N"
errors=$(grep -oP 'Error 1' "$LOGFILE" | tail -n 1)

if [[ -z "$errors" ]]; then
  exit 0
else
  exit 1
fi
