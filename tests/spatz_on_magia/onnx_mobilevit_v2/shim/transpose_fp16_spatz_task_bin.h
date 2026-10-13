// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/*
 * Shim so kernels/spatz_fp16/transpose/src can be compiled unmodified into this test.
 *
 * Each kernel driver includes its own <op>_fp16_spatz_task_bin.h, but one executable
 * gets exactly one Spatz binary, and here that binary holds twelve tasks (the eleven
 * kernels plus mul's broadcast entry point). add_spatz_task generates a single
 * onnx_mobilevit_v2_task_bin.h for it, and extract_task_symbols.sh puts a
 * <SYMBOL>_TASK macro in there for every global task symbol it finds. So redirecting
 * the per-kernel include at that one header is all that is needed; it has an include
 * guard, so all eleven shims including it is fine.
 *
 * shim/ must come before the kernel include dirs on the include path.
 */

#include "onnx_mobilevit_v2_task_bin.h"
