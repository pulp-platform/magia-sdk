// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>
//
// SPATZ_VLD_FENCE: a workaround for two Spatz RTL bugs with vector loads.
// This header is force-included in every Spatz task (see spatz_helpers.cmake).
//
// HOW TO USE IT
//   Put SPATZ_VLD_FENCE right after:
//     1. every strided or indexed vector load (vlse*, vluxei*, ...);
//     2. any vector load whose destination register is read next by a
//        reduction (vfred*, vfwred*).
//   Example:
//     asm volatile("vlse32.v v0, (%0), %1" SPATZ_VLD_FENCE ...);
//
// HOW IT WORKS
//   The fence is a dummy scalar store: "sw zero, -4(sp)".
//   Snitch does not issue a scalar store while a vector memory instruction
//   is still in flight (it stalls on acc_mem_cnt_q != 0, see snitch.sv).
//   So the store, and everything after it, waits until the load has
//   written all its elements. Nothing can chain on the load.
//   The store writes to the word just below sp. Nothing lives there: the
//   RISC-V ABI has no red zone, so the compiler never uses that memory.
//
// BUG 1: strided/indexed loads (half-written words)
//   These loads fill each word of the vector register file in several
//   partial writes. The scoreboard, however, lets a chained instruction
//   read a word as soon as the load has written to it once. The chained
//   instruction can then read a half-written word.
//
// BUG 2: reductions chained on a load (dropped operation)
//   A reduction chained on a load gets its input words as the load writes
//   them, so there can be one-cycle gaps. A gap in the middle of the
//   reduction leaves the FPU stalled with a partial result at its output.
//   The next operation (triggered by a one-cycle pulse of
//   reduction_operand_ready_q, see spatz_vfu.sv) finds the FPU input full
//   and is silently dropped.
//   With the fence, the reduction starts on a fully written register: it
//   gets one word per cycle, with no gaps.
//
// ENABLING / DISABLING
//   On by default. Turn it off (the macro becomes empty) with
//     make build spatz_vld_fence=0      (CMake: -DSPATZ_VLD_FENCE=0)

#ifndef SPATZ_VLD_FENCE_H
#define SPATZ_VLD_FENCE_H

#if defined(SPATZ_VLD_FENCE_EN) && SPATZ_VLD_FENCE_EN
#define SPATZ_VLD_FENCE "\n\tsw zero, -4(sp)"
#else
#define SPATZ_VLD_FENCE ""
#endif

#endif // SPATZ_VLD_FENCE_H
