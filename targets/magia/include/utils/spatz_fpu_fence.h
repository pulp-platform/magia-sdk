// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>
//
// SPATZ_FPU_FENCE: a workaround for a Spatz RTL deadlock.
// This header is force-included in every Spatz task (see spatz_helpers.cmake).
//
// WHAT IT IS FOR
//   Inside the Spatz vector unit (VFU), the FPU (FPnew) has op groups with
//   different latencies. On MAGIA:
//     ADDMUL   (vfadd, vfmul, vfmacc, ...)  3 cycles
//     CONV     (conversions)                2 cycles
//     NONCOMP  (vfmax, vfmin, sign-inj...)  1 cycle
//   Results from different groups can come back out of order. A fast
//   instruction may finish before the slow instruction issued just before it.
//   If the fast one depends on the slow one, the chaining logic stalls it
//   at the FPU output, the slow result gets stuck behind it, and the VFU
//   hangs forever.
//
// HOW TO USE IT
//   Put SPATZ_FPU_FENCE in front of any vector FP instruction that is faster
//   than the FP instruction before it, e.g.
//     asm volatile(SPATZ_FPU_FENCE "vfmax.vf v0, v0, %0" ...);
//   scripts/spatz_fpu_hazards.py scans the task ELFs and lists the
//   instructions that need it.
//
// HOW IT WORKS
//   The fence is a single "vmv.x.s tp, v0" instruction. It is an integer
//   VFU op, and the VFU only switches from FP to integer mode once all FPU
//   pipelines are empty. So everything after the fence starts with a
//   drained FPU.
//   The result is written to tp, which Spatz tasks never use (no TLS, and
//   the compiler does not allocate it). We cannot use x0 because GVSoC's
//   vmv.x.s writes its destination even when it is x0.
//
// DO NOT put it right before a reduction (vfredsum etc.)
//   The VFU may accept the reduction while the fence's result is still
//   pending, and then drop that result. tp then stays marked as "busy"
//   forever and the core hangs.
//   Reductions do not need the fence anyway: the VFU already waits for the
//   FPU to drain before starting one. (A reduction that follows a vector
//   load needs SPATZ_VLD_FENCE instead; see spatz_vld_fence.h.)
//
// ENABLING / DISABLING
//   On by default. Turn it off (the macro becomes empty) with
//     make build spatz_fpu_fence=0      (CMake: -DSPATZ_FPU_FENCE=0)

#ifndef SPATZ_FPU_FENCE_H
#define SPATZ_FPU_FENCE_H

#if defined(SPATZ_FPU_FENCE_EN) && SPATZ_FPU_FENCE_EN
#define SPATZ_FPU_FENCE "vmv.x.s tp, v0\n\t"
#else
#define SPATZ_FPU_FENCE ""
#endif

#endif // SPATZ_FPU_FENCE_H
