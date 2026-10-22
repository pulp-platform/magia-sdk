// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <francesco.conti@chips.it>
//
// FPU fence for Spatz tasks, force-included in every Spatz task compile (spatz_helpers.cmake).
// SPATZ_FPU_FENCE prefixes the asm string of a vector FP instruction whose FPnew op group has a
// lower latency than the FP instruction issued to the VFU before it (MAGIA: ADDMUL 3, CONV 2,
// NONCOMP 1), e.g. asm volatile(SPATZ_FPU_FENCE "vfmax.vf v0, v0, %0" ...).
//
// RTL workaround: FPnew completes out of order across op groups, so the
// shorter instruction's first result can overtake the last result of the one before it; if it
// depends on it, the scoreboard's lockstep chaining holds it at the FPU output with the older
// result queued behind it, and the VFU deadlocks. vmv.x.s is an integer VFU operation (vl = 1,
// result to tp): the VFU switches from its FPU to its IPU state only once the FPUs are no longer
// busy, so everything after it starts with the FPU pipelines drained. The result goes to tp, which
// the Spatz tasks never read (no TLS; the compiler does not allocate it), not to x0: GVSoC's
// vmv.x.s writes its destination without the x0 guard.
//
// Never put it right before a reduction: the VFU can accept the reduction
// while the vmv.x.s result is still in the IPU result register, and drops that result without a
// response, so tp stays reserved forever. A reduction does not need it anyway: the VFU waits for
// the FPU to drain before starting one.
//
// On by default; disabled (empty) with `make build spatz_fpu_fence=0` (CMake -DSPATZ_FPU_FENCE=0).
// scripts/spatz_fpu_hazards.py lists the instructions that need it, from the task ELFs.

#ifndef SPATZ_FPU_FENCE_H
#define SPATZ_FPU_FENCE_H

#if defined(SPATZ_FPU_FENCE_EN) && SPATZ_FPU_FENCE_EN
#define SPATZ_FPU_FENCE "vmv.x.s tp, v0\n\t"
#else
#define SPATZ_FPU_FENCE ""
#endif

#endif // SPATZ_FPU_FENCE_H
