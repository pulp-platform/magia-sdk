// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <francesco.conti@chips.it>
//
// Load fence for Spatz tasks, force-included in every Spatz task compile (spatz_helpers.cmake).
// Every strided or indexed vector load asm string in the Spatz kernels is followed by
// SPATZ_VLD_FENCE.
//
// RTL workaround: a strided/indexed load writes each VRF word in more than
// one partial write, but the scoreboard lets a chained VFU instruction read a word as soon as the
// load wrote to it in the previous cycle, so it can read the word half-written. Snitch does not
// issue a scalar store while an offloaded vector memory instruction is outstanding
// (acc_mem_cnt_q != 0 stalls it, see snitch.sv), so a dummy store right after the load holds
// back everything that follows until the load has written all its elements: no chaining on it.
// The store writes zero to the word below sp, which is not allocated (the RISC-V ABI has no red
// zone, so the compiler keeps nothing there).
//
// On by default; disabled (empty) with `make build spatz_vld_fence=0` (CMake -DSPATZ_VLD_FENCE=0).

#ifndef SPATZ_VLD_FENCE_H
#define SPATZ_VLD_FENCE_H

#if defined(SPATZ_VLD_FENCE_EN) && SPATZ_VLD_FENCE_EN
#define SPATZ_VLD_FENCE "\n\tsw zero, -4(sp)"
#else
#define SPATZ_VLD_FENCE ""
#endif

#endif // SPATZ_VLD_FENCE_H
