// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>
//
// SPATZ_VST_FENCE: a workaround for a Spatz RTL deadlock with vector stores.
// This header is force-included in every Spatz task (see spatz_helpers.cmake).
//
// HOW TO USE IT
//   Put SPATZ_VST_FENCE right after every vector store. Example:
//     asm volatile("vse32.v v0, (%0)" SPATZ_VST_FENCE ...);
//
// THE BUG
//   If Snitch offloads an instruction to Spatz (e.g. a scalar mul) while a
//   vector store is still in flight, the VFU and the VLSU can deadlock.
//
// HOW IT WORKS
//   The fence is a dummy scalar load: "lw zero, 0(sp)".
//   Snitch does not issue a scalar load while a vector store is still in
//   flight (it stalls on acc_mem_str_cnt_q != 0, see snitch.sv).
//   So the load, and everything after it, waits until the store has
//   finished.
//   A plain `fence` instruction does not work here: it only waits for the
//   scalar LSU, not for Spatz.
//
// ENABLING / DISABLING
//   On by default. Turn it off (the macro becomes empty) with
//     make build spatz_vst_fence=0      (CMake: -DSPATZ_VST_FENCE=0)

#ifndef SPATZ_VST_FENCE_H
#define SPATZ_VST_FENCE_H

#if defined(SPATZ_VST_FENCE_EN) && SPATZ_VST_FENCE_EN
#define SPATZ_VST_FENCE "\n\tlw zero, 0(sp)"
#else
#define SPATZ_VST_FENCE ""
#endif

#endif // SPATZ_VST_FENCE_H
