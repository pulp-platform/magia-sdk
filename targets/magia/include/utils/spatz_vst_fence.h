// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <francesco.conti@chips.it>
//
// Store fence for Spatz tasks, force-included in every Spatz task compile (spatz_helpers.cmake).
// Every vector store asm string in the Spatz kernels is followed by SPATZ_VST_FENCE.
//
// RTL workaround: a Snitch instruction offloaded to Spatz (e.g. a scalar mul) while a vector
// store is still in flight can deadlock the VFU against the VLSU. Snitch does not issue a scalar
// load while an offloaded vector store is outstanding (acc_mem_str_cnt_q != 0 stalls it, see
// snitch.sv), so a dummy load right after the store holds back everything that follows until
// the store has finished. `fence` does not do this: it only waits for the scalar LSU.
//
// Enabled with `make build spatz_vst_fence=1` (CMake -DSPATZ_VST_FENCE=1); empty otherwise.

#ifndef SPATZ_VST_FENCE_H
#define SPATZ_VST_FENCE_H

#if defined(SPATZ_VST_FENCE_EN) && SPATZ_VST_FENCE_EN
#define SPATZ_VST_FENCE "\n\tlw zero, 0(sp)"
#else
#define SPATZ_VST_FENCE ""
#endif

#endif // SPATZ_VST_FENCE_H
