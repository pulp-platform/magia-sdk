// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef _TEST_L2_BIG_INCLUDE_GUARD_
#define _TEST_L2_BIG_INCLUDE_GUARD_

// Probed L2 window: [0xC000_0000, 0xC000_0000 + L2_PROBE_SIZE), i.e. 256 MiB.
// The default gvsoc L2 (0x0CFF_0000) ends at 0xCCFF_0000, so the probes above
// it only succeed on a gvsoc built with l2_size=0x10000000 (or larger).
#define L2_PROBE_BASE (0xC0000000)
#define L2_PROBE_SIZE (0x10000000)

// Narrow (core LD/ST) probes: one word per tile at base + hartid * stride
#define N_NARROW_PROBES (4)
static const uint32_t narrow_base[N_NARROW_PROBES] = {
    0xC0000000, // bottom of L2, below the code (valid on the default L2 too)
    0xC8000000, // middle of the 256 MiB
    0xCD000000, // just past the end of the default L2
    0xCFFFF000, // last page of the 256 MiB
};
static const uint32_t narrow_stride[N_NARROW_PROBES] = {64, 4096, 64, 64};

// Wide (iDMA) probe: L1 -> L2 -> L1 round trip of DMA_LEN bytes per tile, at
// DMA_BASE + hartid * 4 KiB so that every transfer stays within one AXI page
#define DMA_BASE (0xCF000000)
#define DMA_LEN  (1024)

#endif
