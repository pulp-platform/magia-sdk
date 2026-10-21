// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "test.h"

#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"

#define WAIT_MODE WFE

static inline uint32_t pattern(uint32_t addr) { return 0xA5000000 ^ addr; }

/**
 * Minimal check of a 256 MiB L2: every tile writes and reads back a few words
 * spread over the whole window with its core (narrow NoC channel), plus one
 * 1 KiB L1 -> L2 -> L1 round trip with the iDMA (wide NoC channel) near the
 * top, cross-checked with core loads. Only a few pages of the L2 are touched.
 */
int main(void)
{
    uint32_t hartid = get_hartid();

    idma_init();
    fsync_init();

    uint32_t l1_tile_base = get_l1_base(hartid);

#if STALLING == 0
    eu_init();
    eu_clear_events(0xFFFFFFFF);
    eu_fsync_init(0);
    eu_idma_init(0);
#endif

    uint32_t errors = 0;

    /**
     * 1. Narrow probes: write all, then read all back.
     */
    for (int i = 0; i < N_NARROW_PROBES; i++) {
        uint32_t addr = narrow_base[i] + hartid * narrow_stride[i];
        *(volatile uint32_t *)addr = pattern(addr);
    }
    if (hartid == 0)
        *(volatile uint32_t *)(L2_PROBE_BASE + L2_PROBE_SIZE - 4) =
            pattern(L2_PROBE_BASE + L2_PROBE_SIZE - 4);

    for (int i = 0; i < N_NARROW_PROBES; i++) {
        uint32_t addr = narrow_base[i] + hartid * narrow_stride[i];
        uint32_t got  = *(volatile uint32_t *)addr;
        if (got != pattern(addr)) {
            printf("[%d] narrow %#x: got %#x, expected %#x\n", hartid, addr, got, pattern(addr));
            errors++;
        }
    }
    if (hartid == 0) {
        uint32_t addr = L2_PROBE_BASE + L2_PROBE_SIZE - 4;
        uint32_t got  = *(volatile uint32_t *)addr;
        if (got != pattern(addr)) {
            printf("[%d] narrow %#x: got %#x, expected %#x\n", hartid, addr, got, pattern(addr));
            errors++;
        }
    }

    /**
     * 2. Wide probe: L1 -> L2 with the iDMA, clear L1, L2 -> L1, compare.
     */
    uint32_t l2_addr      = DMA_BASE + hartid * 4096;
    volatile uint32_t *l1  = (volatile uint32_t *)l1_tile_base;
    for (int i = 0; i < DMA_LEN / 4; i++)
        l1[i] = pattern(l2_addr + 4 * i);

    idma_memcpy_1d(1, l2_addr, l1_tile_base, DMA_LEN);
#if STALLING == 0
    eu_idma_wait_o2a(WAIT_MODE);
#endif

    for (int i = 0; i < DMA_LEN / 4; i++)
        l1[i] = 0;

    idma_memcpy_1d(0, l2_addr, l1_tile_base, DMA_LEN);
#if STALLING == 0
    eu_idma_wait_a2o(WAIT_MODE);
#endif

    uint32_t dma_errors = 0;
    for (int i = 0; i < DMA_LEN / 4; i++)
        if (l1[i] != pattern(l2_addr + 4 * i))
            dma_errors++;
    // Same data seen by the core through the narrow channel
    for (int i = 0; i < DMA_LEN / 4; i += 64) {
        uint32_t addr = l2_addr + 4 * i;
        if (*(volatile uint32_t *)addr != pattern(addr))
            dma_errors++;
    }
    if (dma_errors)
        printf("[%d] idma %#x: %d errors\n", hartid, l2_addr, dma_errors);
    errors += dma_errors;

    /**
     * 3. Wait that all the tiles have finished
     */
    fsync_sync_level(MAX_SYNC_LVL - 1, 0);
#if STALLING == 0
    eu_fsync_wait(WAIT_MODE);
#endif

    if (errors)
        printf("[%d] L2 256 MiB: FAIL, %d errors\n", hartid, errors);
    else
        printf("[%d] L2 256 MiB: PASS\n", hartid);

    return errors;
}
