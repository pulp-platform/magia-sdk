// Copyright 2025-2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Carlotta Chiarini, Fondazione Chips-IT

#include <stdint.h>
#include "test.h"
#include "tile.h"
#include "fsync.h"
#include "eventunit.h"

#define WAIT_MODE WFE

/**
 * This test verifies the multicast over the FlooNoC narrow channel through the collective driver
 * (narrow_mcast_mesh/row/column). The source tile multicasts a single 32-bit word to the selected
 * group (ROW, COLUMN, MESH)
 */
int main(void)
{
    /**
     * 0. Get the mesh-tile's hartid and initialize the controllers for the fsync and
     * the collectives.
     */
    uint32_t hartid = get_hartid();

    uint32_t sender_x = GET_X_ID(SOURCE_HART_ID);
    uint32_t sender_y = GET_Y_ID(SOURCE_HART_ID);

    fsync_config_t fsync_cfg      = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {
        .base = NULL,
        .cfg  = &fsync_cfg,
        .api  = &fsync_api,
    };

    fsync_init(&fsync_ctrl);

#if STALLING == 0
    eu_config_t eu_cfg      = {.hartid = hartid};
    eu_controller_t eu_ctrl = {
        .base = NULL,
        .cfg  = &eu_cfg,
        .api  = &eu_api,
    };
    eu_init(&eu_ctrl);
    eu_clear_events(0xFFFFFFFF);
    eu_fsync_init(&eu_ctrl, 0);
#endif

    /**
     * 1. Word address in this tile's L1.
     */
    uint32_t floo_mcast_addr = get_l1_base(hartid) + MEM_OFFSET;
    uint32_t n_errors        = 0;
    uint32_t receiver        = 0;

    /**
     * 2. MESH narrow test
     */
    mmio32(floo_mcast_addr) = 0;
    receiver                = 1;

    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

    if (hartid == SOURCE_HART_ID)
        floo_narrow_mcast_mesh(BROADCAST_WORD, floo_mcast_addr);
    else if (receiver) {
        while (mmio32(floo_mcast_addr) != BROADCAST_WORD)
            ;
    }

    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

    if (hartid != SOURCE_HART_ID && !receiver) {
        uint32_t detected = mmio32(floo_mcast_addr);
        if (detected != 0) {
            printf("NARROW MESH ERROR: detected 0x%x, expected 0x0\n", detected);
            n_errors++;
        }
    }

    /**
     * 3. ROW narrow test
     */
    mmio32(floo_mcast_addr) = 0;

    if (GET_Y_ID(hartid) == sender_y)
        receiver = 1;
    else
        receiver = 0;

    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

    if (hartid == SOURCE_HART_ID)
        floo_narrow_mcast_row(BROADCAST_WORD, floo_mcast_addr);
    else if (receiver) {
        while (mmio32(floo_mcast_addr) != BROADCAST_WORD)
            ;
    }

    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

    if (hartid != SOURCE_HART_ID && !receiver) {
        uint32_t detected = mmio32(floo_mcast_addr);
        if (detected != 0) {
            printf("NARROW ROW ERROR: detected 0x%x, expected 0x0\n", detected);
            n_errors++;
        }
    }

    /**
     * 4. COLUMN narrow test
     */
    mmio32(floo_mcast_addr) = 0;

    if (GET_X_ID(hartid) == sender_x)
        receiver = 1;
    else
        receiver = 0;

    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

    if (hartid == SOURCE_HART_ID)
        floo_narrow_mcast_column(BROADCAST_WORD, floo_mcast_addr);
    else if (receiver) {
        while (mmio32(floo_mcast_addr) != BROADCAST_WORD)
            ;
    }

    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

    if (hartid != SOURCE_HART_ID && !receiver) {
        uint32_t detected = mmio32(floo_mcast_addr);
        if (detected != 0) {
            printf("NARROW COLUMN ERROR: detected 0x%x, expected 0x0\n", detected);
            n_errors++;
        }
    }

    printf("Finished test with %d errors\n", n_errors);

    return n_errors;
}
