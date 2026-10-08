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
 * This test verifies the synchronization over the FlooNoC narrow channel through the collective
 * driver (narrow_sync_mesh/row/column). Every tile of the selected group (ROW, COLUMN, MESH) sends
 * a 32-bit word to the destination tile, where the words are reduced.
 */
int main(void)
{
    /**
     * 0. Get the mesh-tile's hartid and initialize the controllers for the fsync and
     * the collectives.
     */
    uint32_t hartid = get_hartid();

    uint32_t receiver_x = GET_X_ID(DESTINATION_HART_ID);
    uint32_t receiver_y = GET_Y_ID(DESTINATION_HART_ID);

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
     * 1. Word address in the destination tile's L1.
     */
    uint32_t floo_sync_addr = get_l1_base(DESTINATION_HART_ID) + MEM_OFFSET;
    uint32_t sender         = 0;

    /**
     * 2. MESH Narrow synch.
     */
    sender = 1;

    if (hartid == DESTINATION_HART_ID)
        mmio32(floo_sync_addr) = 0;

    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

    if (sender)
        floo_narrow_sync_mesh(FLOO_SYNC_WORD, floo_sync_addr);
    if (hartid == DESTINATION_HART_ID) {
        while (mmio32(floo_sync_addr) != FLOO_SYNC_WORD)
            ;
    }

    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

    /**
     * 3. ROW Narrow synch.
     */
    if (receiver_y == GET_Y_ID(hartid))
        sender = 1;
    else
        sender = 0;

    if (hartid == DESTINATION_HART_ID)
        mmio32(floo_sync_addr) = 0;

    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

    if (sender)
        floo_narrow_sync_row(FLOO_SYNC_WORD, floo_sync_addr);
    if (hartid == DESTINATION_HART_ID) {
        while (mmio32(floo_sync_addr) != FLOO_SYNC_WORD)
            ;
    }

    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

    /**
     * 4. COLUMN Narrow synch.
     */
    if (receiver_x == GET_X_ID(hartid))
        sender = 1;
    else
        sender = 0;

    if (hartid == DESTINATION_HART_ID)
        mmio32(floo_sync_addr) = 0;

    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

    if (sender)
        floo_narrow_sync_column(FLOO_SYNC_WORD, floo_sync_addr);
    if (hartid == DESTINATION_HART_ID) {
        while (mmio32(floo_sync_addr) != FLOO_SYNC_WORD)
            ;
    }

    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

    return 0;
}
