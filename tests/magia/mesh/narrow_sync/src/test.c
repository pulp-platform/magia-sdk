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
#include "collective.h"

#define WAIT_MODE WFE

/**
 * This test verifies the synchronization over the FlooNoC narrow channel through the collective driver (narrow_sync_mesh/row/column).
 * Every tile of the selected group (ROW, COLUMN, MESH) sends a 32-bit word to the destination tile, where the words are reduced.
 */
int main(void)
{
    /**
     * 0. Get the mesh-tile's hartid and initialize the controllers for the fsync and
     * the collectives.
     */
    uint32_t hartid = get_hartid();

    fsync_config_t fsync_cfg      = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {
        .base = NULL,
        .cfg  = &fsync_cfg,
        .api  = &fsync_api,
    };
    fsync_init(&fsync_ctrl);

    floo_collective_config_t coll_cfg = {.hartid = hartid};
    floo_collective_t coll_ctrl       = {
        .base = NULL,
        .cfg  = &coll_cfg,
        .api  = &floonoc_collective_api,
    };

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

    const uint32_t comm_groups[NUM_COMM_GROUPS] = {MESH, ROW, COLUMN};

    for (uint32_t g = 0; g < NUM_COMM_GROUPS; g++) {

        uint32_t group = comm_groups[g];

        /**
         * Whether this tile is expected to send the word in the current group.
         */
        int sender = (group == MESH) ||
                    (group == ROW    && GET_Y_ID(hartid) == GET_Y_ID(DESTINATION_HART_ID)) ||
                    (group == COLUMN && GET_X_ID(hartid) == GET_X_ID(DESTINATION_HART_ID));

        /**
         * 2. The destination tile clears its word, then wait for all the tiles
         * before starting the synchronization.
         */
        if (hartid == DESTINATION_HART_ID)
            mmio32(floo_sync_addr) = 0;

        fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
        eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

        if (sender) {
            /**
             * 3. The sender tiles send the word to be reduced at destination.
             */
            switch (group) {
            case MESH:
                narrow_sync_mesh(&coll_ctrl, FLOO_SYNC_WORD, floo_sync_addr);
                break;
            case ROW:
                narrow_sync_row(&coll_ctrl, FLOO_SYNC_WORD, floo_sync_addr);
                break;
            case COLUMN:
                narrow_sync_column(&coll_ctrl, FLOO_SYNC_WORD, floo_sync_addr);
                break;
            }
        }

        if (hartid == DESTINATION_HART_ID) {
            /**
             * 4. The destination tile waits for the reception of the word.
             */
            while (mmio32(floo_sync_addr) != FLOO_SYNC_WORD);
        }

        /**
         * 5. Wait that the destination has got the word before the next group.
         */
        fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
        eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif
    }

    return 0;
}
