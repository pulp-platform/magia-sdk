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
 * This test verifies the multicast over the FlooNoC narrow channel through the collective driver (narrow_mcast_mesh/row/column).
 * The source tile multicasts a single 32-bit word to the selected group (ROW, COLUMN, MESH)
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
     * 1. Word address in this tile's L1.
     */
    uint32_t floo_mcast_addr = get_l1_base(hartid) + MEM_OFFSET;
    const uint32_t comm_groups[NUM_COMM_GROUPS] = {MESH, ROW, COLUMN};
    uint32_t n_errors = 0;

    for (uint32_t g = 0; g < NUM_COMM_GROUPS; g++) {

        uint32_t group = comm_groups[g];

        /**
         * Whether this tile is expected to receive the word in the current group.
         */
        int receiver = (group == MESH) ||
                    (group == ROW    && GET_Y_ID(hartid) == GET_Y_ID(SOURCE_HART_ID)) ||
                    (group == COLUMN && GET_X_ID(hartid) == GET_X_ID(SOURCE_HART_ID));

        /**
         * 2. Clear the destination word on every tile, then wait for all the tiles
         * before starting the multicast.
         */
        mmio32(floo_mcast_addr) = 0;

        fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
        eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

        if (hartid == SOURCE_HART_ID) {
            /**
             * 3. The source tile multicasts the word to the selected communicator group.
             */
            switch (group) {
            case MESH:
                narrow_mcast_mesh(&coll_ctrl, BROADCAST_WORD, floo_mcast_addr);
                break;
            case ROW:
                narrow_mcast_row(&coll_ctrl, BROADCAST_WORD, floo_mcast_addr);
                break;
            case COLUMN:
                narrow_mcast_column(&coll_ctrl, BROADCAST_WORD, floo_mcast_addr);
                break;
            }
        } else if (receiver) {
            /**
             * 4. The destination tiles wait for the reception of the word.
             */
            while (mmio32(floo_mcast_addr) != BROADCAST_WORD);
        }

        /**
         * 5. Wait that all the receivers have got the word before checking the others.
         */
        fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
        eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

        /**
         * 6. If the tile is not a receiver we expect its word to be all zeros.
         */
        if (hartid != SOURCE_HART_ID && !receiver) {
            uint32_t detected = mmio32(floo_mcast_addr);
            if (detected != 0) {
                printf("COMM_GROUP %d ERROR: detected 0x%x, expected 0x0\n",
                       group,
                       detected);
                n_errors++;
            }
        }
    }

    printf("Finished test with %d errors\n", n_errors);

    return n_errors;
}
