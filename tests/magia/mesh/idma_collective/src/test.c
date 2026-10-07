// Copyright 2025-2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Carlotta Chiarini, Fondazione Chips-IT

#include <stdint.h>
#include "test.h"
#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"
#include "collective.h"

#define WAIT_MODE WFE



/**
 * This test verifies the iDMA collective API (idma_collective_1d).
 * The source tile multicasts an L1 buffer to the whole mesh, to its row and to its column.
 * In every phase, each tile checks that it received the data only if it belongs to the
 * destination group, and that its buffer was left untouched otherwise.
 */
int main(void)
{
    /**
     * 0. Get the mesh-tile's hartid, mesh-tile coordinates and define its L1 base,
     * also initialize the controllers for the idma and fsync.
     */
    uint32_t hartid = get_hartid();

    idma_config_t idma_cfg      = {.hartid = hartid};
    idma_controller_t idma_ctrl = {
        .base = NULL,
        .cfg  = &idma_cfg,
        .api  = &idma_api,
    };

    idma_init(&idma_ctrl);

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
    eu_idma_init(&eu_ctrl, 0);
#endif

    /**
     * 1. Buffer addresses in this tile's L1.
     */
    uint32_t src_addr = get_l1_base(hartid) + SRC_OFFSET;
    uint32_t dst_addr = get_l1_base(hartid) + DST_OFFSET;

    uint32_t n_errors = 0;

    /**
     * 2. The source tile moves the input data from L2 to its L1.
     */
    if (hartid == SOURCE_HART_ID) {
        printf("iDMA moving data from L2 to L1...\n");
        idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)x_inp, src_addr, BUF_SIZE);
#if STALLING == 0
        eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
#endif
    }

    const uint32_t comm_groups[NUM_COMM_GROUPS] = {MESH, ROW, COLUMN};

    for (uint32_t g = 0; g < NUM_COMM_GROUPS; g++) {

        uint32_t group = comm_groups[g];

        /**
         * Whether this tile is expected to receive the data in the current group.
         */
        int receiver = (group == MESH) ||
                    (group == ROW    && GET_Y_ID(hartid) == GET_Y_ID(SOURCE_HART_ID)) ||
                    (group == COLUMN && GET_X_ID(hartid) == GET_X_ID(SOURCE_HART_ID));


        /**
         * 3. Clear the destination buffer on every tile.
         */
        for (uint32_t i = 0; i < N_ELEMS; i++)
            mmio16(dst_addr + 2 * i) = 0;
        
        
        fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
        eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

        /**
         * 4. The source tile sends its L1 buffer over the NoC.
         */
        if (hartid == SOURCE_HART_ID) {
            idma_collective_1d(&idma_ctrl,
                                dst_addr,
                                src_addr,
                                BUF_SIZE,
                                gen_collective_mask(&coll_ctrl, group),
                                MULTICAST);
#if STALLING == 0
            eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
#endif
        }

        /**
        * 5. Wait that all the tiles have finished before checking data
        */
        fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
        eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

        /**
         * 6. Check results. If the tile is not a receiver we expect all zeros.
         */
        if (hartid != SOURCE_HART_ID) {
            uint32_t group_errors = 0;
            for (uint32_t i = 0; i < N_ELEMS; i++) {
                uint16_t expected = receiver ? x_inp[i] : 0;
                uint16_t detected = mmio16(dst_addr + 2 * i);
                if (detected != expected) {
                    printf("COMM_GROUP %d ERROR: dst[%d] = 0x%x, expected 0x%x\n",
                           group,
                           i,
                           detected,
                           expected);
                    group_errors++;
                }
            }
            n_errors += group_errors;
        }
    }
    printf("Finished test with %d errors\n", n_errors);

    return 0;
}
