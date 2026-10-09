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

#define WAIT_MODE WFE

inline void clear_buffer(uint32_t buffy)
{
    for (uint32_t i = 0; i < N_ELEMS; i++)
        mmio16(buffy + 2 * i) = 0;
}

/**
 * This test verifies the iDMA collective API (idma_collective_1d).
 * The source tile (ID defined in test's header file) multicasts an L1 buffer to the whole mesh, to
 * its row and to its column. In every phase, each tile checks that it received the data only if it
 * belongs to the destination group, and that its buffer was left untouched otherwise.
 */
int main(void)
{
    /**
     * 0. Get the mesh-tile's hartid, mesh-tile coordinates and define its L1 base,
     * also initialize the controllers for the idma and fsync.
     */
    uint32_t hartid = get_hartid();

    uint32_t sender_x = GET_X_ID(SOURCE_HART_ID);
    uint32_t sender_y = GET_Y_ID(SOURCE_HART_ID);

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

    // Reciever flag: if set to 1, it means that this tile has to recieve something in the specific
    // test stage
    int reciever_flag = 0;

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

    /**
     * 3. Test ROW wide broadcast
     */
    clear_buffer(dst_addr);

    if (GET_Y_ID(hartid) == sender_y)
        reciever_flag = 1;
    else
        reciever_flag = 0;

    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

    if (hartid == SOURCE_HART_ID) {
        idma_collective_1d(&idma_ctrl, dst_addr, src_addr, BUF_SIZE, ROW_MASK, MULTICAST);
#if STALLING == 0
        eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
#endif
    }

    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

    if (hartid != SOURCE_HART_ID) {
        uint32_t group_errors = 0;
        for (uint32_t i = 0; i < N_ELEMS; i++) {
            uint16_t expected = reciever_flag ? x_inp[i] : 0;
            uint16_t detected = mmio16(dst_addr + 2 * i);
            if (detected != expected) {
                printf(
                    "ROW MULTICAST ERROR: dst[%d] = 0x%x, expected 0x%x\n", i, detected, expected);
                group_errors++;
            }
        }
        n_errors += group_errors;
    }

    /**
     * 4. Test COLUMN wide broadcast
     */
    clear_buffer(dst_addr);

    if (GET_X_ID(hartid) == sender_x)
        reciever_flag = 1;
    else
        reciever_flag = 0;

    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

    if (hartid == SOURCE_HART_ID) {
        idma_collective_1d(&idma_ctrl, dst_addr, src_addr, BUF_SIZE, COLUMN_MASK, MULTICAST);
#if STALLING == 0
        eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
#endif
    }

    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

    if (hartid != SOURCE_HART_ID) {
        uint32_t group_errors = 0;
        for (uint32_t i = 0; i < N_ELEMS; i++) {
            uint16_t expected = reciever_flag ? x_inp[i] : 0;
            uint16_t detected = mmio16(dst_addr + 2 * i);
            if (detected != expected) {
                printf("COLUMN MULTICAST ERROR: dst[%d] = 0x%x, expected 0x%x\n",
                       i,
                       detected,
                       expected);
                group_errors++;
            }
        }
        n_errors += group_errors;
    }

    /**
     * 5. Test MESH wide broadcast
     */
    clear_buffer(dst_addr);

    reciever_flag = 1;

    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

    if (hartid == SOURCE_HART_ID) {
        idma_collective_1d(&idma_ctrl, dst_addr, src_addr, BUF_SIZE, MESH_MASK, MULTICAST);
#if STALLING == 0
        eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
#endif
    }

    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

    if (hartid != SOURCE_HART_ID) {
        uint32_t group_errors = 0;
        for (uint32_t i = 0; i < N_ELEMS; i++) {
            uint16_t expected = reciever_flag ? x_inp[i] : 0;
            uint16_t detected = mmio16(dst_addr + 2 * i);
            if (detected != expected) {
                printf("COLUMN MULTICAST ERROR: dst[%d] = 0x%x, expected 0x%x\n",
                       i,
                       detected,
                       expected);
                group_errors++;
            }
        }
        n_errors += group_errors;
    }

    printf("Finished test with %d errors\n", n_errors);

    return n_errors;
}
