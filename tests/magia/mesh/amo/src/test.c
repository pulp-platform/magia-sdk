// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Alberto Dequino <alberto.dequino@unibo.it>

#include <stdint.h>
#include "test.h"

#include "tile.h"
#include "fsync.h"
#include "utils/performance_utils.h"

#define INITIAL_VALUE 1234
#define N_ITERS 1000


int main(void){
    /** 
     * 0. Get the mesh-tile's hartid, mesh-tile coordinates and define its L1 base, 
     * also initialize the controllers for the peripherals.
     */
    uint32_t hartid = get_hartid();

    fsync_config_t          fsync_cfg = {.hartid = hartid};
    fsync_controller_t      fsync_ctrl = {
        .base = NULL,
        .cfg = &fsync_cfg,
        .api = &fsync_api,
    };

    fsync_init(&fsync_ctrl);

    uint32_t y_id           = GET_Y_ID(hartid);
    uint32_t x_id           = GET_X_ID(hartid);
    uint32_t l1_tile_base   = get_l1_base(hartid);

    /** 
     * 1. Initialize the counter in L1
     */
    mmio32(l1_tile_base) = (uint32_t) INITIAL_VALUE;

    // Synch all the tiles
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);

    /**
     * 2. Main loop.
     * Iterate N_ITERS times over the L1 counters of all the mesh tiles, increasing them by 1.
     * No errors and contention should occur by using AMO.
     */
    sentinel_start();
    for(uint32_t i = 0; i < N_ITERS; i++){
        for(uint8_t j = 0; j < NUM_HARTS; j++){
            amo_add_immediate(get_l1_base(j), 1);
        }
    }
    sentinel_end();

    // Synch all the tiles
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);

    /**
     * 3. Check if the counter has the correct value.
     */
    if(*(volatile uint32_t*)(l1_tile_base) != (INITIAL_VALUE + N_ITERS * NUM_HARTS)){
        printf("Error: expected %d but got %d\n", (INITIAL_VALUE + N_ITERS * NUM_HARTS), *(volatile uint32_t*)(l1_tile_base));
        magia_return(hartid, 1);
        return 1;  
    }
    else{
        printf("Correct value!\n");
    }

    magia_return(hartid, 0);
    return 0;
}