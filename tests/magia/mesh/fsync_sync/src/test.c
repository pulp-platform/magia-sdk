// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Alberto Dequino <alberto.dequino@unibo.it>

#include <stdint.h>

#include "test.h"
#include "tile.h"
#include "fsync.h"

/**
 * Compares the value written in L1 memory with the value written in L1 memory of the other tiles listed.
 */
int check_values(uint32_t *ids, uint8_t n_tiles){
    uint32_t hartid = get_hartid();
    uint8_t val = *(volatile uint8_t*)(get_l1_base(hartid));
    uint8_t val2;
    for(uint8_t i = 0; i < n_tiles; i++){
        if(hartid == ids[i])
            continue;
        val2 = *(volatile uint8_t*)(get_l1_base(ids[i]));
        if(val != val2){
            printf("Error detected: val=%d val2=%d (id of other tile:%d)", val, val2, ids[i]);
            return 1;
        }
    }
    return 0;
}

/**
 * This test checks the correctness of the Fractal Sync mechanism for synchronizing an arbitrary vector of mesh tiles.
 * Each tiles listed writes an identical value in its L1 memory.
 * After the write, each tile reads the value written by the others to check their correctness.
 * Each tile has a possible delay on the write proportional to its ID, making the synchronization mechanism mandatory. 
 */
int main(void){
    /** 
     * 0. Get the tile's hartid and define its L1 base, also initialize the controllers for fsync and sync tree direction.
     */
    uint32_t hartid = get_hartid();

    fsync_config_t fsync_cfg = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {
        .base = NULL,
        .cfg = &fsync_cfg,
        .api = &fsync_api,
    };

    fsync_init(&fsync_ctrl);

    uint32_t l1_tile_base = get_l1_base(hartid);
    uint8_t groupid;
    uint8_t flag = 0;
    uint8_t dir = 0;

    /**
     * 1. Select which tiles to be synchronized.
     */
    uint8_t N_TILES = 2;
    uint32_t ids[] = {27, 28};

    /**
     * 2. Cycle over the ids. If the current tile is part of the ids, test the synchronization.
     */
    for(uint8_t i = 0; i < N_TILES; i++){
        if(hartid == ids[i]){
            /**
             * 2a. Write value in memory
             */
            //wait_nop(1000 * hartid);
            mmio8(get_l1_base(hartid)) = (uint8_t) 123;

            /**
             * 2b. Synchronize with the other tiles
             */
            if(fsync_sync(&fsync_ctrl, ids, N_TILES, 0, 0))
                printf("Error in synchronization.");

            /**
             * 2c. Check that all tiles wrote the same value
             */
            if(!check_values(ids, N_TILES))
                printf("No errors detected for arbitrary sync!");
            
            break;
        }
    }

    magia_return(hartid, PASS_EXIT_CODE);
    
    return 0;
}