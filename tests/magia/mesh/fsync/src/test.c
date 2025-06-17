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
 * Writes a value to the L1 memory address to be used to verify correct synchronization.
 * Delays the write by an increasing number of nops depending on the core id.
 * The value written is calculated based on the tile's X and Y position in the mesh and the current synchronization level.
 * The value is a "mock-id" of the synchronized area the tile is part of, plus an offset depending of the current sync level.
 * Refer to MAGIA pubblication and documentation for a detailed explanation of the different synchronization level geometries. 
 */
int write_delayed(uint8_t lvl, uint32_t id, uint32_t x, uint32_t y, uint32_t addr){
    uint8_t val = (uint8_t)((x >> ((lvl + 2) / 2)) + ((y >> ((lvl + 1) / 2))*(MESH_X_TILES >> ((lvl + 2) / 2))));
    if(lvl){
        for(uint8_t i = lvl; i > 0; i--){
            val += (NUM_HARTS >> i);
        }
    }
    wait_nop(100 * id);
    mmio8(addr) = val;
    return 0;
}

/**
 * Compares the value written in L1 memory with the value written in L1 memory of the tile_0 of the same synched mesh area.
 * To locate which tile is the tile_0, the value stored in L1 (the "mock-id") is used to calculate the X and Y of tile_0 (and its ID).
 */
int check_values(uint8_t lvl, uint32_t id, uint32_t x, uint32_t y, uint32_t addr){
    uint8_t val = *(volatile uint8_t*)(addr);
    uint8_t val2 = val;
    if(lvl){
        for(uint8_t i = lvl; i > 0; i--){
            val2 = val2 - (NUM_HARTS >> i);
        }
    }
    uint8_t id_0 = (((val2 % (MESH_X_TILES >> ((lvl + 2) / 2))) << ((lvl + 2) / 2)) + (((val2 / (MESH_X_TILES >> ((lvl + 2) / 2))) << ((lvl + 1) / 2)) * MESH_X_TILES));
    uint8_t val_0 = *(volatile uint8_t*)(L1_BASE + (id_0 * L1_TILE_OFFSET));
    uint8_t flag = 0;
    if(val_0 != val){
        printf("Error detected at sync level %d - val is: %d but val_0 (id_0:%d) is %d", lvl, val, id_0, val_0);
        flag = 1;
    }
    //else
        //printf("DEBUG lvl %d - val_0:%d val:%d", lvl, val_0, val);
    return flag;
}

/**
 * This test checks the correctness of the Fractal Sync mechanism for synchronizing mesh tiles at different tree levels.
 * Each tiles of the same synchronization level writes an identical value in its L1 memory.
 * After the write, each tile reads the value written by the others in the same synchronization level to check their correctness.
 * Each tile has a delay on the write proportional to its ID, making the synchronization mechanism mandatory. 
 */
int main(void){
    /** 
     * 0. Get the tile's hartid, mesh coordinates and define its L1 base, also initialize the controllers for fsync.
     */
    uint32_t hartid = get_hartid();

    fsync_config_t fsync_cfg = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {
        .base = NULL,
        .cfg = &fsync_cfg,
        .api = &fsync_api,
    };

    fsync_init(&fsync_ctrl);

    uint32_t y_id = GET_Y_ID(hartid);
    uint32_t x_id = GET_X_ID(hartid);
    uint32_t l1_tile_base = L1_BASE + hartid * L1_TILE_OFFSET;
    uint8_t flag = 0;

    /**
     * 1. Cycle over the synchronization levels.
     * Increasing the synchronization level increases the mesh area that has to be synchronized.
     */
    for(uint8_t i = 0; i < MAX_SYNC_LVL; i++){
        //printf("Entering synchronization level %d", i);
        /**
        * 1_a. Synchronize before write (waits the read of the previous cycle).
        */
        fsync_sync(&fsync_ctrl, (uint32_t) i);

        /**
        * 1_b. Write value.
        */
        write_delayed(i, hartid, x_id, y_id, l1_tile_base);

        /**
        * 1_c. Synchronize on the current level.
        */
        fsync_sync(&fsync_ctrl, (uint32_t) i);

        /**
        * 1_d. Check if the other tiles have written the correct value.
        */
        if(check_values(i, hartid, x_id, y_id, l1_tile_base))
            flag=1;    
    }

    if(!flag){
        printf("No errors detected for all synchronization levels! (MAX LEVEL: %d)\n", (MAX_SYNC_LVL-1));
    }

    magia_return(hartid, PASS_EXIT_CODE);
    
    return 0;
}