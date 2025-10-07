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
 * Compares the value written in L1 memory with the value written in L1 memory of the tile_0 of the same synched row or column.
 */
int check_values(uint8_t val, uint32_t hartid){
    if(val == (uint8_t) (GET_X_ID(hartid) + MESH_X_TILES)){
        uint32_t id_0 = GET_ID(0, val - MESH_X_TILES);
        uint8_t val_0 = *(volatile uint8_t*)get_l1_base(id_0);
        if(val != val_0){
            printf("Column Error detected: val=%d val_0=%d (id_0=%d)\n", val, val_0, id_0);
            return 1;
        }
        return 0;
    }
    else if(val == (uint8_t) GET_Y_ID(hartid)){
        uint32_t id_0 = GET_ID(val, 0);
        uint8_t val_0 = *(volatile uint8_t*)get_l1_base(id_0);
        if(val != val_0){
            printf("Row Error detected: val=%d val_0=%d (id_0=%d)\n", val, val_0, id_0);
            return 1;
        }
        return 0;
    }
    else
        printf("Error in check_values: val is invalid (val=%d)\n", val);
    return 1;
}

/**
 * This test checks the correctness of the Fractal Sync mechanism for synchronizing mesh tiles row and column wise.
 * Each tiles writes an identical value in its L1 memory, depending on its coordinates.
 * After the write, each tile reads the value written by the others in the same row (or column) to check their correctness.
 * Each tile has a delay on the write proportional to its ID, making the synchronization mechanism mandatory. 
 */
int main(void){
    /** 
     * 0. Get the tile's hartid, coordinates, and define its L1 base. Also initialize the controllers for fsync.
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
    uint8_t x_id = (uint8_t) GET_X_ID(hartid);
    uint8_t y_id = (uint8_t) GET_Y_ID(hartid);
    uint8_t flag = 0;

    
    /** 
     * 1. Test row synchronization.
     * 1a - Write the row value in L1 after a delay (depending on ID).
     * 1b - Wait for the other tiles in the row to write.
     * 1c - Check the correctness of all the other tile values.
     * 1d - Wait for the read before proceding to the next test.
     */
    //wait_nop(100 * hartid);
    mmio8(l1_tile_base) = y_id;
    fsync_sync_row(&fsync_ctrl);
    flag = check_values(y_id, hartid);
    fsync_sync_row(&fsync_ctrl);
    if(!flag){
        printf("No errors detected in row synch!\n");
    }
    else{
        printf("Errors detected in row synch!\n");
        magia_return(hartid, 1);
        return 1;
    }
    

    /** 
     * 2. Test column synchronization.
     * 2a - Write the column value in L1 after a delay (depending on ID).
     * 2b - Wait for the other tiles in the column to write.
     * 2c - Check the correctness of all the other tile values.
     * 2d - Wait for the read before proceding to the next test.
     */
    //wait_nop(100 * hartid);
    uint8_t val = x_id + (uint8_t) MESH_X_TILES;
    mmio8(l1_tile_base) = val;
    fsync_sync_col(&fsync_ctrl);
    flag = check_values(val, hartid);
    fsync_sync_col(&fsync_ctrl);
    if(!flag){
        printf("No errors detected in column synch!\n");
    }
    else{
        printf("Errors detected in column synch!\n");
        return 1;
    }

    return 0;
}