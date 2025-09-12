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
 * Compares the value written in L1 memory with the value written in L1 memory of the tile_0 of the diagonal.
 */
int check_values(uint8_t val){
    uint32_t id_0 = GET_ID(0, 0);
    uint8_t val_0 = *(volatile uint8_t*)get_l1_base(id_0);
    if(val == val_0)
        return 0;
    else{
        printf("Error in diagonal sync - val=%d, val_0=%d", val, val_0);
        return 1;
    }
}

/**
 * This test checks the correctness of the Fractal Sync mechanism for synchronizing the mesh tiles on the diagonal.
 * Each tiles writes an identical value in its L1 memory, depending on its coordinates.
 * After the write, each tile reads the value written by the others in the diagonal to check their correctness.
 * Each tile can have a delay on the write proportional to its ID, making the synchronization mechanism mandatory. 
 */
int main(void){
    /** 
     * 0. Get the tile's hartid, coordinates, and define its L1 base. Also initialize the controller for fsync.
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
     * 1. Test diagonal synchronization.
     * 1a - Write 123 in L1 after a delay (depending on ID).
     * 1b - Wait for the other tiles in the diagonal to write.
     * 1c - Check the correctness of all the other tile values.
     * 1d - Wait for the read before returning.
     */
    if(x_id == y_id){
        //wait_nop(100 * hartid);
        mmio8(l1_tile_base) = (uint8_t) 123;
        fsync_sync_diag(&fsync_ctrl);
        flag = check_values((uint8_t) 123);
        fsync_sync_diag(&fsync_ctrl);
        if(!flag){
            printf("No errors detected in diagonal!\n");
            magia_return(hartid, 0);
            return 0;
        }
        else{
            printf("Errors detected in diagonal!\n");
            magia_return(hartid, 1);
            return 1;
        }        
    }

    magia_return(hartid, 0);
    return 0; 
}