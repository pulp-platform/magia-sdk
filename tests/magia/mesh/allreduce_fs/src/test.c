// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Victor Isachi <victor.isachi@unibo.it>
// Alberto Dequino <alberto.dequino@unibo.it>

#include <stdint.h>
#include "tile.h"
#include "fsync.h"

#define WIDTH (MESH_X_TILES)
#define MAX(x, y) (((int)(x) > (int)(y)) ? (x) : (y))
#define MIN(x, y) (((int)(x) < (int)(y)) ? (x) : (y))
#define GTH(x, y, t) (((int)(x) >= (int)(t)) ? (x) : (y))
#define LTH(x, y, t) (((int)(x) <= (int)(t)) ? (x) : (y))

/**
 * This test implements the AllReduce kernel using FractalSync for synchronization. 
 */
int main(void){
    // sentinel_start();   // Total execution

    /** 
     * 0. Get the mesh-tile's hartid, mesh-tile coordinates and define its L1 base, 
     * also initialize the controllers for fsync.
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
    uint32_t l1_tile_base = get_l1_base(hartid);

    // Wait for all tiles to be awake and ready to start the kernel
    stnl_snc_s();
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    stnl_snc_f();

    sentinel_start();   // Execution time after wakeup
    stnl_ts_s();        // Initial timeslot

    /**
     * The AllReduce takes number of timeslots proportional to the mesh dimension.
     */
    uint8_t timeslots = 2*(WIDTH-1);

    /**
     * 1. Initialize reduction data.
     */
    mmio32(l1_tile_base)   = 0;         // Result
    mmio32(l1_tile_base+4) = hartid+1;  // Datum 0
    mmio32(l1_tile_base+8) = 0;         // Datum 1

    // printf("Result: %d, Datum 0: %d, Datum 1: %d\n", mmio32(l1_tile_base), mmio32(l1_tile_base+4), mmio32(l1_tile_base+8));

    uint32_t mesh_dst_x_id = x_id%2 ? MAX(x_id-2, 0) : MIN(x_id+2, WIDTH-1);
    uint32_t mesh_dst_y_id = y_id%2 ? MAX(y_id-2, 0) : MIN(y_id+2, WIDTH-1);
    uint32_t mesh_src_x_id = x_id%2 ? LTH(x_id+2, WIDTH-2, WIDTH-1) : GTH(x_id-2, 1, 0);
    uint32_t mesh_src_y_id = y_id%2 ? LTH(y_id+2, WIDTH-2, WIDTH-1) : GTH(y_id-2, 1, 0);
    uint32_t horizontal_dst_id = GET_ID(y_id, mesh_dst_x_id);
    uint32_t vertical_dst_id   = GET_ID(mesh_dst_y_id, x_id);
    uint32_t horizontal_src_id = GET_ID(y_id, mesh_src_x_id);
    uint32_t vertical_src_id   = GET_ID(mesh_src_y_id, x_id);
    // printf("Mesh DST X ID: %d, Mesh DST Y ID: %d, Horizontal DST ID: %d, Vertical DST ID: %d\n", mesh_dst_x_id, mesh_dst_y_id, horizontal_dst_id, vertical_dst_id);
    // printf("Mesh SRC X ID: %d, Mesh SRC Y ID: %d, Horizontal SRC ID: %d, Vertical SRC ID: %d\n", mesh_src_x_id, mesh_src_y_id, horizontal_src_id, vertical_src_id);

    stnl_ts_f();    // Initial timeslot

    /**
     * 2. Cycle over the timeslots.
     * For each timeslot, the mesh-tile will:
     * a - Load the input to reduce for the next timeslot
     * b - reduce the input of the current timeslot
     */
    for(int i = 0; i < timeslots; i++){
        if (i < timeslots/2){
            stnl_snc_s();
            fsync_sync_row(&fsync_ctrl);
            stnl_snc_f();

            stnl_ts_s();
            if(i % 2){
                mmio32(l1_tile_base+4) = mmio32(get_l1_base(horizontal_src_id)+8);
                mmio32(l1_tile_base)  += mmio32(l1_tile_base+8);
                if (i == timeslots/2 - 1){
                    mmio32(l1_tile_base)  += mmio32(l1_tile_base+4);
                    mmio32(l1_tile_base+4) =  mmio32(l1_tile_base);
                    mmio32(l1_tile_base+8) =  mmio32(l1_tile_base);
                }
            }
            else{
                mmio32(l1_tile_base+8) = mmio32(get_l1_base(horizontal_src_id)+4);
                mmio32(l1_tile_base)  += mmio32(l1_tile_base+4);
                if (i == timeslots/2 - 1){
                    mmio32(l1_tile_base)  += mmio32(l1_tile_base+8);
                    mmio32(l1_tile_base+4) =  mmio32(l1_tile_base);
                    mmio32(l1_tile_base+8) =  mmio32(l1_tile_base);
                }
            }
            stnl_ts_f();
        } else {
            stnl_snc_s();
            fsync_sync_col(&fsync_ctrl);
            stnl_snc_f();

            stnl_ts_s();
            if(i % 2){
                mmio32(l1_tile_base+4) = mmio32(get_l1_base(vertical_src_id)+8);
                if (i != timeslots/2){
                    mmio32(l1_tile_base) += mmio32(l1_tile_base+8);
                }
                if (i == timeslots - 1){
                    mmio32(l1_tile_base) += mmio32(l1_tile_base+4);
                }
            }
            else{
                mmio32(l1_tile_base+8) = mmio32(get_l1_base(vertical_src_id)+4);
                if (i != timeslots/2){
                    mmio32(l1_tile_base) += mmio32(l1_tile_base+4);
                }
                if (i == timeslots - 1){
                    mmio32(l1_tile_base) += mmio32(l1_tile_base+8);
                }
            }
            stnl_ts_f();
        }
        // printf("Result: %d, Current Data: %d, Next Data: %d\n", mmio32(l1_tile_base), mmio32(l1_tile_base+4), mmio32(l1_tile_base+8));
    }

    sentinel_end(); // Execution time after wakeup
    asm volatile("nop" ::); // Needed to detect same instruction consecutively

    // sentinel_end(); // Total execution

    stnl_snc_r();
    
    // if (get_hartid() == 0){
    //   stnl_r();
    //   stnl_ts_r();
    // }

    /**
     * 3. Check results
     */
    unsigned int expected_result = (unsigned int)((WIDTH*WIDTH)*(WIDTH*WIDTH+1)/2);
    if (mmio32(l1_tile_base) == expected_result){
        return 0;
    } else {
         printf("[ERROR] - Expected: %d, Detected: %d\n", expected_result, mmio32(l1_tile_base));
        return 1;
    } 
}