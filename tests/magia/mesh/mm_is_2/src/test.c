// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Alberto Dequino <alberto.dequino@unibo.it>

#include <stdint.h>
#include "test.h"

#include "tile.h"
#include "fsync.h"
#include "idma.h"
#include "redmule.h"
#include "eventunit.h"

#define N_ITERATIONS 1
#define WAIT_MODE WFE

/**
 * This test aims to verify the functionality of MAGIA as a systolic array for matrix multiplications,
 * following the input-static mechanism. 
 */
int main(void){
    /** 
     * 0. Get the mesh-tile's hartid, mesh-tile coordinates and define its L1 base, 
     * also initialize the controllers for the idma, redmule and fsync
     */
    uint32_t hartid = get_hartid();

    idma_config_t idma_cfg = {.hartid = hartid};
    idma_controller_t idma_ctrl = {
        .base = NULL,
        .cfg = &idma_cfg,
        .api = &idma_api,
    };

    redmule_config_t redmule_cfg = {.hartid = hartid};
    redmule_controller_t redmule_ctrl = {
        .base = NULL,
        .cfg = &redmule_cfg,
        .api = &redmule_api,
    };

    fsync_config_t fsync_cfg = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {
        .base = NULL,
        .cfg = &fsync_cfg,
        .api = &fsync_api,
    };

    fsync_init(&fsync_ctrl);
    idma_init(&idma_ctrl);
    redmule_init(&redmule_ctrl);

    #if STALLING == 0
    eu_config_t eu_cfg = {.hartid = hartid};
    eu_controller_t eu_ctrl = {
        .base = NULL,
        .cfg = &eu_cfg,
        .api = &eu_api,
    };
    eu_init(&eu_ctrl);
    eu_redmule_init(&eu_ctrl, 0);
    eu_idma_init(&eu_ctrl, 0);
    eu_fsync_init(&eu_ctrl, 0);
    #endif

    uint32_t y_id = GET_Y_ID(hartid);
    uint32_t x_id = GET_X_ID(hartid);
    uint32_t l1_tile_base = get_l1_base(hartid);

    /**
     * 1. Calculate the static input data-tile dimensions.
     */
    uint32_t tile_h_max = ((M_SIZE + MESH_Y_TILES - 1) /  MESH_Y_TILES);
    uint32_t tile_w_max = ((N_SIZE + MESH_X_TILES - 1) /  MESH_X_TILES);
    int32_t tile_h;
    int32_t tile_w;

    if(((tile_h_max * y_id) + tile_h_max) > M_SIZE){
        tile_h = M_SIZE - (tile_h_max * y_id);    
    }    
    else{
        tile_h = tile_h_max;
    }

    if(((tile_w_max * x_id) + tile_w_max) > N_SIZE){
        tile_w = N_SIZE - (tile_w_max * x_id);
    }    
    else{
        tile_w = tile_w_max;
    }

    if(tile_h < 1 || tile_w < 1){
        return 0;
    }
    else{
        //printf("ID:%d, Mesh-Tile-X:%d, Mesh-Tile-Y:%d, Data-Tile w: %d, Data-Tile h: %d", hartid, x_id, y_id, tile_w, tile_h);
    }

    /**
     * 1a. Set the t_size, how many timeslots we want to divide the temporal dimension into.
     * t_start is the first timeslot in which it is possible to elaborate the output.
     * t_end is the last timeslot.
     */
    uint8_t timeslots = 16;
    uint8_t t_size = K_SIZE / timeslots;
    uint8_t t_start = x_id * 2;
    uint8_t t_end = t_start + timeslots;
    uint8_t total_timeslots = (MESH_X_TILES - 1) * 2 + timeslots + 1;
    //printf("timeslots=%d, t_start=%d, t_end=%d\n", timeslots, t_start, t_end);

    /**
     * 2. Initalize the IDMA transfer variables for input data-tile transfers.
     */
    uint32_t len_x = tile_w * 2;
    uint32_t std_x = N_SIZE * 2;
    uint32_t reps_x = (uint32_t) tile_h;
    uint32_t obi_addr_x = (l1_tile_base);
    uint32_t axi_addr_x = (uint32_t) x_inp + (y_id * N_SIZE * tile_h_max * 2) + (tile_w_max * x_id * 2);

    /**
     * 2a. Initalize the IDMA transfer variables for weight data-tile transfers.
     */
    uint32_t len_w = (uint32_t) (t_size * 2);
    uint32_t std_w = (uint32_t) (K_SIZE * 2);
    uint32_t reps_w = (uint32_t) tile_w;
    uint32_t obi_addr_w_0 = obi_addr_x + (tile_h * tile_w * 2);
    uint32_t obi_addr_w_1 = obi_addr_w_0 + (tile_w * t_size * 2);
    uint32_t obi_addr_w_2 = obi_addr_w_1 + (tile_w * t_size * 2);
    uint32_t axi_addr_w = (uint32_t) w_inp + (x_id * K_SIZE * tile_w_max * 2);

    /**
     * 2b. Initalize the IDMA transfer variables for output data-tile transfers.
     */
    uint32_t len_y = (uint32_t) (t_size * 2);
    uint32_t std_y = (uint32_t) (K_SIZE * 2);
    uint32_t reps_y = (uint32_t) tile_h;
    uint32_t obi_addr_y_0 = obi_addr_w_2 + (tile_w * t_size * 2);
    uint32_t obi_addr_y_1 = obi_addr_y_0 + (tile_h * t_size * 2);
    uint32_t obi_addr_y_2 = obi_addr_y_1 + (tile_h * t_size * 2);
    uint32_t axi_addr_y = (uint32_t) y_inp + (y_id * K_SIZE * tile_h_max * 2);
    //uint32_t axi_addr_y_out = (uint32_t) y_out + (y_id * K_SIZE * tile_h_max * 2);

    uint8_t pt = 0;
    volatile uint32_t output_pt;
    volatile uint32_t weight_pt;
    volatile uint32_t output_pt_prev;
    volatile uint32_t weight_pt_next;
    //redmule_mcnfig((uint16_t) t_size, (uint16_t) tile_h, (uint16_t) tile_w);
    
    /**
     * TEST LOOP - REPEAT THE TEST N_ITERATION TIMES.
     */
    for(int z = 0; z < N_ITERATIONS; z++){
        // TIMESLOT t = -1
        // Initial static input load
        idma_memcpy_2d(&idma_ctrl, 0, axi_addr_x, obi_addr_x, len_x, std_x, reps_x);
        #if STALLING == 0
        eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
        #endif

        // First weight load
        idma_memcpy_2d(&idma_ctrl, 0, axi_addr_w, obi_addr_w_0, len_w, std_w, reps_w);
        #if STALLING == 0
        eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
        #endif

        fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
        #if STALLING == 0
        eu_fsync_wait(&eu_ctrl, WAIT_MODE);
        #endif

        /**
         * 3. Cycle over the timeslots.
         */
        for(int t = 0; t < total_timeslots; t++){
            //printf("TIMESLOT N: %d\n", t);
            /**
             * 3a. Skip the timeslot if outside the range
             */
            if(t < t_start || t > t_end){
                fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
                #if STALLING == 0
                eu_fsync_wait(&eu_ctrl, WAIT_MODE);
                #endif
                continue;
            }

            /**
             * 3b. Select correct buffer for current, previous and next timeslot
             */
            switch (pt % 3){
                case 0:
                    // printf("Case 0\n");
                    output_pt = obi_addr_y_0;
                    output_pt_prev = obi_addr_y_2;
                    weight_pt = obi_addr_w_0;
                    weight_pt_next = obi_addr_w_1;
                    break;
                case 1:
                    // printf("Case 1\n");
                    output_pt = obi_addr_y_1;
                    output_pt_prev = obi_addr_y_0;
                    weight_pt = obi_addr_w_1;
                    weight_pt_next = obi_addr_w_2;
                    break;
                case 2:
                    // printf("Case 2\n");
                    output_pt = obi_addr_y_2;
                    output_pt_prev = obi_addr_y_1;
                    weight_pt = obi_addr_w_2;
                    weight_pt_next = obi_addr_w_0;
                    break;
            }

            /**
             * 3c. Load L2 output of current timeslot (if leftmost tile)
             */
            if(x_id == 0 && t < (t_end)){
                idma_memcpy_2d(&idma_ctrl, 0, axi_addr_y + (pt * t_size * 2), output_pt, len_y, std_y, reps_y);
                #if STALLING == 0
                eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
                #endif
                // printf("Received this data: %x, %x\n", *(volatile uint16_t*)(output_pt), *(volatile uint16_t*)(output_pt + 2));
            }

            /**
             * 3d. 
             * If we are in the START timeslot:
             *      - IDMA-in to load weight tile for next timeslot
             *      - REDMULE to execute current timeslot output
             * If we are at the END - 1 timeslot:
             *      - IDMA-out to push the previous timeslot output onto the right tile (or L2)
             *      - REDMULE to execute the current timeslot output
             * If we are at the END timeslot:
             *      - IDMA-out to push the previous timeslot output onto the right tile (or L2)
             * Otherwise:
             *      - IDMA-in to load weight tile for next timeslot
             *      - IDMA-out to push the previous timeslot output onto the right tile (or L2)
             *      - REDMULE to execute current timeslot output
             */
            if(pt < timeslots - 1)
                idma_memcpy_2d(&idma_ctrl, 0, axi_addr_w + (t_size * (pt + 1) * 2), weight_pt_next, len_w, std_w, reps_w);
            if(pt > 0){
                if(x_id == (MESH_X_TILES-1)){
                    idma_memcpy_2d(&idma_ctrl, 1, axi_addr_y + ((pt - 1) * t_size * 2), output_pt_prev, len_y, std_y, reps_y);
                }
                else{
                    switch (pt % 3){
                        case 0:
                            idma_memcpy_1d(&idma_ctrl, 1, get_l1_base(hartid + 1) + (tile_h * tile_w * 2) + (tile_w * t_size * 6) + (tile_h * t_size * 4), output_pt_prev, tile_h * t_size * 2);
                            break;
                        case 1:
                            idma_memcpy_1d(&idma_ctrl, 1, get_l1_base(hartid + 1) + (tile_h * tile_w * 2) + (tile_w * t_size * 6), output_pt_prev, tile_h * t_size * 2);
                            break;
                        case 2:
                            idma_memcpy_1d(&idma_ctrl, 1, get_l1_base(hartid + 1) + (tile_h * tile_w * 2) + (tile_w * t_size * 6) + (tile_h * t_size * 2), output_pt_prev, tile_h * t_size * 2);
                            break;
                    }
                }
            }
            if(pt < timeslots)
                redmule_gemm(&redmule_ctrl, obi_addr_x, weight_pt, output_pt, (uint16_t) tile_h, (uint16_t) tile_w, (uint16_t) t_size);
            
            #if STALLING == 0
            if(pt < timeslots - 1)
                eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
            if(pt > 0)
                eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
            if(pt < timeslots)
                eu_redmule_wait(&eu_ctrl, WAIT_MODE);
            #endif

            /**
             * 3f. Sync before next timeslot
             */
            pt++;
            fsync_sync_global(&fsync_ctrl);
            #if STALLING == 0
            eu_fsync_wait(&eu_ctrl, WAIT_MODE);
            #endif
        }
    }
    
    
    /**
     * 5. Check results
     */
    volatile uint32_t errors=0;
    fsync_sync_row(&fsync_ctrl);
    #if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
    #endif
    
    if(x_id == MESH_X_TILES - 1){
        uint16_t computed, expected, diff = 0;
        for(int i = (y_id * tile_h_max); i < (y_id * tile_h_max + tile_h); i++){
            for(int j = 0; j < K_SIZE; j++){
                computed = *(volatile uint16_t*)(y_inp + (i * K_SIZE + j));
                expected = *(volatile uint16_t*)(z_out + (i * K_SIZE + j));
                diff = (computed > expected) ? (computed - expected) : (expected - computed);
                if(diff > 0x0011){
                    #if EVAL == 1
                    if(y_id == 0)
                        printf("Error detected at coordinates[%d][%d]: Y=%x Z=%x\n", i, j, *(volatile uint16_t*)(y_inp + (i * K_SIZE + j)), *(volatile uint16_t*)(z_out + (i * K_SIZE + j)));
                    #endif    
                    errors++;
                }       
            }
        }
        printf("Number of errors: %d\n", errors);
    }

    return errors;  
}