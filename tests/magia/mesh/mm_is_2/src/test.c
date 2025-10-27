// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Alberto Dequino <alberto.dequino@unibo.it>
// Victor Isachi <victor.isachi@unibo.it>

#define SIZE_64x64x64
// #define SIZE_128x128x128
// #define SIZE_256x256x256
// #define SIZE_512x512x512
// #define SIZE_1024x1024x1024

#include <stdint.h>

#if defined(SIZE_64x64x64)
#include "mat_64x64x64.h"
#elif defined(SIZE_128x128x128)
#include "mat_128x128x128.h"
#elif defined(SIZE_256x256x256)
#include "mat_256x256x256.h"
#elif defined(SIZE_512x512x512)
#include "mat_512x512x512.h"
#elif defined(SIZE_1024x1024x1024)
#include "mat_1024x1024x1024.h"
#endif

#include "tile.h"
#include "fsync.h"
#include "idma.h"
#include "redmule.h"

#define N_ITERATIONS 1

/**
 * This test aims to verify the functionality of MAGIA as a systolic array for matrix multiplications,
 * following the input-static mechanism. 
 */
int main(void){
    sentinel_start();   // Total execution
    
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

    uint32_t y_id = GET_Y_ID(hartid);
    uint32_t x_id = GET_X_ID(hartid);
    uint32_t l1_tile_base = get_l1_base(hartid);
    
    // Wait for all tiles to be awake and ready to start the kernel
    stnl_snc_s();
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    stnl_snc_f();

    sentinel_start();   // Execution time after wakeup
    asm volatile("nop" ::); // Needed to detect same instruction consecutively
    sentinel_start();   // Initial step overhead

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
#if defined(SIZE_64x64x64)
    uint8_t timeslots = 4;
#elif defined(SIZE_128x128x128)
    uint8_t timeslots = 8;
#elif defined(SIZE_256x256x256)
    uint8_t timeslots = 16;
#elif defined(SIZE_512x512x512)
    uint8_t timeslots = 32;
#elif defined(SIZE_1024x1024x1024)
    uint8_t timeslots = 64;
#endif
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
    volatile uint32_t obi_addr_x = (l1_tile_base);
    volatile uint32_t axi_addr_x = (uint32_t) x_in + (y_id * N_SIZE * tile_h_max * 2) + (tile_w_max * x_id * 2);

    /**
     * 2a. Initalize the IDMA transfer variables for weight data-tile transfers.
     */
    uint32_t len_w = (uint32_t) (t_size * 2);
    uint32_t std_w = (uint32_t) (K_SIZE * 2);
    uint32_t reps_w = (uint32_t) tile_w;
    volatile uint32_t obi_addr_w_0 = obi_addr_x + (tile_h * tile_w * 2);
    volatile uint32_t obi_addr_w_1 = obi_addr_w_0 + (tile_w * t_size * 2);
    volatile uint32_t obi_addr_w_2 = obi_addr_w_1 + (tile_w * t_size * 2);
    volatile uint32_t axi_addr_w = (uint32_t) w_in + (x_id * K_SIZE * tile_w_max * 2);

    /**
     * 2b. Initalize the IDMA transfer variables for output data-tile transfers.
     */
    uint32_t len_y = (uint32_t) (t_size * 2);
    uint32_t std_y = (uint32_t) (K_SIZE * 2);
    uint32_t reps_y = (uint32_t) tile_h;
    volatile uint32_t obi_addr_y_0 = obi_addr_w_2 + (tile_w * t_size * 2);
    volatile uint32_t obi_addr_y_1 = obi_addr_y_0 + (tile_h * t_size * 2);
    volatile uint32_t obi_addr_y_2 = obi_addr_y_1 + (tile_h * t_size * 2);
    volatile uint32_t axi_addr_y = (uint32_t) y_in + (y_id * K_SIZE * tile_h_max * 2);
    volatile uint32_t axi_addr_y_out = (uint32_t) y_out + (y_id * K_SIZE * tile_h_max * 2);

    uint8_t pt = 0;
    volatile uint32_t output_pt;
    volatile uint32_t weight_pt;
    volatile uint32_t output_pt_prev;
    volatile uint32_t weight_pt_next;
    redmule_mcnfig((uint16_t) t_size, (uint16_t) tile_h, (uint16_t) tile_w);

    sentinel_end(); // Initial step overhead

    // if (hartid == 0) printf("TEST PARAMETERS:\n \ttimeslots: %0d\n \tt_size: %0d\n \ttile_h: %0d\n \ttile_w: %0d\n", timeslots, t_size, tile_h, tile_w);

    /**
     * TEST LOOP - REPEAT THE TEST N_ITERATION TIMES.
     */
    for(uint8_t z = 0; z < N_ITERATIONS; z++){
        stnl_ts_s();

        // TIMESLOT t = -1
        // Initial static input load
        idma_conf_in();
        idma_set_addr_len_in(obi_addr_x, axi_addr_x, len_x);
        idma_set_std2_rep2_in(len_x, std_x, reps_x);
        idma_set_std3_rep3_in(0, 0, 1);

        stnl_cmi_s();
        idma_start_in();
        idma_wait();
        stnl_par_f();

        // First weight load
        idma_conf_in();
        idma_set_addr_len_in(obi_addr_w_0, axi_addr_w, len_w);
        idma_set_std2_rep2_in(len_w, std_w, reps_w);
        idma_set_std3_rep3_in(0, 0, 1);

        stnl_cmi_s();
        idma_start_in();
        idma_wait();
        stnl_par_f();

        stnl_ts_f();

        stnl_snc_s();
        fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
        stnl_snc_f();

        /**
         * 3. Cycle over the timeslots.
         */
        for(int t = 0; t < total_timeslots; t++){
            stnl_ts_s();
            
            //printf("TIMESLOT N: %d\n", t);
            /**
             * 3a. Skip the timeslot if outside the range
             */
            if(t < t_start || t > t_end){
                stnl_ts_f();
                
                stnl_snc_s();
                fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
                stnl_snc_f();

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
                idma_conf_in();
                idma_set_addr_len_in(output_pt, axi_addr_y + (pt * t_size * 2), len_y);
                idma_set_std2_rep2_in(len_y, std_y, reps_y);
                idma_set_std3_rep3_in(0, 0, 1);

                stnl_cmi_s();
                idma_start_in();
                idma_wait();
                stnl_par_f();
                // printf("Received this data: %x, %x\n", *(volatile uint16_t*)(output_pt), *(volatile uint16_t*)(output_pt + 2));
            }

            /**
             * 3d. 
             * If we are in the START timeslot:
             *      - Set IDMA-in to load weight tile for next timeslot
             *      - Set REDMULE to execute current timeslot output
             * If we are at the END - 1 timeslot:
             *      - Set IDMA-out to push the previous timeslot output onto the right tile (or L2)
             *      - Set REDMULE to execute the current timeslot output
             * If we are at the END timeslot:
             *      - Set IDMA-out to push the previous timeslot output onto the right tile (or L2)
             * Otherwise:
             *      - Set IDMA-in to load weight tile for next timeslot
             *      - Set IDMA-out to push the previous timeslot output onto the right tile (or L2)
             *      - Set REDMULE to execute current timeslot output
             */
            if(pt < timeslots - 1){
                idma_conf_in();
                idma_set_addr_len_in(weight_pt_next, axi_addr_w + (t_size * (pt + 1) * 2), len_w);
                idma_set_std2_rep2_in(len_w, std_w, reps_w);
                idma_set_std3_rep3_in(0, 0, 1);
            }
            if(pt > 0){
                if(x_id == (MESH_X_TILES-1)){
                    idma_conf_out();
                    idma_set_addr_len_out(axi_addr_y_out + ((pt - 1) * t_size * 2), output_pt_prev, len_y);
                    idma_set_std2_rep2_out(std_y, len_y, reps_y);
                    idma_set_std3_rep3_out(0, 0, 1);
                }
                else{
                    //TODO: This test assumes tile_h and tile_w are the same for each mesh tile, which may not be true.
                    idma_conf_out();
                    switch (pt % 3){
                        case 0:
                            idma_set_addr_len_out(get_l1_base(hartid + 1) + (tile_h * tile_w * 2) + (tile_w * t_size * 6) + (tile_h * t_size * 4), output_pt_prev, tile_h * t_size * 2);
                            // printf("IDMA out len: %d\n", tile_h * t_size * 2);
                            break;
                        case 1:
                            idma_set_addr_len_out(get_l1_base(hartid + 1) + (tile_h * tile_w * 2) + (tile_w * t_size * 6), output_pt_prev, tile_h * t_size * 2);
                            // printf("IDMA out len: %d\n", tile_h * t_size * 2);
                            break;
                        case 2:
                            idma_set_addr_len_out(get_l1_base(hartid + 1) + (tile_h * tile_w * 2) + (tile_w * t_size * 6) + (tile_h * t_size * 2), output_pt_prev, tile_h * t_size * 2);
                            // printf("IDMA out len: %d\n", tile_h * t_size * 2);
                            break;
                    }
                    idma_set_std2_rep2_out(0, 0, 1);
                    idma_set_std3_rep3_out(0, 0, 1);
                }
            }

            /**
             * 3e. Activate IDMA and Redmule
             */
            if(t == t_start){
                //printf("Received this data: %x, %x\n", *(volatile uint16_t*)(output_pt), *(volatile uint16_t*)(output_pt + 2));
                stnl_cmp_s();
                redmule_marith(output_pt, weight_pt, obi_addr_x);
                stnl_cmi_s();
                idma_start_in();
                redmule_wait();
                stnl_par_f();
                idma_wait();
                stnl_par_f();
                // printf("Redmule output: %x, %x\n", *(volatile uint16_t*)(output_pt), *(volatile uint16_t*)(output_pt + 2));

            }
            else if(t == (t_end - 1)){
                //printf("Received this data: %x, %x\n", *(volatile uint16_t*)(output_pt), *(volatile uint16_t*)(output_pt + 2));
                stnl_cmp_s();
                redmule_marith(output_pt, weight_pt, obi_addr_x);
                stnl_cmo_s();
                idma_start_out();
                redmule_wait();
                stnl_par_f();
                idma_wait();
                stnl_par_f();
                // printf("Redmule output: %x, %x\n", *(volatile uint16_t*)(output_pt), *(volatile uint16_t*)(output_pt + 2));

                //printf("Sent this data: %x, %x\n", *(volatile uint16_t*)(output_pt_prev), *(volatile uint16_t*)(output_pt_prev + 2));
            }
            else if(t == t_end){
                stnl_cmo_s();
                idma_start_out();
                idma_wait();
                stnl_par_f();
                // printf("Sent this data: %x, %x\n", *(volatile uint16_t*)(output_pt_prev), *(volatile uint16_t*)(output_pt_prev + 2));
            }
            else{
                //printf("Received this data: %x, %x\n", *(volatile uint16_t*)(output_pt), *(volatile uint16_t*)(output_pt + 2));
                stnl_cmp_s();
                redmule_marith(output_pt, weight_pt, obi_addr_x);
                stnl_cmo_s();
                idma_start_out();
                stnl_cmi_s();
                idma_start_in();
                redmule_wait();
                stnl_par_f();
                idma_wait();
                stnl_par_f();
                idma_wait();
                stnl_par_f();
                // printf("Redmule output: %x, %x\n", *(volatile uint16_t*)(output_pt), *(volatile uint16_t*)(output_pt + 2));

                // printf("Sent this data: %x, %x\n", *(volatile uint16_t*)(output_pt_prev), *(volatile uint16_t*)(output_pt_prev + 2));
            }

            /**
             * 3f. Sync before next timeslot
             */
            pt++;

            stnl_ts_f();

            stnl_snc_s();
            fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
            stnl_snc_f();
        }
    }

    sentinel_end(); // Execution time after wakeup
    asm volatile("nop" ::); // Needed to detect same instruction consecutively
    sentinel_end(); // Total execution

    stnl_cmi_r();
    stnl_cmo_r();
    stnl_cmp_r();
    stnl_snc_r();
    
    if (get_hartid() == 0){
      stnl_r();
      stnl_ts_r();
    }

    /**
     * 5. Check results
     */
    uint32_t errors=0;
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    if(x_id == MESH_X_TILES - 1){
        uint16_t computed, expected, diff = 0;
        for(uint8_t i = (y_id * tile_h_max); i < (y_id * tile_h_max + tile_h); i++){
            for(uint8_t j = 0; j < K_SIZE; j++){
                computed = *(volatile uint16_t*)(y_out + (i * K_SIZE + j));
                expected = *(volatile uint16_t*)(z_out + (i * K_SIZE + j));
                diff = (computed > expected) ? (computed - expected) : (expected - computed);
                if(diff > 0x0011){
                    if(y_id == 0)
                        printf("Error detected at coordinates[%d][%d]: Y=%x Z=%x\n", i, j, *(volatile uint16_t*)(y_out+ (i * K_SIZE + j)), *(volatile uint16_t*)(z_out + (i * K_SIZE + j)));
                    errors++;
                }       
            }
        }
        printf("Number of errors: %d\n", errors);
    }

    return errors;  
}