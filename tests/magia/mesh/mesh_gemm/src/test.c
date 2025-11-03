// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Victor Isachi <victor.isachi@unibo.it>
// Alberto Dequino <alberto.dequino@unibo.it>

#define SIZE_64x64x64
// #define SIZE_128x128x128
// #define SIZE_256x256x256
// #define SIZE_512x512x512
// #define SIZE_1024x1024x1024

#define GEMM_WIDTH (MESH_X_TILES)
#define GEMM_MAX(x, y) (((int)(x) > (int)(y)) ? (x) : (y))
#define GEMM_MIN(x, y) (((int)(x) < (int)(y)) ? (x) : (y))

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
#include "idma.h"
#include "redmule.h"
#include "fsync.h"

/**
 * This test implements the optimal GeMM algorithm for MAGIA (and for mesh-based architectures in general). 
 * See "WaferLLM: Large Language Model Inference at Wafer Scale" paper.
 */
int main(void){
    sentinel_start();   // Total execution

    /** 
     * 0. Get the mesh-tile's hartid, mesh-tile coordinates and define its L1 base, 
     * also initialize the controllers for the idma, redmule and fsync.
     */
    uint32_t hartid = get_hartid();

    idma_config_t idma_cfg = {.hartid = hartid};
    idma_controller_t idma_ctrl = {
        .base = NULL,
        .cfg = &idma_cfg,
        .api = &idma_api,
    };
    idma_init(&idma_ctrl);

    redmule_config_t redmule_cfg = {.hartid = hartid};
    redmule_controller_t redmule_ctrl = {
        .base = NULL,
        .cfg = &redmule_cfg,
        .api = &redmule_api,
    };
    redmule_init(&redmule_ctrl);

    fsync_config_t          fsync_cfg = {.hartid = hartid};
    fsync_controller_t      fsync_ctrl = {
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
     * 1. Calculate the static output data-tile dimensions.
     * If M and K are perfect multiples of the number of mesh-tiles in their respective axis, 
     * tile_h and tile_w will be the same for each mesh-tile.
     * Otherwise, these dimensions will be smaller for the mesh-tiles closer to the right and lower 
     * border of the mesh. 
     */
    uint32_t tile_h_max = ((M_SIZE + MESH_Y_TILES - 1) /  MESH_Y_TILES);
    uint32_t tile_w_max = ((K_SIZE + MESH_X_TILES - 1) /  MESH_X_TILES);
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
        // printf("ID:%d, Mesh-Tile-X:%d, Mesh-Tile-Y:%d, Data-Tile w: %d, Data-Tile h: %d\n", hartid, x_id, y_id, tile_w, tile_h);
    }

    /**
     * The MeshGeMM is implemented with the number of timeslots equal to the mesh dimension.
     */
#if defined(SIZE_64x64x64)
    uint8_t timeslots = 2;
#elif defined(SIZE_128x128x128)
    uint8_t timeslots = 4;
#elif defined(SIZE_256x256x256)
    uint8_t timeslots = 8;
#elif defined(SIZE_512x512x512)
    uint8_t timeslots = 16;
#elif defined(SIZE_1024x1024x1024)
    uint8_t timeslots = 32;
#endif
    uint8_t t_size = N_SIZE / timeslots;

    /**
     * 2. Use iDMA to transfer static output data-tile.
     */
    uint32_t len_y = tile_w * 2;
    uint32_t std_y = K_SIZE * 2;
    uint32_t reps_y = (uint32_t) tile_h;
    uint32_t obi_addr_y = (l1_tile_base);
    uint32_t axi_addr_y = (uint32_t) y_in + (y_id * K_SIZE * tile_h_max * 2) + (tile_w_max * x_id * 2); 
    
    // printf("Doing initial output L2 idma memcpy\n");
    stnl_cmi_s();
    idma_conf_in();
    idma_set_addr_len_in(obi_addr_y, axi_addr_y, len_y);
    idma_set_std2_rep2_in(len_y, std_y, reps_y);
    idma_set_std3_rep3_in(0, 0, 1);
    idma_start_in();
    idma_wait();
    stnl_par_f();
    
    /**
     * 2a. Initalize and run IDMA transfer variables for initial L2 input data-tile transfers.
     */
    // This index calculates the initial contribution for the current tile
    int32_t index = ((y_id+1)/2)*(y_id%2 ? -1 : 1) + ((x_id+1)/2)*(x_id%2 ? -1 : 1);
    index = index < 0 ? index+GEMM_WIDTH : index;
    index = index >= GEMM_WIDTH/2 ? GEMM_WIDTH-(2*index-GEMM_WIDTH+1) : 2*index;
    // printf("Index: %d\n", index);

    uint32_t obi_addr_x_0 = obi_addr_y + (tile_h * tile_w * 2);
    uint32_t obi_addr_x_1 = obi_addr_x_0 + (tile_h * t_size * 2);

    uint32_t len_x = (uint32_t) (t_size * 2);
    uint32_t std_x = (uint32_t) (N_SIZE * 2);
    uint32_t reps_x = (uint32_t) tile_h;
    uint32_t axi_addr_x = (uint32_t) x_in + (y_id * N_SIZE * tile_h_max * 2) + (index * t_size * 2);
    // printf("Doing initial input L2 idma memcpy\n");
    stnl_cmi_s();
    idma_conf_in();
    idma_set_addr_len_in(obi_addr_x_0, axi_addr_x, len_x);
    idma_set_std2_rep2_in(len_x, std_x, reps_x);
    idma_set_std3_rep3_in(0, 0, 1);
    idma_start_in();
    idma_wait();
    stnl_par_f();

    /**
     * 2b. Initalize and run IDMA transfer variables for initial L2 weight data-tile transfers.
     */
    uint32_t obi_addr_w_0 = obi_addr_x_1 + (tile_h * t_size * 2);
    uint32_t obi_addr_w_1 = obi_addr_w_0 + (tile_w * t_size * 2);

    uint32_t len_w = (uint32_t) (tile_w * 2);
    uint32_t std_w = (uint32_t) (K_SIZE * 2);
    uint32_t reps_w = (uint32_t) t_size;
    uint32_t axi_addr_w = (uint32_t) w_in + (x_id * tile_w_max * 2) + (index * t_size * K_SIZE * 2);
    // printf("Doing initial weight L2 idma memcpy\n");
    stnl_cmi_s();
    idma_conf_in();
    idma_set_addr_len_in(obi_addr_w_0, axi_addr_w, len_w);
    idma_set_std2_rep2_in(len_w, std_w, reps_w);
    idma_set_std3_rep3_in(0, 0, 1);
    idma_start_in();
    idma_wait();
    stnl_par_f();

    volatile uint32_t input_pt;
    volatile uint32_t weight_pt;
    volatile uint32_t input_pt_next;
    volatile uint32_t weight_pt_next;

    uint32_t mesh_x_id = x_id%2 ? GEMM_MAX(x_id-2, 0) : GEMM_MIN(x_id+2, GEMM_WIDTH-1);
    uint32_t mesh_y_id = y_id%2 ? GEMM_MAX(y_id-2, 0) : GEMM_MIN(y_id+2, GEMM_WIDTH-1);
    uint32_t horizontal_id = GET_ID(y_id, mesh_x_id);
    uint32_t vertical_id = GET_ID(mesh_y_id, x_id);
    // printf("Mesh X ID: %d, Mesh Y ID: %d, Horizontal ID: %d, Vertical ID: %d\n", mesh_x_id, mesh_y_id, horizontal_id, vertical_id);

    // printf("tile_h = %d, tile_w = %d, t_size = %d\n", tile_h, tile_w, t_size);

    redmule_mcnfig((uint16_t) tile_w, (uint16_t) tile_h, (uint16_t) t_size);

    stnl_ts_f();    // Initial timeslot

    /**
     * 3. Cycle over the timeslots.
     * For each timeslot, the mesh-tile will:
     * a - Load the weight and input data-tile for the current timeslot
     * b - Multiply and add
     * Synchronization is not required.
     */
    for(int i = 0; i < timeslots; i++){
        if (i > 0){
            stnl_ts_f();
        }
        if (i < (timeslots - 1)){
            //printf("Syncing\n");
            // if(hartid == 0)
            //     printf("TIMESLOT NUMBER %d\n", i);
            fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);

            stnl_ts_s();
        }
        
        /**
         * 3a. Choose which of the 2 buffers use (double buffering is in effect)
         */
        if(i % 2){
            input_pt = obi_addr_x_1;
            weight_pt = obi_addr_w_1;
            input_pt_next = obi_addr_x_0;
            weight_pt_next = obi_addr_w_0;
        }
        else{
            input_pt = obi_addr_x_0;
            weight_pt = obi_addr_w_0;
            input_pt_next = obi_addr_x_1;
            weight_pt_next = obi_addr_w_1;
        }

        /**
         * 3b. IDMA to load the input and weight data-tile for next timeslot (if there is one)
         */
        if(i != (timeslots - 1)){
            // printf("Loading next input tile\n");
            idma_conf_out();
            idma_set_addr_len_out(get_l1_base(vertical_id) + (tile_h * tile_w * 2) + (tile_h * t_size * 4) + (tile_w * t_size * 2 * (((i + 1) % 2))), weight_pt, tile_w * t_size * 2);
            idma_set_std2_rep2_out(0, 0, 1);
            idma_set_std3_rep3_out(0, 0, 1);
            idma_conf_in();
            idma_set_addr_len_in(input_pt_next, get_l1_base(horizontal_id) + (tile_h * tile_w * 2) + (tile_h * t_size * 2 * (i % 2)), tile_h * t_size * 2);
            idma_set_std2_rep2_in(0, 0, 1);
            idma_set_std3_rep3_in(0, 0, 1);

            stnl_cmp_s();
            redmule_marith(obi_addr_y, weight_pt, input_pt);
            stnl_cmi_s();
            idma_start_in();
            stnl_cmo_s();
            idma_start_out();
            redmule_wait();
            stnl_par_f();
            idma_wait();
            stnl_par_f();
            idma_wait();
            stnl_par_f();
        }
        else{
            sentinel_start();   // Last CMP overhead
            stnl_cmp_s();
            redmule_marith(obi_addr_y, weight_pt, input_pt);
            redmule_wait();
            stnl_par_f();
            sentinel_end(); // Last CMP overhead
        }
    }

    /**
     * 4. Store the output data-tile back to L2
     */
    sentinel_start();   // Last CMO overhead
    stnl_cmo_s();
    idma_conf_out();
    idma_set_addr_len_out(axi_addr_y, obi_addr_y, len_y);
    idma_set_std2_rep2_out(std_y, len_y, reps_y);
    idma_set_std3_rep3_out(0, 0, 1);
    idma_start_out();
    idma_wait();
    stnl_par_f();
    sentinel_end(); // Last CMO overhead
    asm volatile("nop" ::); // Needed to detect same instruction consecutively

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
    uint16_t computed, expected, diff = 0;
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    for(int i = (y_id * tile_h_max); i < (y_id * tile_h_max + tile_h); i++){
        for(int j = (x_id * tile_w_max); j < (x_id * tile_w_max) + tile_w; j++){
            computed = *(volatile uint16_t*)(y_in + (i * K_SIZE + j));
            expected = *(volatile uint16_t*)(z_out + (i * K_SIZE + j));
            diff = (computed > expected) ? (computed - expected) : (expected - computed);
            if(diff > 0x01FF){
                if(y_id == 0)
                    printf("Error detected at coordinates[%d][%d]: Y=%x Z=%x\n", i, j, *(volatile uint16_t*)(y_in+ (i * K_SIZE + j)), *(volatile uint16_t*)(z_out + (i * K_SIZE + j)));
                errors++;
            }       
        }
    }
    
    printf("Number of errors: %d\n", errors);

    return errors;  
}