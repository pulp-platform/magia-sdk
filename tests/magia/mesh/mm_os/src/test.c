// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Alberto Dequino <alberto.dequino@unibo.it>

#include <stdint.h>
#include "test.h"

#include "tile.h"
#include "idma.h"
#include "redmule.h"

/**
 * This test aims to verify the functionality of MAGIA as a systolic array for matrix multiplications,
 * following the output-static mechanism. 
 */
int main(void){
    /** 
     * 0. Get the mesh-tile's hartid, mesh-tile coordinates and define its L1 base, 
     * also initialize the controllers for the idma and redmule.
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

    idma_init(&idma_ctrl);
    redmule_init(&redmule_ctrl);

    uint32_t y_id = GET_Y_ID(hartid);
    uint32_t x_id = GET_X_ID(hartid);
    uint32_t l1_tile_base = get_l1_base(hartid);

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
        //printf("ID:%d, Mesh-Tile-X:%d, Mesh-Tile-Y:%d, Data-Tile w: %d, Data-Tile h: %d\n", hartid, x_id, y_id, tile_w, tile_h);
    }

    /**
     * 1a. Set the t_size, how many timeslots we want to divide the temporal dimension into.
     * The final L1 memory requirement will be:
     * Input data-tile: (tile_h x t_size) * data_dim
     * Weight data-tile: (t_size x tile_w) * data_dim
     * Output data-tile: ((tile_h x tile_w) * data_dim)
     */
    uint8_t timeslots = 2;
    uint8_t t_size = N_SIZE / timeslots;

    /**
     * 2. Use IDMA to transfer static output data-tile
     */
    uint32_t len_y = tile_w * 2;
    uint32_t std_y = K_SIZE * 2;
    uint32_t reps_y = (uint32_t) tile_h;
    uint32_t obi_addr_y = (l1_tile_base);
    uint32_t axi_addr_y = (uint32_t) y_inp + (y_id * K_SIZE * tile_h_max * 2) + (tile_w_max * x_id * 2); 
    
    /**
     * 2a. Initalize the IDMA transfer variables for input data-tile transfers.
     */
    uint32_t len_x = (uint32_t) (t_size * 2);
    uint32_t std_x = (uint32_t) (N_SIZE * 2);
    uint32_t reps_x = (uint32_t) tile_h;
    uint32_t obi_addr_x = obi_addr_y + (tile_h * tile_w * 2);
    uint32_t axi_addr_x = (uint32_t) x_inp + (y_id * N_SIZE * tile_h_max * 2);
    
    /**
     * 2b. Initalize the IDMA transfer variables for weight data-tile transfers.
     */
    uint32_t len_w = (uint32_t) (tile_w * 2);
    uint32_t std_w = (uint32_t) (K_SIZE * 2);
    uint32_t reps_w = (uint32_t) t_size;
    uint32_t obi_addr_w = obi_addr_x + (t_size * tile_h * 2);
    uint32_t axi_addr_w = (uint32_t) w_inp + (x_id * tile_w_max * 2);

    //printf("tile_h = %d, tile_w = %d, t_size = %d\n", tile_h, tile_w, t_size);

    //sentinel_start();
    //printf("Doing idma memcpy y\n");
    idma_memcpy_2d(&idma_ctrl, 0, axi_addr_y, obi_addr_y, len_y, std_y, reps_y);
    idma_wait();

    /**
     * 3. Cycle over the timeslots.
     * For each timeslot, the mesh-tile will:
     * a - Load the weight and input data-tile for the current timeslot
     * b - Multiply and add
     * Synchronization is not required.
     */
    for(uint8_t i = 0; i < timeslots; i++){
        /**
         * 3a. IDMA to load the input and weight data-tile for current timeslot
         */
        // printf("Doing idma memcpy x\n");
        idma_memcpy_2d(&idma_ctrl, 0, (axi_addr_x + (t_size * i * 2)), obi_addr_x, len_x, std_x, reps_x);
        idma_wait();
        // printf("Doing idma memcpy w\n");
        idma_memcpy_2d(&idma_ctrl, 0, (axi_addr_w + (t_size * K_SIZE * i * 2)), obi_addr_w, len_w, std_w, reps_w);
        idma_wait();
        
        /**
         * 3b. Evoke the RED MULE 
         * https://www.youtube.com/watch?v=RG-bRbBuaBI&list=PLTLXyHxNV4azQtL26W-7l6fTrOa3rJgLo&index=35
         */
        // printf("Doing redmule\n");
        redmule_gemm(&redmule_ctrl, obi_addr_x, obi_addr_w, obi_addr_y, (uint16_t) tile_h, (uint16_t) t_size, (uint16_t) tile_w);
        redmule_wait();
    }

    /**
     * 4. Store the output data-tile back to L2
     */
    idma_memcpy_2d(&idma_ctrl, 1, axi_addr_y, obi_addr_y, len_y, std_y, reps_y);
    idma_wait();

    //sentinel_end();

    /**
     * 5. Check results
     */
    uint32_t errors=0;
    uint16_t computed, expected, diff = 0;
    for(uint8_t i = (y_id * tile_h_max); i < (y_id * tile_h_max + tile_h); i++){
        for(uint8_t j = (x_id * tile_w_max); j < (x_id * tile_w_max) + tile_w; j++){
            computed = *(volatile uint16_t*)(y_inp + (i * K_SIZE + j));
            expected = *(volatile uint16_t*)(z_out + (i * K_SIZE + j));
            diff = (computed > expected) ? (computed - expected) : (expected - computed);
            if(diff > 0x0011){
                #if EVAL == 1
                if(y_id == 0)
                    printf("Error detected at coordinates[%d][%d]: Y=%x Z=%x", i, j, *(volatile uint16_t*)(y_inp+ (i * K_SIZE + j)), *(volatile uint16_t*)(z_out + (i * K_SIZE + j)));
                #endif
                errors++;
            }       
        }
    }
    printf("Number of errors: %d\n", errors);

    return errors;  
}