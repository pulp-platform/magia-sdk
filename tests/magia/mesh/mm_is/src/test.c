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

/**
 * This test aims to verify the functionality of MAGIA as a systolic array for matrix multiplications,
 * following the input-static mechanism. 
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

    /**
     * 1. Calculate the static input data-tile dimensions.
     * If M and N are perfect multiples of the number of mesh-tiles in their respective axis, 
     * tile_h and tile_w will be the same for each mesh-tile.
     * Otherwise, these dimensions will be smaller for the mesh-tiles closer to the right and lower 
     * border of the mesh. 
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
     * The final L1 memory requirement will be:
     * Input data-tile: (tile_h x tile_w) * data_dim
     * Weight data-tile: (tile_w x t_size) * data_dim
     * Output data-tile: ((tile_h x t_size) * data_dim) * 2 (Double buffering)
     */
    uint8_t timeslots = 16;
    uint8_t t_size = K_SIZE / timeslots;

    /**
     * 2. Use IDMA to transfer static input data-tile
     */
    uint32_t len_x = tile_w * 2;
    uint32_t std_x = N_SIZE * 2;
    uint32_t reps_x = (uint32_t) tile_h;
    uint32_t obi_addr_x = (l1_tile_base);
    uint32_t axi_addr_x = (uint32_t) x_inp + (y_id * N_SIZE * tile_h_max * 2) + (tile_w_max * x_id * 2); 
    
    idma_memcpy_2d(&idma_ctrl, 0, axi_addr_x, obi_addr_x, len_x, std_x, reps_x);
    
    /**
     * 2a. Initalize the IDMA transfer variables for weight data-tile transfers.
     */
    uint32_t len_w = (uint32_t) (t_size * 2);
    uint32_t std_w = (uint32_t) (K_SIZE * 2);
    uint32_t reps_w = (uint32_t) tile_w;
    uint32_t obi_addr_w = obi_addr_x + (tile_h * tile_w * 2);
    uint32_t axi_addr_w = (uint32_t) w_inp + (x_id * K_SIZE * tile_w_max * 2);
    
    /**
     * 2b. Initialize the obi addresses for the output buffers.
     */
    uint32_t obi_addr_y_0 = obi_addr_w + (tile_w * t_size * 2);
    uint32_t obi_addr_y_1 = obi_addr_y_0 + (tile_h * t_size * 2);

    /**
     * 3. Cycle over the timeslots.
     * For each timeslot, the mesh-tile will:
     * a - Load the weight data-tile for the current timeslot
     * b - Load the output data-tile for the current timeslot
     * c - Multiply and add
     * d - Store the output data-tile
     * The output data-tile is loaded from the previous mesh-tile, and stored in the next one.
     * Synchronization is required.
     * If the mesh-tile is the leftmost of the row: output data-tile is loaded from L2 memory.
     * If the mesh-tile is the rightmost of the row: output data-tile is stored in L2 memory.
     */
    for(uint8_t i = 0; i < timeslots; i++){
        /**
         * 3a. IDMA to load the weight data-tile for current timeslot
         */
        idma_memcpy_2d(&idma_ctrl, 0, (axi_addr_w + (t_size * i * 2)), obi_addr_w, len_w, std_w, reps_w);

        /**
         * 3b. Load the output data-tile
         * If leftmost mesh-tile: load from L2 (IDMA transfer)
         * Else: sync from the previous tile, then copy its L1 buffer.
         * 0 and even timeslots: load in buffer 0; odd timeslots: load in buffer 1.
         */
        if(x_id == 0){
            //if(hartid == 0)
                //printf("Storing the output data-tile...");
            uint32_t len_y = (uint32_t) (t_size * 2);
            uint32_t std_y = (uint32_t) (K_SIZE * 2);
            uint32_t reps_y = (uint32_t) tile_h;
            uint32_t axi_addr_y = (uint32_t) y_inp + (y_id * K_SIZE * tile_h_max * 2) + (i * t_size * 2);

            if(i % 2)
                idma_memcpy_2d(&idma_ctrl, 0, axi_addr_y, obi_addr_y_1, len_y, std_y, reps_y);
            else
                idma_memcpy_2d(&idma_ctrl, 0, axi_addr_y, obi_addr_y_0, len_y, std_y, reps_y);
            //printf("Loaded data from L2: %x, %x, %x, %x", *(volatile uint16_t*)(obi_addr_y), *(volatile uint16_t*)(obi_addr_y + 2), *(volatile uint16_t*)(obi_addr_y + 4), *(volatile uint16_t*)(obi_addr_y + 6));
        }
        else{
            if(fsync_sync_left(&fsync_ctrl))
                printf("Error when synchronizing with left tile.");

            if(i % 2){
                uint32_t src_addr = get_l1_base(hartid - 1) + (tile_h_max * tile_w_max * 2) + (tile_w_max * t_size * 2) + (tile_h_max * t_size * 2);
                idma_memcpy_1d(&idma_ctrl, 0, src_addr, obi_addr_y_1, tile_h * t_size * 2);
            }                
            else{
                uint32_t src_addr = get_l1_base(hartid - 1) + (tile_h_max * tile_w_max * 2) + (tile_w_max * t_size * 2);
                idma_memcpy_1d(&idma_ctrl, 0, src_addr, obi_addr_y_0, tile_h * t_size * 2);
            }
            //printf("Received this data: %x, %x, %x, %x", *(volatile uint16_t*)(obi_addr_y), *(volatile uint16_t*)(obi_addr_y + 2), *(volatile uint16_t*)(obi_addr_y + 4), *(volatile uint16_t*)(obi_addr_y + 6));
        }
        
        /**
         * 3c. Evoke the RED MULE 
         * https://www.youtube.com/watch?v=RG-bRbBuaBI&list=PLTLXyHxNV4azQtL26W-7l6fTrOa3rJgLo&index=35
         */
        if(i % 2)
            redmule_gemm(&redmule_ctrl, obi_addr_x, obi_addr_w, obi_addr_y_1, (uint16_t) tile_h, (uint16_t) tile_w, (uint16_t) t_size);
        else
            redmule_gemm(&redmule_ctrl, obi_addr_x, obi_addr_w, obi_addr_y_0, (uint16_t) tile_h, (uint16_t) tile_w, (uint16_t) t_size);

        /**
         * 3d. Sync with the next tile to ready the data.
         * On the rightmost tile, store in L2 memory instead.
         */
        if(x_id == (MESH_X_TILES-1)){
            uint32_t len_y = (uint32_t) (t_size * 2);
            uint32_t std_y = (uint32_t) (K_SIZE * 2);
            uint32_t reps_y = (uint32_t) tile_h;
            uint32_t axi_addr_y = (uint32_t) y_inp + (y_id * K_SIZE * tile_h_max * 2) + (i * t_size * 2);

            if(i % 2)
                idma_memcpy_2d(&idma_ctrl, 1, axi_addr_y, obi_addr_y_1, len_y, std_y, reps_y);
            else
                idma_memcpy_2d(&idma_ctrl, 1, axi_addr_y, obi_addr_y_0, len_y, std_y, reps_y);   
        }
        else{
            //printf("Sending this data: %x, %x, %x, %x", *(volatile uint16_t*)(obi_addr_y), *(volatile uint16_t*)(obi_addr_y + 2), *(volatile uint16_t*)(obi_addr_y + 4), *(volatile uint16_t*)(obi_addr_y + 6));
            if(fsync_sync_right(&fsync_ctrl))
                printf("Error when synchronizing with right tile");
        }
    }

    /**
     * 5. Check results
     */
    fsync_sync_row(&fsync_ctrl);
    if(x_id == MESH_X_TILES - 1){
        uint32_t errors=0;
        uint16_t computed, expected, diff = 0;
        for(uint8_t i = (y_id * tile_h_max); i < (y_id * tile_h_max + tile_h); i++){
            for(uint8_t j = 0; j < K_SIZE; j++){
                computed = *(volatile uint16_t*)(y_inp + (i * K_SIZE + j));
                expected = *(volatile uint16_t*)(z_out + (i * K_SIZE + j));
                diff = (computed > expected) ? (computed - expected) : (expected - computed);
                if(diff > 0x0011){
                    //if(y_id == 0)
                        //printf("Error detected at coordinates[%d][%d]: Y=%x Z=%x", i, j, *(volatile uint16_t*)(y_inp+ (i * K_SIZE + j)), *(volatile uint16_t*)(z_out + (i * K_SIZE + j)));
                    errors++;
                }       
            }
        }
        printf("Number of errors: %d", errors);
    }

    magia_return(hartid, PASS_EXIT_CODE);
    return 0;  
}