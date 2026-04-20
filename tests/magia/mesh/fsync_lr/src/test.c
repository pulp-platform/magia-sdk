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

#define WAIT_MODE WFE

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

    #if STALLING == 0
    eu_config_t eu_cfg = {.hartid = hartid};
    eu_controller_t eu_ctrl = {
        .base = NULL,
        .cfg = &eu_cfg,
        .api = &eu_api,
    };
    eu_init(&eu_ctrl);
    eu_fsync_init(&eu_ctrl, 0);
    eu_redmule_init(&eu_ctrl, 0);
    eu_idma_init(&eu_ctrl, 0);
    #endif

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
     * 
     * I was lazy and copypasted stuff from another test, just ignore this part...
     */
    volatile uint8_t timeslots = 2;
    volatile uint8_t t_size = K_SIZE / timeslots;

    /**
     * 2b. Initalize the IDMA transfer variables for input data-tile transfer
     */
    volatile uint32_t len_x = tile_w * 2;
    volatile uint32_t std_x = N_SIZE * 2;
    volatile uint32_t reps_x = (uint32_t) tile_h;
    volatile uint32_t obi_addr_x = (l1_tile_base);
    volatile uint32_t axi_addr_x = (uint32_t) x_inp;
    
    /**
     * 2b. Initalize the IDMA transfer variables for weight data-tile transfer.
     */
    volatile uint32_t len_w = (uint32_t) (t_size * 2);
    volatile uint32_t std_w = (uint32_t) (K_SIZE * 2);
    volatile uint32_t reps_w = (uint32_t) tile_w;
    volatile uint32_t obi_addr_w = obi_addr_x + (tile_h * tile_w * 2);
    volatile uint32_t axi_addr_w = (uint32_t) w_inp;

    /**
     * 2c. Initalize the IDMA transfer variables for output data-tile transfer.
     */
    volatile uint32_t len_y = (uint32_t) (t_size * 2);
    volatile uint32_t std_y = (uint32_t) (K_SIZE * 2);
    volatile uint32_t reps_y = (uint32_t) tile_h;
    volatile uint32_t obi_addr_y = obi_addr_w + (tile_w * t_size * 2);
    volatile uint32_t axi_addr_y = (uint32_t) y_inp;

    /**
     * 3. Load variables and then compute matmul
     */
    idma_memcpy_2d(&idma_ctrl, 0, axi_addr_x, obi_addr_x, len_x, std_x, reps_x);
    #if STALLING == 0
    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    #endif

    idma_memcpy_2d(&idma_ctrl, 0, axi_addr_w, obi_addr_w, len_w, std_w, reps_w);
    #if STALLING == 0
    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    #endif

    idma_memcpy_2d(&idma_ctrl, 0, axi_addr_y, obi_addr_y, len_y, std_y, reps_y);
    #if STALLING == 0
    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    #endif

    redmule_gemm(&redmule_ctrl, obi_addr_x, obi_addr_w, obi_addr_y, tile_h, tile_w, t_size);
    #if STALLING == 0
    eu_redmule_wait(&eu_ctrl, WAIT_MODE);
    #endif

    /**
     * 4. Epic stall based on id
     */
    wait_nop(hartid * 100);

    /**
     * 5. Check results with the neighbor on the right.
     */
    uint32_t errors=0;
    fsync_sync_row(&fsync_ctrl);
    #if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
    #endif

    uint16_t computed, expected, diff = 0;
    if(x_id != (MESH_X_TILES - 1)){
        uint32_t right_l1_base = get_l1_base(hartid + 1) + (tile_h * tile_w * 2) + (tile_w * t_size * 2);
        for(uint32_t i = 0; i < tile_h; i++){
            for(uint32_t j = 0; j < t_size; j++){
                computed = *(volatile uint16_t*)(obi_addr_y + (i * t_size * 2) + (j * 2));
                expected = *(volatile uint16_t*)(right_l1_base + (i * t_size * 2) + (j * 2));
                diff = (computed > expected) ? (computed - expected) : (expected - computed);
                if(diff > 0x0011){
                    #if EVAL == 1
                    if(y_id == 0)
                        printf("Error detected at coordinates[%d][%d]: Y=%x Z=%x\n", i, j, *(volatile uint16_t*)(y_inp+ (i * K_SIZE + j)), *(volatile uint16_t*)(z_out + (i * K_SIZE + j)));
                    #endif    
                    errors++;
                }       
            }
        }
        printf("Number of errors: %d\n", errors);
    }
    

    return errors;  
}