// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Alberto Dequino <alberto.dequino@unibo.it>

#include <stdint.h>
#include "test.h"

#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"

/**
 * This test aims to verify the functionality of MAGIA as a systolic array for matrix multiplications,
 * following the output-static mechanism. 
 */
int main(void){
    /** 
     * 0. Get the mesh-tile's hartid, mesh-tile coordinates and define its L1 base, 
     * also initialize the controllers for the idma and fsync.
     */
    uint32_t hartid = get_hartid();

    idma_config_t idma_cfg = {.hartid = hartid};
    idma_controller_t idma_ctrl = {
        .base = NULL,
        .cfg = &idma_cfg,
        .api = &idma_api,
    };

    idma_init(&idma_ctrl);

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

    #if STALLING == 0
    eu_config_t eu_cfg = {.hartid = hartid};
    eu_controller_t eu_ctrl = {
        .base = NULL,
        .cfg = &eu_cfg,
        .api = &eu_api,
    };

    eu_init(&eu_ctrl);
    eu_clear_events(0xFFFFFFFF);
    eu_fsync_init(&eu_ctrl, 0);
    eu_idma_init(&eu_ctrl, 0);
    #endif

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
        //printf("ID:%d, Mesh-Tile-X:%d, Mesh-Tile-Y:%d, Data-Tile w: %d, Data-Tile h: %d", hartid, x_id, y_id, tile_w, tile_h);
    }

    /**
     * 2. Use IDMA to transfer output data-tile in L1 memory
     */
    uint32_t len = tile_w * 2;
    uint32_t std = K_SIZE * 2;
    uint32_t reps = (uint32_t) tile_h;
    uint32_t obi_addr = (l1_tile_base);
    uint32_t axi_addr_z = (uint32_t) z_out + (y_id * K_SIZE * tile_h_max * 2) + (tile_w_max * x_id * 2); 
    uint32_t axi_addr_y = (uint32_t) y_inp + (y_id * K_SIZE * tile_h_max * 2) + (tile_w_max * x_id * 2); 

    for(int i = 0; i < reps; i++){
        idma_memcpy_1d(&idma_ctrl, 0, axi_addr_z + ((std) * i), (l1_tile_base) + (len * i), len);
        #if STALLING == 0
        eu_idma_wait_a2o(&eu_ctrl, POLLING);
        #endif
    }

    /**
     * 3. Use IDMA to write the L1 data in the input vector in L2.
     */
    for(int i = 0; i < reps; i++){
        idma_memcpy_1d(&idma_ctrl, 1, axi_addr_y + ((std) * i), (l1_tile_base) + (len * i), len);
        #if STALLING == 0
        eu_idma_wait_o2a(&eu_ctrl, POLLING);
        #endif
    }

    /**
     * 4. Wait that all the tiles have finished
     */
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    #if STALLING == 0
    eu_fsync_wait(&eu_ctrl, POLLING);
    #endif


    /**
     * 5. Check results
     */
    uint32_t errors=0;
    if(hartid==0){
        uint16_t computed, expected, diff = 0;
        for(uint8_t i = 0; i < M_SIZE; i++){
            for(uint8_t j = 0; j < K_SIZE; j++){
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
    }
    
    return errors;  
}