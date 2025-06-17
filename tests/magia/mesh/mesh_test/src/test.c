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
 * This test splits a GEMM into the N^2 tiles of the NxN MAGIA mesh architecture and then checks its correctness.
 * This test has been written to be compatible to any mesh and input dimensions.
 * The workload has been split as evenly as possible on each tile, dividing it on the output dimensions.
 * I.E. Each tile of a 2x2 computes one of the four corners of the output matrix (tile 0 the top left corner, tile 3 the bottom right).
 */
int main(void) {
    /** 
     * 0. Get the tile's hartid, mesh coordinates and define its L1 base, also initialize the controllers for idma and redmule.
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
    uint32_t L1_TILE_BASE = L1_BASE + hartid * L1_TILE_OFFSET;

    /**
     * 1. Calculate how many lines and columns the tile has to compute.
     * If M and K are multiple of the number of tiles in both axis, tile_h and tile_w are the same for each tile.
     * Otherwise, these dimensions will be different for the tiles closer to the right and lower border of the mesh. 
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

    if(((tile_w_max * x_id) + tile_w_max) > K_SIZE){
        tile_w = K_SIZE - (tile_w_max * x_id);
    }    
    else{
        tile_w = tile_w_max;
    }

    if(tile_h < 1 || tile_w < 1){
        return 0;
    }
    else{
        printf("ID:%d, Tile-X:%d, Tile-Y:%d, Tile w: %d, tile h: %d", hartid, x_id, y_id, tile_w, tile_h);
    }

    /** 
     * 2. Set and do the 2D AXI to OBI (L2 to L1) transfers for each tile
     * Three transfers are necessary, for X, W, and Y.
     * We use the hartid and tile coordinates to calculate the correct addresses and strides.
     * len and std are byte-sizes, so we multiply by 2 since we are working with 16-bit data.
     */
    uint32_t len_x = N_SIZE * 2;
    uint32_t std_x = N_SIZE * 2;
    uint32_t reps_x = (uint32_t) tile_h;
    uint32_t obi_addr_x = (L1_TILE_BASE);
    uint32_t axi_addr_x = (uint32_t) x_inp + (y_id * N_SIZE * tile_h_max * 2); 
    
    idma_memcpy_2d(&idma_ctrl, 0, axi_addr_x, obi_addr_x, len_x, std_x, reps_x);

    uint32_t len_w = (uint32_t) (tile_w * 2);
    uint32_t std_w = K_SIZE * 2;
    uint32_t reps_w = N_SIZE;
    uint32_t obi_addr_w = (L1_TILE_BASE + (len_x * reps_x));
    uint32_t axi_addr_w = (uint32_t) w_inp + (x_id * tile_w_max * 2); 
    
    idma_memcpy_2d(&idma_ctrl, 0, axi_addr_w, obi_addr_w, len_w, std_w, reps_w);

    uint32_t len_y = (uint32_t) tile_w * 2;
    uint32_t std_y = K_SIZE * 2;
    uint32_t reps_y = (uint32_t) tile_h;
    uint32_t obi_addr_y = (L1_TILE_BASE + (len_x * reps_x) + (len_w * reps_w));
    uint32_t axi_addr_y = (uint32_t) y_inp + (x_id * tile_w_max * 2) + (y_id * K_SIZE * tile_h_max * 2); 
    
    idma_memcpy_2d(&idma_ctrl, 0, axi_addr_y, obi_addr_y, len_y, std_y, reps_y);

    /** 
     * 3. Set and do the GEMM on Redmule for each tile.
     */
    redmule_gemm(&redmule_ctrl, obi_addr_x, obi_addr_w, obi_addr_y, (uint16_t) tile_h, (uint16_t) N_SIZE, (uint16_t) tile_w);

    /** 
     * 4. Copy the result back to L2.
     */
    idma_memcpy_2d(&idma_ctrl, 1, axi_addr_y, obi_addr_y, len_y, std_y, reps_y);

    /**
     * 5. Check results.
     */
    uint32_t errors=0;
    uint16_t computed, expected, diff = 0;
    uint32_t axi_addr_z = (uint32_t) z_out + (x_id * tile_w_max * 2) + (y_id * K_SIZE * tile_h_max * 2);
    for(uint8_t i = 0; i < tile_h; i++){
        for(uint8_t j = 0; j < tile_w; j++){
            computed = *(volatile uint16_t*)(axi_addr_y + (i * K_SIZE + j) * 2);
            expected = *(volatile uint16_t*)(axi_addr_z + (i * K_SIZE + j) * 2);
            diff = (computed > expected) ? (computed - expected) : (expected - computed);
            if(diff > 0x0011){
                //printf("ERROR DETECTED!\n");
                if(hartid==0){
                    printf("Error detected at coordinates[%d][%d]: Y=%x Z=%x", (tile_h_max * y_id + i), (tile_w_max * x_id + j), *(volatile uint16_t*)(axi_addr_y + (i * K_SIZE + j) * 2), *(volatile uint16_t*)(axi_addr_z + (i * K_SIZE + j) * 2));
                }
                errors++;
            }
        }
    }

    printf("Number of errors: %d", errors);


    magia_return(hartid, PASS_EXIT_CODE);
    return 0;
}