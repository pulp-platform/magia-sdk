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
 * Element-wise comparison of the max vectors.
 * Saves in the curr buffer the bigger values.
 */
int max_compare(uint32_t prev, uint32_t curr, uint32_t dim){
    for(uint32_t i = 0; i < dim; i++){
        if((*(volatile uint16_t*)(prev + (i * 2))) > (*(volatile uint16_t*)(curr + (i * 2))))
            mmio16(curr + (i * 2)) = (*(volatile uint16_t*)(prev + (i * 2)));
    }
}

/**
 * Finds the max value for each row and saves it in the result in the maxes buffer.
 */
int rowmax(uint32_t s, uint32_t maxes, uint32_t dim_h, uint32_t dim_w){
    for(uint32_t i = 0; i < dim_h; i++){
        uint32_t row = s + i * dim_w * 2;
        uint16_t rowmax = 0;
        for(uint32_t j = 0; j < dim_w; j++){
            if((*(volatile uint16_t*)(row + j * 2)) > rowmax)
                rowmax = *(volatile uint16_t*)(row + j * 2);
        }
        mmio16(maxes + i * 2) = rowmax;
    }
}

/**
 * For each row i of the input h x w matrix "s", substract the i-th element of the "m" vector. 
 */
int rowdiff(uint32_t s, uint32_t m, uint32_t h, uint32_t w){
    for(uint32_t i = 0; i < h; i++){
        uint32_t row = s + i * w * 2;
        uint16_t diff = *(volatile uint16_t*)(m + i * 2);
        for(uint32_t j = 0; j < w; j++){
            mmio16(row + j * 2) = (*(volatile uint16_t*)(row + j * 2)) - diff;
        }
    }
}

/**
 * Sum the rows
 */
int rowsum(uint32_t s, uint32_t l, uint32_t h, uint32_t w){
    for(uint32_t i = 0; i < h; i++){
        uint32_t row = s + i * 2 * w;
        uint16_t sum = 0;
        for(uint32_t j = 0; j < w; j++){
            sum = sum + *(volatile uint16_t*)(row + j * 2);
        }
        mmio16(l + i * 2) = sum;
    }
}

/**
 * Element wise sum of v2 into v1
 */
int vect_sum(uint32_t v1, uint32_t v2, uint32_t dim){
    for(uint32_t i = 0; i < dim; i++){
        mmio16(v1 + i * 2) = *(volatile uint16_t*)(v1 + i * 2) + *(volatile uint16_t*)(v2 + i * 2);
    }
}

/**
 * This test aims to verify the functionality of MAGIA as a tile group for the FlatAttention algorithm.
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
     * 1a. Set arbitrary B value, dimension of the square block to be computed by the mesh at each cycle 
     * Calculate number of blocks, and dimension of the leftover block
     */
    uint32_t B_SIZE = 16;
    uint32_t T = S_SIZE / B_SIZE;

    /** TODO: if S_SIZE is not a multiple of B_SIZE, we should do something for the leftover.
     * "We" is most likely Future Me.
     * Fuck you, Future Me! - signed Past Me.
     * Here is my only help: this is the size of the leftover block, the rest is up to YOU! Who is probably ME!
     * But if you are not me, tell me to get a girlfriend.
     * 
     * uint32_t B_leftover = S_SIZE % B_SIZE;
     */

    /**
     * 1c. Calculate tile_h and tile_w
     */
    uint32_t tile_h_max = ((B_SIZE + MESH_Y_TILES - 1) /  MESH_Y_TILES);
    uint32_t tile_w_max = ((B_SIZE + MESH_X_TILES - 1) /  MESH_X_TILES);
    int32_t tile_h;
    int32_t tile_w;

    if(((tile_h_max * y_id) + tile_h_max) > B_SIZE){
        tile_h = B_SIZE - (tile_h_max * y_id);    
    }    
    else{
        tile_h = tile_h_max;
    }

    if(((tile_w_max * x_id) + tile_w_max) > B_SIZE){
        tile_w = B_SIZE - (tile_w_max * x_id);
    }    
    else{
        tile_w = tile_w_max;
    }

    /**
     * 1d. Initialize the L1 addresses for the tiles
     */
    uint32_t obi_addr_m = (l1_tile_base);
    uint32_t obi_addr_q = obi_addr_m + (tile_h * 2);
    uint32_t obi_addr_k = obi_addr_q + (tile_h * D_SIZE * 2);
    uint32_t obi_addr_s = obi_addr_k + (tile_w * D_SIZE * 2);
    uint32_t obi_addr_m_prev = obi_addr_s + (tile_h * tile_w * 2);
    uint32_t obi_addr_l = obi_addr_m_prev + (tile_h * 2);

    /**
     * 1e. Initialize the L2 addresses for the tiles
     */
    uint32_t len_q = D_SIZE * 2;
    uint32_t std_q = D_SIZE * 2;
    uint32_t reps_q = (uint32_t) tile_h;
    uint32_t axi_addr_q = (uint32_t) q_inp + (y_id * tile_h_max * D_SIZE * 2);
    
    uint32_t len_k = tile_w * 2;
    uint32_t std_k = S_SIZE * 2;
    uint32_t reps_k = (uint32_t) D_SIZE;
    uint32_t axi_addr_k = (uint32_t) k_inp + (x_id * tile_w_max * 2);

    /**
     * 2. Cycle over the blocks.
     *      i-cycle -> cycle over the blocks rows of the attention map
     *          j-cycle -> cycle over the blocks columns of the attention map
     */
    for(uint8_t i = 0; i < T; i++){
        /**
         * 2a. Flush the output buffer
         * TODO: Actually code flush lmao
         */
        //flush(obi_addr_o, tile_dim * tile_dim);

        /**
         * 2b. Load the Q data-tile required for the i-th block row
         */
        idma_memcpy_2d(&idma_ctrl, 0, axi_addr_q + (i * B_SIZE * D_SIZE * 2), obi_addr_q, len_q, std_q, reps_q);
        idma_wait(&idma_ctrl);

        /**
         * 3. Cycle over the blocks columns of the attention map
         */
        for(uint8_t j = 0; j < T; j++){
            /**
             * 3b. Load the K (transposed) data-tile required for the j-th block column
             */
            idma_memcpy_2d(&idma_ctrl, 0, axi_addr_k + (j * B_SIZE * 2), obi_addr_k, len_k, std_k, reps_k);
            idma_wait(&idma_ctrl);

            /**
             * 3c. Q * Kt
             */
            redmule_gemm(&redmule_ctrl, obi_addr_q, obi_addr_k, obi_addr_s, (uint16_t) tile_h, (uint16_t) D_SIZE, (uint16_t) tile_w);
            redmule_wait(&redmule_ctrl);

            /**
             * 3d. Find row maxes.
             * Each tile compares with the max of the previous tile, and propagates them to the following tile.
             * When reaching the rightmost tile, propagate the values back to the entire row.
             */
            rowmax(obi_addr_s, obi_addr_m, tile_h, tile_w);
            if(x_id != 0){
                fsync_sync_left(&fsync_ctrl);
                max_compare(get_l1_base(hartid - 1), obi_addr_m, tile_h);
            }   
            if(x_id != (MESH_X_TILES - 1)){
                fsync_sync_right(&fsync_ctrl);
            }
            fsync_sync_row(&fsync_ctrl);
            if(x_id != (MESH_X_TILES - 1))
                max_compare(get_l1_base(GET_ID(y_id, (MESH_X_TILES - 1))), obi_addr_m, tile_h);

            /**
             * 3e. Check if the max of the current block is lower than the max calculated in a previous block and update if so.
             */
            if(j > 0){
                max_compare(obi_addr_m_prev, obi_addr_m, tile_h);
            }

            /**
             * 3f. Element wise substraction for each row with their maximum
             */
            rowdiff(obi_addr_s, obi_addr_m, tile_h, tile_w);

            /**
             * 3g. Exponential on the scores
             * TODO: Actually implement the exponential function lol
             */
            exponential(obi_addr_s, tile_h, tile_w);

            /**
             * 3h. Row wise summation of the elements.
             * After summing, add the contribution of the previous tile and send them forward.
             * When reaching the end, "broadcast" it back.
             */
            rowsum(obi_addr_s, obi_addr_l, tile_h, tile_w);

            if(x_id != 0){
                fsync_sync_left(&fsync_ctrl);
                vect_sum(obi_addr_l, get_l1_base(hartid - 1) + 2 * ((tile_h * 2) + D_SIZE * (tile_h + tile_w_max) + (tile_h * tile_w_max)));
            }
            if(x_id != (MESH_X_TILES - 1)){
                fsync_sync_right(&fsync_ctrl);
            }
            fsync_sync_row(&fsync_ctrl);
            if(x_id != (MESH_X_TILES - 1))
                idma_memcpy_1d(&idma_ctrl, 0, get_l1_base(GET_ID(y_id, (MESH_X_TILES - 1))) + 2 * ((tile_h * 2) + D_SIZE * (tile_h + tile_w_max) + (tile_h * tile_w_max)), obi_addr_l, tile_h * 2);


        }
    }

    magia_return(hartid, PASS_EXIT_CODE);
    return 0;  
}