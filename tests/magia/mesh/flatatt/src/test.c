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
 * Flush
 */
int flush(uint32_t o, uint32_t dim){
    for(uint32_t i = 0; i < dim; i++)
        mmio16(o + i * 2) = 0x0000;
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

    idma_config_t           idma_cfg = {.hartid = hartid};
    idma_controller_t       idma_ctrl = {
        .base = NULL,
        .cfg = &idma_cfg,
        .api = &idma_api,
    };

    redmule_config_t        redmule_cfg = {.hartid = hartid};
    redmule_controller_t    redmule_ctrl = {
        .base = NULL,
        .cfg = &redmule_cfg,
        .api = &redmule_api,
    };

    fsync_config_t          fsync_cfg = {.hartid = hartid};
    fsync_controller_t      fsync_ctrl = {
        .base = NULL,
        .cfg = &fsync_cfg,
        .api = &fsync_api,
    };

    fsync_init(&fsync_ctrl);
    idma_init(&idma_ctrl);
    redmule_init(&redmule_ctrl);

    uint32_t y_id           = GET_Y_ID(hartid);
    uint32_t x_id           = GET_X_ID(hartid);
    uint32_t l1_tile_base   = get_l1_base(hartid);

    /**
     * 1a. Set arbitrary B value, dimension of the square block to be computed by the mesh at each cycle 
     * Calculate number of blocks, and dimension of the leftover block
     */
    uint32_t B_SIZE         = S_SIZE;
    uint32_t T              = S_SIZE / B_SIZE;

    /**
     * 1b. Set arbitrary timestep value to further divide the Q and K tiles in subtiles.
     */
    uint32_t n_timesteps    = 1;
    uint32_t t_size         = D_SIZE / n_timesteps;

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
    uint32_t    tile_h_max = ((B_SIZE + MESH_Y_TILES - 1) /  MESH_Y_TILES);
    uint32_t    tile_w_max = ((B_SIZE + MESH_X_TILES - 1) /  MESH_X_TILES);
    int32_t     tile_h;
    int32_t     tile_w;

    if(((tile_h_max * y_id) + tile_h_max) > B_SIZE)
        tile_h = B_SIZE - (tile_h_max * y_id);       
    else
        tile_h = tile_h_max;

    if(((tile_w_max * x_id) + tile_w_max) > B_SIZE)
        tile_w = B_SIZE - (tile_w_max * x_id);
    else
        tile_w = tile_w_max;

    /**
     * 1d. Initialize the L1 addresses for the tiles
     * Buffers point to one of these tiles, swap out current and prev buffer at each cycle, as each cycle requires
     * data from the previous one.
     */
    uint32_t obi_addr_m_0   = l1_tile_base;
    uint32_t obi_addr_m_1   = obi_addr_m_0  +   (tile_h * 2);
    uint32_t obi_addr_l_0   = obi_addr_m_1  +   (tile_h * 2);
    uint32_t obi_addr_l_1   = obi_addr_l_0  +   (tile_h * 2);
    uint32_t obi_addr_o_0   = obi_addr_l_1  +   (tile_h * 2);
    uint32_t obi_addr_o_1   = obi_addr_o_0  +   (tile_h * D_SIZE * 2);
    uint32_t obi_addr_q     = obi_addr_o_1  +   (tile_h * D_SIZE * 2);
    uint32_t obi_addr_k     = obi_addr_q    +   (tile_h * t_size * 2);
    uint32_t obi_addr_v     = obi_addr_k    +   (tile_w * t_size * 2);
    uint32_t obi_addr_s     = obi_addr_v    +   (tile_w * t_size * 2);
    uint32_t obi_addr_sb    = obi_addr_s    +   (tile_h * tile_w * 2);
    

    uint32_t max_buffer;
    uint32_t prev_max_buffer;
    uint32_t sum_buffer;
    uint32_t prev_sum_buffer;
    uint32_t output_buffer;
    uint32_t prev_output_buffer;

    /**
     * 1e. Initialize the L2 addresses for the tiles
     */
    uint32_t len_q      = t_size * 2;
    uint32_t std_q      = D_SIZE * 2;
    uint32_t reps_q     = (uint32_t) tile_h;
    uint32_t axi_addr_q = (uint32_t) q_inp + (y_id * tile_h_max * D_SIZE * 2);
    
    uint32_t len_k      = tile_w * 2;
    uint32_t std_k      = S_SIZE * 2;
    uint32_t reps_k     = (uint32_t) t_size;
    uint32_t axi_addr_k = (uint32_t) k_inp + (x_id * tile_w_max * 2);

    uint32_t len_v      = t_size * 2;
    uint32_t std_v      = D_SIZE * 2;
    uint32_t reps_v     = (uint32_t) tile_w;
    uint32_t axi_addr_v = (uint32_t) v_inp + (x_id * tile_w_max * D_SIZE * 2);

    /**
     * 2. Cycle over the blocks.
     *      i-cycle -> cycle over the blocks rows of the attention map
     *          j-cycle -> cycle over the blocks columns of the attention map
     */
    for(uint8_t i = 0; i < T; i++){
        /**
         * 2a. Flush the output buffers
         */
        flush(obi_addr_o_0, tile_h * D_SIZE);
        flush(obi_addr_o_1, tile_h * D_SIZE);

        axi_addr_q = axi_addr_q + (i * B_SIZE * D_SIZE * 2);

        /**
         * 3. Cycle over the blocks columns of the attention map
         */
        for(uint8_t j = 0; j < T; j++){
            /**
             * 3a. Set which buffers we are using, and which ones refers to the previous block. 
             */
            if(j % 2){
                max_buffer          = obi_addr_m_1;
                prev_max_buffer     = obi_addr_m_0;
                sum_buffer          = obi_addr_l_1;
                prev_sum_buffer     = obi_addr_l_0;
                output_buffer       = obi_addr_o_1;
                prev_output_buffer  = obi_addr_o_0;
            }
            else{
                max_buffer          = obi_addr_m_0;
                prev_max_buffer     = obi_addr_m_1;
                sum_buffer          = obi_addr_l_0;
                prev_sum_buffer     = obi_addr_l_1;
                output_buffer       = obi_addr_o_0;
                prev_output_buffer  = obi_addr_o_1;
            }

            /**
             * 3b. Output static matmul Q * Kt
             */
            for(uint32_t k = 0; k < t_size; k++){
                /**
                * 3ba. IDMA to load the input and weight data-tile for current timeslot
                */
                idma_memcpy_2d(&idma_ctrl, 0, (axi_addr_q + (t_size * k * 2)), obi_addr_q, len_q, std_q, reps_q);
                idma_wait();
                idma_memcpy_2d(&idma_ctrl, 0, (axi_addr_k + (j * B_SIZE * 2) + (t_size * S_SIZE * i * 2)), obi_addr_k, len_k, std_k, reps_k);
                idma_wait();

                /**
                * 3bb. Evoke the RED MULE 
                * https://www.youtube.com/watch?v=RG-bRbBuaBI&list=PLTLXyHxNV4azQtL26W-7l6fTrOa3rJgLo&index=35
                */
                redmule_gemm(&redmule_ctrl, obi_addr_q, obi_addr_k, obi_addr_s, (uint16_t) tile_h, (uint16_t) t_size, (uint16_t) tile_w);
                redmule_wait();
            }

            /**
             * 3c. Find row maxes.
             * Each tile compares with the max of the previous tile, and propagates them to the following tile.
             * When reaching the rightmost tile, compare with the maxes of the previous block.
             * Then, propagate the values back to the entire row.
             */
            rowmax(obi_addr_s, max_buffer, tile_h, tile_w);
            if(x_id != 0){
                fsync_sync_left(&fsync_ctrl);
                if(j % 2)
                    max_compare(max_buffer, get_l1_base(hartid - 1) + (tile_h * 2), tile_h);
                else
                    max_compare(max_buffer, get_l1_base(hartid - 1), tile_h);
            }   
            if(x_id != (MESH_X_TILES - 1))
                fsync_sync_right(&fsync_ctrl);
            else if(j > 0)
                max_compare(max_buffer, prev_max_buffer, tile_h);
            fsync_sync_row(&fsync_ctrl);
            if(x_id != (MESH_X_TILES - 1)){
                if(j % 2)
                    max_compare(max_buffer, get_l1_base(GET_ID(y_id, (MESH_X_TILES - 1))) + (tile_h * 2), tile_h);
                else
                    max_compare(max_buffer, get_l1_base(GET_ID(y_id, (MESH_X_TILES - 1))), tile_h);
            }
            fsync_sync_row(&fsync_ctrl);

            /**
             * 3d. Element wise substraction for each row with their maximum
             */
            rowdiff(obi_addr_s, max_buffer, tile_h, tile_w);

            /**
             * 3e. Exponential on the scores
             */
            exponential(obi_addr_s, (uint32_t) tile_h, (uint32_t) tile_w);

            /**
             * 3f. Row wise summation of the elements.
             * After summing, add the contribution of the previous tile and send them forward.
             * When reaching the end, "broadcast" it back.
             */
            rowsum(obi_addr_s, sum_buffer, tile_h, tile_w);
            if(x_id != 0){
                fsync_sync_left(&fsync_ctrl);
                if(j % 2)
                    vect_sum(sum_buffer, get_l1_base(hartid - 1) + (tile_h * 6), tile_h);
                else
                    vect_sum(sum_buffer, get_l1_base(hartid - 1) + (tile_h * 4), tile_h);
            }
            if(x_id != (MESH_X_TILES - 1))
                fsync_sync_right(&fsync_ctrl);
            fsync_sync_row(&fsync_ctrl);
            if(x_id != (MESH_X_TILES - 1)){
                if(j % 2)
                    idma_memcpy_1d(&idma_ctrl, 0, get_l1_base(GET_ID(y_id, (MESH_X_TILES - 1))) + (tile_h * 6), sum_buffer, tile_h * 2);
                else
                    idma_memcpy_1d(&idma_ctrl, 0, get_l1_base(GET_ID(y_id, (MESH_X_TILES - 1))) + (tile_h * 4), sum_buffer, tile_h * 2);
                idma_wait();
            }
            fsync_sync_row(&fsync_ctrl);

            /**
             * 3g. Add retroactive contribution of previous blocks
             */
            if(j > 0){
                /**
                 * 3ga. Do the difference between current block's maxes vector and the previous one.
                 * You'll get a bunch of zeroes if the previous block had bigger maxes 
                 * (because they replaced the current block's in step 3d)
                 */
                vect_diff(prev_max_buffer, max_buffer, tile_h);

                /**
                 * 3gb. Exponential of the max difference.
                 * 1 in the rows dimension as the input is a vector.
                 */
                exponential(prev_max_buffer, 1, (uint32_t) tile_h);

                /**
                 * 3gc. Element-wise multiply the exponential with the previous block's sum vector.
                 */
                vect_prod(prev_sum_buffer, prev_max_buffer, tile_h);

                /**
                 * 3gd. Sum previous block's sum on current block's sum
                 */
                vect_sum(sum_buffer, prev_sum_buffer, tile_h);

                /**
                 * 3ge. Row-wise divide each row of the previous block's output by the exponential of the difference
                 * between the current block rows' maxes and the previous block's ones.
                 * (Sadly, I'm serious.)
                 */
                rowdiv(prev_output_buffer, prev_max_buffer, tile_h, D_SIZE);
            }

            /**
             * 3h. Input static Activation * V
             */
            for(uint32_t k = 0; k < n_timesteps; k++){
                /**
                * 3ha. Load V data-tile required for the j-th block column and k-th timestep
                */
                idma_memcpy_2d(&idma_ctrl, 0, axi_addr_v + (j * B_SIZE * D_SIZE * 2) + (k * t_size * 2), obi_addr_v, len_v, std_v, reps_v);
                idma_wait();

                /**
                 * 3hb. Evoke REDMULE
                 * https://www.youtube.com/watch?v=xDbIDKel-O4
                 */
                redmule_gemm(&redmule_ctrl, obi_addr_s, obi_addr_v, obi_addr_sb, (uint16_t) tile_h, (uint16_t) tile_w, (uint16_t) t_size);
                redmule_wait();

                /**
                 * 3hc. Strided store of the current timestep buffer in the output
                 */
                idma_memcpy_2d(&idma_ctrl, 0, obi_addr_sb, output_buffer + (k * t_size * 2), t_size * 2, D_SIZE * 2, tile_h);
                idma_wait();
            }
            

            /**
             * 3j. Add in the previous blocks' contribution
             */
            if(j > 0)
                vect_sum(output_buffer, prev_output_buffer, tile_h * D_SIZE);
        }
        
        /**
         * 4. Divide the output buffer by the accumulated sum buffer
         */
        rowdiv(output_buffer, sum_buffer, tile_h, D_SIZE);

        /**
         * 5. Propagate and sum all the output buffers in a systolic way
         */
        if(x_id != 0){
            fsync_sync_left(&fsync_ctrl);
            if(T % 2)
                vect_sum(output_buffer, get_l1_base(hartid - 1) + (8 * tile_h), tile_h * D_SIZE);
            else
                vect_sum(output_buffer, get_l1_base(hartid - 1) + (8 * tile_h + D_SIZE * tile_h), tile_h * D_SIZE);
        }
        if(x_id != (MESH_X_TILES - 1))
            fsync_sync_right(&fsync_ctrl);
        else{
            idma_memcpy_2d(&idma_ctrl, 1, o_out, output_buffer, D_SIZE, D_SIZE, tile_h);
            idma_wait();
        }
        fsync_sync_row(&fsync_ctrl);
    }

    magia_return(hartid, PASS_EXIT_CODE);
    return 0;  
}