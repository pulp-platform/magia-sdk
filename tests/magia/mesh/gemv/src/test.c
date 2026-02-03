// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Victor Isachi <victor.isachi@unibo.it>
// Alberto Dequino <alberto.dequino@unibo.it>

#include <stdint.h>

//#define BASELINE_K2
#define K_LOGN

#include "tile.h"
#include "idma.h"
#include "redmule.h"
#include "fsync.h"
#include "eventunit.h"

#define WAIT_MODE WFE

#include "test.h"

/**
 * This test implements the optimal GeMV algorithm for MAGIA (and for mesh-based architectures in general) using FractalSync for synchronization. 
 * See "WaferLLM: Large Language Model Inference at Wafer Scale" paper.
 */
int main(void){
    // sentinel_start();   // Total execution

    /** 
     * 0. Get the mesh-tile's hartid, mesh-tile coordinates and define its L1 base. 
     * Initialize the controllers for the idma, redmule, fsync and event unit.
     * Move in identity matrix.
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

    fsync_config_t fsync_cfg = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {
        .base = NULL,
        .cfg = &fsync_cfg,
        .api = &fsync_api,
    };
    fsync_init(&fsync_ctrl);

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

    uint32_t y_id = GET_Y_ID(hartid);
    uint32_t x_id = GET_X_ID(hartid);
    uint32_t l1_tile_base = get_l1_base(hartid);

    /**
     * 0. I'll try to explain in the easiest way possible.
     * 
     * Basically, we want to add all the contributions of the tiles on each row in a "parallelized" 
     * fashion, instead of computing the partial result and then propagate to the next tile.
     * 
     * So every tile computes a different contribution of the Gemv. 
     * After a first step, ALL the contributions for the gemv are *somewhere* on the mesh, 
     * but need to be summed and reordered to have the final result.
     * 
     * Hence, we divide this process in phases.
     * 
     * "reduce_phases" stands for how many phases of this bullshit are required 
     * to have the complete result.
     * It's basically how many times the outer loop is called.
     * 
     * "reduce_degree" is a constant. 
     * It is used for finding out on which row's tiles the partial results are gonna be stored.
     * For each phase, these tiles change!
     * In the last phase, all the contributions will end up in the leftmost tiles of the mesh.
     * It also says how many tiles are summing their contributions with each other, in each phase.
     * 
     * In the "BASELINE_K2" mode, we ALWAYS do 2 phases. 
     * (unless we are working on a 2x2 mesh, in which case we just need 1 phase)
     * So in bigger meshes, in each phase we will sum 
     * the contribution of many tiles on certain tiles, making the workload imbalanced.
     * 
     * In the "K_LOGN" mode, in each phase we sum the contributions in parallel "pairs".
     * This will increase the number of phases, but the workload for each phase
     * will be much lighter and broadly distributed.  
     */
    #if defined(BASELINE_K2)
    uint32_t reduce_degree = (MESH_2_POWER == 1 ? 2 : MESH_2_POWER);
    uint32_t reduce_phases = (MESH_2_POWER == 1 ? 1 : 2);
    #elif defined(K_LOGN)
    uint32_t reduce_degree = 2;
    uint32_t reduce_phases = MESH_2_POWER;
    #endif

    /**
     * 1. Calculate blocking dimensions.
     * It is assumed that blocks can be equally divided among all tiles. 
     */
    uint32_t tile_h = N_SIZE/MESH_X_TILES;
    uint32_t tile_w = K_SIZE/MESH_Y_TILES;
    uint32_t tile_m = M_SIZE;

    // printf("Blocking dimensions: height %0d, width %0d\n", tile_h, tile_w);

    /**
     * 2. Use iDMA to transfer indentity matrix.
     */
    uint32_t len_id  = tile_w * 2;
    uint32_t std_id  = K_SIZE * 2;
    uint32_t reps_id = tile_w;
    uint32_t obi_addr_id = (l1_tile_base);
    uint32_t axi_addr_id = (uint32_t) id_mat; 

    idma_memcpy_2d(&idma_ctrl, 0, axi_addr_id, obi_addr_id, len_id, std_id, reps_id);
    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

    /**
     * 2a. Use iDMA to transfer bias blocks.
     * To avoid accumulating the bias multiple times only one tile per row fetches it, the rest fetch the 0 matrix.
     * The leftmost tile is the root of the reduction tree so it fetches the bias.
     */
    uint32_t len_y = tile_w * 2;
    uint32_t obi_addr_y = obi_addr_id + (tile_w * tile_w * 2);
    uint32_t axi_addr_y = (x_id == 0) ? (uint32_t) y_in + (y_id * tile_w * 2) : (uint32_t) y_out + (y_id * tile_w * 2);
    
    idma_memcpy_1d(&idma_ctrl, 0, axi_addr_y, obi_addr_y, len_y);
    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

    /**
     * 2b. Use iDMA to transfer weight matrix blocks.
     */
    uint32_t len_w  = tile_w*2;
    uint32_t std_w  = K_SIZE*2;
    uint32_t reps_w = (uint32_t) tile_h;
    uint32_t obi_addr_w = obi_addr_y + (tile_w * 2);
    uint32_t axi_addr_w = (uint32_t) w_in + (x_id * tile_h * K_SIZE * 2) + (y_id * tile_w * 2); 

    idma_memcpy_2d(&idma_ctrl, 0, axi_addr_w, obi_addr_w, len_w, std_w, reps_w);
    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

    /**
     * 2c. Use iDMA to transfer input vector blocks.
     */
    uint32_t len_x = tile_h * 2;
    uint32_t obi_addr_x = obi_addr_w + (tile_w * tile_h * 2);
    uint32_t axi_addr_x = (uint32_t) x_in + (x_id * tile_h * 2); 

    idma_memcpy_1d(&idma_ctrl, 0, axi_addr_x, obi_addr_x, len_x);
    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    
    /**
     * 3. Compute partial GeMV.
     */
    redmule_gemm(&redmule_ctrl, obi_addr_x, obi_addr_w, obi_addr_y, tile_m, tile_h, tile_w);
    eu_redmule_wait(&eu_ctrl, WAIT_MODE);


    // Wait for all tiles to be awake and ready to start the kernel
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /**
    * 4. Reduce partial GeMV.
    */
    uint32_t log_tree_mask = 1;
    uint32_t log_tree_bit  = 1;
    for (int i = 0; i < reduce_phases; i++){
    #if defined(BASELINE_K2)
        if (i == 0) {   // First level of the tree
            if (x_id % reduce_degree == 0) {  // Tile is this phase's group leader.
                fsync_sync_row(&fsync_ctrl);
                eu_fsync_wait(&eu_ctrl, WAIT_MODE);

                /**
                * 4b. Sum partial GeMV on the phases group leader.
                */
                for (int j = 0; j < (reduce_degree - 1); j++){
                    if ((x_id + 1 + j) <= (MESH_X_TILES - 1)){
                        uint32_t partial_gemv_addr = obi_addr_x + (j * len_y);
                        redmule_gemm(&redmule_ctrl, partial_gemv_addr, obi_addr_id, obi_addr_y, tile_m, tile_w, tile_w);
                        eu_redmule_wait(&eu_ctrl, WAIT_MODE);
                    }
                }
            } else {    // The non-leader tiles write on the L1 of their group leader.
                /**
                * 4a. Scatter partial GeMV.
                */
                uint32_t group_id = reduce_degree * (x_id / reduce_degree);
                //printf("Group id: %d\n", group_id);
                uint32_t partial_gemv_addr = get_l1_base(GET_ID(y_id, group_id)) + (tile_w * tile_w * 2) + (tile_w * tile_h * 2) + (tile_w * 2) + (((x_id % reduce_degree) - 1) * len_y);

                idma_memcpy_1d(&idma_ctrl, 1, partial_gemv_addr, obi_addr_y, len_y);
                eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);

                fsync_sync_row(&fsync_ctrl);
                eu_fsync_wait(&eu_ctrl, WAIT_MODE);
            }
        } 
        else {    // Second level of the tree
            if (x_id == 0) {    // Leftmost tile
                fsync_sync_row(&fsync_ctrl);
                eu_fsync_wait(&eu_ctrl, WAIT_MODE);

                /**
                * 4b. Sum partial GeMV on leftmost tile.
                */
                for (int j = 0; j < (reduce_degree - 1); j++){
                    if ((x_id + 1 + j) <= (MESH_X_TILES - 1)){
                        uint32_t partial_gemv_addr = obi_addr_x + (j * len_y);
                        redmule_gemm(&redmule_ctrl, partial_gemv_addr, obi_addr_id, obi_addr_y, tile_m, tile_w, tile_w);
                        eu_redmule_wait(&eu_ctrl, WAIT_MODE);
                    }
                }
            } 
            else if (x_id % reduce_degree == 0) {    // Previous phase leaders write on leftmost tile's L1
                /**
                * 4a. Scatter partial GeMV.
                */
                uint32_t partial_gemv_addr = get_l1_base(GET_ID(y_id, 0)) + (tile_w * tile_w * 2) + (tile_w * tile_h * 2) + (tile_w * 2) + (((x_id / reduce_degree)-1) * len_y);
                
                idma_memcpy_1d(&idma_ctrl, 1, partial_gemv_addr, obi_addr_y, len_y);
                eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);

                fsync_sync_row(&fsync_ctrl);
                eu_fsync_wait(&eu_ctrl, WAIT_MODE);
            } 
            else {
                fsync_sync_row(&fsync_ctrl);
                eu_fsync_wait(&eu_ctrl, WAIT_MODE);
            }
        }
    #elif defined(K_LOGN)
        if ((x_id & log_tree_mask) == 0){ // Current phase leader
            /**
            * 4a. Calculate the tile from which add the partial gemv
            */
            uint32_t partial_gemv_addr = get_l1_base(GET_ID(y_id, x_id^log_tree_bit)) + (tile_w*tile_h*2);
            idma_memcpy_1d(&idma_ctrl, 0, partial_gemv_addr, obi_addr_x, len_y);
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

            /**
            * 4b. Sum partial GeMV.
            */
            redmule_gemm(&redmule_ctrl, obi_addr_x, obi_addr_id, obi_addr_y, tile_m, tile_w, tile_w);
            eu_redmule_wait(&eu_ctrl, WAIT_MODE);
        }
        log_tree_mask = (log_tree_mask << 1) | 1;
        log_tree_bit <<= 1;
    #endif
        if (i == (reduce_phases-1)){
            if (x_id == 0){
                /**
                * 5. Store result in memory.
                */
                axi_addr_y = (uint32_t) y_out + (y_id*tile_w*2);
                idma_memcpy_1d(&idma_ctrl, 1, axi_addr_y, obi_addr_y, len_y);
                eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
            }
        }
        fsync_sync_row(&fsync_ctrl);
        eu_fsync_wait(&eu_ctrl, WAIT_MODE);
    }

    //printf("I'm done dog\n");
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /**
    * 7. Check results.
    */
    uint32_t num_errors = 0;
    if (hartid == 0){
        uint16_t computed, expected, diff;
        for(int i = 0; i < M_SIZE*K_SIZE; i++){
            computed = y_out[i];
            expected = z_out[i];
            diff = (computed > expected) ? (computed - expected) : (expected - computed);
            if(diff > 0x0011){
                num_errors++;
                #if EVAL == 1
                printf("**ERROR**: Y[%0d](=0x%4x) != Z[%0d](=0x%4x)\n", i, computed, i, expected);
                #endif
            }
        }
        printf("Finished test with %0d errors\n", num_errors);
    }

    return num_errors;
}