// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Victor Isachi <victor.isachi@unibo.it>

#define SIZE_1x64x64
// #define SIZE_1x128x128
// #define SIZE_1x256x256
// #define SIZE_1x512x512
// #define SIZE_1x1024x1024

#include <stdint.h>

#if defined(SIZE_1x64x64)
#include "mat_vec_1x64x64.h"
#elif defined(SIZE_1x128x128)
#include "mat_vec_1x128x128.h"
#elif defined(SIZE_1x256x256)
#include "mat_vec_1x256x256.h"
#elif defined(SIZE_1x512x512)
#include "mat_vec_1x512x512.h"
#elif defined(SIZE_1x1024x1024)
#include "mat_vec_1x1024x1024.h"
#endif

#define BASELINE_K2
// #define K_LOGN

#include "tile.h"
#include "idma.h"
#include "redmule.h"
#include "fsync.h"

/**
 * This test implements the optimal GeMV algorithm for MAGIA (and for mesh-based architectures in general) using FractalSync for synchronization. 
 * See "WaferLLM: Large Language Model Inference at Wafer Scale" paper.
 */
int main(void){
    // sentinel_start();   // Total execution

    /** 
     * 0. Get the mesh-tile's hartid, mesh-tile coordinates and define its L1 base. 
     * Initialize the controllers for the idma, redmule and fsync.
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

    uint32_t y_id = GET_Y_ID(hartid);
    uint32_t x_id = GET_X_ID(hartid);
    uint32_t l1_tile_base = get_l1_base(hartid);

    /**
     * The MeshGeMV is implemented with a constant number of timeslots (1), each partial GeMV computed in a single go.
     * The reduce phase is implemented as (i) a 2-Tree as indicated in the paper as the baseline, (ii) a logarithmic tree with degree 2.
     */
#if defined(SIZE_1x64x64)
#if defined(BASELINE_K2)
    uint32_t reduce_degree = 2;
    uint32_t reduce_phases = 1;
#elif defined(K_LOGN)
    uint32_t reduce_degree = 2;
    uint32_t reduce_phases = 1;
#endif
#elif defined(SIZE_1x128x128)
#if defined(BASELINE_K2)
    uint32_t reduce_degree = 2;
    uint32_t reduce_phases = 2;
#elif defined(K_LOGN)
    uint32_t reduce_degree = 2;
    uint32_t reduce_phases = 2;
#endif
#elif defined(SIZE_1x256x256)
#if defined(BASELINE_K2)
    uint32_t reduce_degree = 3;
    uint32_t reduce_phases = 2;
#elif defined(K_LOGN)
    uint32_t reduce_degree = 2;
    uint32_t reduce_phases = 3;
#endif
#elif defined(SIZE_1x512x512)
#if defined(BASELINE_K2)
    uint32_t reduce_degree = 4;
    uint32_t reduce_phases = 2;
#elif defined(K_LOGN)
    uint32_t reduce_degree = 2;
    uint32_t reduce_phases = 4;
#endif
#elif defined(SIZE_1x1024x1024)
#if defined(BASELINE_K2)
    uint32_t reduce_degree = 6;
    uint32_t reduce_phases = 2;
#elif defined(K_LOGN)
    uint32_t reduce_degree = 2;
    uint32_t reduce_phases = 5;
#endif
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
    uint32_t len_id  = tile_w*2;
    uint32_t std_id  = K_SIZE*2;
    uint32_t reps_id = tile_h;
    uint32_t obi_addr_id = (l1_tile_base);
    uint32_t axi_addr_id = (uint32_t) id_mat; 
    // printf("Transfering identity matrix: src_addr 0x%0x, dst_addr 0x%0x, lenght (bytes) %0d, stride %0d, reps %0d\n", axi_addr_id, obi_addr_id, len_id, std_id, reps_id);
    stnl_cmi_s();
    idma_conf_in();
    idma_set_addr_len_in(obi_addr_id, axi_addr_id, len_id);
    idma_set_std2_rep2_in(len_id, std_id, reps_id);
    idma_set_std3_rep3_in(0, 0, 1);
    idma_start_in();
    idma_wait();
    stnl_par_f();

    // Wait for all tiles to be awake and ready to start the kernel
    stnl_snc_s();
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    stnl_snc_f();

    sentinel_start();   // Execution time after wakeup
    stnl_ts_s();        // Parallel GeMV timeslot

    /**
     * 2a. Use iDMA to transfer bias blocks.
     * To avoid accumulating the bias multiple times only one tile per row fetches it, the rest fetch the 0 matrix.
     * The leftmost tile is the root of the reduction tree so it fetches the bias.
     */
    uint32_t len_y = tile_w*2;
    uint32_t obi_addr_y = obi_addr_id + (tile_w*tile_h*2);
    uint32_t axi_addr_y = (x_id == 0) ? (uint32_t) y_in + (y_id*tile_w*2) : (uint32_t) y_out + (y_id*tile_w*2);
    
    // printf("Transfering bias: src_addr 0x%0x, dst_addr 0x%0x, lenght (bytes) %0d\n", axi_addr_y, obi_addr_y, len_y);
    stnl_cmi_s();
    idma_conf_in();
    idma_set_addr_len_in(obi_addr_y, axi_addr_y, len_y);
    idma_set_std2_rep2_in(0, 0, 1);
    idma_set_std3_rep3_in(0, 0, 1);
    idma_start_in();
    idma_wait();
    stnl_par_f();

    /**
     * 2b. Use iDMA to transfer input matrix blocks.
     */
    uint32_t len_w  = tile_w*2;
    uint32_t std_w  = K_SIZE*2;
    uint32_t reps_w = (uint32_t) tile_h;
    uint32_t obi_addr_w = obi_addr_y + (tile_w*2);
    uint32_t axi_addr_w = (uint32_t) w_in + (x_id*tile_h*K_SIZE*2) + (y_id*tile_w*2); 
    
    // printf("Transfering input matrix: src_addr 0x%0x, dst_addr 0x%0x, lenght (bytes) %0d, stride %0d, reps %0d\n", axi_addr_w, obi_addr_w, len_w, std_w, reps_w);
    stnl_cmi_s();
    idma_conf_in();
    idma_set_addr_len_in(obi_addr_w, axi_addr_w, len_w);
    idma_set_std2_rep2_in(len_w, std_w, reps_w);
    idma_set_std3_rep3_in(0, 0, 1);
    idma_start_in();
    idma_wait();
    stnl_par_f();

    /**
     * 2c. Use iDMA to transfer input vector blocks.
     */
    uint32_t len_x = tile_h*2;
    uint32_t obi_addr_x = obi_addr_w + (tile_w*tile_h*2);
    uint32_t axi_addr_x = (uint32_t) x_in + (x_id*tile_h*2); 
    
    // printf("Transfering input vector: src_addr 0x%0x, dst_addr 0x%0x, lenght (bytes) %0d\n", axi_addr_x, obi_addr_x, len_x);
    stnl_cmi_s();
    idma_conf_in();
    idma_set_addr_len_in(obi_addr_x, axi_addr_x, len_x);
    idma_set_std2_rep2_in(0, 0, 1);
    idma_set_std3_rep3_in(0, 0, 1);
    idma_start_in();
    idma_wait();
    stnl_par_f();
    
    /**
     * 3. Compute partial GeMV.
     */
    // printf("RedMulE block sizes: tile_m = %0d, tile_h = %0d, tile_w = %0d\n", tile_m, tile_h, tile_w);
    redmule_mcnfig((uint16_t) tile_w, (uint16_t) tile_m, (uint16_t) tile_h);

    stnl_cmp_s();
    redmule_marith(obi_addr_y, obi_addr_w, obi_addr_x);
    redmule_wait();
    stnl_par_f();

    stnl_ts_f();    // Parallel GeMV timeslot

    /**
    * 4. Reduce partial GeMV.
    */
    uint32_t num_baseline_k2_lvl1_leafs = 0;
    for (int i = 0; i < reduce_degree-1; i++){
        if ((x_id+1+i) < MESH_X_TILES){
            num_baseline_k2_lvl1_leafs++;
        }
    }
    uint32_t log_tree_mask = 1;
    uint32_t log_tree_bit  = 1;
    for (int i = 0; i < reduce_phases; i++){
#if defined(BASELINE_K2)
        if (i == 0) {   // First level of the tree
            // Synchronize all local subsets of the tree that have data-dependencies
            stnl_snc_s();
            if (x_id%reduce_degree == 0) {  // DST
                 // Wait for all SRCs to request synchronization
                while (mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) < num_baseline_k2_lvl1_leafs);

                // Reset barrier counter
                mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) = 0;

                // Send synchronization response to all SRCs
                for (int i = 0; i < num_baseline_k2_lvl1_leafs; i++) amo_increment(SYNC_BASE + GET_ID(y_id, x_id+1+i)*L1_TILE_OFFSET, 1);
            } else {    // SRC
                // Send synchronization request to DST
                amo_increment(SYNC_BASE + GET_ID(y_id, reduce_degree*(x_id/reduce_degree))*L1_TILE_OFFSET, 1);

                // Wait for DST synchronization response
                while (mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) < 1);

                // Reset barrier counter
                mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) = 0;
            }
            stnl_snc_f();

            stnl_ts_s();    // Reduction timeslots

            if (x_id%reduce_degree == 0) {  // DST
                // Synchronize all local subsets of the tree that have data-dependencies
                stnl_snc_s();
                if (x_id%reduce_degree == 0) {  // DST
                    // Wait for all SRCs to request synchronization
                    while (mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) < num_baseline_k2_lvl1_leafs);

                    // Reset barrier counter
                    mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) = 0;

                    // Send synchronization response to all SRCs
                    for (int i = 0; i < num_baseline_k2_lvl1_leafs; i++) amo_increment(SYNC_BASE + GET_ID(y_id, x_id+1+i)*L1_TILE_OFFSET, 1);
                } else {    // SRC
                    // Send synchronization request to DST
                    amo_increment(SYNC_BASE + GET_ID(y_id, reduce_degree*(x_id/reduce_degree))*L1_TILE_OFFSET, 1);

                    // Wait for DST synchronization response
                    while (mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) < 1);

                    // Reset barrier counter
                    mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) = 0;
                }
                stnl_snc_f();

                /**
                * 4b. Sum partial GeMV.
                */
                for (int j = 0; j < reduce_degree-1; j++){
                    if ((x_id+1+j) > (MESH_X_TILES-1)) break; // Non full-degree node
                    uint32_t partial_gemv_addr = obi_addr_x + (j*len_y);
                    // printf("Summing partial GeMV: y_addr 0x%0x, x_addr 0x%0x\n", obi_addr_y, partial_gemv_addr);
                    stnl_cmp_s();
                    redmule_marith(obi_addr_y, obi_addr_id, partial_gemv_addr);
                    redmule_wait();
                    stnl_par_f();
                }
            } else {    // SRC
                /**
                * 4a. Scatter partial GeMV.
                */
                uint32_t partial_gemv_addr = get_l1_base(GET_ID(y_id, reduce_degree*(x_id/reduce_degree))) + 2*(tile_w*tile_h*2) + (tile_w*2) + (((x_id%reduce_degree)-1)*len_y);
                // printf("Transfering partial GeMV to neighbor: src_addr 0x%0x, dst_addr 0x%0x, lenght (bytes) %0d\n", obi_addr_y, partial_gemv_addr, len_y);
                stnl_cmo_s();
                idma_conf_out();
                idma_set_addr_len_out(partial_gemv_addr, obi_addr_y, len_y);
                idma_set_std2_rep2_out(0, 0, 1);
                idma_set_std3_rep3_out(0, 0, 1);
                idma_start_out();
                idma_wait();
                stnl_par_f();

                // Synchronize all local subsets of the tree that have data-dependencies
                stnl_snc_s();
                if (x_id%reduce_degree == 0) {  // DST
                    // Wait for all SRCs to request synchronization
                    while (mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) < num_baseline_k2_lvl1_leafs);

                    // Reset barrier counter
                    mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) = 0;

                    // Send synchronization response to all SRCs
                    for (int i = 0; i < num_baseline_k2_lvl1_leafs; i++) amo_increment(SYNC_BASE + GET_ID(y_id, x_id+1+i)*L1_TILE_OFFSET, 1);
                } else {    // SRC
                    // Send synchronization request to DST
                    amo_increment(SYNC_BASE + GET_ID(y_id, reduce_degree*(x_id/reduce_degree))*L1_TILE_OFFSET, 1);

                    // Wait for DST synchronization response
                    while (mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) < 1);

                    // Reset barrier counter
                    mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) = 0;
                }
                stnl_snc_f();
            }
        } else {    // Second level of the tree
            // Synchronize all local subsets of the tree that have data-dependencies
            stnl_snc_s();
            if (x_id == 0) {  // DST
                // Wait for all SRCs to request synchronization
                while (mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) < reduce_degree-1);

                // Reset barrier counter
                mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) = 0;

                // Send synchronization response to all SRCs
                for (int i = 1; i < reduce_degree; i++) amo_increment(SYNC_BASE + GET_ID(y_id, i*reduce_degree)*L1_TILE_OFFSET, 1);
            } else if (x_id%reduce_degree == 0) {    // SRC
                // Send synchronization request to DST
                amo_increment(SYNC_BASE + GET_ID(y_id, 0)*L1_TILE_OFFSET, 1);

                // Wait for DST synchronization response
                while (mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) < 1);

                // Reset barrier counter
                mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) = 0;
            }
            stnl_snc_f();

            stnl_ts_s();    // Reduction timeslots

            if (x_id == 0) {    // DST
                // Synchronize all local subsets of the tree that have data-dependencies
                stnl_snc_s();
                if (x_id == 0) {  // DST
                    // Wait for all SRCs to request synchronization
                    while (mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) < reduce_degree-1);

                    // Reset barrier counter
                    mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) = 0;

                    // Send synchronization response to all SRCs
                    for (int i = 1; i < reduce_degree; i++) amo_increment(SYNC_BASE + GET_ID(y_id, i*reduce_degree)*L1_TILE_OFFSET, 1);
                } else if (x_id%reduce_degree == 0) {    // SRC
                    // Send synchronization request to DST
                    amo_increment(SYNC_BASE + GET_ID(y_id, 0)*L1_TILE_OFFSET, 1);

                    // Wait for DST synchronization response
                    while (mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) < 1);

                    // Reset barrier counter
                    mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) = 0;
                }
                stnl_snc_f();

                /**
                * 4b. Sum partial GeMV.
                */
                for (int j = 0; j < reduce_degree-1; j++){
                    if ((x_id+1+j) > (MESH_X_TILES-1)) break; // Non full-degree node
                    uint32_t partial_gemv_addr = obi_addr_x + (j*len_y);
                    // printf("Summing partial GeMV: y_addr 0x%0x, x_addr 0x%0x\n", obi_addr_y, partial_gemv_addr);
                    stnl_cmp_s();
                    redmule_marith(obi_addr_y, obi_addr_id, partial_gemv_addr);
                    redmule_wait();
                    stnl_par_f();
                }
            } else if (x_id%reduce_degree == 0) {    // SRC
                /**
                * 4a. Scatter partial GeMV.
                */
                uint32_t partial_gemv_addr = get_l1_base(GET_ID(y_id, 0)) + 2*(tile_w*tile_h*2) + (tile_w*2) + (((x_id/reduce_degree)-1)*len_y);
                // printf("Transfering partial GeMV to neighbor: src_addr 0x%0x, dst_addr 0x%0x, lenght (bytes) %0d\n", obi_addr_y, partial_gemv_addr, len_y);
                stnl_cmo_s();
                idma_conf_out();
                idma_set_addr_len_out(partial_gemv_addr, obi_addr_y, len_y);
                idma_set_std2_rep2_out(0, 0, 1);
                idma_set_std3_rep3_out(0, 0, 1);
                idma_start_out();
                idma_wait();
                stnl_par_f();

                // Synchronize all local subsets of the tree that have data-dependencies
                stnl_snc_s();
                if (x_id == 0) {  // DST
                    // Wait for all SRCs to request synchronization
                    while (mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) < reduce_degree-1);

                    // Reset barrier counter
                    mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) = 0;

                    // Send synchronization response to all SRCs
                    for (int i = 1; i < reduce_degree; i++) amo_increment(SYNC_BASE + GET_ID(y_id, i*reduce_degree)*L1_TILE_OFFSET, 1);
                } else if (x_id%reduce_degree == 0) {    // SRC
                    // Send synchronization request to DST
                    amo_increment(SYNC_BASE + GET_ID(y_id, 0)*L1_TILE_OFFSET, 1);

                    // Wait for DST synchronization response
                    while (mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) < 1);

                    // Reset barrier counter
                    mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) = 0;
                }
                stnl_snc_f();
            }
        }

        stnl_ts_f();    // Reduction timeslots

        if (i == (reduce_phases-1)){
            if (x_id == 0){
                /**
                * 5. Store result in memory.
                */
                axi_addr_y = (uint32_t) y_out + (y_id*tile_w*2);
                // printf("Transfering result: src_addr 0x%0x, dst_addr 0x%0x, lenght (bytes) %0d\n", obi_addr_y, axi_addr_y, len_y);
                stnl_cmo_s();
                idma_conf_out();
                idma_set_addr_len_out(axi_addr_y, obi_addr_y, len_y);
                idma_set_std2_rep2_out(0, 0, 1);
                idma_set_std3_rep3_out(0, 0, 1);
                idma_start_out();
                idma_wait();
                stnl_par_f();
            }
        }
#elif defined(K_LOGN)
        // Synchronize all local subsets of the tree that have data-dependencies
        stnl_snc_s();
        if ((x_id&log_tree_mask) == 0) {  // DST
            // Wait for all SRCs to request synchronization
            while (mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) < 1);

            // Reset barrier counter
            mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) = 0;

            // Send synchronization response to all SRCs
            amo_increment(SYNC_BASE + GET_ID(y_id, x_id^log_tree_bit)*L1_TILE_OFFSET, 1);
        } else if (((x_id^log_tree_bit)&log_tree_mask) == 0) {    // SRC
            // Send synchronization request to DST
            amo_increment(SYNC_BASE + GET_ID(y_id, x_id^log_tree_bit)*L1_TILE_OFFSET, 1);

            // Wait for DST synchronization response
            while (mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) < 1);

            // Reset barrier counter
            mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) = 0;
        }
        stnl_snc_f();

        stnl_ts_s();    // Reduction timeslots

        // printf("Log Tree Mask: 0x%0x, Log Tree Bit: 0x%0x\n", log_tree_mask, log_tree_bit);
        
        if ((x_id&log_tree_mask) == 0){
            /**
            * 4a. Gather all partial GeMV.
            */
            uint32_t partial_gemv_addr = get_l1_base(GET_ID(y_id, x_id^log_tree_bit)) + (tile_w*tile_h*2);
            // printf("Transfering partial GeMV from neighbor: src_addr 0x%0x, dst_addr 0x%0x, lenght (bytes) %0d\n", partial_gemv_addr, obi_addr_x, len_x);
            stnl_cmi_s();
            idma_conf_in();
            idma_set_addr_len_in(obi_addr_x, partial_gemv_addr, len_x);
            idma_set_std2_rep2_in(0, 0, 1);
            idma_set_std3_rep3_in(0, 0, 1);
            idma_start_in();
            idma_wait();
            stnl_par_f();

            /**
            * 4b. Sum partial GeMV.
            */
            stnl_cmp_s();
            redmule_marith(obi_addr_y, obi_addr_id, obi_addr_x);
            redmule_wait();
            stnl_par_f();
        }
        log_tree_mask = (log_tree_mask << 1) | 1;
        log_tree_bit <<= 1;

        stnl_ts_f();    // Reduction timeslots

        if (i == (reduce_phases-1)){
            if (x_id == 0){
                /**
                * 5. Store result in memory.
                */
                axi_addr_y = (uint32_t) y_out + (y_id*tile_w*2);
                // printf("Transfering result: src_addr 0x%0x, dst_addr 0x%0x, lenght (bytes) %0d\n", obi_addr_y, axi_addr_y, len_y);
                stnl_cmo_s();
                idma_conf_out();
                idma_set_addr_len_out(axi_addr_y, obi_addr_y, len_y);
                idma_set_std2_rep2_out(0, 0, 1);
                idma_set_std3_rep3_out(0, 0, 1);
                idma_start_out();
                idma_wait();
                stnl_par_f();
            }
        }
#endif
    }

    sentinel_end(); // Execution time after wakeup
    // asm volatile("nop" ::); // Needed to detect same instruction consecutively
    // sentinel_end(); // Total execution

    stnl_cmi_r();
    stnl_cmo_r();
    stnl_cmp_r();
    stnl_snc_r();
    
    // if (get_hartid() == 0){
    //   stnl_r();
    //   stnl_ts_r();
    // }

    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);

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
                printf("**ERROR**: Y[%0d](=0x%4x) != Z[%0d](=0x%4x)\n", i, computed, i, expected);
            }
        }
        printf("Finished test with %0d errors\n", num_errors);
    }

    return num_errors;
}