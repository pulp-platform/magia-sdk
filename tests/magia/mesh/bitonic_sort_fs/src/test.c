// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Victor Isachi <victor.isachi@unibo.it>
// Alberto Dequino <alberto.dequino@unibo.it>

#include <stdint.h>
#include "tile.h"
#include "idma.h"
#include "fsync.h"

#define BS_SIZE_16384
// #define BS_SIZE_65536
// #define BS_SIZE_262144
// #define BS_SIZE_1048576
// #define BS_SIZE_4194304

#if defined(BS_SIZE_16384)
#include "bs_array_16384.h"
#elif defined(BS_SIZE_65536)
#include "bs_array_65536.h"
#elif defined(BS_SIZE_262144)
#include "bs_array_262144.h"
#elif defined(BS_SIZE_1048576)
#include "bs_array_1048576.h"
#elif defined(BS_SIZE_4194304)
#include "bs_array_4194304.h"
#endif

#define VERBOSE (0)

#define BLOCK_SIZE      (BS_ARRAY_SIZE/NUM_HARTS)
#define LOG2_BLOCK_SIZE (__builtin_ctz(BLOCK_SIZE))
#define CHUNK_SIZE      (32)
#define DATA_BYTES      (4)
#define NUM_BUFF        (2)

#define DMA_IN  (0)
#define DMA_OUT (1)

#define CHUNK_OWNER(global_index) ((global_index) >> LOG2_BLOCK_SIZE)
#define LOCAL_INDEX(global_index) ((global_index) & (BLOCK_SIZE-1))

void idma_1d(idma_controller_t *ctrl, uint8_t dir, uint32_t axi_addr, uint32_t obi_addr, uint32_t len){
    idma_mm_conf(dir);
    if(dir){    // OBI to AXI (L1 to L2)
        idma_mm_set_addr_len(dir, axi_addr, obi_addr, len);
    }
    else{       // AXI to OBI (L2 to L1)
        idma_mm_set_addr_len(dir, obi_addr, axi_addr, len);
    }
    idma_mm_set_std2_rep2(dir, 0, 0, 1);
    idma_mm_set_std3_rep3(dir, 0, 0, 1);
    idma_mm_start(dir);
}

    // This test implements the Bitonic Sort kernel using FractalSync for synchronization. 
int main(void){

    // Get the mesh-tile's hartid, mesh-tile coordinates and define its L1 base, 
    // also initialize the controllers for idma and fsync.
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

    uint32_t local_block_idx_start = hartid*BLOCK_SIZE;
    uint32_t local_block_idx_end   = local_block_idx_start + BLOCK_SIZE;

    uint32_t local_block_addr = l1_tile_base;
    uint32_t local_block_size = BLOCK_SIZE*DATA_BYTES;

    uint32_t buff0_addr = local_block_addr + local_block_size;
    uint32_t buff0_size = CHUNK_SIZE*DATA_BYTES;
    uint32_t buff1_addr = buff0_addr + buff0_size;
    uint32_t buff1_size = CHUNK_SIZE*DATA_BYTES;

#if VERBOSE > 100
    if(hartid == 0){
        printf("Initial array:\n");
        for(unsigned int i = 0; i < BS_ARRAY_SIZE; i++){
            printf("bs_array[%2d]: %0d\n", i, bs_array[i]);
        }
    }
#endif

    // Wait for all tiles to be awake and ready to start the kernel
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    
    // Load chunck of global array that needs to be ordered
    idma_1d(&idma_ctrl, DMA_IN, (uint32_t)&bs_array[hartid*BLOCK_SIZE], local_block_addr, local_block_size);

    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);

#if VERBOSE > 100
    printf("Loaded partial data:\n");
    for(int i = 0; i < BLOCK_SIZE; i++){
        printf("local_data[%2d]: %0d\n", i, mmio32(local_block_addr+i*DATA_BYTES));
    }
#endif

    // Iterate through bitonic sort stages
    for(uint32_t bitonic_len = 2; bitonic_len <= BS_ARRAY_SIZE; bitonic_len <<= 1){
        for(uint32_t stride = bitonic_len >> 1; stride > 0; stride >>= 1){

#if VERBOSE > 10
            printf("Bitonic sort iteration with paraleters: bitonic_len = %0u, stride = %0u\n", bitonic_len, stride);
#endif

            const uint32_t two_stride = 2*stride;

            uint32_t curr_idx  = local_block_idx_start;
            uint32_t curr_slot = 0;
            uint32_t curr_have = 0;

            uint32_t left_base[NUM_BUFF]       = {0, 0};
            uint32_t right_base[NUM_BUFF]      = {0, 0};
            uint32_t right_owner[NUM_BUFF]     = {0, 0};
            uint32_t right_local_idx[NUM_BUFF] = {0, 0};
            uint32_t count[NUM_BUFF]           = {0, 0};
            uint32_t ascending[NUM_BUFF]       = {0 ,0};

            for(;;){
                if(curr_idx >= local_block_idx_end){curr_have = 0; break;}

                uint32_t block_base = curr_idx & ~(two_stride - 1);
                uint32_t left_end   = block_base + stride;
                uint32_t next_block = block_base + two_stride;

                if(curr_idx >= left_end){curr_idx = next_block; continue;}

                left_base[curr_slot]       = curr_idx;
                right_base[curr_slot]      = left_base[curr_slot] + stride;
                right_owner[curr_slot]     = CHUNK_OWNER(right_base[curr_slot]);
                right_local_idx[curr_slot] = LOCAL_INDEX(right_base[curr_slot]);

                uint32_t right_owner_end = (right_owner[curr_slot] + 1) * BLOCK_SIZE;
                uint32_t limit = left_end;
                if(local_block_idx_end < limit)              {limit = local_block_idx_end;}
                if(left_base[curr_slot] + CHUNK_SIZE < limit){limit = left_base[curr_slot] + CHUNK_SIZE;}
                if(right_owner_end - stride < limit)         {limit = right_owner_end - stride;}

                count[curr_slot] = limit - left_base[curr_slot];
                curr_idx = limit;

                ascending[curr_slot] = ((left_base[curr_slot] & bitonic_len) == 0);
                left_base[curr_slot] = local_block_addr + (left_base[curr_slot] - local_block_idx_start) * DATA_BYTES;

                if(right_owner[curr_slot] == hartid){
                    right_base[curr_slot] = local_block_addr + (right_base[curr_slot] - local_block_idx_start) * DATA_BYTES;

                    uint32_t left_elem;
                    uint32_t right_elem;
                    uint32_t left_ptr  = left_base[curr_slot];
                    uint32_t right_ptr = right_base[curr_slot];
                    if(ascending[curr_slot]){
                        for(uint32_t i = 0; i < count[curr_slot]; i++){
                            left_elem  = mmio32(left_ptr);
                            right_elem = mmio32(right_ptr);
                            mmio32(left_ptr)  = (left_elem <= right_elem) ? left_elem  : right_elem;
                            mmio32(right_ptr) = (left_elem <= right_elem) ? right_elem : left_elem;
                            left_ptr  += DATA_BYTES;
                            right_ptr += DATA_BYTES;
                        }
                    }else{
                        for(uint32_t i = 0; i < count[curr_slot]; i++){
                            left_elem  = mmio32(left_ptr);
                            right_elem = mmio32(right_ptr);
                            mmio32(left_ptr)  = (left_elem >= right_elem) ? left_elem  : right_elem;
                            mmio32(right_ptr) = (left_elem >= right_elem) ? right_elem : left_elem;
                            left_ptr  += DATA_BYTES;
                            right_ptr += DATA_BYTES;
                        }
                    }
                    continue;
                }

                right_base[curr_slot] = (curr_slot == 0) ? buff0_addr : buff1_addr;

                uint32_t src_addr = get_l1_base(right_owner[curr_slot])+(right_local_idx[curr_slot]*DATA_BYTES);
                uint32_t dst_addr = right_base[curr_slot];
                idma_1d(&idma_ctrl, DMA_IN, src_addr, dst_addr, count[curr_slot]*DATA_BYTES);
                curr_have = 1;
                break;
            }

            while(curr_have){
                uint32_t next_slot = 1 - curr_slot;
                uint32_t next_have = 0;

                for(;;){
                    if(curr_idx >= local_block_idx_end){next_have = 0; break;}

                    uint32_t block_base = curr_idx & ~(two_stride - 1);
                    uint32_t left_end   = block_base + stride;
                    uint32_t next_block = block_base + two_stride;

                    if(curr_idx >= left_end){curr_idx = next_block; continue;}

                    left_base[next_slot]       = curr_idx;
                    right_base[next_slot]      = left_base[next_slot] + stride;
                    right_owner[next_slot]     = CHUNK_OWNER(right_base[next_slot]);
                    right_local_idx[next_slot] = LOCAL_INDEX(right_base[next_slot]);

                    uint32_t right_owner_end = (right_owner[next_slot] + 1) * BLOCK_SIZE;
                    uint32_t limit = left_end;
                    if(local_block_idx_end < limit)              {limit = local_block_idx_end;}
                    if(left_base[next_slot] + CHUNK_SIZE < limit){limit = left_base[next_slot] + CHUNK_SIZE;}
                    if(right_owner_end - stride < limit)         {limit = right_owner_end - stride;}

                    count[next_slot] = limit - left_base[next_slot];
                    curr_idx = limit;

                    ascending[next_slot] = ((left_base[next_slot] & bitonic_len) == 0);
                    left_base[next_slot] = local_block_addr + (left_base[next_slot] - local_block_idx_start) * DATA_BYTES;

                    if(right_owner[next_slot] == hartid){
                        right_base[next_slot] = local_block_addr + (right_base[next_slot] - local_block_idx_start) * DATA_BYTES;

                        uint32_t left_elem;
                        uint32_t right_elem;
                        uint32_t left_ptr  = left_base[next_slot];
                        uint32_t right_ptr = right_base[next_slot];
                        if(ascending[next_slot]){
                            for(uint32_t i = 0; i < count[next_slot]; i++){
                                left_elem  = mmio32(left_ptr);
                                right_elem = mmio32(right_ptr);
                                mmio32(left_ptr)  = (left_elem <= right_elem) ? left_elem  : right_elem;
                                mmio32(right_ptr) = (left_elem <= right_elem) ? right_elem : left_elem;
                                left_ptr  += DATA_BYTES;
                                right_ptr += DATA_BYTES;
                            }
                        }else{
                            for(uint32_t i = 0; i < count[next_slot]; i++){
                                left_elem  = mmio32(left_ptr);
                                right_elem = mmio32(right_ptr);
                                mmio32(left_ptr)  = (left_elem >= right_elem) ? left_elem  : right_elem;
                                mmio32(right_ptr) = (left_elem >= right_elem) ? right_elem : left_elem;
                                left_ptr  += DATA_BYTES;
                                right_ptr += DATA_BYTES;
                            }
                        }
                        continue;
                    }

                    right_base[next_slot] = (next_slot == 0) ? buff0_addr : buff1_addr;
                    
                    uint32_t src_addr = get_l1_base(right_owner[next_slot])+(right_local_idx[next_slot]*DATA_BYTES);
                    uint32_t dst_addr = right_base[next_slot];
                    idma_1d(&idma_ctrl, DMA_IN, src_addr, dst_addr, count[next_slot]*DATA_BYTES);
                    next_have = 1;
                    break;
                }

                uint32_t left_elem;
                uint32_t right_elem;
                for(uint32_t i = 0; i < count[curr_slot]; i++){
                    left_elem  = mmio32(left_base[curr_slot]+i*DATA_BYTES);
                    right_elem = mmio32(right_base[curr_slot]+i*DATA_BYTES);
                    if(ascending[curr_slot]){
                        mmio32(left_base[curr_slot]+i*DATA_BYTES)  = (left_elem <= right_elem) ? left_elem  : right_elem;
                        mmio32(right_base[curr_slot]+i*DATA_BYTES) = (left_elem <= right_elem) ? right_elem : left_elem;
                    }else{
                        mmio32(left_base[curr_slot]+i*DATA_BYTES)  = (left_elem >= right_elem) ? left_elem  : right_elem;
                        mmio32(right_base[curr_slot]+i*DATA_BYTES) = (left_elem >= right_elem) ? right_elem : left_elem;
                    }
                }

                uint32_t src_addr = right_base[curr_slot];
                uint32_t dst_addr = get_l1_base(right_owner[curr_slot])+(right_local_idx[curr_slot]*DATA_BYTES);
                idma_1d(&idma_ctrl, DMA_OUT, dst_addr, src_addr, count[curr_slot]*DATA_BYTES);

                curr_slot = next_slot;
                curr_have = next_have;
            }

            fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
        }
    }

#if VERBOSE > 100
    printf("Result partial data:\n");
    for(int i = 0; i < BLOCK_SIZE; i++){
        printf("local_data[%2d]: %0d\n", i, mmio32(local_block_addr+i*DATA_BYTES));
    }
#endif

    // Store results back into the global array
    idma_1d(&idma_ctrl, DMA_OUT, (uint32_t)&bs_array[hartid*BLOCK_SIZE], local_block_addr, local_block_size);

    // Check results
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    if(hartid == 0){
#if VERBOSE > 100
        printf("Sorted array:\n");
        for(unsigned int i = 0; i < BS_ARRAY_SIZE; i++){
            printf("bs_array[%2d]: %0d\n", i, bs_array[i]);
        }
#endif

        for(int i = 0; i < BS_ARRAY_SIZE-1; i++){
            if(bs_array[i] > bs_array[i+1]){
                printf("[ERROR] - The array does not seem to be ordered: element %0d (%0d) > element %0d (%0d)\n", i, bs_array[i], i+1, bs_array[i+1]);
                return 1;
            }
        }
        return 0;
    }else{
        return 0;
    }
}