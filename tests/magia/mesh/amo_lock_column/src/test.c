// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Alberto Dequino <alberto.dequino@unibo.it>

#include <stdint.h>
#include "test.h"

#include "tile.h"
#include "fsync.h"
#include "eventunit.h"
#include "utils/cache_fill.h"

#define WAIT_MODE POLLING
#define CACHE_HEAT_CYCLES (3)

int main(void){
    // Filling up the cache
    fill_icache();

    /** 
     * 0. Get the mesh-tile's hartid, and also initialize fsync + eu
     */
    uint32_t hartid = get_hartid();
    uint32_t y_id   = GET_Y_ID(hartid);
    uint32_t x_id   = GET_X_ID(hartid);
    //Assuming the mesh dimensions are a power of 2
    uint32_t centre_id = GET_ID(((MESH_Y_TILES / 2) - 1), x_id);
    //printf("Centre id is: %x\n", centre_id);

    fsync_config_t fsync_cfg = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {
        .base = NULL,
        .cfg = &fsync_cfg,
        .api = &fsync_api,
    };

    fsync_init(&fsync_ctrl);

    #if STALLING == 0
    eu_config_t eu_cfg = {.hartid = hartid};
    eu_controller_t eu_ctrl = {
        .base = NULL,
        .cfg = &eu_cfg,
        .api = &eu_api,
    };
    eu_init(&eu_ctrl);
    eu_fsync_init(&eu_ctrl, 0);
    #endif

    uint32_t l1_tile_base = get_l1_base(hartid);

    /** 
     * 1a. Initialize the personal lock
     */
    uint32_t mynode = l1_tile_base;
    ((lock_node*)(mynode)) -> next   = NULL;
    ((lock_node*)(mynode)) -> locked = 0;


    /**
     * 1b.Get the address of the tail (it's in the "central" tile).
     * Also initialize the tail to NULL.
     */
    uint32_t tail_a = get_l1_base(centre_id) + sizeof(lock_node);
    if(hartid == centre_id)
        mmio32(tail_a) = NULL;
    // Synch all the tiles
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);

    #if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
    #endif
    for (int i = 0; i < CACHE_HEAT_CYCLES; i++) {
        sentinel_start();
        
        /**
         * 2a. Amo lock test, get the lock to enter the protected code area.
         */
        amo_lock(tail_a, mynode);
        //amo_lock_naive(tail_a);

        /**
         * 2b. Protected code area.
         * Write own hartid on shared global value.
         * Wait a bit.
         * Check if the value is still the same, and nobody else got inside.  
         */
        mmio32(&value + (uint32_t)(4 * x_id)) = hartid;
        wait_nop(100);
        if(mmio32(&value + (uint32_t)(4 * x_id)) != hartid){
            printf("We fucking loooooost... on core %d\n", hartid);
        }

        /**
         * 2c. UNLOCK https://libraryofruina.wiki.gg/wiki/Unlock-%E2%85%A0
         */
        amo_unlock(tail_a, mynode);
        //amo_unlock_naive(tail_a);
        sentinel_end();

        /**
         * 3. Synch all the tiles and return
         */
        // Synch all the tiles
        fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);

        #if STALLING == 0
        eu_fsync_wait(&eu_ctrl, WAIT_MODE);
        #endif
    }

    if(hartid == 0)
        printf("Uh I guess the lock works?\n");

    return 0;
}