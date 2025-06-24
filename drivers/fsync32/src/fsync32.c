// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Viviane Potocnik <vivianep@iis.ee.ethz.ch>
// Alberto Dequino <alberto.dequino@unibo.it>
//
// This file provides the strong (driver-specific) implementations for the
// Fractal Sync functions using 32-bits levels.
// These functions override the weak HAL symbols.
// This is a WIP and might be redundant, as the moment of writing there is only one FSync configuration tested on MAGIA.


#include <stdint.h>
#include "fsync32.h"
#include "regs/tile_ctrl.h"
#include "addr_map/tile_addr_map.h"
#include "utils/fsync_isa_utils.h"
#include "utils/tinyprintf.h"
#include "utils/magia_utils.h"

int fsync32_init(fsync_controller_t *ctrl) {
    irq_en(1<<IRQ_FSYNC_DONE);
    return 0;
}

/**
 * Synchronize the tile with the others of the selected horizzontal synchronization tree level.
 * Level 0 will synchronize with the neighbor tile.
 * Increasing the level will synchronize with more neighborhoods following the horizzontal fsync tree.
 * At maximum level (log_2(N_tiles)) the entire mesh is synchronized.
 * Uncomment the asm volatile line only if you enabled interrupt mode when building the MAGIA architecture.
 *
 * @param level Horizzontal tree level over which synchronize.
 */
int fsync32_sync_level_h(fsync_controller_t *ctrl, uint32_t level){
    if(level >= MAX_SYNC_LVL){
        printf("Error: synchronization level is too high! Maximum level is: %d", MAX_SYNC_LVL);
        return 1;
    }
    //printf("Calling sync level %d, using %d", level, (uint32_t) (0xFFFFFFFF >> (31 - level)));
    //fsync((uint8_t) (1 << (level)));
    fsync(0, (uint32_t) (0xFFFFFFFF >> (31 - level)));
    //asm volatile("wfi" ::: "memory");
    return 0;
}

/**
 * Gets the group ID of the current tile in the selected synchronization level on the horizzontal fsync tree.
 * When calling sync_level_h(level), all the tiles with the same horizzontal group ID of that level will synchronize.
 *
 * @param level Synch level to get the horizzontal group ID from.
 */
int fsync32_getgroup_level_h(fsync_controller_t *ctrl, uint32_t level){
    uint32_t id = get_hartid();
    return ((GET_X_ID(id) >> ((level + 2) / 2)) + ((GET_Y_ID(id) >> ((level + 1) / 2))*(MESH_X_TILES >> ((level + 2) / 2))));
}

extern int fsync_init(fsync_controller_t *ctrl)
    __attribute__((alias("fsync32_init"), used, visibility("default")));
extern int fsync_sync_level_h(fsync_controller_t *ctrl, uint32_t level)
    __attribute__((alias("fsync32_sync_level_h"), used, visibility("default")));
extern int fsync_getgroup_level_h(fsync_controller_t *ctrl, uint32_t level)
    __attribute__((alias("fsync32_getgroup_level_h"), used, visibility("default")));

/* Export the FSYNC-specific controller API */
fsync_controller_api_t fsync_api = {
    .init = fsync32_init,
    .sync_level_h = fsync32_sync_level_h,
    .getgroup_level_h = fsync32_getgroup_level_h,
};

