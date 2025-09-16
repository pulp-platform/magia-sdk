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
 * Synchronize the tile with the others of the selected synchronization tree level.
 * Level 0 will synchronize with the neighbor tile.
 * Increasing the level will synchronize with more neighborhoods following the fsync tree.
 * At maximum level (log_2(N_tiles)) the entire mesh is synchronized.
 * Uncomment the asm volatile line only if you enabled interrupt mode when building the MAGIA architecture.
 *
 * @param level Tree level over which synchronize.
 * @param dir Direction of the tree, 0 = Horizontal 1 = Vertical
 */
int fsync32_sync_level(fsync_controller_t *ctrl, uint32_t level, uint8_t dir){
    if(level >= MAX_SYNC_LVL){
        printf("Error: synchronization level is too high! Maximum level is: %d\n", MAX_SYNC_LVL - 1);
        return 1;
    }
    //printf("Calling sync level %d, using %d", level, (uint32_t) (0xFFFFFFFF >> (31 - level)));
    //fsync((uint8_t) (1 << (level)));
    fsync(dir, (uint32_t) (0xFFFFFFFF >> (31 - level)));
    //asm volatile("wfi" ::: "memory");
    return 0;
}

/**
 * Gets the group ID of the the tile ID in the selected synchronization level on the fsync tree.
 * When calling sync_level(level, dir), all the tiles with the same group ID of that level will synchronize,
 * as long as the tree direction is the same.
 *
 * @param level Synch level to get the horizzontal group ID from.
 * @param id Tile ID we want to know the synch group of.
 * @param dir Synchronization tree direction 0 = Horizontal 1 = Vertical
 */
int fsync32_getgroup_level(fsync_controller_t *ctrl, uint32_t level, uint32_t id, uint8_t dir){
    if(dir == 0)
        return ((GET_X_ID(id) >> ((level + 2) / 2)) + ((GET_Y_ID(id) >> ((level + 1) / 2))*(MESH_X_TILES >> ((level + 2) / 2))));
    else if(dir == 1)
        return ((GET_X_ID(id) >> ((level + 1) / 2)) + ((GET_Y_ID(id) >> ((level + 2) / 2))*(MESH_X_TILES >> ((level + 1) / 2))));
    else{
        printf("Direction value incorrect. Correct values: 0 for horizontal tree, 1 for vertical tree.\n");
        return -1;
    }
}

/**
 * Synchronize the tile with the others of the same mesh row.
 * ID is the tile row id multiplied by 2 (even=horizontal fsync).
 * Rows past the middle point can use the same ID as the ones before it.
 * Aggregate is 1 in even levels, 0 in odd levels.
 * Since each architecture rows/columns are a power of 2 (up to 32), I'm shifting 2 bits for each power difference.
 * It just works - Todd howard
 */
int fsync32_sync_row(fsync_controller_t *ctrl) {
    uint32_t y_id = GET_Y_ID(get_hartid()) % (MESH_Y_TILES/2);
    fsync(y_id * 2, (0b101010101 >> ((5 - MESH_2_POWER) * 2)));
    return 0;
}

/**
 * Synchronize the tile with the others of the same mesh column.
 */
int fsync32_sync_col(fsync_controller_t *ctrl) {
    uint32_t x_id = GET_X_ID(get_hartid()) % (MESH_X_TILES/2);
    fsync(((x_id * 2) + 1), (0b101010101 >> ((5 - MESH_2_POWER) * 2)));
    return 0;
}

/**
 * Synchronize the tile with the others of the diagonal.
 */
int fsync32_sync_diag(fsync_controller_t *ctrl) {
    if(GET_X_ID(get_hartid()) != GET_Y_ID(get_hartid())){
        printf("Error: non-diagonal tile attempted to synchronize with the diagonal.\n");
        return 1;
    }
    fsync(0, (0b1010101010 >> ((5 - MESH_2_POWER) * 2)));
    return 0;
}

/**
 * Synchronizes an arbitrary subset of tiles listed by a vector.
 * This algorithm follows the same logic you should be utilizing for writing the fsync function on your own,
 * but it does it automatically.
 * 
 * @param ids Vector list of the ids of the tiles we want to synchronize.
 * @param n_tiles Number of tiles to be synchronized.
 * @param dir Fractalsync tree direction (0=horizontal, 1=vertical)
 * @param bid Barrier ID for synchronization
 */
int fsync32_sync(fsync_controller_t *ctrl, uint32_t *ids, uint8_t n_tiles, uint8_t dir, uint8_t bid) {
    uint32_t aggregate = 0x00000000;
    uint32_t hartid = get_hartid();
    if(n_tiles <= 2){
        for(uint8_t i = 0; i < n_tiles; i++){
            if(hartid == ids[i])
                continue; 
            uint32_t x_diff = GET_X_ID(hartid) - GET_X_ID(ids[i]);
            uint32_t y_diff = GET_Y_ID(hartid) - GET_Y_ID(ids[i]);
            x_diff = x_diff * x_diff;
            y_diff = y_diff * y_diff;
            if(x_diff == 0 && y_diff == 1){
                if(fsync32_getgroup_level(ctrl, 0, hartid, 1) == fsync32_getgroup_level(ctrl, 0, ids[i], 1))
                    fsync(1, 0b1);
                else
                    fsync(3, 0b1);
                return 0;
            }
            else if(x_diff == 1 && y_diff == 0){
                if(fsync32_getgroup_level(ctrl, 0, hartid, 0) == fsync32_getgroup_level(ctrl, 0, ids[i], 0))
                    fsync(0, 0b1);
                else
                    fsync(2, 0b1);
                return 0;
            }
        }
    }
    for(uint8_t i = 0; i < n_tiles; i++){
        if(hartid == ids[i])
            continue;
        for(uint8_t j = 0; j < MAX_SYNC_LVL; j++){
            if(fsync32_getgroup_level(ctrl, j, hartid, dir) == fsync32_getgroup_level(ctrl, j, ids[i], dir)){
                aggregate = aggregate | ((uint32_t)(1 << j));
                break;
            }
        }
    }
    if(aggregate){
        fsync((uint32_t)((bid * 2) + dir), aggregate);
        return 0;
    }    
    return 1;
}

/**
 * Synchronizes with the tile on the left.
 */
int fsync32_sync_left(fsync_controller_t *ctrl){
    uint32_t hartid = get_hartid();
    if(GET_X_ID(hartid) == 0)
        return 1;
    if(fsync32_getgroup_level(ctrl, 0, hartid, 0) == fsync32_getgroup_level(ctrl, 0, (hartid - 1), 0))
        fsync(0, 0b1);
    else
        fsync(2, 0b1);
    return 0; 
}

/**
 * Synchronizes with the tile on the right.
 */
int fsync32_sync_right(fsync_controller_t *ctrl){
    uint32_t hartid = get_hartid();
    if(GET_X_ID(hartid) == MESH_X_TILES - 1)
        return 1;
    if(fsync32_getgroup_level(ctrl, 0, hartid, 0) == fsync32_getgroup_level(ctrl, 0, (hartid + 1), 0))
        fsync(0, 0b1);
    else
        fsync(2, 0b1);
    return 0; 
}

/**
 * Synchronizes with the tile above.
 */
int fsync32_sync_up(fsync_controller_t *ctrl){
    uint32_t hartid = get_hartid();
    if(GET_Y_ID(hartid) == 0)
        return 1;
    if(fsync32_getgroup_level(ctrl, 0, hartid, 1) == fsync32_getgroup_level(ctrl, 0, (hartid - MESH_X_TILES), 1))
        fsync(1, 0b1);
    else
        fsync(3, 0b1);
    return 0; 
}

/**
 * Synchronizes with the tile below.
 */
int fsync32_sync_down(fsync_controller_t *ctrl){
    uint32_t hartid = get_hartid();
    if(GET_Y_ID(hartid) == MESH_Y_TILES - 1)
        return 1;
    if(fsync32_getgroup_level(ctrl, 0, hartid, 1) == fsync32_getgroup_level(ctrl, 0, (hartid + MESH_X_TILES), 1))
        fsync(1, 0b1);
    else
        fsync(3, 0b1);
    return 0; 
}

void fsync32_hnbr(fsync_controller_t *ctrl){
  fsync(_FS_HNBR_ID, _FS_HNBR_AGGR);
}

void fsync32_vnbr(fsync_controller_t *ctrl){
  fsync(_FS_VNBR_ID, _FS_VNBR_AGGR);
}

void fsync32_hring(fsync_controller_t *ctrl){
  uint32_t hartid   = get_hartid();
  uint32_t hartid_x = GET_X_ID(hartid);
  uint32_t hartid_y = GET_Y_ID(hartid);
  if ((hartid_x == 0) || (hartid_x == MESH_X_TILES-1)){
    uint32_t id = row_id_lookup(hartid_y);
    fsync(id, _FS_RC_LVL);
  } else {
    fsync(_FS_HRING_ID, _FS_HRING_AGGR);
  }
}

void fsync32_vring(fsync_controller_t *ctrl){
  uint32_t hartid   = get_hartid();
  uint32_t hartid_x = GET_X_ID(hartid);
  uint32_t hartid_y = GET_Y_ID(hartid);
  if ((hartid_y == 0) || (hartid_y == MESH_Y_TILES-1)){
    uint32_t id = col_id_lookup(hartid_x);
    fsync(id, _FS_RC_LVL);
  } else {
    fsync(_FS_VRING_ID, _FS_VRING_AGGR);
  }
}

extern int fsync_init(fsync_controller_t *ctrl)
    __attribute__((alias("fsync32_init"), used, visibility("default")));
extern int fsync_sync_level(fsync_controller_t *ctrl, uint32_t level, uint8_t dir)
    __attribute__((alias("fsync32_sync_level"), used, visibility("default")));
extern int fsync_getgroup_level(fsync_controller_t *ctrl, uint32_t level, uint32_t id, uint8_t dir)
    __attribute__((alias("fsync32_getgroup_level"), used, visibility("default")));
extern int fsync_sync_row(fsync_controller_t *ctrl)
    __attribute__((alias("fsync32_sync_row"), used, visibility("default")));
extern int fsync_sync_col(fsync_controller_t *ctrl)
    __attribute__((alias("fsync32_sync_col"), used, visibility("default")));
extern int fsync_sync_diag(fsync_controller_t *ctrl)
    __attribute__((alias("fsync32_sync_diag"), used, visibility("default")));
extern int fsync_sync(fsync_controller_t *ctrl, uint32_t *ids, uint8_t n_tiles, uint8_t dir, uint8_t bid)
    __attribute__((alias("fsync32_sync"), used, visibility("default")));
extern int fsync_sync_left(fsync_controller_t *ctrl)
    __attribute__((alias("fsync32_sync_left"), used, visibility("default")));
extern int fsync_sync_right(fsync_controller_t *ctrl)
    __attribute__((alias("fsync32_sync_right"), used, visibility("default")));
extern int fsync_sync_up(fsync_controller_t *ctrl)
    __attribute__((alias("fsync32_sync_up"), used, visibility("default")));
extern int fsync_sync_down(fsync_controller_t *ctrl)
    __attribute__((alias("fsync32_sync_down"), used, visibility("default")));
extern void fsync_hnbr(fsync_controller_t *ctrl)
    __attribute__((alias("fsync32_hnbr"), used, visibility("default")));
extern void fsync_vnbr(fsync_controller_t *ctrl)
    __attribute__((alias("fsync32_vnbr"), used, visibility("default")));
extern void fsync_hring(fsync_controller_t *ctrl)
    __attribute__((alias("fsync32_hring"), used, visibility("default")));
extern void fsync_vring(fsync_controller_t *ctrl)
    __attribute__((alias("fsync32_vring"), used, visibility("default")));

/* Export the FSYNC-specific controller API */
fsync_controller_api_t fsync_api = {
    .init = fsync32_init,
    .sync_level = fsync32_sync_level,
    .getgroup_level = fsync32_getgroup_level,
    .sync_col = fsync32_sync_col,
    .sync_row = fsync32_sync_row,
    .sync_diag = fsync32_sync_diag,
    .sync = fsync32_sync,
    .sync_left = fsync32_sync_left,
    .sync_right = fsync32_sync_right,
    .sync_up = fsync32_sync_up,
    .sync_down = fsync32_sync_down,
    .hnbr = fsync32_hnbr,
    .vnbr = fsync32_vnbr,
    .hring = fsync32_hring,
    .vring = fsync32_vring,
};

