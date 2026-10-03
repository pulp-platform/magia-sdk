// Copyright 2024-2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Viviane Potocnik <vivianep@iis.ee.ethz.ch>
// Alberto Dequino <alberto.dequino@unibo.it>

#ifndef HAL_FSYNC_H
#define HAL_FSYNC_H
/**
 * Opens and initializes the fsync interface.
 */
extern int fsync_init();

/**
 * Synchronizes the current tile with the ones of the same synchronization level.
 */
extern int fsync_sync_level(uint32_t level, uint8_t dir);

/**
 * Gets the current tile's group ID for the selected synchronization level.
 */
extern int fsync_getgroup_level(uint32_t level, uint32_t id, uint8_t dir);

/**
 * Synchronizes the current tile with the ones of the same column.
 */
extern int fsync_sync_col();

/**
 * Synchronizes the current tile with the ones of the same row.
 */
extern int fsync_sync_row();

/**
 * Synchronizes mesh diagonal.
 */
extern int fsync_sync_diag();

/**
 * Synchronizes an arbitrary subset of tiles selected by the ids vector.
 */
extern int fsync_sync(uint32_t *ids, uint8_t n_tiles, uint8_t dir, uint8_t bid);

/**
 * Synchronizes with the tile on the left.
 */
extern int fsync_sync_left();

/**
 * Synchronizes with the tile on the right.
 */
extern int fsync_sync_right();

/**
 * Synchronizes with the tile above.
 */
extern int fsync_sync_up();

/**
 * Synchronizes with the tile below.
 */
extern int fsync_sync_down();

/**
 * Synchronizes the entire mesh.
 */
extern int fsync_sync_global();

extern void fsync_hnbr();
extern void fsync_vnbr();
extern void fsync_hring();
extern void fsync_vring();

#endif