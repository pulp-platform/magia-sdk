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
// This is a WIP and might be redundant, as the moment of writing there is only one collective
// configuration tested on MAGIA.

#include <stdint.h>
#include "collective.h"
#include "regs/tile_ctrl.h"
#include "addr_map/tile_addr_map.h"
// #include "utils/tinyprintf.h"
#include "utils/printf.h"
#include "utils/magia_utils.h"



static void floo_set_collective_mask(uint32_t collective_mask) {
    mmio32(COLLECTIVE_MASK_OFFSET) = collective_mask;
}

static void floo_set_collective_op(uint32_t collective_op) {
    mmio32(COLLECTIVE_OP_OFFSET) = collective_op;
}

/**
 * Build the collective mask for the given collective group of communicators.
 */
int floo_gen_collective_mask(floo_collective_t *ctrl, uint32_t group)
{
    uint32_t x_bits = __builtin_ctz(MESH_Y_TILES);
    uint32_t x_mask = (MESH_Y_TILES - 1) << MASK_OFFSET;
    uint32_t y_mask = (MESH_X_TILES - 1) << (MASK_OFFSET + x_bits);

    switch (group) {
    case MESH:
        return x_mask | y_mask;
    case ROW:
        return x_mask;
    case COLUMN:
        return y_mask;
    default:
        return 0;
    }
}


/**
 * Broadcast word on a single row
 *
 * @param collective_word 32-bit word broadcasted to all the tiles in the comm_group
 * @param addr_offset Destination address offset
 */
int floo_narrow_mcast_row(floo_collective_t *ctrl, uint32_t collective_word, uint32_t addr_offset)
{
    floo_set_collective_mask(gen_collective_mask(ctrl, ROW));
    floo_set_collective_op(MULTICAST);
    mmio32(COLLECTIVE_ADDR_OFFSET + addr_offset) = collective_word;

    return 0;
}

/**
 * Broadcast word on a single column
 *
 * @param collective_word 32-bit word broadcasted to all the tiles in the comm_group
 * @param addr_offset Destination address offset
 */
int floo_narrow_mcast_column(floo_collective_t *ctrl, uint32_t collective_word, uint32_t addr_offset)
{
    floo_set_collective_mask(gen_collective_mask(ctrl, COLUMN));
    floo_set_collective_op(MULTICAST);
    mmio32(COLLECTIVE_ADDR_OFFSET + addr_offset) = collective_word;

    return 0;
}

/**
 * Broadcast word on entire mesh
 *
 * @param collective_word 32-bit word broadcasted to all the tiles in the comm_group
 * @param addr_offset Destination address offset
 */
int floo_narrow_mcast_mesh(floo_collective_t *ctrl, uint32_t collective_word, uint32_t addr_offset)
{
    floo_set_collective_mask(gen_collective_mask(ctrl, MESH));
    floo_set_collective_op(MULTICAST);
    mmio32(COLLECTIVE_ADDR_OFFSET + addr_offset) = collective_word;

    return 0;
}

/**
 * Syncronization on single row
 *
 * @param collective_word Tile-local word participating in the collective AND operation
 * @param addr_offset Destination address offset
 */
int floo_narrow_sync_row(floo_collective_t *ctrl, uint32_t collective_word, uint32_t addr_offset)
{
    floo_set_collective_mask(gen_collective_mask(ctrl, ROW));
    floo_set_collective_op(LSBAND);
    mmio32(COLLECTIVE_ADDR_OFFSET + addr_offset) = collective_word;
    // Compiler barrier
    asm volatile("" :::"memory");
    
    return 0;
}


/**
 * Syncronization on single column
 *
 * @param collective_word Tile-local word participating in the collective AND operation
 * @param addr_offset Destination address offset
 */
int floo_narrow_sync_column(floo_collective_t *ctrl, uint32_t collective_word, uint32_t addr_offset)
{
    floo_set_collective_mask(gen_collective_mask(ctrl, COLUMN));
    floo_set_collective_op(LSBAND);
    mmio32(COLLECTIVE_ADDR_OFFSET + addr_offset) = collective_word;
    // Compiler barrier
    asm volatile("" :::"memory");

    return 0;
}


/**
 * Syncronization on entire mesh
 *
 * @param collective_word Tile-local word participating in the collective AND operation
 * @param addr_offset Destination address offset
 */
int floo_narrow_sync_mesh(floo_collective_t *ctrl, uint32_t collective_word, uint32_t addr_offset)
{
    floo_set_collective_mask(gen_collective_mask(ctrl, MESH));
    floo_set_collective_op(LSBAND);
    mmio32(COLLECTIVE_ADDR_OFFSET + addr_offset) = collective_word;
    // Compiler barrier
    asm volatile("" :::"memory");

    return 0;
}

extern int gen_collective_mask(floo_collective_t *ctrl, uint32_t group)
    __attribute__((alias("floo_gen_collective_mask"), used, visibility("default")));
extern int narrow_mcast_row(floo_collective_t *ctrl, uint32_t collective_word, uint32_t addr_offset)
    __attribute__((alias("floo_narrow_mcast_row"), used, visibility("default")));
extern int narrow_mcast_column(floo_collective_t *ctrl, uint32_t collective_word, uint32_t addr_offset)
    __attribute__((alias("floo_narrow_mcast_column"), used, visibility("default")));
extern int narrow_mcast_mesh(floo_collective_t *ctrl, uint32_t collective_word, uint32_t addr_offset)
    __attribute__((alias("floo_narrow_mcast_mesh"), used, visibility("default")));
extern int narrow_sync_row(floo_collective_t *ctrl, uint32_t collective_word, uint32_t addr_offset)
    __attribute__((alias("floo_narrow_sync_row"), used, visibility("default")));
extern int narrow_sync_column(floo_collective_t *ctrl, uint32_t collective_word, uint32_t addr_offset)
    __attribute__((alias("floo_narrow_sync_column"), used, visibility("default")));
extern int narrow_sync_mesh(floo_collective_t *ctrl, uint32_t collective_word, uint32_t addr_offset)
    __attribute__((alias("floo_narrow_sync_mesh"), used, visibility("default")));


/* Export the FlooNoC collective controller API */
floo_collective_api_t floonoc_collective_api = {
    .gen_collective_mask = floo_gen_collective_mask,
    .mcast_row           = floo_narrow_mcast_row,
    .mcast_column        = floo_narrow_mcast_column,
    .mcast_mesh          = floo_narrow_mcast_mesh,
    .sync_row            = floo_narrow_sync_row,
    .sync_column         = floo_narrow_sync_column,
    .sync_mesh           = floo_narrow_sync_mesh,
};
