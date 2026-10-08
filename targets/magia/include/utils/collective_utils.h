/*
 * Copyright (C) 2026 ETH Zurich and University of Bologna and ChipsIT
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Authors: Carlotta Chiarini <carlotta.charini@chips.it>
 * Alberto Dequino <alberto.dequino@unibo.it>
 *
 * MAGIA Collectives - Generic utilities for FlooNoC collective functionalities
 */

#ifndef COLLECTIVE_UTILS_H
#define COLLECTIVE_UTILS_H

#include <stdint.h>
#include "magia_tile_utils.h"

static inline void floo_set_collective_mask(uint32_t collective_mask) {
    mmio32(COLLECTIVE_MASK_OFFSET) = collective_mask;
}

static inline void floo_set_collective_op(uint32_t collective_op) {
    mmio32(COLLECTIVE_OP_OFFSET) = collective_op;
}

/**
 * Broadcast word on a single row
 *
 * @param collective_word 32-bit word broadcasted to all the tiles in the comm_group
 * @param addr_offset Destination address offset
 */
static inline void floo_narrow_mcast_row(uint32_t collective_word, uint32_t addr_offset)
{
    floo_set_collective_mask(ROW_MASK);
    floo_set_collective_op(MULTICAST);
    mmio32(COLLECTIVE_ADDR_OFFSET + addr_offset) = collective_word;
}

/**
 * Broadcast word on a single column
 *
 * @param collective_word 32-bit word broadcasted to all the tiles in the comm_group
 * @param addr_offset Destination address offset
 */
static inline void floo_narrow_mcast_column(uint32_t collective_word, uint32_t addr_offset)
{
    floo_set_collective_mask(COLUMN_MASK);
    floo_set_collective_op(MULTICAST);
    mmio32(COLLECTIVE_ADDR_OFFSET + addr_offset) = collective_word;
}

/**
 * Broadcast word on the entire mesh
 *
 * @param collective_word 32-bit word broadcasted to all the tiles in the comm_group
 * @param addr_offset Destination address offset
 */
static inline void floo_narrow_mcast_mesh(uint32_t collective_word, uint32_t addr_offset)
{
    floo_set_collective_mask(MESH_MASK);
    floo_set_collective_op(MULTICAST);
    mmio32(COLLECTIVE_ADDR_OFFSET + addr_offset) = collective_word;
}


/**
 * Syncronization on single row
 *
 * @param collective_word Tile-local word participating in the collective AND operation
 * @param addr_offset Destination address offset
 */
static inline void floo_narrow_sync_row(uint32_t collective_word, uint32_t addr_offset)
{
    floo_set_collective_mask(ROW_MASK);
    floo_set_collective_op(LSBAND);
    mmio32(COLLECTIVE_ADDR_OFFSET + addr_offset) = collective_word;
    // Compiler barrier
    asm volatile("" :::"memory");
}

/**
 * Syncronization on single column
 *
 * @param collective_word Tile-local word participating in the collective AND operation
 * @param addr_offset Destination address offset
 */
static inline void floo_narrow_sync_column(uint32_t collective_word, uint32_t addr_offset)
{
    floo_set_collective_mask(COLUMN_MASK);
    floo_set_collective_op(LSBAND);
    mmio32(COLLECTIVE_ADDR_OFFSET + addr_offset) = collective_word;
    // Compiler barrier
    asm volatile("" :::"memory");
}

/**
 * Syncronization on entire mesh
 *
 * @param collective_word Tile-local word participating in the collective AND operation
 * @param addr_offset Destination address offset
 */
int floo_narrow_sync_mesh(uint32_t collective_word, uint32_t addr_offset)
{
    floo_set_collective_mask(MESH_MASK);
    floo_set_collective_op(LSBAND);
    mmio32(COLLECTIVE_ADDR_OFFSET + addr_offset) = collective_word;
    // Compiler barrier
    asm volatile("" :::"memory");
}

#endif