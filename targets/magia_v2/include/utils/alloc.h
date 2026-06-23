/*
 * Copyright (C) 2023-2026 ETH Zurich and University of Bologna
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
 * Authors: Alberto Dequino <alberto.dequino@unibo.it>
 *
 * MAGIA Attention ISA Utils
 */
#ifndef ALLOC_H_
#define ALLOC_H_

#include "magia_utils.h"

#define ALIGNMENT      (4)
#define ALIGN_4B(addr) (((addr) + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1))

#define L1_TILE_BASE   (L1_BASE + (get_hartid() * L1_TILE_OFFSET))
#define L1_TILE_END    (L1_TILE_BASE + L1_SIZE)

#define L1_TAIL        (L1_TILE_BASE)
#define L1_TAIL_START  ALIGN_4B(L1_TILE_BASE + (sizeof(uint32_t))) /* Reserves 4 Bytes for L1_TAIL \
                                                                    */

static inline void l1_alloc_init(void)
{
    mmio32(L1_TAIL) = L1_TAIL_START;
}

static inline void *l1_alloc(size_t size)
{
    uint32_t current_offset = mmio32(L1_TAIL);
    uint32_t next_offset    = ALIGN_4B(current_offset + size);

    /* check: out-of-memory || overflow (if 'size' is too big) */
    if (next_offset > L1_TILE_END || next_offset < current_offset) {
        return NULL;
    }

    mmio32(L1_TAIL) = next_offset;

    return (void *)(current_offset);
}

#endif /* ALLOG_H_ */