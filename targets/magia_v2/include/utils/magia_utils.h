/*
 * Copyright (C) 2023-2025 ETH Zurich and University of Bologna
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
 * Authors: Victor Isachi <victor.isachi@unibo.it>
 * Alberto Dequino <alberto.dequino@unibo.it>
 *
 * MAGIA Utils
 */

#ifndef MAGIA_UTILS_H
#define MAGIA_UTILS_H

#include "magia_tile_utils.h"
#include "addr_map/tile_addr_map.h"
#include "regs/tile_ctrl.h"

#define h_pprintf(x) (h_psprint(get_hartid(), x))
#define n_pprintf(x) (n_psprint(get_hartid(), x))
#define   pprintf(x) (  psprint(get_hartid(), x))
#define   pprintln   (  pprintf("\n"))

inline uint32_t get_hartid(){
    uint32_t hartid;
    asm volatile("csrr %0, mhartid"
                 :"=r"(hartid):);
    return hartid;
}

// Lookup table indicating the id of row synchronization
inline uint32_t row_id_lookup(volatile uint32_t hartid_y){
  if (hartid_y < MESH_Y_TILES/2) return 2*hartid_y;
  else                           return 2*(hartid_y-MESH_Y_TILES/2);
}

// Lookup table indicating the id of column synchronization
inline uint32_t col_id_lookup(volatile uint32_t hartid_x){
  if (hartid_x < MESH_X_TILES/2) return 2*hartid_x+1;
  else                           return 2*(hartid_x-MESH_X_TILES/2)+1;
}

inline uint32_t get_l1_base(uint32_t hartid){
    return L1_BASE + hartid * L1_TILE_OFFSET;
}

#endif /* MAGIA_UTILS_H */
