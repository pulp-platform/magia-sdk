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

static char* utoa(unsigned int value, unsigned int base, char* result) {
    if (base < 2 || base > 16){
        *result = '\0'; 
        return result; 
    }

    char *ptr0 = result;
    char *ptr1 = result;
    char tmp_char;
    unsigned int tmp_value;

    do {
        tmp_value = value;
        value    /= base;
        *ptr0++   = "0123456789ABCDEF"[tmp_value - value * base];
    } while (value);

    *ptr0-- = '\0';
  
    // Reverse the string
    while(ptr1 < ptr0) {
        tmp_char = *ptr0;
        *ptr0--  = *ptr1;
        *ptr1++  = tmp_char;
    }
    return result;
}

static char* bs(uint32_t x) {
    uint32_t hartid = get_hartid();
    char *address = STR_BASE + L1_TILE_OFFSET*hartid;

    return utoa(x, 2, address);
}

static char* ds(uint32_t x) {
    uint32_t hartid = get_hartid();
    char *address = STR_BASE + L1_TILE_OFFSET*hartid;

    return utoa(x, 10, address);
}

static char* hs(uint32_t x) {
    uint32_t hartid = get_hartid();
    char *address = STR_BASE + L1_TILE_OFFSET*hartid;

    return utoa(x, 16, address);
}

static void h_psprint(uint32_t hartid, const char* string){
    mmio8(0xFFFF0004 + (hartid*4)) = '[';
    mmio8(0xFFFF0004 + (hartid*4)) = 'm';
    mmio8(0xFFFF0004 + (hartid*4)) = 'h';
    mmio8(0xFFFF0004 + (hartid*4)) = 'a';
    mmio8(0xFFFF0004 + (hartid*4)) = 'r';
    mmio8(0xFFFF0004 + (hartid*4)) = 't';
    mmio8(0xFFFF0004 + (hartid*4)) = 'i';
    mmio8(0xFFFF0004 + (hartid*4)) = 'd';
    mmio8(0xFFFF0004 + (hartid*4)) = ' ';
    char*   mhardid_str = ds(hartid);
    uint8_t mhartid_idx = 0;
    while (mhardid_str[mhartid_idx] != '\0')
        mmio8(0xFFFF0004 + (hartid*4)) = mhardid_str[mhartid_idx++];
    mmio8(0xFFFF0004 + (hartid*4)) = ']';
    mmio8(0xFFFF0004 + (hartid*4)) = ' ';

    uint8_t index = 0;
    while (string[index] != '\0')
        mmio8(0xFFFF0004 + (hartid*4)) = string[index++];
}

static void n_psprint(uint32_t hartid, const char* string){
    uint8_t index = 0;
    while (string[index] != '\0')
        mmio8(0xFFFF0004 + (hartid*4)) = string[index++];
    mmio8(0xFFFF0004 + (hartid*4)) = '\n';
}

static void psprint(uint32_t hartid, const char* string){
    uint8_t index = 0;
    while (string[index] != '\0')
        mmio8(0xFFFF0004 + (hartid*4)) = string[index++];
}

static void magia_return(uint32_t hartid, uint32_t exit_code){
    printf("Tile %d returned.\n", hartid);
    mmio16(TEST_END_ADDR + hartid*2) = (uint16_t) (exit_code - get_hartid());
}

#endif /*MAGIA_UTILS_H*/