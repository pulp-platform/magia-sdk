/*
 * Copyright (C) 2025 ETH Zurich and University of Bologna
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
 * MAGIA Tile Address Map
 */

#ifndef _TILE_ADDR_MAP_INCLUDE_GUARD_
#define _TILE_ADDR_MAP_INCLUDE_GUARD_


#define NUM_L1_BANKS (32)
#define WORDS_BANK   (8192)
#define BITS_WORD    (32)
#define BITS_BYTE    (8)

#define RESERVED_START (0x00000000)
#define RESERVED_END   (0x0000FFFF)
#define STACK_START    (0x00010000)
#define STACK_END      (0x0001FFFF)
#define L1_BASE        (0x00020000)
#define L1_SIZE        (0x000DFFFF)
#define L1_TILE_OFFSET (0x00100000)
#define MHARTID_OFFSET (0x00100000)
#define L2_BASE        (0xCC000000)
#define TEST_END_ADDR  (0xCC030000)
#define PRINT_ADDR     (0xFFFF0004)

#define MESH_X_TILES 2
#define MESH_Y_TILES 2
#define NUM_HARTS    (MESH_X_TILES*MESH_Y_TILES)
#define MAX_SYNC_LVL 2
#define MESH_2_POWER 1

#define STR_OFFSET  (0x00000000)
#define STR_BASE    (RESERVED_START + STR_OFFSET)
#define SYNC_OFFSET (0x0000F000)
#define SYNC_BASE   (RESERVED_START + SYNC_OFFSET)
#define SYNC_EN     (SYNC_BASE + 0x4)

#define GET_X_ID(mhartid)  ((mhartid)%MESH_Y_TILES)
#define GET_Y_ID(mhartid)  ((mhartid)/MESH_Y_TILES)
#define GET_ID(y_id, x_id) (((y_id)*MESH_Y_TILES)+(x_id))

#endif // _TILE_ADDR_MAP_INCLUDE_GUARD_