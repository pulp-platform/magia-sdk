/*
 * Copyright (C) 2023-2024 ETH Zurich and University of Bologna
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
 * MAGIA FractalSync ISA Utils
 */

#ifndef FSYNC_ISA_UTILS_H
#define FSYNC_ISA_UTILS_H

#include "utils/tinyprintf.h"
#include "addr_map/tile_addr_map.h"

#define _FS_GLOBAL_AGGR (0xFFFFFFFF >> (1+__builtin_clz(NUM_HARTS)))
#define _FS_GLOBAL_ID   (-1)
#define _FS_HNBR_AGGR   (0x1)
#define _FS_HNBR_ID     (0)
#define _FS_VNBR_AGGR   (0x1)
#define _FS_VNBR_ID     (1)
#define _FS_HRING_AGGR  (0x1)
#define _FS_HRING_ID    (2)
#define _FS_VRING_AGGR  (0x1)
#define _FS_VRING_ID    (3)
#define _FS_RC_LVL      (0x1 << (29-__builtin_clz(NUM_HARTS)))
#define _FS_RC_AGGR     (0x155 >> (__builtin_clz(NUM_HARTS)-21))

#ifdef FSYNC_MM
#define FSYNC_MM_AGGR_REG_OFFSET    (0x00)
#define FSYNC_MM_ID_REG_OFFSET      (0x04)
#define FSYNC_MM_CONTROL_REG_OFFSET (0x08)
#define FSYNC_MM_STATUS_REG_OFFSET  (0x0C)

/* Status register bits */
#define FSYNC_MM_STATUS_BUSY_MASK   (1 << 2)
#endif

/* synch instruction */
  // asm volatile(
  //      ".word (0x0       << 20) | \     /* Reserved - 0x0 */
  //             (0b00101   << 15) | \     /* R1 - t0 */
  //             (0b010     << 12) | \     /* FUNC3 */
  //             (0x0       <<  7) | \     /* Reserved - 0x0 */
  //             (0b1011011 <<  0)   \n"); /* OPCODE */
inline void fsync_legacy(volatile uint32_t level){
  asm volatile("addi t0, %0, 0" ::"r"(level));
  asm volatile(
       ".word (0x0       << 20) | \
              (0b00101   << 15) | \
              (0b010     << 12) | \
              (0x0       <<  7) | \
              (0b1011011 <<  0)   \n");
}

/* synch instruction */
  // asm volatile(
  //      ".word (0x0       << 25) | \     /* Reserved - 0x0 */
  //             (0b00110   << 20) | \     /* R2 - t1 */
  //             (0b00101   << 15) | \     /* R1 - t0 */
  //             (0b010     << 12) | \     /* FUNC3 */
  //             (0x0       <<  7) | \     /* Reserved - 0x0 */
  //             (0b1011011 <<  0)   \n"); /* OPCODE */
/**
 * This ISA instruction is the bread and butter for synchronizing the current tile with an arbitrary subset of 
 * other tiles in the MAGIA mesh. The functionality of this instruction is NOT easy to understand or correctly 
 * utilize, it is heavily recomended to avoid directly using this instruction and instead use the API wrappers  
 * hiding this behemoth.
 * 
 * But if you insist to use it for your little, special experiment, then be my fucking guest.
 * 
 * ID and AGGREGATE are the two values that guide the entire synchronization process.
 * 
 * Starting from the tile this instruction is called, the fractal sync tree is explored in reverse order, towards 
 * the root node.
 * 
 * If ID is even (or zero), the HORIZONTAL tree is explored. If ID is odd, the VERTICAL tree is explored.
 * 
 * The number of nodes traversed is equal to the number of significant bits passed in the AGGREGATE value. 
 * In fact, you should always pass AGGREGATE as a bit string to know what you are exactly doing. (I.E. 0b110)
 * 
 * The MSB is the highest level explored, the LSB is the lowest.
 * 
 * If there is at least one tile you want to synchronize with, which visible from a certain level, you should put 1 in 
 * the bit associated for that level. Instead, if there are no tiles you have to synchronize with in a certain 
 * level THAT YOU HAVEN'T SYNCHRONIZED WITH IN A PREVIOUS ONE, then you put 0 in its associated bit.
 * 
 * If you mess up and put 1 in a level in which there are no other tiles to synchronize with, 
 * or in which there are only tiles you already synchronized in a previous level, you are in Deadlock City baby. 
 * 
 * Told you to use the fucking APIs!
 * 
 * If the AGGREGATE value is EXACTLY 1, then a special behavior is activated. 
 * 
 * Depending on the ID value, the current tile will synchronize with:
 * 
 * ID == 0 : The tile at synchronization level 0 in the horizontal fsync tree
 * ID == 1 : The tile at synchronization level 0 in the vertical fsync tree
 * ID == 2 : The tile horizontally neighboring NOT visible at synch level 0 in the horizontal fsync tree
 * ID == 3 : The tile vertically neighboring NOT visible at synch level 0 in the vertical fsync tree
 * 
 * If you have no fucking clue on what you just read means, then I'm sorry but you are not in the right place.
 * Try the MAGIA repository README file!
 * 
 * There is more! The ID value, in fact, isn't just used to decide whether we explore the horizontal or vertical
 * tree. In fact, through a single synchronization tree node there might be multiple fsync calls going through in
 * parallel at the same time. 
 * 
 * To avoid collisions, multiple barriers are set up for each node, the one on which the instruction waits is 
 * chosen by the ID value.
 * 
 * It's up to YOU to make sure that the tiles that are synchronizing are calling the fsync over the same barrier ID.
 * 
 * Unfortunately, you can't choose whichever ID you want. There is a physical ID limit equal to:
 * 
 * - For the horizontal fsync tree: 2 * ((2^L) - 1)
 * - For the vertical fsync tree: 2 * ((2^L) - 1) + 1
 * - For the nodes that are in both fsync trees: ((2^L) - 1)
 * 
 * Where L is the level of the MSB in the AGGREGATE value. This means that you must have a solid idea of the 
 * nature of the highest node you are synchronizing with. And of the fsync map, in general. If you mess up, you're screwed.
 * 
 * I'd gladly write down some examples, but my ASCII art skills are dogshit, so instead just open the APIs and
 * take a look on how it is possible to do stuff like synchronizing specific rows, columns, diagnolas, the outer
 * ring, and from there you should have a rough idea on how to do this.
 * 
 * Maybe.
 * 
 * Good luck!
 */
inline void fsync(volatile uint32_t id, volatile uint32_t aggregate){
  #if FSYNC_MM == 0
  asm volatile("addi t1, %0, 0" ::"r"(id));
  asm volatile("addi t0, %0, 0" ::"r"(aggregate));
  asm volatile(
       ".word (0x0       << 25) | \
              (0b00110   << 20) | \
              (0b00101   << 15) | \
              (0b010     << 12) | \
              (0x0       <<  7) | \
              (0b1011011 <<  0)   \n");
  #else
  volatile char *fsync_base = (volatile char *)(FSYNC_BASE);
  
  *(volatile uint32_t *)(fsync_base + FSYNC_MM_AGGR_REG_OFFSET) = aggregate;
  *(volatile uint32_t *)(fsync_base + FSYNC_MM_ID_REG_OFFSET) = id;
  *(volatile uint32_t *)(fsync_base + FSYNC_MM_CONTROL_REG_OFFSET) = 1;
  
  #if STALLING == 1
  // Polling mode - wait for completion
  volatile uint32_t status;
  do {
    status = *(volatile uint32_t *)(fsync_base + FSYNC_MM_STATUS_REG_OFFSET);
  } while (status & FSYNC_MM_STATUS_BUSY_MASK);
  #endif
  #endif
}
  

#endif /*FSYNC_ISA_UTILS_H*/
