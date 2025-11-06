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
 * Luca Balboni <luca.balboni10@studio.unibo.it>
 * 
 * MAGIA Tile Address Map
 */

#ifndef _TILE_ADDR_MAP_INCLUDE_GUARD_
#define _TILE_ADDR_MAP_INCLUDE_GUARD_


#define NUM_L1_BANKS (32)
#define WORDS_BANK   (8192)
#define BITS_WORD    (32)
#define BITS_BYTE    (8)


#define REDMULE_BASE   (0x00000100)
#define REDMULE_END    (0x000001FF)
#define IDMA_BASE      (0x00000200)
#define IDMA_END       (0x000005FF)
#define FSYNC_BASE     (0x00000600)
#define FSYNC_END      (0x000006FF)
#define EU_BASE        (0x00000700)
#define EU_END         (0x000016FF)
#define RESERVED_START (0x00001700)   
#define RESERVED_END   (0x0000FFFF)   
#define STACK_START    (0x00010000)
#define STACK_END      (0x0001FFFF)
#define L1_BASE        (0x00020000)
#define L1_SIZE        (0x000DFFFF)
#define L1_TILE_OFFSET (0x00100000)
#define L2_BASE        (0xCC000000)
#define TEST_END_ADDR  (0xCC030000)
#define PRINT_ADDR     (0xFFFF0004)
#define MHARTID_OFFSET (0x00100000)

#define IDMA_BASE_AXI2OBI (IDMA_BASE)           // direction=0, L2 to L1
#define IDMA_BASE_OBI2AXI (IDMA_BASE + 0x200)   // direction=1, L1 to L2

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

//=============================================================================
// Event Unit Register Map - Base addresses and offsets
//=============================================================================

// Core Event Unit registers - Main control and status
#define EU_CORE_MASK                        (EU_BASE + 0x00)          // R/W: Event mask (enables event lines)
#define EU_CORE_MASK_AND                    (EU_BASE + 0x04)          // W: Clear bits in mask
#define EU_CORE_MASK_OR                     (EU_BASE + 0x08)          // W: Set bits in mask
#define EU_CORE_IRQ_MASK                    (EU_BASE + 0x0C)          // R/W: IRQ event mask
#define EU_CORE_IRQ_MASK_AND                (EU_BASE + 0x10)          // W: Clear IRQ mask bits
#define EU_CORE_IRQ_MASK_OR                 (EU_BASE + 0x14)          // W: Set IRQ mask bits
#define EU_CORE_STATUS                      (EU_BASE + 0x18)          // R: Core clock status
#define EU_CORE_BUFFER                      (EU_BASE + 0x1C)          // R: Event buffer
#define EU_CORE_BUFFER_MASKED               (EU_BASE + 0x20)          // R: Buffer with mask applied
#define EU_CORE_BUFFER_IRQ_MASKED           (EU_BASE + 0x24)          // R: Buffer with IRQ mask
#define EU_CORE_BUFFER_CLEAR                (EU_BASE + 0x28)          // W: Clear received events
#define EU_CORE_SW_EVENTS_MASK              (EU_BASE + 0x2C)          // R/W: SW event target mask
#define EU_CORE_SW_EVENTS_MASK_AND          (EU_BASE + 0x30)          // W: Clear SW target bits
#define EU_CORE_SW_EVENTS_MASK_OR           (EU_BASE + 0x34)          // W: Set SW target bits

// Core Event Unit wait registers - Sleep functionality
#define EU_CORE_EVENT_WAIT                  (EU_BASE + 0x38)          // R: Sleep until event
#define EU_CORE_EVENT_WAIT_CLEAR            (EU_BASE + 0x3C)          // R: Sleep + clear buffer

// Hardware barrier registers (0x20 * barr_id offset)
#define HW_BARR_TRIGGER_MASK                (EU_BASE + 0x400)         // R/W: Barrier trigger mask
#define HW_BARR_STATUS                      (EU_BASE + 0x404)         // R: Barrier status
#define HW_BARR_TARGET_MASK                 (EU_BASE + 0x40C)         // R/W: Barrier target mask
#define HW_BARR_TRIGGER                     (EU_BASE + 0x410)         // W: Manual barrier trigger
#define HW_BARR_TRIGGER_SELF                (EU_BASE + 0x414)         // R: Automatic trigger
#define HW_BARR_TRIGGER_WAIT                (EU_BASE + 0x418)         // R: Trigger + sleep
#define HW_BARR_TRIGGER_WAIT_CLEAR          (EU_BASE + 0x41C)         // R: Trigger + sleep + clear

// Software event trigger registers (0x04 * sw_event_id offset)
#define EU_CORE_TRIGG_SW_EVENT              (EU_BASE + 0x600)         // W: Generate SW event
#define EU_CORE_TRIGG_SW_EVENT_WAIT         (EU_BASE + 0x640)         // R: Generate event + sleep
#define EU_CORE_TRIGG_SW_EVENT_WAIT_CLEAR   (EU_BASE + 0x680)         // R: Generate event + sleep + clear

// SoC event FIFO register
#define EU_CORE_CURRENT_EVENT               (EU_BASE + 0x700)         // R: SoC event FIFO

// Hardware mutex registers (0x04 * mutex_id offset)
#define EU_CORE_HW_MUTEX                    (EU_BASE + 0x0C0)         // R/W: HW mutex management

#endif // _TILE_ADDR_MAP_INCLUDE_GUARD_