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
 * MAGIA Tile Control registers and IRQ
 */

#ifndef _TILE_REG_DEFS_
#define _TILE_REG_DEFS_

#define DEFAULT_EXIT_CODE            (0xDEFC)
#define PASS_EXIT_CODE               (0xAAAA)
#define FAIL_EXIT_CODE               (0xFFFF)
#define IRQ_REDMULE_EVT_0            (31)
#define IRQ_REDMULE_EVT_1            (30)
#define IRQ_A2O_ERROR                (29)
#define IRQ_O2A_ERROR                (28)
#define IRQ_A2O_DONE                 (27)
#define IRQ_O2A_DONE                 (26)
#define IRQ_A2O_START                (25)
#define IRQ_O2A_START                (24)
#define IRQ_A2O_BUSY                 (23)
#define IRQ_O2A_BUSY                 (22)
#define IRQ_REDMULE_BUSY             (21)
#define IRQ_FSYNC_DONE               (20)
#define IRQ_FSYNC_ERROR              (19)

//=============================================================================
// Event Bit Mapping - Based on cluster_event_map.sv
//=============================================================================

// DMA Events [3:2] - dma_events_i mapping
#define EU_DMA_EVT_0_BIT             (2)                         // DMA event 0 (completion)
#define EU_DMA_EVT_1_BIT             (3)                         // DMA event 1 (error/status)
#define EU_DMA_EVT_MASK              (0x0000000C)                // bits 3:2

// Timer Events [5:4] - timer_events_i mapping  
#define EU_TIMER_EVT_0_BIT           (4)                         // Timer event 0
#define EU_TIMER_EVT_1_BIT           (5)                         // Timer event 1
#define EU_TIMER_EVT_MASK            (0x00000030)                // bits 5:4

// Accelerator Events [11:8] - acc_events_i mapping
#define EU_ACC_EVT_0_BIT             (8)                         // Accelerator event 0 (always zero)
#define EU_ACC_EVT_1_BIT             (9)                         // Accelerator event 1 (busy)
#define EU_ACC_EVT_2_BIT             (10)                        // Accelerator event 2 (completion)
#define EU_ACC_EVT_3_BIT             (11)                        // Accelerator event 3 (additional)
#define EU_ACC_EVT_MASK              (0x00000F00)                // bits 11:8

// RedMulE specific event mapping (within accelerator events)
#define EU_REDMULE_BUSY_BIT          EU_ACC_EVT_1_BIT           // bit 9 - RedMulE busy
#define EU_REDMULE_DONE_BIT          EU_ACC_EVT_2_BIT           // bit 10 - RedMulE completion
#define EU_REDMULE_EVT1_BIT          EU_ACC_EVT_3_BIT           // bit 11 - RedMulE additional event
#define EU_REDMULE_DONE_MASK         (1 << EU_REDMULE_DONE_BIT) // 0x400
#define EU_REDMULE_BUSY_MASK         (1 << EU_REDMULE_BUSY_BIT) // 0x200
#define EU_REDMULE_EVT1_MASK         (1 << EU_REDMULE_EVT1_BIT) // 0x800
#define EU_REDMULE_ALL_MASK          (EU_ACC_EVT_MASK)          // 0xF00

// iDMA specific event mapping (within DMA events)
// Based on magia_tile.sv: assign dma_events_array[0] = {idma_o2a_done, idma_a2o_done};
#define EU_IDMA_A2O_DONE_BIT         EU_DMA_EVT_0_BIT          // bit 2 - iDMA AXI2OBI (L2->L1) completion
#define EU_IDMA_O2A_DONE_BIT         EU_DMA_EVT_1_BIT          // bit 3 - iDMA OBI2AXI (L1->L2) completion
#define EU_IDMA_A2O_DONE_MASK        (1 << EU_IDMA_A2O_DONE_BIT) // 0x04 - L2->L1 done
#define EU_IDMA_O2A_DONE_MASK        (1 << EU_IDMA_O2A_DONE_BIT) // 0x08 - L1->L2 done
#define EU_IDMA_ALL_DONE_MASK        (EU_IDMA_A2O_DONE_MASK | EU_IDMA_O2A_DONE_MASK) // 0x0C
#define EU_IDMA_ALL_MASK             (EU_DMA_EVT_MASK)         // 0x0C

// Legacy compatibility (uses A2O done by default)
#define EU_IDMA_DONE_BIT             EU_IDMA_A2O_DONE_BIT      // bit 2 - Default to A2O done
#define EU_IDMA_ERROR_BIT            EU_IDMA_O2A_DONE_BIT      // bit 3 - Legacy "error" was O2A done
#define EU_IDMA_DONE_MASK            EU_IDMA_A2O_DONE_MASK     // 0x04 - Legacy compatibility
#define EU_IDMA_ERROR_MASK           EU_IDMA_O2A_DONE_MASK     // 0x08 - Legacy compatibility

// iDMA extended status via cluster events [31:26]
#define EU_IDMA_A2O_ERROR_BIT        (26)                        // iDMA AXI2OBI error
#define EU_IDMA_O2A_ERROR_BIT        (27)                        // iDMA OBI2AXI error  
#define EU_IDMA_A2O_START_BIT        (28)                        // iDMA AXI2OBI start
#define EU_IDMA_O2A_START_BIT        (29)                        // iDMA OBI2AXI start
#define EU_IDMA_A2O_BUSY_BIT         (30)                        // iDMA AXI2OBI busy
#define EU_IDMA_O2A_BUSY_BIT         (31)                        // iDMA OBI2AXI busy
#define EU_IDMA_A2O_ERROR_MASK       (1 << EU_IDMA_A2O_ERROR_BIT) // 0x04000000
#define EU_IDMA_O2A_ERROR_MASK       (1 << EU_IDMA_O2A_ERROR_BIT) // 0x08000000
#define EU_IDMA_A2O_START_MASK       (1 << EU_IDMA_A2O_START_BIT) // 0x10000000
#define EU_IDMA_O2A_START_MASK       (1 << EU_IDMA_O2A_START_BIT) // 0x20000000
#define EU_IDMA_A2O_BUSY_MASK        (1 << EU_IDMA_A2O_BUSY_BIT)  // 0x40000000
#define EU_IDMA_O2A_BUSY_MASK        (1 << EU_IDMA_O2A_BUSY_BIT)  // 0x80000000
#define EU_IDMA_STATUS_MASK          (0xFC000000)                // All iDMA status bits [31:26]

// FSync specific event mapping (via cluster_events_i[25:24])
// Based on magia_tile.sv: fsync_error, fsync_done at bits [25:24]
#define EU_FSYNC_DONE_BIT            (24)                        // FSync completion event
#define EU_FSYNC_ERROR_BIT           (25)                        // FSync error event
#define EU_FSYNC_DONE_MASK           (1 << EU_FSYNC_DONE_BIT)  // 0x01000000
#define EU_FSYNC_ERROR_MASK          (1 << EU_FSYNC_ERROR_BIT) // 0x02000000
#define EU_FSYNC_ALL_MASK            (EU_FSYNC_DONE_MASK | EU_FSYNC_ERROR_MASK) // 0x03000000

// Legacy compatibility - use DONE by default
#define EU_FSYNC_EVT_BIT             EU_FSYNC_DONE_BIT         // bit 24 - Legacy compatibility
#define EU_FSYNC_EVT_MASK            EU_FSYNC_DONE_MASK        // 0x01000000 - Legacy compatibility

// Synchronization and barrier events [1:0]
#define EU_SYNC_EVT_BIT              (0)                         // Synchronization/barrier event
#define EU_DISPATCH_EVT_BIT          (1)                         // Dispatch event
#define EU_SYNC_EVT_MASK             (0x00000001)                // bit 0
#define EU_DISPATCH_EVT_MASK         (0x00000002)                // bit 1


//=============================================================================
// IDMA Register Addresses
//=============================================================================

#define IDMA_CONF_OFFSET          (0x00)
#define IDMA_STATUS_OFFSET        (0x04)  
#define IDMA_NEXT_ID_OFFSET       (0x44)  
#define IDMA_DONE_ID_OFFSET       (0x84)  
#define IDMA_DST_ADDR_LOW_OFFSET  (0xD0)
#define IDMA_SRC_ADDR_LOW_OFFSET  (0xD8)
#define IDMA_LENGTH_LOW_OFFSET    (0xE0)
#define IDMA_DST_STRIDE_2_LOW_OFFSET (0xE8)
#define IDMA_SRC_STRIDE_2_LOW_OFFSET (0xF0)
#define IDMA_REPS_2_LOW_OFFSET    (0xF8)
#define IDMA_DST_STRIDE_3_LOW_OFFSET (0x100)
#define IDMA_SRC_STRIDE_3_LOW_OFFSET (0x108)
#define IDMA_REPS_3_LOW_OFFSET    (0x110)

// Register Addresses - now direction-aware
#define IDMA_CONF_ADDR(is_l1_to_l2)          ((is_l1_to_l2) ? (IDMA_BASE_OBI2AXI + IDMA_CONF_OFFSET) : (IDMA_BASE_AXI2OBI + IDMA_CONF_OFFSET))
#define IDMA_STATUS_ADDR(is_l1_to_l2, id)    ((is_l1_to_l2) ? (IDMA_BASE_OBI2AXI + IDMA_STATUS_OFFSET + ((id) * 4)) : (IDMA_BASE_AXI2OBI + IDMA_STATUS_OFFSET + ((id) * 4)))
#define IDMA_NEXT_ID_ADDR(is_l1_to_l2, id)   ((is_l1_to_l2) ? (IDMA_BASE_OBI2AXI + IDMA_NEXT_ID_OFFSET + ((id) * 4)) : (IDMA_BASE_AXI2OBI + IDMA_NEXT_ID_OFFSET + ((id) * 4)))
#define IDMA_DONE_ID_ADDR(is_l1_to_l2, id)   ((is_l1_to_l2) ? (IDMA_BASE_OBI2AXI + IDMA_DONE_ID_OFFSET + ((id) * 4)) : (IDMA_BASE_AXI2OBI + IDMA_DONE_ID_OFFSET + ((id) * 4)))
#define IDMA_DST_ADDR_LOW_ADDR(is_l1_to_l2)  ((is_l1_to_l2) ? (IDMA_BASE_OBI2AXI + IDMA_DST_ADDR_LOW_OFFSET) : (IDMA_BASE_AXI2OBI + IDMA_DST_ADDR_LOW_OFFSET))
#define IDMA_SRC_ADDR_LOW_ADDR(is_l1_to_l2)  ((is_l1_to_l2) ? (IDMA_BASE_OBI2AXI + IDMA_SRC_ADDR_LOW_OFFSET) : (IDMA_BASE_AXI2OBI + IDMA_SRC_ADDR_LOW_OFFSET))
#define IDMA_LENGTH_LOW_ADDR(is_l1_to_l2)    ((is_l1_to_l2) ? (IDMA_BASE_OBI2AXI + IDMA_LENGTH_LOW_OFFSET) : (IDMA_BASE_AXI2OBI + IDMA_LENGTH_LOW_OFFSET))
#define IDMA_DST_STRIDE_2_LOW_ADDR(is_l1_to_l2) ((is_l1_to_l2) ? (IDMA_BASE_OBI2AXI + IDMA_DST_STRIDE_2_LOW_OFFSET) : (IDMA_BASE_AXI2OBI + IDMA_DST_STRIDE_2_LOW_OFFSET))
#define IDMA_SRC_STRIDE_2_LOW_ADDR(is_l1_to_l2) ((is_l1_to_l2) ? (IDMA_BASE_OBI2AXI + IDMA_SRC_STRIDE_2_LOW_OFFSET) : (IDMA_BASE_AXI2OBI + IDMA_SRC_STRIDE_2_LOW_OFFSET))
#define IDMA_REPS_2_LOW_ADDR(is_l1_to_l2)    ((is_l1_to_l2) ? (IDMA_BASE_OBI2AXI + IDMA_REPS_2_LOW_OFFSET) : (IDMA_BASE_AXI2OBI + IDMA_REPS_2_LOW_OFFSET))
#define IDMA_DST_STRIDE_3_LOW_ADDR(is_l1_to_l2) ((is_l1_to_l2) ? (IDMA_BASE_OBI2AXI + IDMA_DST_STRIDE_3_LOW_OFFSET) : (IDMA_BASE_AXI2OBI + IDMA_DST_STRIDE_3_LOW_OFFSET))
#define IDMA_SRC_STRIDE_3_LOW_ADDR(is_l1_to_l2) ((is_l1_to_l2) ? (IDMA_BASE_OBI2AXI + IDMA_SRC_STRIDE_3_LOW_OFFSET) : (IDMA_BASE_AXI2OBI + IDMA_SRC_STRIDE_3_LOW_OFFSET))
#define IDMA_REPS_3_LOW_ADDR(is_l1_to_l2)    ((is_l1_to_l2) ? (IDMA_BASE_OBI2AXI + IDMA_REPS_3_LOW_OFFSET) : (IDMA_BASE_AXI2OBI + IDMA_REPS_3_LOW_OFFSET))


//=============================================================================
// RedMule Register Addresses
//=============================================================================

/* Register offsets (RedMulE hwpe-ctrl) */
#define REDMULE_REG_OFFS     0x00
#define REDMULE_TRIGGER      0x00
#define REDMULE_ACQUIRE      0x04
#define REDMULE_EVT_ENABLE   0x08
#define REDMULE_STATUS       0x0C
#define REDMULE_RUNNING_JOB  0x10
#define REDMULE_SOFT_CLEAR   0x14

/* RedMulE configuration registers */
#define REDMULE_REG_X_PTR    0x40
#define REDMULE_REG_W_PTR    0x44
#define REDMULE_REG_Z_PTR    0x48
#define REDMULE_MCFG0_PTR    0x4C
#define REDMULE_MCFG1_PTR    0x50
#define REDMULE_ARITH_PTR    0x54


#endif  // _TILE_REG_DEFS_