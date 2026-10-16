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
 *
 * PULP Cluster Utility Functions (CV32 control side)
 */
#ifndef MAGIA_PULP_UTILS_H
#define MAGIA_PULP_UTILS_H

#include <stdint.h>
#include "magia_tile_utils.h"
#include "addr_map/tile_addr_map.h"

/* ClusterRegs PULP offsets from PULP_CTRL_BASE (0x1740)
 *   0x00: PULP_CLK_EN           — broadcast enable: write 1 starts ALL cores, 0 stops all
 *   0x04: PULP_BINARY           — binary entry point address
 *   0x08: PULP_NB_CORES_TO_WAIT — number of PULP cores participating in this task run
 *   0x0C: PULP_DONE             — (legacy) fired by ClusterRegs when all ACKs received
 *   0x10: PULP_TASKBIN          — task function address; all cores read this via MMIO
 *   0x14: PULP_DATA             — context pointer passed to the task function
 *   0x18: PULP_START            — CV32 writes one-hot core_mask → per-core IRQ edge pulse;
 *                                 PULP cores write 0 to ack; register clears when all done
 *   0x1C: PULP_READY            — each PULP hart writes 1 when booted; CV32 polls until all ready
 */
#define PULP_FETCH_EN           (PULP_CTRL_BASE + 0x00)
#define PULP_BINARY             (PULP_CTRL_BASE + 0x04)
#define PULP_DONE               (PULP_CTRL_BASE + 0x08)
#define PULP_TASKBIN            (PULP_CTRL_BASE + 0x0C)
#define PULP_DATA               (PULP_CTRL_BASE + 0x10)
#define PULP_START              (PULP_CTRL_BASE + 0x14)
#define PULP_READY              (PULP_CTRL_BASE + 0x18)
#define PULP_RETURN             (PULP_CTRL_BASE + 0x1C)
#define PULP_RETURN_CRASHED_BIT (0x80000000u) /* Set when core 0 trapped instead of returning */

static inline void pulp_fetch_en(void)
{
    mmio32(PULP_FETCH_EN) = 1;
}

static inline void pulp_set_binary(uint32_t addr)
{
    mmio32(PULP_BINARY) = addr;
}

static inline void pulp_set_func(uint32_t addr)
{
    mmio32(PULP_TASKBIN) = addr;
}

static inline void pulp_pass_params(uint32_t params_ptr)
{
    mmio32(PULP_DATA) = params_ptr;
}

/* Write entry point, broadcast clock enable to ALL cores, poll until all booted. */
static inline void pulp_init(uint32_t binary_start)
{
    pulp_set_binary(binary_start);
    pulp_fetch_en();
    while ((mmio32(PULP_READY) & 1u) == 0u) {
    }
}

/* Dispatch a task to core 0, returns once core 0 acknowledges PULP_START */
static inline void pulp_run_task(uint32_t task_addr)
{
    pulp_set_func(task_addr);
    mmio32(PULP_START) = 1;
    while (mmio32(PULP_START) != 0u) {
    }
}

/* Pass context pointer and dispatch task in one call. */
static inline void pulp_run_task_with_params(uint32_t task_addr, uint32_t params_ptr)
{
    pulp_pass_params(params_ptr);
    pulp_run_task(task_addr);
}

static inline int32_t pulp_get_return_value(void)
{
    return (int32_t)mmio32(PULP_RETURN);
}

static inline int pulp_task_crashed(void)
{
    return (mmio32(PULP_RETURN) & PULP_RETURN_CRASHED_BIT) != 0;
}

static inline uint32_t pulp_get_mcause(void)
{
    return mmio32(PULP_RETURN) & ~PULP_RETURN_CRASHED_BIT;
}

#endif /* MAGIA_PULP_UTILS_H */
