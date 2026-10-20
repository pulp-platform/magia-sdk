/*
 * Copyright (C) 2023-2026 ETH Zurich, University of Bologna and Fondazione Chips-IT
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
 * Cluster Event Unit Utilities for MAGIA.
 */

#ifndef CLUSTER_EVENT_UNIT_UTILS_H
#define CLUSTER_EVENT_UNIT_UTILS_H

#include <stdint.h>
#include "magia_tile_utils.h"
#include "eu_isa_utils.h"

static inline void cl_eu_sw_event_trigger(unsigned int id, uint32_t core_mask)
{
    mmio32(CLUSTER_EU_SW_EVT_TRIG(id)) = core_mask;
}

static inline void cl_eu_sw_event_wait(unsigned int id)
{
    uint32_t bit                          = 1u << CLUSTER_EU_SW_EVT_BIT(id);
    mmio32(CLUSTER_EU_DIRECT_BASE + 0x08) = bit;     /* EU_CORE_MASK_OR */
    (void)evt_read32(CLUSTER_EU_DIRECT_BASE + 0x3C); /* EU_EVENT_WAIT_CLR */
}

// The ID of the Cluster domain
static inline uint32_t cl_core_id(void)
{
    uint32_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    return (hartid - PULP_HARTID_BASE) % PULP_CORE_COUNT;
}

// The ID of the Control Core connected to the current Cluster
static inline uint32_t cl_tile_id(void)
{
    uint32_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    return (hartid - PULP_HARTID_BASE) / PULP_CORE_COUNT;
}

static inline void cl_eu_team_critical_enter(void)
{
    (void)evt_read32(CLUSTER_EU_MUTEX0_ADDR);
}

static inline void cl_eu_team_critical_exit(void)
{
    mmio32(CLUSTER_EU_MUTEX0_ADDR) = 0;
    __asm__ __volatile__(
        "" ::
            : "memory"); /* compiler barrier: the unlock store has no memory clobber */
}

static inline void cl_eu_team_barrier_id(unsigned int barrier_id)
{
    uint32_t addr =
        CLUSTER_EU_DIRECT_BASE + CLUSTER_EU_BARRIER_OFFSET + barrier_id * CLUSTER_EU_BARRIER_SIZE;
    (void)evt_read32(addr + CLUSTER_EU_BARR_WAIT_CLEAR);
}

static inline void cl_eu_team_barrier(void)
{
    (void)evt_read32(CLUSTER_EU_BARRIER0_ADDR + CLUSTER_EU_BARR_WAIT_CLEAR);
}

static inline void cl_eu_team_fork(int nb_cores, void (*entry)(void *), void *arg)
{
    uint32_t core_mask = (nb_cores >= 32) ? 0xFFFFFFFFu : ((1u << nb_cores) - 1u);

    if (nb_cores) {
        mmio32(CLUSTER_EU_DISPATCH_TEAM_CFG)                            = core_mask;
        mmio32(CLUSTER_EU_BARRIER0_ADDR + CLUSTER_EU_BARR_TRIGGER_MASK) = core_mask;
        mmio32(CLUSTER_EU_BARRIER0_ADDR + CLUSTER_EU_BARR_TARGET_MASK)  = core_mask;
    }

    mmio32(CLUSTER_EU_DISPATCH_FIFO) = (uint32_t)entry;
    mmio32(CLUSTER_EU_DISPATCH_FIFO) = (uint32_t)arg;

    /* The calling core is itself a member of the team. */
    entry(arg);

    cl_eu_team_barrier();
}

static inline void cl_eu_team_push_other(uint32_t core_mask, void (*entry)(void *), void *arg)
{
    mmio32(CLUSTER_EU_DISPATCH_TEAM_CFG) = core_mask;
    mmio32(CLUSTER_EU_DISPATCH_FIFO) =
        ((uint32_t)entry) | 1u; /* bit 0 set: other entry, no auto barrier */
    mmio32(CLUSTER_EU_DISPATCH_FIFO) = (uint32_t)arg;
}

static inline void cl_eu_team_barrier_setup(unsigned int barrier_id, uint32_t core_mask)
{
    uint32_t addr =
        CLUSTER_EU_DIRECT_BASE + CLUSTER_EU_BARRIER_OFFSET + barrier_id * CLUSTER_EU_BARRIER_SIZE;
    mmio32(addr + CLUSTER_EU_BARR_TRIGGER_MASK) = core_mask;
    mmio32(addr + CLUSTER_EU_BARR_TARGET_MASK)  = core_mask;
}

#endif /* CLUSTER_EVENT_UNIT_UTILS_H */
