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

static inline void eu_cl_sw_event_trigger(unsigned int id, uint32_t core_mask) {
    mmio32(CLUSTER_EU_SW_EVT_TRIG(id)) = core_mask;
}

static inline void pi_cl_sw_event_wait(unsigned int id) {
    uint32_t bit = 1u << CLUSTER_EU_SW_EVT_BIT(id);
    mmio32(CLUSTER_EU_DIRECT_BASE + 0x08) = bit;                 /* EU_CORE_MASK_OR */
    (void)evt_read32(CLUSTER_EU_DIRECT_BASE, 0x3C);               /* EU_EVENT_WAIT_CLR */
}

static inline uint32_t cluster_core_id(void) {
    uint32_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    return (hartid - PULP_HARTID_BASE) % PULP_CORE_COUNT;
}

static inline uint32_t cluster_tile_id(void) {
    uint32_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    return (hartid - PULP_HARTID_BASE) / PULP_CORE_COUNT;
}

static inline uint32_t pi_core_id(void) {
    return cluster_core_id();
}

#endif /* CLUSTER_EVENT_UNIT_UTILS_H */
