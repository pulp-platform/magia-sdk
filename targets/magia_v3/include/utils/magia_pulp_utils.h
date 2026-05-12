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

/* ClusterRegs offsets from PULP_CTRL_BASE (0x1740) */
#define PULP_CLK_EN  (PULP_CTRL_BASE + 0x00)  /* W: 1=enable PULP cluster clock */
#define PULP_BINARY  (PULP_CTRL_BASE + 0x04)  /* W: binary entry point address   */
#define PULP_DONE    (PULP_CTRL_BASE + 0x08)  /* W: each PULP hart writes 1 here */

static inline void pulp_clk_en(void) {
    mmio32(PULP_CLK_EN) = 1;
}

static inline void pulp_clk_dis(void) {
    mmio32(PULP_CLK_EN) = 0;
}

static inline void pulp_set_binary(uint32_t addr) {
    mmio32(PULP_BINARY) = addr;
}

/* Write entry point then enable clock — PULP harts boot immediately */
static inline void pulp_init(uint32_t binary_start) {
    pulp_set_binary(binary_start);
    pulp_clk_en();
}

#endif /* MAGIA_PULP_UTILS_H */
