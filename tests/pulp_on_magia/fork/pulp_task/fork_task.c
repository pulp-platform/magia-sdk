/*
 * Copyright (C) 2026 Fondazione Chips-IT
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
 * Authors: Luca Balboni <luca.balboni@chips.it>
 * Alberto Dequino <alberto.dequino@unibo.it>
 *
 * fork test: a simple element wise add to check if the fork and barrier stuff works
 */

#include <stdint.h>
#include "tile.h"

typedef struct {
    uint32_t src_a;
    uint32_t src_b;
    uint32_t dst;
    uint32_t len;
} pulp_task_params_t;

static void vector_add(uint32_t a, uint32_t b, uint32_t out, uint32_t len){
    volatile uint32_t cid = cl_core_id();

    if(cid == 1)
        wait_nop(100);
    mmio32(out + cid * 4) = mmio32(a + cid * 4) + mmio32(b + cid * 4);

    return;
}

static void fork_entry(void *arg) {
    pulp_task_params_t *p = (pulp_task_params_t *)arg;
    vector_add(p->src_a, p->src_b, p->dst, p->len);
}

int fork_task(void *data) {
    cl_eu_team_fork(NUM_PULP_CORES, fork_entry, data);
    return 0;
}