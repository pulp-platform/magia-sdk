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
 * fork test - dispatch a vector addition to the cluster core 0, who then forks it to the 
 * rest of the cores of the cluster.
 */

#include "tile.h"
#include "eventunit.h"
#include "fork_params.h"
#include "test_fork_task_bin.h"

#define TILE_L1_BASE     (get_l1_base(get_hartid()))
#define PULP_PARAMS_ADDR (TILE_L1_BASE + 0x100) /* pulp_task_params_t */
#define IN_A_ADDR        (TILE_L1_BASE + 0x120) /* int32_t[8]         */
#define IN_B_ADDR        (TILE_L1_BASE + 0x140) /* int32_t[8]         */
#define OUT_ADDR         (TILE_L1_BASE + 0x160) /* int32_t[8]         */
#define GOLDEN_ADDR      (TILE_L1_BASE + 0x180) /* int32_t[8]         */

static void
setup_pulp_params(uint32_t in_a, uint32_t in_b, uint32_t out, uint32_t len)
{
    volatile pulp_task_params_t *p = (volatile pulp_task_params_t *)PULP_PARAMS_ADDR;
    p->in_a                        = in_a;
    p->in_b                        = in_b;
    p->out                         = out;
    p->len                         = len;
}

int main(void){
    uint32_t hartid = get_hartid();
    uint32_t errors = 0;
    //printf("[CV32 %d] Fork Test\n", hartid);

    //printf("[CV32 %d] Initializing PULP cluster (binary @ 0x%08x)\n", hartid, PULP_BINARY_START);
    pulp_init(PULP_BINARY_START);

    //printf("[CV32 %d] Initializing Event Unit for accepting the cluster\n", hartid);
    eu_init();
    eu_pulp_init(0);

    //printf("[CV32 %d] Initializing Data pointers\n", hartid);
    volatile uint32_t a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    volatile uint32_t b[8] = {3, 4, 32, 41, 53, 64, 17, 328};
    volatile uint32_t golden[8] = {4, 6, 35, 45, 58, 70, 24, 336};
    volatile uint32_t in_a    = IN_A_ADDR;
    volatile uint32_t in_b    = IN_B_ADDR;
    volatile uint32_t out     = OUT_ADDR;
    volatile uint32_t gold    = GOLDEN_ADDR;
    for(int i = 0; i < 8; i++){
        mmio32(in_a + i * 4) = a[i];
        mmio32(in_b + i * 4) = b[i];
        mmio32(gold + i * 4) = golden[i];
    }
    setup_pulp_params(IN_A_ADDR, IN_B_ADDR, OUT_ADDR, 8);

    //printf("[CV32 %d] Dispatching FORK_TASK with params to PULP cluster (core 0)\n", hartid);
    pulp_run_task_with_params(FORK_TASK, PULP_PARAMS_ADDR);

    eu_pulp_wait(WFE);

    //printf("[CV32 %d] PULP cluster done, checking errors...\n");

    for(int i = 0; i < 8; i++){
        if(mmio32(out + i * 4) != mmio32(gold + i * 4))
            errors++;
    }
    printf("[CV32 %d] Number of errors: %d\n", hartid, errors);

    return errors;
}