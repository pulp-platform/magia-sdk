/*
 * Copyright (C) 2025 ETH Zurich and University of Bologna
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 *
 * PULP-only multi-task test: 4 sequential PULP task dispatches on different
 * core pairs (0+1, 2+3, 4+5, 6+7). This is the multi_task test WITHOUT the
 * Spatz part, to isolate whether the PULP path alone works on the mesh.
 */
#include "tile.h"
#include "eventunit.h"
#include "multi_task_params.h"
#include "multi_task_bin.h"

#define TILE_L1_BASE     (get_l1_base(get_hartid()))
#define PULP_PARAMS_ADDR (TILE_L1_BASE + 0x100) /* pulp_task_params_t */
#define IN_A_ADDR        (TILE_L1_BASE + 0x120) /* int32_t[8]         */
#define IN_B_ADDR        (TILE_L1_BASE + 0x140) /* int32_t[8]         */
#define OUT_ADDR         (TILE_L1_BASE + 0x160) /* int32_t[8]         */

static void
setup_pulp_params(uintptr_t in_a, uintptr_t in_b, uintptr_t out, uint32_t len, uint32_t scale)
{
    volatile pulp_task_params_t *p = (volatile pulp_task_params_t *)PULP_PARAMS_ADDR;
    p->in_a                        = in_a;
    p->in_b                        = in_b;
    p->out                         = out;
    p->len                         = len;
    p->scale                       = scale;
}

int main(void)
{
    int errors = 0;
    eu_config_t eu_cfg;
    eu_controller_t eu_ctrl;

    printf("[CV32] ===== Multi-task PULP-only test =====\n");

    eu_cfg.hartid = get_hartid();
    eu_ctrl.base  = NULL;
    eu_ctrl.cfg   = &eu_cfg;
    eu_ctrl.api   = &eu_api;

    eu_init(&eu_ctrl);
    eu_pulp_init(&eu_ctrl, 0);

    printf("[CV32] Init PULP cluster\n");
    pulp_init(PULP_BINARY_START);

    /* --- Task 1: vec_sum on cores 0+1 (mask=0x03) --------------------- */
    {
        static const int32_t a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        volatile int32_t *in_a    = (volatile int32_t *)IN_A_ADDR;
        volatile int32_t *out     = (volatile int32_t *)OUT_ADDR;
        for (int i = 0; i < 8; i++) {
            in_a[i] = a[i];
            out[i]  = 0;
        }
        setup_pulp_params(IN_A_ADDR, 0, OUT_ADDR, 8, 0);

        printf("[CV32] PULP task 1: vec_sum (cores 0+1)\n");
        pulp_run_task_with_params(VEC_SUM_PULP_TASK, PULP_PARAMS_ADDR, 0x03);
        eu_pulp_wait(&eu_ctrl, WFE);

        int32_t sum = out[0] + out[1];
        if (sum != 36) {
            printf("[CV32] SUM FAIL: got %d expected 36\n", (int)sum);
            errors++;
        } else {
            printf("[CV32] PULP SUM OK (%d)\n", (int)sum);
        }
    }

    /* --- Task 2: vec_dot on cores 2+3 (mask=0x0C) --------------------- */
    {
        /* dot([1,1,1,1,1,1,1,1], [1,2,3,4,5,6,7,8]) = 36 */
        static const int32_t a[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        static const int32_t b[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        volatile int32_t *in_a    = (volatile int32_t *)IN_A_ADDR;
        volatile int32_t *in_b    = (volatile int32_t *)IN_B_ADDR;
        volatile int32_t *out     = (volatile int32_t *)OUT_ADDR;
        for (int i = 0; i < 8; i++) {
            in_a[i] = a[i];
            in_b[i] = b[i];
            out[i]  = 0;
        }
        setup_pulp_params(IN_A_ADDR, IN_B_ADDR, OUT_ADDR, 8, 0);

        printf("[CV32] PULP task 2: vec_dot (cores 2+3)\n");
        pulp_run_task_with_params(VEC_DOT_PULP_TASK, PULP_PARAMS_ADDR, 0x0C);
        eu_pulp_wait(&eu_ctrl, WFE);

        int32_t dot = out[0] + out[1];
        if (dot != 36) {
            printf("[CV32] DOT FAIL: got %d expected 36\n", (int)dot);
            errors++;
        } else {
            printf("[CV32] PULP DOT OK (%d)\n", (int)dot);
        }
    }

    /* --- Task 3: vec_scale on cores 4+5 (mask=0x30) ------------------- */
    {
        /* [1..8] × 3 → out[0]=3, out[7]=24 */
        static const int32_t a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        volatile int32_t *in_a    = (volatile int32_t *)IN_A_ADDR;
        volatile int32_t *out     = (volatile int32_t *)OUT_ADDR;
        for (int i = 0; i < 8; i++) {
            in_a[i] = a[i];
            out[i]  = 0;
        }
        setup_pulp_params(IN_A_ADDR, 0, OUT_ADDR, 8, 3);

        printf("[CV32] PULP task 3: vec_scale (cores 4+5, scale=3)\n");
        pulp_run_task_with_params(VEC_SCALE_PULP_TASK, PULP_PARAMS_ADDR, 0x30);
        eu_pulp_wait(&eu_ctrl, WFE);

        if (out[0] != 3 || out[7] != 24) {
            printf("[CV32] SCALE FAIL: out[0]=%d (exp 3), out[7]=%d (exp 24)\n",
                   (int)out[0],
                   (int)out[7]);
            errors++;
        } else {
            printf("[CV32] PULP SCALE OK\n");
        }
    }

    /* --- Task 4: vec_absmax on cores 6+7 (mask=0xC0) ------------------ */
    {
        /* absmax([-5,3,-1,4,-2,-7,6,-3]) = 7 */
        static const int32_t a[8] = {-5, 3, -1, 4, -2, -7, 6, -3};
        volatile int32_t *in_a    = (volatile int32_t *)IN_A_ADDR;
        volatile int32_t *out     = (volatile int32_t *)OUT_ADDR;
        for (int i = 0; i < 8; i++) {
            in_a[i] = a[i];
            out[i]  = 0;
        }
        setup_pulp_params(IN_A_ADDR, 0, OUT_ADDR, 8, 0);

        printf("[CV32] PULP task 4: vec_absmax (cores 6+7)\n");
        pulp_run_task_with_params(VEC_ABSMAX_PULP_TASK, PULP_PARAMS_ADDR, 0xC0);
        eu_pulp_wait(&eu_ctrl, WFE);

        int32_t absmax = out[0] > out[1] ? out[0] : out[1];
        if (absmax != 7) {
            printf("[CV32] ABSMAX FAIL: got %d expected 7\n", (int)absmax);
            errors++;
        } else {
            printf("[CV32] PULP ABSMAX OK (%d)\n", (int)absmax);
        }
    }

    printf("[CV32] ===== %s (%d error%s) =====\n",
           errors ? "FAILED" : "PASSED",
           errors,
           errors == 1 ? "" : "s");
    return errors;
}
