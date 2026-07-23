/*
 * Copyright (C) 2025 ETH Zurich and University of Bologna
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Multi-task test: 2 sequential Spatz tasks (fp16 vec-add, relu) followed by
 * 4 sequential PULP task dispatches on different core pairs (0+1, 2+3, 4+5, 6+7).
 */
#include "tile.h"
#include "eventunit.h"
#include "multi_task_params.h"
#include "multi_spatz_task_bin.h"
#include "multi_pulp_task_bin.h"

/* ---- L1 memory layout -------------------------------------------------- */
#define SPATZ_ADD_PARAMS_ADDR  (L1_BASE + 0x000) /* spatz_add_params_t  */
#define SPATZ_RELU_PARAMS_ADDR (L1_BASE + 0x020) /* spatz_relu_params_t */
#define A_ADDR                 (L1_BASE + 0x040) /* fp16[8] add input A */
#define B_ADDR                 (L1_BASE + 0x050) /* fp16[8] add input B */
#define C_ADDR                 (L1_BASE + 0x060) /* fp16[8] add output  */
#define X_ADDR                 (L1_BASE + 0x070) /* fp16[8] relu input  */
#define Y_ADDR                 (L1_BASE + 0x080) /* fp16[8] relu output */
#define PULP_PARAMS_ADDR       (L1_BASE + 0x100) /* pulp_task_params_t  */
#define IN_A_ADDR              (L1_BASE + 0x120) /* int32_t[8]          */
#define IN_B_ADDR              (L1_BASE + 0x140) /* int32_t[8]          */
#define OUT_ADDR               (L1_BASE + 0x160) /* int32_t[8]          */

#define VEC_LEN                8

/* fp16 IEEE 754 half-precision bit patterns */
#define FP16_0_0               ((uint16_t)0x0000) /*  0.0 */
#define FP16_1_0               ((uint16_t)0x3C00) /*  1.0 */
#define FP16_2_0               ((uint16_t)0x4000) /*  2.0 */
#define FP16_3_0               ((uint16_t)0x4200) /*  3.0 */
#define FP16_4_0               ((uint16_t)0x4400) /*  4.0 */
#define FP16_5_0               ((uint16_t)0x4500) /*  5.0 */
#define FP16_6_0               ((uint16_t)0x4600) /*  6.0 */
#define FP16_7_0               ((uint16_t)0x4700) /*  7.0 */
#define FP16_8_0               ((uint16_t)0x4800) /*  8.0 */
#define FP16_9_0               ((uint16_t)0x4880) /*  9.0 */
#define FP16_N1_0              ((uint16_t)0xBC00) /* -1.0 */
#define FP16_N2_0              ((uint16_t)0xC000) /* -2.0 */
#define FP16_N3_0              ((uint16_t)0xC200) /* -3.0 */
#define FP16_N4_0              ((uint16_t)0xC400) /* -4.0 */

static inline void write_fp16(uint32_t addr, uint16_t bits)
{
    *(volatile uint16_t *)addr = bits;
}

static inline uint16_t read_fp16(uint32_t addr)
{
    return *(volatile uint16_t *)addr;
}

static void init_spatz_data(void)
{
    /* A = [1, 2, 3, 4, 5, 6, 7, 8], B = [8, 7, 6, 5, 4, 3, 2, 1] → C = [9, ..., 9] */
    static const uint16_t A[8] = {
        FP16_1_0, FP16_2_0, FP16_3_0, FP16_4_0, FP16_5_0, FP16_6_0, FP16_7_0, FP16_8_0};
    static const uint16_t B[8] = {
        FP16_8_0, FP16_7_0, FP16_6_0, FP16_5_0, FP16_4_0, FP16_3_0, FP16_2_0, FP16_1_0};
    /* X = [-4, 2, -1, 3, 0, -2, 5, -3] → Y = [0, 2, 0, 3, 0, 0, 5, 0] */
    static const uint16_t X[8] = {
        FP16_N4_0, FP16_2_0, FP16_N1_0, FP16_3_0, FP16_0_0, FP16_N2_0, FP16_5_0, FP16_N3_0};

    for (int i = 0; i < VEC_LEN; i++) {
        write_fp16(A_ADDR + i * 2, A[i]);
        write_fp16(B_ADDR + i * 2, B[i]);
        write_fp16(C_ADDR + i * 2, FP16_0_0);
        write_fp16(X_ADDR + i * 2, X[i]);
        write_fp16(Y_ADDR + i * 2, FP16_0_0);
    }

    volatile spatz_add_params_t *add_p = (volatile spatz_add_params_t *)SPATZ_ADD_PARAMS_ADDR;
    add_p->A                           = A_ADDR;
    add_p->B                           = B_ADDR;
    add_p->C                           = C_ADDR;
    add_p->len                         = VEC_LEN;

    volatile spatz_relu_params_t *relu_p = (volatile spatz_relu_params_t *)SPATZ_RELU_PARAMS_ADDR;
    relu_p->X                            = X_ADDR;
    relu_p->Y                            = Y_ADDR;
    relu_p->len                          = VEC_LEN;
}

static int check_spatz_add(void)
{
    /* C[i] must all be 9.0 (0x4900) */
    for (int i = 0; i < VEC_LEN; i++) {
        uint16_t got = read_fp16(C_ADDR + i * 2);
        if (got != FP16_9_0) {
            printf("[CV32] ADD FAIL: C[%d]=0x%04x expected 0x%04x\n", i, got, FP16_9_0);
            return -1;
        }
    }
    return 0;
}

static int check_spatz_relu(void)
{
    static const uint16_t expected[8] = {
        FP16_0_0, FP16_2_0, FP16_0_0, FP16_3_0, FP16_0_0, FP16_0_0, FP16_5_0, FP16_0_0};
    for (int i = 0; i < VEC_LEN; i++) {
        uint16_t got = read_fp16(Y_ADDR + i * 2);
        if (got != expected[i]) {
            printf("[CV32] RELU FAIL: Y[%d]=0x%04x expected 0x%04x\n", i, got, expected[i]);
            return -1;
        }
    }
    return 0;
}

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

    printf("[CV32] ===== Multi-task Spatz+PULP test =====\n");

    eu_cfg.hartid = get_hartid();
    eu_ctrl.base  = NULL;
    eu_ctrl.cfg   = &eu_cfg;
    eu_ctrl.api   = &eu_api;

    eu_init(&eu_ctrl);
    eu_pulp_init(&eu_ctrl, 0);
    eu_spatz_init(&eu_ctrl, 0);

    /* ------------------------------------------------------------------
     * Spatz: 2 sequential fp16 tasks
     * ------------------------------------------------------------------ */
    printf("[CV32] Init Spatz\n");
    spatz_init(SPATZ_BINARY_START);
    init_spatz_data();

    printf("[CV32] Spatz task 1: vec_add\n");
    spatz_run_task_with_params(VEC_ADD_SPATZ_TASK, SPATZ_ADD_PARAMS_ADDR);
    eu_spatz_wait(&eu_ctrl, WFE);
    if (spatz_get_exit_code() != 0) {
        printf("[CV32] Spatz ADD error (code=0x%x)\n", spatz_get_exit_code());
        errors++;
    } else if (check_spatz_add() != 0) {
        errors++;
    } else {
        printf("[CV32] Spatz ADD OK\n");
    }

    printf("[CV32] Spatz task 2: vec_relu\n");
    spatz_run_task_with_params(VEC_RELU_SPATZ_TASK, SPATZ_RELU_PARAMS_ADDR);
    eu_spatz_wait(&eu_ctrl, WFE);
    if (spatz_get_exit_code() != 0) {
        printf("[CV32] Spatz RELU error (code=0x%x)\n", spatz_get_exit_code());
        errors++;
    } else if (check_spatz_relu() != 0) {
        errors++;
    } else {
        printf("[CV32] Spatz RELU OK\n");
    }

    spatz_clk_dis();

    /* ------------------------------------------------------------------
     * PULP: 4 sequential int32 tasks, one pair of cores each time
     * ------------------------------------------------------------------ */
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
