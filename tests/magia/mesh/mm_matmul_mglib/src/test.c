// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include <stdint.h>
#include "test.h"

#include "tile.h"
#include "mg_redmule.h"
#include "mg_event.h"

#define WAIT_MODE WFE
#define GARBAGE   0x7bff
/* RedMulE stores whole 8-row blocks of Z: room for M rounded up to 8 rows */
#define Z_ROWS    ((M_SIZE + 7) / 8 * 8)

static void l1_copy(uint32_t dst, const uint16_t *src, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++)
        ((volatile uint16_t *)dst)[i] = src ? src[i] : GARBAGE;
}

static uint32_t l1_check(uint32_t hartid, const char *what, uint32_t res, const uint16_t *golden)
{
    uint32_t errors = 0;
    for (uint32_t i = 0; i < M_SIZE * K_SIZE; i++) {
        uint16_t computed = ((volatile uint16_t *)res)[i];
        if (computed != golden[i]) {
#if EVAL == 1
            printf("TILE %d: %s error at [%d][%d]: %x != %x\n",
                   hartid,
                   what,
                   i / K_SIZE,
                   i % K_SIZE,
                   computed,
                   golden[i]);
#endif
            errors++;
        }
    }
    return errors;
}

/**
 * This test verifies the non-accumulating RedMulE MATMUL (Z = X @ W) exposed by
 * mglib, on every tile:
 * 1. a one-shot mg_redmule_matmul into an output full of garbage, which must not
 *    be read;
 * 2. a MATMUL and a GEMM (Z = X @ W + Z) enqueued back to back in the hardware
 *    queue, committed together and waited through their own events.
 */
int main(void)
{
    uint32_t hartid = get_hartid();

    redmule_init();
    eu_init();
    eu_redmule_init(0);

    uint32_t l1_x  = get_l1_base(hartid);
    uint32_t l1_w  = l1_x + M_SIZE * N_SIZE * 2;
    uint32_t l1_z0 = l1_w + N_SIZE * K_SIZE * 2;
    uint32_t l1_z1 = l1_z0 + Z_ROWS * K_SIZE * 2;

    mg_event_t evt0, evt1;
    uint32_t errors = 0;

    l1_copy(l1_x, x_inp, M_SIZE * N_SIZE);
    l1_copy(l1_w, w_inp, N_SIZE * K_SIZE);

    /* 1. One-shot MATMUL */
    l1_copy(l1_z0, NULL, M_SIZE * K_SIZE);
    mg_redmule_matmul(WAIT_MODE,
                      l1_x,
                      l1_w,
                      l1_z0,
                      (uint16_t)M_SIZE,
                      (uint16_t)N_SIZE,
                      (uint16_t)K_SIZE,
                      &evt0,
                      NULL);
    mg_redmule_wait(WAIT_MODE, &evt0);
    errors += l1_check(hartid, "matmul", l1_z0, z_mm);

    /* 2. MATMUL and GEMM queued back to back */
    l1_copy(l1_z0, NULL, M_SIZE * K_SIZE);
    l1_copy(l1_z1, y_inp, M_SIZE * K_SIZE);
    mg_redmule_matmul_enqueue(WAIT_MODE,
                              l1_x,
                              l1_w,
                              l1_z0,
                              (uint16_t)M_SIZE,
                              (uint16_t)N_SIZE,
                              (uint16_t)K_SIZE,
                              &evt0,
                              NULL);
    mg_redmule_gemm_commit();
    mg_redmule_gemm_enqueue(WAIT_MODE,
                            l1_x,
                            l1_w,
                            l1_z1,
                            (uint16_t)M_SIZE,
                            (uint16_t)N_SIZE,
                            (uint16_t)K_SIZE,
                            &evt1,
                            NULL);
    mg_redmule_gemm_commit_start();
    mg_redmule_wait(WAIT_MODE, &evt0);
    mg_redmule_wait(WAIT_MODE, &evt1);
    errors += l1_check(hartid, "queued matmul", l1_z0, z_mm);
    errors += l1_check(hartid, "queued gemm", l1_z1, z_gemm);

    printf("TILE %d: Number of errors: %d\n", hartid, errors);

    return errors;
}
