// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Viviane Potocnik <vivianep@iis.ee.ethz.ch>
// Alberto Dequino <alberto.dequino@unibo.it>
//
// This file provides the strong (driver-specific) implementations for the
// Redmule functions on FP16 vector data.
// These functions override the weak HAL symbols.
// This is a WIP and might be redundant, as the moment of writing there is only one Redmule configuration tested on MAGIA.

#include <stdint.h>
#include "redmule16.h"
#include "regs/tile_ctrl.h"
#include "utils/redmule_isa_utils.h"
#include "utils/magia_utils.h"
#include "utils/tinyprintf.h"
#include "utils/performance_utils.h"

int redmule16_init(redmule_controller_t *ctrl) {
    irq_en(1<<IRQ_REDMULE_EVT_0);
    return 0;
}

/* static inline void redmule16_wait() {
    asm volatile("wfi" ::: "memory");
}
 */

/**
 * This function prepares and execute an accelerated generic matrix multiplication.
 * (N x M * M x K) + (N x K) = (N x K)
 *
 * @param x First matrix initial pointer.
 * @param w Second matrix initial pointer.
 * @param y Bias Matrix initial pointer. The GEMM output is also stored here.
 * @param m Number of rows of x and y.
 * @param n Number of columns of x, and number of rows of w.
 * @param k Number of columns of w and y.
 * 
 * @return 0 if successful
 * 
 */
int redmule16_gemm(redmule_controller_t *ctrl, uint32_t x, uint32_t w, uint32_t y, uint16_t m, uint16_t n, uint16_t k){
    printf("RedMulE with parameter: x=0x%0x, w=0x%0x, y=0x%0x, m=%0d, n=%0d, k=%0d\n", x, w, y, m, n, k);
    sentinel_start();
    //printf("Redmule GEMM!");
    redmule_mcnfig(k, m, n);
    redmule_marith(y, w, x);
    //printf("Redmule GEMM: Detected IRQ...\n");
    redmule_wait();
    sentinel_end();

    return 0;
}

extern int redmule_init(redmule_controller_t *ctrl)
    __attribute__((alias("redmule16_init"), used, visibility("default")));
/* extern void redmule_wait()
    __attribute__((alias("redmule16_wait"), used, visibility("default"))); */
extern int redmule_gemm(redmule_controller_t *ctrl, uint32_t x, uint32_t w, uint32_t y, uint16_t m, uint16_t n, uint16_t k)
    __attribute__((alias("redmule16_gemm"), used, visibility("default")));


/* Export the Redmule-specific controller API */
redmule_controller_api_t redmule_api = {
    .init = redmule16_init,
/*     .wait = redmule16_wait, */
    .gemm = redmule16_gemm,
};