// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Viviane Potocnik <vivianep@iis.ee.ethz.ch>
// Alberto Dequino <alberto.dequino@unibo.it>
//
// This file provides the strong (driver-specific) implementations for the
// RedmulE functions on FP16 vector data.
// These functions override the weak HAL symbols.
// This is a WIP and might be redundant, as the moment of writing there is only one RedmulE
// configuration tested on MAGIA.

#include <stdint.h>
#include "redmule16.h"
#include "regs/tile_ctrl.h"
#include "utils/redmule_isa_utils.h"
#include "utils/magia_utils.h"
// #include "utils/tinyprintf.h"
#include "utils/printf.h"

int redmule16_init(redmule_controller_t *ctrl)
{
    irq_en(1 << IRQ_REDMULE_EVT_0);
    return 0;
}

/* static inline void redmule16_wait() {
    asm volatile("wfi" ::: "memory");
}
 */

/**
 * Configure and launch an FP16 GEMM on the RedMulE accelerator.
 * Computes: Y = X * W + Y  where X is [M x N], W is [N x K], Y is [M x K].
 *
 * The function issues the configuration and arithmetic commands to RedMulE
 * via custom RISC-V instructions (REDMULE_MM=0) or memory-mapped HWPE
 * registers (REDMULE_MM=1), then returns immediately. The caller must wait
 * for completion (e.g. via eu_redmule_wait).
 *
 * @param ctrl RedMulE controller handle (unused internally, reserved for API consistency).
 * @param x    OBI (L1) address of input matrix X [M x N].
 * @param w    OBI (L1) address of weight matrix W [N x K].
 * @param y    OBI (L1) address of bias/output matrix Y [M x K]. Accumulated in-place.
 * @param m    Number of rows of X and Y.
 * @param n    Number of columns of X / rows of W (inner dimension).
 * @param k    Number of columns of W and Y.
 *
 * @return 0 on successful dispatch.
 */
int redmule16_gemm(redmule_controller_t *ctrl,
                   uint32_t x,
                   uint32_t w,
                   uint32_t y,
                   uint16_t m,
                   uint16_t n,
                   uint16_t k)
{
#if REDMULE_MM == 0
    redmule_mcnfig(k, m, n); // Set GEMM dimensions via custom RISC-V instruction
    redmule_marith(y, w, x); // Launch GEMM with matrix addresses via custom RISC-V instruction
#else
    redmule_mm_mcnfig(k, m, n); // Set GEMM dimensions via memory-mapped HWPE registers
    redmule_mm_marith(
        y, w, x); // Launch GEMM with matrix addresses via memory-mapped HWPE registers
#endif

    return 0;
}

extern int redmule_init(redmule_controller_t *ctrl)
    __attribute__((alias("redmule16_init"), used, visibility("default")));
/* extern void redmule_wait()
    __attribute__((alias("redmule16_wait"), used, visibility("default"))); */
extern int redmule_gemm(redmule_controller_t *ctrl,
                        uint32_t x,
                        uint32_t w,
                        uint32_t y,
                        uint16_t m,
                        uint16_t n,
                        uint16_t k)
    __attribute__((alias("redmule16_gemm"), used, visibility("default")));

/* Export the RedmulE-specific controller API */
redmule_controller_api_t redmule_api = {
    .init = redmule16_init,
    /*     .wait = redmule16_wait, */
    .gemm = redmule16_gemm,
};
