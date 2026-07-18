// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Authors: Viviane Potocnik <vivianep@iis.ee.ethz.ch>
//          Alberto Dequino <alberto.dequino@unibo.it>

#pragma once

#include <stdint.h>
#include "redmule.h"
#include "utils/redmule_isa_utils.h" // redmule_mcnfig/marith, redmule_mm_* helpers

// The enqueue/commit/start helpers live here as static inline (rather than
// out-of-line in redmule16.c) so they fold into their mglib call sites
// (mg_redmule.c), which are compiled -O2 -flto: redmule16.c is not built with
// LTO, so an out-of-line definition there could not be inlined into mglib. This
// mirrors the eventunit32.h pattern (eu32_redmule_wait etc.).

/**
 * Configure and launch an FP16 GEMM on the RedMulE accelerator.
 * Computes: Y = X * W + Y  where X is [M x N], W is [N x K], Y is [M x K].
 *
 * The function issues the configuration and arithmetic commands to RedMulE
 * via custom RISC-V instructions (REDMULE_MM=0) or memory-mapped HWPE
 * registers (REDMULE_MM=1), then returns immediately. The caller must wait
 * for completion (e.g. via eu_redmule_wait).
 *
 * This variant enqueues the job without triggering it.
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
static inline __attribute__((always_inline)) int redmule16_gemm_enqueue(redmule_controller_t *ctrl,
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

/**
 * Commits execution of an enqueued job on the HW queue.
 *
 * @param ctrl RedMulE controller handle (unused internally, reserved for API consistency).
 *
 * @return 0 on successful dispatch.
 */
static inline __attribute__((always_inline)) int redmule16_gemm_commit(redmule_controller_t *ctrl)
{
#if REDMULE_MM != 0
    redmule_mm_commit();
#endif
    return 0;
}

/**
 * Commits and starts execution of an enqueued job on the HW queue.
 *
 * @param ctrl RedMulE controller handle (unused internally, reserved for API consistency).
 *
 * @return 0 on successful dispatch.
 */
static inline __attribute__((always_inline)) int redmule16_gemm_commit_start(redmule_controller_t *ctrl)
{
#if REDMULE_MM != 0
    redmule_mm_commit_trigger();
#endif
    return 0;
}

/**
 * Starts execution of an enqueued set of jobs.
 *
 * @param ctrl RedMulE controller handle (unused internally, reserved for API consistency).
 *
 * @return 0 on successful dispatch.
 */
static inline __attribute__((always_inline)) int redmule16_gemm_start(redmule_controller_t *ctrl)
{
#if REDMULE_MM != 0
    redmule_mm_trigger();
#endif
    return 0;
}
