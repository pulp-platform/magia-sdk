// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include <stdint.h>
#include "mg_redmule.h"
#include "redmule16.h" // static inline redmule16_gemm_enqueue/_commit/_start

/**
 * Acquire a hardware job id for the next job and stamp `event` with it. If the
 * hardware queue (depth 2) is full, wait for a completion pulse to free a slot
 * before retrying the acquire.
 *
 * Shared by all the job-issuing calls below; always inlined, so each of them
 * compiles exactly as if the loop were written in place.
 */
static inline __attribute__((always_inline)) void
mg_redmule_acquire_event(eu_wait_mode_t mode, mg_event_t *event, mg_event_callback_t callback)
{
    int32_t id;
    while ((id = redmule_acquire()) < 0) {
        eu32_redmule_wait(mode);
    }
    mg_event_init(event, id, callback);
}

void mg_redmule_gemm(eu_wait_mode_t mode,
                     uint32_t x,
                     uint32_t w,
                     uint32_t y,
                     uint16_t m,
                     uint16_t n,
                     uint16_t k,
                     mg_event_t *event,
                     mg_event_callback_t callback)
{
    mg_redmule_acquire_event(mode, event, callback);
    redmule_gemm(x, w, y, m, n, k);
}

void mg_redmule_gemm_enqueue(eu_wait_mode_t mode,
                             uint32_t x,
                             uint32_t w,
                             uint32_t y,
                             uint16_t m,
                             uint16_t n,
                             uint16_t k,
                             mg_event_t *event,
                             mg_event_callback_t callback)
{
    mg_redmule_acquire_event(mode, event, callback);
    redmule16_gemm_enqueue(x, w, y, m, n, k);
}

#if REDMULE_MM != 0
void mg_redmule_matmul(eu_wait_mode_t mode,
                       uint32_t x,
                       uint32_t w,
                       uint32_t y,
                       uint16_t m,
                       uint16_t n,
                       uint16_t k,
                       mg_event_t *event,
                       mg_event_callback_t callback)
{
    mg_redmule_acquire_event(mode, event, callback);
    redmule16_matmul_enqueue(x, w, y, m, n, k);
    redmule16_gemm_commit_start();
}

void mg_redmule_matmul_enqueue(eu_wait_mode_t mode,
                               uint32_t x,
                               uint32_t w,
                               uint32_t y,
                               uint16_t m,
                               uint16_t n,
                               uint16_t k,
                               mg_event_t *event,
                               mg_event_callback_t callback)
{
    mg_redmule_acquire_event(mode, event, callback);
    redmule16_matmul_enqueue(x, w, y, m, n, k);
}
#endif

void mg_redmule_gemm_commit_start()
{
    redmule16_gemm_commit_start();
}

void mg_redmule_gemm_commit()
{
    redmule16_gemm_commit();
}

void mg_redmule_gemm_start()
{
    redmule16_gemm_start();
}

// mg_redmule_wait() is defined as a static inline in mg_redmule.h so it folds
// into its call sites (removing the call/return and the one-time cold I-cache
// miss, and letting constant-propagation drop the callback tail where NULL).
