// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include <stdint.h>
#include "mg_redmule.h"
#include "redmule16.h" // static inline redmule16_gemm_enqueue/_commit/_start

/**
 * Completion counter. Only the low 8 bits are significant (see mg_seq_ge()
 * in mg_event.h): the number of RedMulE jobs ever in flight at once is
 * always far below 256, so wrapping is safe.
 *
 * Advanced from two places: mg_redmule_gemm()'s backpressure spin (below)
 * and mg_redmule_wait(). That is safe because RedMulE jobs retire strictly
 * in FIFO order regardless of which call site happens to observe a given
 * completion pulse.
 *
 * Per-tile completion bookkeeping, so it lives in per-tile-aliased .tile_bss
 * (private per tile, zeroed by every hart in crt0) rather than the shared-L2
 * .bss where all tiles would collide on one copy - hence no initializer here.
 */
// External linkage (declared in mg_redmule.h): shared with the now-inlined
// mg_redmule_wait() so both the acquire/backpressure path (here) and the wait
// path observe a single completion counter.
uint8_t mg_redmule_completed __attribute__((section(".tile_bss")));

void mg_redmule_gemm(redmule_controller_t *ctrl,
                     eu_controller_t *eu,
                     eu_wait_mode_t mode,
                     uint32_t x,
                     uint32_t w,
                     uint32_t y,
                     uint16_t m,
                     uint16_t n,
                     uint16_t k,
                     mg_event_t *event,
                     mg_event_callback_t callback)
{
    int32_t id;
    while ((id = redmule_acquire(ctrl)) < 0) {
        // Hardware queue (depth 2) is full: drain one completion pulse to
        // free a slot before retrying the acquire.
        if (eu32_redmule_wait(eu, mode)) {
            mg_redmule_completed++;
        }
    }
    mg_event_init(event, id, callback);
    redmule_gemm(ctrl, x, w, y, m, n, k);
}

void mg_redmule_gemm_enqueue(redmule_controller_t *ctrl,
                             eu_controller_t *eu,
                             eu_wait_mode_t mode,
                             uint32_t x,
                             uint32_t w,
                             uint32_t y,
                             uint16_t m,
                             uint16_t n,
                             uint16_t k,
                             mg_event_t *event,
                             mg_event_callback_t callback)
{
    int32_t id;
    while ((id = redmule_acquire(ctrl)) < 0) {
        // Hardware queue (depth 2) is full: drain one completion pulse to
        // free a slot before retrying the acquire.
        if (eu32_redmule_wait(eu, mode)) {
            mg_redmule_completed++;
        }
    }
    mg_event_init(event, id, callback);
    redmule16_gemm_enqueue(ctrl, x, w, y, m, n, k);
}

void mg_redmule_gemm_commit_start(redmule_controller_t *ctrl)
{
    redmule16_gemm_commit_start(ctrl);
}

void mg_redmule_gemm_commit(redmule_controller_t *ctrl)
{
    redmule16_gemm_commit(ctrl);
}

void mg_redmule_gemm_start(redmule_controller_t *ctrl)
{
    redmule16_gemm_start(ctrl);
}

// mg_redmule_wait() is defined as a static inline in mg_redmule.h so it folds
// into its call sites (removing the call/return and the one-time cold I-cache
// miss, and letting constant-propagation drop the callback tail where NULL).
