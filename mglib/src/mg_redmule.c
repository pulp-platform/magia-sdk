// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include <stdint.h>
#include "mg_redmule.h"

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
static uint8_t mg_redmule_completed __attribute__((section(".tile_bss")));

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

void mg_redmule_wait(eu_controller_t *eu, eu_wait_mode_t mode, mg_event_t *event)
{
    uint8_t target = (uint8_t)(event->id + 1);

    // the event may already be done - e.g. its completion pulse was consumed
    // while waiting on a later id - in which case we must not wait on the
    // hardware at all.
    while (!mg_seq_ge(mg_redmule_completed, target)) {
        // not yet the right one: spin back into the hardware wait.
        if (eu32_redmule_wait(eu, mode)) {
            // update the completion counter whenever a pulse was seen: it
            // always retires exactly one FIFO-ordered job, whether or not it
            // is the one we are waiting for.
            mg_redmule_completed++;
        }
    }

    // our event is the one that just completed (or had already completed).
    mg_event_trigger(event);
}
