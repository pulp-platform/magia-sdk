// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include <stdint.h>
#include "mg_idma.h"

/**
 * Per-direction issue/completion counters (index 0 = A2O/L2->L1, 1 = O2A/L1->L2).
 * The number of iDMA transfers ever in flight at once is always far below 256, so
 * wrapping these counters is safe.
 *
 * These are per-tile completion bookkeeping, so they must live in per-tile-aliased
 * memory rather than the shared-L2 .bss (where every tile would collide on one
 * copy). The .tile_bss section is private per tile and zeroed by every hart in
 * crt0, so no initializer is given here.
 */
static uint8_t mg_idma_issued[2]    __attribute__((section(".tile_bss")));
static uint8_t mg_idma_completed[2] __attribute__((section(".tile_bss")));

static inline __ALWAYS_INLINE_ void mg_idma_issue(uint8_t dir, mg_event_t *event, mg_event_callback_t callback)
{
    uint8_t idx = dir ? 1 : 0;
    mg_event_init(event, (int32_t) mg_idma_issued[idx], callback);
    mg_idma_issued[idx]++;
}

void mg_idma_memcpy_1d(idma_controller_t *idma,
                        uint8_t dir,
                        uint32_t axi_addr,
                        uint32_t obi_addr,
                        uint32_t len,
                        mg_event_t *event,
                        mg_event_callback_t callback)
{
    mg_idma_issue(dir, event, callback);
    // Triggers the transfer; may block if the hardware queue for this
    // direction is full.
    idma_memcpy_1d(idma, dir, axi_addr, obi_addr, len);
}

void mg_idma_memcpy_2d(idma_controller_t *idma,
                        uint8_t dir,
                        uint32_t axi_addr,
                        uint32_t obi_addr,
                        uint32_t len,
                        uint32_t std,
                        uint32_t reps,
                        mg_event_t *event,
                        mg_event_callback_t callback)
{
    mg_idma_issue(dir, event, callback);
    idma_memcpy_2d(idma, dir, axi_addr, obi_addr, len, std, reps);
}

void mg_idma_wait(eu_controller_t *eu, uint8_t dir, eu_wait_mode_t mode, mg_event_t *event)
{
    uint8_t idx    = dir ? 1 : 0;
    uint8_t target = (uint8_t) (event->id + 1);

    // the event may already be done - e.g. its completion pulse was
    // consumed while waiting on a later id - in which case we must not
    // wait on the hardware at all.
    while (!mg_seq_ge(mg_idma_completed[idx], target)) {
        // dnot yet the right one: spin back into the hardware wait.
        uint32_t done;
        if(dir) {
            done = eu_idma_wait_o2a(eu, mode);
        } else {
            done = eu_idma_wait_a2o(eu, mode);
        }
        if (done) {
            // update the completion counter whenever a pulse was seen: it
            // always retires exactly one FIFO-ordered transfer, whether or
            // not it is the one we are waiting for.
            mg_idma_completed[idx]++;
        }
    }

    // our event is the one that just completed (or had already completed).
    mg_event_trigger(event);
}
