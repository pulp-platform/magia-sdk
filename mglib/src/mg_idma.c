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
static uint8_t mg_idma_issued[2] __attribute__((section(".tile_bss")));
// External linkage (declared in mg_idma.h): shared with the now-inlined
// mg_idma_wait() so both the issue path (here) and the wait path observe a
// single completion counter. Only the issue path touches mg_idma_issued, so it
// stays file-static.
uint8_t mg_idma_completed[2] __attribute__((section(".tile_bss")));

// WORKAROUND: the current iDMA HW (at least in the GVSOC model) has no job
// queue and holds only one in-flight transfer per direction; issuing a new
// transfer while the engine is busy raises
// "read IDMA_NEXT_ID ... when not in IDLE state". We emulate a depth-1
// per-direction job queue in software (see mg_idma_issue()) so callers can
// pipeline issues without manually interleaving waits. Raise this once the
// HW/model gains real job queueing.
#define MG_IDMA_HW_QUEUE_DEPTH 1

static inline __ALWAYS_INLINE_ void mg_idma_issue(eu_controller_t *eu,
                                                  eu_wait_mode_t mode,
                                                  uint8_t dir,
                                                  mg_event_t *event,
                                                  mg_event_callback_t callback)
{
    uint8_t idx = dir ? 1 : 0;
    // WORKAROUND: backpressure - block until this direction has a free slot in
    // the (software-emulated) HW job queue before issuing. Draining a
    // completion pulse here shares mg_idma_completed[idx] with mg_idma_wait();
    // that is safe because same-direction transfers retire strictly FIFO.
    while ((uint8_t)(mg_idma_issued[idx] - mg_idma_completed[idx]) >= MG_IDMA_HW_QUEUE_DEPTH) {
        uint32_t done = dir ? eu32_idma_wait_o2a(eu, mode) : eu32_idma_wait_a2o(eu, mode);
        if (done) {
            mg_idma_completed[idx]++;
        }
    }
    mg_event_init(event, (int32_t)mg_idma_issued[idx], callback);
    mg_idma_issued[idx]++;
}

void mg_idma_memcpy_1d(idma_controller_t *idma,
                       eu_controller_t *eu,
                       eu_wait_mode_t mode,
                       uint8_t dir,
                       uint32_t axi_addr,
                       uint32_t obi_addr,
                       uint32_t len,
                       mg_event_t *event,
                       mg_event_callback_t callback)
{
    // May block until a prior same-direction transfer drains (see
    // mg_idma_issue() and the MG_IDMA_HW_QUEUE_DEPTH workaround).
    mg_idma_issue(eu, mode, dir, event, callback);
    // Triggers the transfer.
    idma_memcpy_1d(idma, dir, axi_addr, obi_addr, len);
}

void mg_idma_memcpy_2d(idma_controller_t *idma,
                       eu_controller_t *eu,
                       eu_wait_mode_t mode,
                       uint8_t dir,
                       uint32_t axi_addr,
                       uint32_t obi_addr,
                       uint32_t len,
                       uint32_t std,
                       uint32_t reps,
                       mg_event_t *event,
                       mg_event_callback_t callback)
{
    // May block until a prior same-direction transfer drains (see
    // mg_idma_issue() and the MG_IDMA_HW_QUEUE_DEPTH workaround).
    mg_idma_issue(eu, mode, dir, event, callback);
    idma_memcpy_2d(idma, dir, axi_addr, obi_addr, len, std, reps);
}

// mg_idma_wait() is defined as a static inline in mg_idma.h so it folds into
// its call sites (removing the call/return and the one-time cold I-cache miss,
// and letting constant-propagation drop the callback tail where it is NULL).
