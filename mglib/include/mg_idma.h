// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#pragma once

#include <stdint.h>
#include "eventunit32.h"
#include "idma.h"
#include "mg_event.h"

/**
 * Issue an asynchronous 1D iDMA transfer and stamp `event` with the id needed
 * to later wait for its completion via mg_idma_wait().
 *
 * WORKAROUND: the current iDMA HW holds only one in-flight transfer per
 * direction, so this may block (per `mode`, draining `eu` completion pulses)
 * until a previously issued same-direction transfer has drained before it can
 * issue - a software-emulated depth-1 job queue. See MG_IDMA_HW_QUEUE_DEPTH in
 * mg_idma.c.
 */
extern void mg_idma_memcpy_1d(idma_controller_t *idma,
                              eu_controller_t *eu,
                              eu_wait_mode_t mode,
                              uint8_t dir,
                              uint32_t axi_addr,
                              uint32_t obi_addr,
                              uint32_t len,
                              mg_event_t *event,
                              mg_event_callback_t callback);

/**
 * Issue an asynchronous 2D iDMA transfer and stamp `event` with the id needed
 * to later wait for its completion via mg_idma_wait().
 *
 * WORKAROUND: as for mg_idma_memcpy_1d(), this may block (per `mode`, draining
 * `eu` completion pulses) until a previously issued same-direction transfer has
 * drained before it can issue - a software-emulated depth-1 job queue.
 */
extern void mg_idma_memcpy_2d(idma_controller_t *idma,
                              eu_controller_t *eu,
                              eu_wait_mode_t mode,
                              uint8_t dir,
                              uint32_t axi_addr,
                              uint32_t obi_addr,
                              uint32_t len,
                              uint32_t std,
                              uint32_t reps,
                              mg_event_t *event,
                              mg_event_callback_t callback);

/**
 * Per-direction completion counter (index 0 = A2O/L2->L1, 1 = O2A/L1->L2).
 * Defined in mg_idma.c and advanced from both the issue path (mg_idma_issue)
 * and mg_idma_wait() below; shared so both observe a single instance.
 */
extern uint8_t mg_idma_completed[2];

/**
 * Block (per `mode`) until `event` (as produced by mg_idma_memcpy_1d/2d for
 * direction `dir`) has completed, then run its callback if set.
 *
 * iDMA transfers on the same direction complete strictly in issue order, so
 * this consumes hardware completion pulses one at a time - advancing a
 * per-direction completion counter - until that counter has caught up with
 * `event`'s id. That may happen immediately, if the transfer already
 * completed while waiting on a later event, or only after spinning on new
 * pulses.
 *
 * Defined here as a static inline (rather than out-of-line in mg_idma.c) so it
 * folds into its call sites under -O/-flto.
 */
static inline __ALWAYS_INLINE_ void
mg_idma_wait(eu_controller_t *eu, uint8_t dir, eu_wait_mode_t mode, mg_event_t *event)
{
    uint8_t idx    = dir ? 1 : 0;
    uint8_t target = (uint8_t)(event->id + 1);

    // the event may already be done - e.g. its completion pulse was
    // consumed while waiting on a later id - in which case we must not
    // wait on the hardware at all.
    while (!mg_seq_ge(mg_idma_completed[idx], target)) {
        // dnot yet the right one: spin back into the hardware wait.
        uint32_t done;
        if (dir) {
            done = eu32_idma_wait_o2a(eu, mode);
        } else {
            done = eu32_idma_wait_a2o(eu, mode);
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
