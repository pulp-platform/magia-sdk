// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#pragma once

#include <stdint.h>
#include "eventunit32.h"
#include "redmule.h"
#include "mg_event.h"
#include "addr_map/tile_addr_map.h" // REDMULE_BASE, used by mg_redmule_hw_done()

/**
 * Issue an asynchronous RedMulE GEMM job and stamp `event` with the id needed
 * to later wait for its completion via mg_redmule_wait().
 *
 * The event id is the real hardware-issued job id, obtained by reading the
 * ACQUIRE register (redmule_api.acquire()). If the hardware's job queue
 * (depth 2) is already full, this call blocks - draining completion pulses
 * through `eu`/`mode`, exactly like mg_redmule_wait() does - until a slot
 * frees and a fresh id can be acquired.
 */
extern void mg_redmule_gemm(redmule_controller_t *ctrl,
                            eu_controller_t *eu,
                            eu_wait_mode_t mode,
                            uint32_t x,
                            uint32_t w,
                            uint32_t y,
                            uint16_t m,
                            uint16_t n,
                            uint16_t k,
                            mg_event_t *event,
                            mg_event_callback_t callback);
extern void mg_redmule_gemm_enqueue(redmule_controller_t *ctrl,
                                    eu_controller_t *eu,
                                    eu_wait_mode_t mode,
                                    uint32_t x,
                                    uint32_t w,
                                    uint32_t y,
                                    uint16_t m,
                                    uint16_t n,
                                    uint16_t k,
                                    mg_event_t *event,
                                    mg_event_callback_t callback);
extern void mg_redmule_gemm_commit_start(redmule_controller_t *ctrl);
extern void mg_redmule_gemm_start(redmule_controller_t *ctrl);
extern void mg_redmule_gemm_commit(redmule_controller_t *ctrl);

/**
 * Completion counter. Defined in mg_redmule.c. Kept coherent with
 * mg_redmule_hw_done() by the acquire/backpressure path (mg_redmule_gemm/enqueue)
 * and mg_redmule_wait() so external readers still see a sane value, but it is no
 * longer the wait predicate (see mg_redmule_hw_done()). Only the low 8 bits are
 * significant (see mg_seq_ge()).
 */
extern uint8_t mg_redmule_completed;

/**
 * Authoritative RedMulE job-completion count, read from the HWPE controller's
 * RUNNING_JOB register. In hwpe_ctrl_target this is `job_running_id_q`: an 8-bit
 * up-counter that resets to 0 and is bumped once per job_done pulse (same event
 * that pops the job FIFO and frees a queue slot). Despite the register name it
 * is a retired-job count, and it is already 0-based - no offset to subtract,
 * unlike the iDMA DONE_ID (see mg_idma_hw_done).
 *
 * WHY NOT THE EVENT UNIT: "a RedMulE job finished" reaches the core as one
 * OR-latched Event Unit buffer bit. With the depth-2 job queue two jobs can
 * retire while the core is parked elsewhere - e.g. in an iDMA wait whose cv.elw
 * keeps waking on the still-latched iDMA bit - and their two completions then
 * collapse into a single edge. A wait that counts edges falls permanently one
 * behind and its next cv.elw never wakes. RUNNING_JOB counts every completion
 * and cannot coalesce.
 */
static inline __ALWAYS_INLINE_ uint8_t mg_redmule_hw_done(void)
{
#if defined(REDMULE_MM) && (REDMULE_MM == 1)
    return (uint8_t)mmio32(REDMULE_BASE + REDMULE_REG_OFFS + REDMULE_RUNNING_JOB);
#else
    // Custom-instruction path: no HW job queue, only ever one job in flight, so
    // the SW pulse count cannot coalesce - fall back to it.
    return mg_redmule_completed;
#endif
}

/**
 * Block (per `mode`) until `event` (as produced by mg_redmule_gemm) has
 * completed, then run its callback if set.
 *
 * The predicate is the HWPE RUNNING_JOB completion counter, not the Event Unit
 * done-latch. In WFE mode cv.elw is used only to sleep between checks: the
 * RedMulE done-latch is cleared before each sleep so the pending/next job
 * completion is guaranteed to produce a fresh rising edge, and the counter is
 * re-tested after the clear in case that completion raced it. A stale or
 * coalesced EU edge can then only cost an extra spin, never a permanent stall.
 *
 * Defined here as a static inline (rather than out-of-line in mg_redmule.c) so
 * it folds into its call sites under -O/-flto.
 */
static inline __ALWAYS_INLINE_ void
mg_redmule_wait(eu_controller_t *eu, eu_wait_mode_t mode, mg_event_t *event)
{
    (void)eu;
    uint8_t target = (uint8_t)(event->id + 1);

    while (!mg_seq_ge(mg_redmule_hw_done(), target)) {
        if (mode == WFE) {
            eu_clear_events(EU_REDMULE_DONE_MASK);
            if (mg_seq_ge(mg_redmule_hw_done(), target))
                break;
            evt_read32(EU_CORE_EVENT_WAIT);
        }
        // POLLING: fall through and re-read the HW counter.
    }
    mg_redmule_completed = mg_redmule_hw_done();

#if PROFILE_CMP == 1
    stnl_cmp_f();
#endif

    mg_event_trigger(event);
}
