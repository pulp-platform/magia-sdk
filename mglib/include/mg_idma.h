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
#include "addr_map/tile_addr_map.h" // IDMA_BASE_AXI2OBI/OBI2AXI, used by IDMA_DONE_ID_ADDR()

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
 * Defined in mg_idma.c. Kept coherent with mg_idma_hw_done() by both the issue
 * path (mg_idma_issue) and mg_idma_wait() so external readers still see a sane
 * value, but it is no longer the wait predicate (see mg_idma_hw_done()).
 */
extern uint8_t mg_idma_completed[2];

/**
 * Authoritative count of retired transfers for direction `dir` (0 = A2O/L2->L1,
 * 1 = O2A/L1->L2), taken straight from the iDMA's own transfer-id generator
 * (idma_transfer_id_gen), exposed as the DONE_ID register.
 *
 * NOTE ON THE -1: idma_transfer_id_gen resets `completed_q` to 1 and bumps it
 * once per retired transfer, so DONE_ID == (retired transfers) + 1. mglib's
 * issue counter and event ids are 0-based, so we subtract that offset here to
 * get a plain retired-count aligned with `mg_idma_issued[]` / `event->id + 1`.
 *
 * WHY NOT THE EVENT UNIT: the tile Event Unit reports "an iDMA transfer of this
 * direction finished" as a single OR-latched buffer bit. When two same-direction
 * transfers retire before software clears that bit - e.g. the core is parked in
 * a RedMulE wait whose cv.elw keeps waking on the still-latched RedMulE bit -
 * the two completions collapse into one observable edge; a wait that counts
 * edges then falls permanently one behind and its next cv.elw sleeps forever.
 * DONE_ID is bumped once per retired transfer by the iDMA itself and never
 * coalesces. It is a 32-bit up-counter; the low 8 bits are compared circularly
 * via mg_seq_ge().
 */
static inline __ALWAYS_INLINE_ uint8_t mg_idma_hw_done(uint8_t dir)
{
    return (uint8_t)(mmio32(IDMA_DONE_ID_ADDR(dir ? 1 : 0, 0)) - 1u);
}

/**
 * Block (per `mode`) until `event` (as produced by mg_idma_memcpy_1d/2d for
 * direction `dir`) has completed, then run its callback if set.
 *
 * The predicate is the iDMA's own non-coalescing retired-transfer count, not the
 * Event Unit done-latch. In WFE mode cv.elw is used only to sleep between
 * checks: the relevant done-latch is cleared before each sleep so the
 * pending/next transfer completion is guaranteed to produce a fresh rising edge,
 * and the counter is re-tested after the clear in case that completion raced it.
 * A stale or coalesced EU edge can then only cost an extra spin, never a
 * permanent stall.
 *
 * Defined here as a static inline (rather than out-of-line in mg_idma.c) so it
 * folds into its call sites under -O/-flto.
 */
static inline __ALWAYS_INLINE_ void
mg_idma_wait(eu_controller_t *eu, uint8_t dir, eu_wait_mode_t mode, mg_event_t *event)
{
    (void)eu;
    uint8_t  idx    = dir ? 1 : 0;
    uint8_t  target = (uint8_t)(event->id + 1);
    uint32_t clr_ms = dir ? EU_IDMA_O2A_DONE_MASK : EU_IDMA_A2O_DONE_MASK;

    while (!mg_seq_ge(mg_idma_hw_done(dir), target)) {
        if (mode == WFE) {
            eu_clear_events(clr_ms);
            if (mg_seq_ge(mg_idma_hw_done(dir), target))
                break;
            evt_read32(EU_CORE_EVENT_WAIT);
        }
        // POLLING: fall through and re-read the HW counter.
    }
    mg_idma_completed[idx] = mg_idma_hw_done(dir);

#if PROFILE_CMI == 1
    if (!dir)
        stnl_cmi_f();
#endif
#if PROFILE_CMO == 1
    if (dir)
        stnl_cmo_f();
#endif

    mg_event_trigger(event);
}
