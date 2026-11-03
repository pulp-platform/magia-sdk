// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#pragma once

#include <stdint.h>
#include "eventunit32.h"
#include "idma.h"
#include "addr_map/tile_addr_map.h"
#include "regs/tile_ctrl.h"
#include "mg_event.h"

/**
 * Issue an asynchronous 1D iDMA transfer and stamp `event` with the id needed
 * to later wait for its completion via mg_idma_wait().
 *
 * The event id is the real hardware-issued job id, returned by the NEXT_ID read
 * that launches the transfer. Each direction queues its jobs in hardware
 * (depth 16): if that queue is full, the launch stalls until a slot frees.
 * `mode` is unused here, and kept for symmetry with mg_idma_wait().
 */
extern void mg_idma_memcpy_1d(eu_wait_mode_t mode,
                              uint8_t dir,
                              uint32_t axi_addr,
                              uint32_t obi_addr,
                              uint32_t len,
                              mg_event_t *event,
                              mg_event_callback_t callback);

/**
 * Issue an asynchronous 2D iDMA transfer and stamp `event` with the id needed
 * to later wait for its completion via mg_idma_wait(). Queued in hardware as
 * mg_idma_memcpy_1d().
 */
extern void mg_idma_memcpy_2d(eu_wait_mode_t mode,
                              uint8_t dir,
                              uint32_t axi_addr,
                              uint32_t obi_addr,
                              uint32_t len,
                              uint32_t std,
                              uint32_t reps,
                              mg_event_t *event,
                              mg_event_callback_t callback);

/**
 * Block (per `mode`) until `event` (as produced by mg_idma_memcpy_1d/2d for
 * direction `dir`) has completed, then run its callback if set.
 *
 * iDMA transfers on the same direction complete strictly in issue order, so
 * the transfer is done once the channel's DONE_ID (the id of the last
 * completed job) has caught up with `event`'s id. That may hold immediately,
 * if the transfer already completed while waiting on a later event; otherwise
 * this sleeps on the direction's completion event and reads DONE_ID again.
 * Completions are counted only from DONE_ID, never from the pulses: a pulse
 * left latched by a transfer already counted would otherwise count one still
 * running. A stale pulse only costs one more read, and a completion between
 * the read and the sleep leaves its pulse latched, so it is not lost.
 *
 * Defined here as a static inline (rather than out-of-line in mg_idma.c) so it
 * folds into its call sites under -O/-flto.
 */
static inline __ALWAYS_INLINE_ void
mg_idma_wait(uint8_t dir, eu_wait_mode_t mode, mg_event_t *event)
{
    uint32_t target = (uint32_t)event->id;

    // DONE_ID and the job ids wrap at 32 bits: compared modulo 2^32 (see mg_seq_ge32())
    while (!mg_seq_ge32(*(volatile uint32_t *)IDMA_DONE_ID_ADDR(dir, 0), target)) {
        if (dir) {
            eu32_idma_wait_o2a(mode);
        } else {
            eu32_idma_wait_a2o(mode);
        }
    }

    mg_event_trigger(event);
}
