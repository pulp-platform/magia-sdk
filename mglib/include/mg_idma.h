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
static inline __ALWAYS_INLINE_ void mg_idma_memcpy_1d(eu_wait_mode_t mode,
                                                      uint8_t dir,
                                                      uint32_t axi_addr,
                                                      uint32_t obi_addr,
                                                      uint32_t len,
                                                      mg_event_t *event,
                                                      mg_event_callback_t callback)
{
    (void)mode;
    mg_event_init(event, idma_memcpy_1d(dir, axi_addr, obi_addr, len), callback);
}

/**
 * Issue an asynchronous 2D iDMA transfer and stamp `event` with the id needed
 * to later wait for its completion via mg_idma_wait(). Queued in hardware like
 * mg_idma_memcpy_1d().
 */
static inline __ALWAYS_INLINE_ void mg_idma_memcpy_2d(eu_wait_mode_t mode,
                                                      uint8_t dir,
                                                      uint32_t axi_addr,
                                                      uint32_t obi_addr,
                                                      uint32_t len,
                                                      uint32_t std,
                                                      uint32_t reps,
                                                      mg_event_t *event,
                                                      mg_event_callback_t callback)
{
    (void)mode;
    mg_event_init(event, idma_memcpy_2d(dir, axi_addr, obi_addr, len, std, reps), callback);
}

/**
 * As mg_idma_memcpy_2d, with a stride on both sides: after each block of `len`
 * bytes the AXI address advances by `axi_std` and the L1 address by `obi_std`,
 * e.g. rows loaded into an L1 buffer with wider rows (zero padding around them).
 */
static inline __ALWAYS_INLINE_ void mg_idma_memcpy_2d_strided(eu_wait_mode_t mode,
                                                              uint8_t dir,
                                                              uint32_t axi_addr,
                                                              uint32_t obi_addr,
                                                              uint32_t len,
                                                              uint32_t axi_std,
                                                              uint32_t obi_std,
                                                              uint32_t reps,
                                                              mg_event_t *event,
                                                              mg_event_callback_t callback)
{
    (void)mode;
    mg_event_init(event,
                  idma_memcpy_2d_strided(dir, axi_addr, obi_addr, len, axi_std, obi_std, reps),
                  callback);
}

/**
 * Native row / column multicast: copy `len` bytes from this tile's L1 at
 * `obi_addr` to the same L1 offset as `dst_addr` (an address in this tile's
 * L1) on every other tile of this tile's row / column. Each contiguous run of
 * destination tiles (before and after this tile) is one 2D iDMA job reading
 * the source with stride 0 and writing with the L1 stride between the tiles,
 * so the bursts towards successive tiles overlap in the NoC instead of each
 * unicast paying the full pipeline fill. Same arguments as mg_idma_memcpy_1d
 * (`dir` must be 1, L1 -> remote L1); `event` tracks the last job issued, and
 * same-direction jobs complete in order, so waiting on it waits for all.
 * Built with `make hw_mcast=1` (MAGIA v4), a multicast whose `dst_addr` is
 * `obi_addr` is instead one NoC hardware multicast job (iDMA multicast mask).
 */
extern void mg_idma_mcast_row(eu_wait_mode_t mode,
                              uint8_t dir,
                              uint32_t dst_addr,
                              uint32_t obi_addr,
                              uint32_t len,
                              mg_event_t *event,
                              mg_event_callback_t callback);

extern void mg_idma_mcast_col(eu_wait_mode_t mode,
                              uint8_t dir,
                              uint32_t dst_addr,
                              uint32_t obi_addr,
                              uint32_t len,
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

/*
 * Blocking transfers: launch, then sleep (per `mode`) until the direction's next completion. Only
 * valid when no other transfer of direction `dir` is in flight, as in a kernel that loads its
 * operands, computes and stores in sequence; with several transfers in flight use the event calls
 * above. Cheaper than mg_idma_memcpy_*() + mg_idma_wait(), which reads DONE_ID before and after
 * sleeping.
 */
static inline __ALWAYS_INLINE_ void mg_idma_sync_wait(eu_wait_mode_t mode, uint8_t dir)
{
    if (dir) {
        eu32_idma_wait_o2a(mode);
    } else {
        eu32_idma_wait_a2o(mode);
    }
}

static inline __ALWAYS_INLINE_ void
mg_idma_memcpy_1d_sync(eu_wait_mode_t mode, uint8_t dir, uint32_t axi_addr, uint32_t obi_addr, uint32_t len)
{
    idma_memcpy_1d(dir, axi_addr, obi_addr, len);
    mg_idma_sync_wait(mode, dir);
}

static inline __ALWAYS_INLINE_ void mg_idma_memcpy_2d_sync(eu_wait_mode_t mode,
                                                           uint8_t dir,
                                                           uint32_t axi_addr,
                                                           uint32_t obi_addr,
                                                           uint32_t len,
                                                           uint32_t std,
                                                           uint32_t reps)
{
    idma_memcpy_2d(dir, axi_addr, obi_addr, len, std, reps);
    mg_idma_sync_wait(mode, dir);
}

static inline __ALWAYS_INLINE_ void mg_idma_memcpy_2d_strided_sync(eu_wait_mode_t mode,
                                                                   uint8_t dir,
                                                                   uint32_t axi_addr,
                                                                   uint32_t obi_addr,
                                                                   uint32_t len,
                                                                   uint32_t axi_std,
                                                                   uint32_t obi_std,
                                                                   uint32_t reps)
{
    idma_memcpy_2d_strided(dir, axi_addr, obi_addr, len, axi_std, obi_std, reps);
    mg_idma_sync_wait(mode, dir);
}
