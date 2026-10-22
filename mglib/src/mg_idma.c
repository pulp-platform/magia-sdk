// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include <stdint.h>
#include "mg_idma.h"
#include "addr_map/tile_addr_map.h"

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

static inline __ALWAYS_INLINE_ void
mg_idma_issue(eu_wait_mode_t mode, uint8_t dir, mg_event_t *event, mg_event_callback_t callback)
{
    // WORKAROUND: backpressure - block until this direction has a free slot in
    // the (software-emulated) HW job queue before issuing. Draining a
    // completion pulse here shares mg_idma_completed[idx] with mg_idma_wait();
    // that is safe because same-direction transfers retire strictly FIFO.
    while ((uint8_t)(mg_idma_issued[dir] - mg_idma_completed[dir]) >= MG_IDMA_HW_QUEUE_DEPTH) {
        uint32_t done = dir ? eu32_idma_wait_o2a(mode) : eu32_idma_wait_a2o(mode);
        if (done) {
            mg_idma_completed[dir]++;
        }
    }
    mg_event_init(event, (int32_t)mg_idma_issued[dir], callback);
    mg_idma_issued[dir]++;
}

void mg_idma_memcpy_1d(eu_wait_mode_t mode,
                       uint8_t dir,
                       uint32_t axi_addr,
                       uint32_t obi_addr,
                       uint32_t len,
                       mg_event_t *event,
                       mg_event_callback_t callback)
{
    // May block until a prior same-direction transfer drains (see
    // mg_idma_issue() and the MG_IDMA_HW_QUEUE_DEPTH workaround).
    mg_idma_issue(mode, dir, event, callback);
    // Triggers the transfer.
    idma_memcpy_1d(dir, axi_addr, obi_addr, len);
}

void mg_idma_memcpy_2d(eu_wait_mode_t mode,
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
    mg_idma_issue(mode, dir, event, callback);
    idma_memcpy_2d(dir, axi_addr, obi_addr, len, std, reps);
}

// Push one L1 block to the `n` tiles (of this tile's row or column) at mesh
// positions 0..n-1 except `me`, which are `tile_std` bytes of L1 apart starting
// at `first`: one 2D job for the tiles before `me` and one for those after it.
static void mg_idma_mcast(eu_wait_mode_t mode,
                          uint32_t first,
                          uint32_t tile_std,
                          uint32_t me,
                          uint32_t n,
                          uint32_t obi_addr,
                          uint32_t len,
                          mg_event_t *event,
                          mg_event_callback_t callback)
{
    if (me > 0) {
        mg_idma_issue(mode, 1, event, callback);
        idma_memcpy_2d_strided(1, first, obi_addr, len, tile_std, 0, me);
    }
    if (me + 1 < n) {
        mg_idma_issue(mode, 1, event, callback);
        idma_memcpy_2d_strided(1, first + (me + 1) * tile_std, obi_addr, len, tile_std, 0, n - me - 1);
    }
}

static inline __ALWAYS_INLINE_ uint32_t mg_idma_hartid(void)
{
    uint32_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    return hartid;
}

void mg_idma_mcast_row(eu_wait_mode_t mode,
                       uint8_t dir,
                       uint32_t dst_addr,
                       uint32_t obi_addr,
                       uint32_t len,
                       mg_event_t *event,
                       mg_event_callback_t callback)
{
    (void)dir;
    uint32_t hartid = mg_idma_hartid();
    uint32_t y = GET_Y_ID(hartid), x = GET_X_ID(hartid);
    uint32_t off   = dst_addr - (L1_BASE + hartid * L1_TILE_OFFSET);
    uint32_t first = L1_BASE + GET_ID(y, 0) * L1_TILE_OFFSET + off;
    uint32_t std   = (GET_ID(y, 1) - GET_ID(y, 0)) * L1_TILE_OFFSET;
    mg_idma_mcast(mode, first, std, x, MESH_X_TILES, obi_addr, len, event, callback);
}

void mg_idma_mcast_col(eu_wait_mode_t mode,
                       uint8_t dir,
                       uint32_t dst_addr,
                       uint32_t obi_addr,
                       uint32_t len,
                       mg_event_t *event,
                       mg_event_callback_t callback)
{
    (void)dir;
    uint32_t hartid = mg_idma_hartid();
    uint32_t y = GET_Y_ID(hartid), x = GET_X_ID(hartid);
    uint32_t off   = dst_addr - (L1_BASE + hartid * L1_TILE_OFFSET);
    uint32_t first = L1_BASE + GET_ID(0, x) * L1_TILE_OFFSET + off;
    uint32_t std   = (GET_ID(1, x) - GET_ID(0, x)) * L1_TILE_OFFSET;
    mg_idma_mcast(mode, first, std, y, MESH_Y_TILES, obi_addr, len, event, callback);
}

// mg_idma_wait() is defined as a static inline in mg_idma.h so it folds into
// its call sites (removing the call/return and the one-time cold I-cache miss,
// and letting constant-propagation drop the callback tail where it is NULL).
