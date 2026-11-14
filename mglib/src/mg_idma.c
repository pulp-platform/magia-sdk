// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include <stdint.h>
#include "mg_idma.h"
#include "addr_map/tile_addr_map.h"
#include "regs/tile_ctrl.h"

#ifndef MG_HW_MCAST
#define MG_HW_MCAST 0
#endif

#if MG_HW_MCAST
// Hardware multicast (MAGIA v4, NoC collectives): one wide iDMA job written to
// `dst_addr`; the NoC copies it to every tile whose address matches dst_addr
// outside the mask (address bits 20 and up), at the same offset.
// The multicast mask is encoded in the AXI user field.
static inline __ALWAYS_INLINE_ void
mg_idma_mcast_hw(uint32_t dst_addr, uint32_t obi_addr, uint32_t len, uint32_t mask,
                 mg_event_t *event, mg_event_callback_t callback)
{
    uint32_t job_id = idma_collective_1d(dst_addr, obi_addr, len, mask, MULTICAST);
    mg_event_init(event, job_id, callback);
    // set back to unicast mode
    mmio32(IDMA_COLLECTIVE_SEL_ADDR(1)) = 0;
    mmio32(IDMA_MULTICAST_MASK_ADDR(1)) = 0;
}
#endif

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
    (void)mode;
    if (me > 0) {
        mg_event_init(
            event, idma_memcpy_2d_strided(1, first, obi_addr, len, tile_std, 0, me), callback);
    }
    if (me + 1 < n) {
        mg_event_init(event,
                      idma_memcpy_2d_strided(
                          1, first + (me + 1) * tile_std, obi_addr, len, tile_std, 0, n - me - 1),
                      callback);
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
#if MG_HW_MCAST
    // HW multicast targets also the *source* tile, deviating from the semantics
    // of the SW multicast.
    // Therefore, use HW multicast only when `dst_addr` == `src_addr`, which is
    // generally the case exploited by mgc.
    if (dst_addr == obi_addr) {
        if (MESH_X_TILES > 1)
            mg_idma_mcast_hw(dst_addr, obi_addr, len, ROW_MASK, event, callback);
        return;
    }
#endif
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
#if MG_HW_MCAST
    // HW multicast targets also the *source* tile, deviating from the semantics
    // of the SW multicast.
    // Therefore, use HW multicast only when `dst_addr` == `src_addr`, which is
    // generally the case exploited by mgc.
    if (dst_addr == obi_addr) {
        if (MESH_Y_TILES > 1)
            mg_idma_mcast_hw(dst_addr, obi_addr, len, COLUMN_MASK, event, callback);
        return;
    }
#endif
    mg_idma_mcast(mode, first, std, y, MESH_Y_TILES, obi_addr, len, event, callback);
}

// mg_idma_wait() is defined as a static inline in mg_idma.h so it folds into
// its call sites (removing the call/return and the one-time cold I-cache miss,
// and letting constant-propagation drop the callback tail where it is NULL).
