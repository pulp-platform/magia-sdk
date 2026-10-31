// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>
//
// L1 kernels called (on the CV32 control core) by the mgc-generated test.c.
// All pointer arguments are L1 addresses.
//
// The ring holds CONV_NSLOT expanded input rows: input row iy is in slot iy % CONV_NSLOT, as CONV_CP rows of
// CONV_KC = CONV_K * CONV_CIN values, row p the patch row of output column x0 + p: input columns
// (x0 + p) * CONV_S - CONV_PAD .. + CONV_K - 1, all channels (zeros outside the image).

#include <stdint.h>

#include "test.h"
#include "tile.h"
#include "mg_idma.h"
#include "mg_redmule.h"

#define SLOT_BYTES (CONV_CP * CONV_KC * 2)

/* The loads in flight (started by l1_e_load, waited by l1_e_load_wait in the same step): input rows pend_lo ..
   pend_hi of the block of columns from pend_x0 (none if pend_lo > pend_hi); the last one's event, as same-direction
   transfers complete in order. */
static mg_event_t load_ev __attribute__((section(".tile_bss")));
static int32_t pend_lo __attribute__((section(".tile_bss")));
static int32_t pend_hi __attribute__((section(".tile_bss")));
static uint32_t pend_x0 __attribute__((section(".tile_bss")));

static inline uint32_t slot_addr(uint32_t ring, int32_t iy)
{
    return ring + (uint32_t)(iy % CONV_NSLOT) * SLOT_BYTES;
}

/**
 * Starts the loads into `ring` of the expanded input rows that output row `row` needs and output row row - 1 did not
 * (all of them if `row` is the first row of the pipeline, r0), inside the image, for the CONV_CP output columns from
 * x0. Each is one 2-D transfer: CONV_CP patch rows of CONV_KC values, CONV_S * CONV_CIN values apart in L2 (they
 * overlap). The patch rows at the left and right edges of the image read past the input row (the previous or next
 * row, or the L2 next to x_inp): l1_e_load_wait zeroes those parts once the data has landed.
 */
void l1_e_load(uint32_t ring, uint32_t row, uint32_t r0, uint32_t x0)
{
    int32_t hi = (int32_t)row * CONV_S - CONV_PAD + CONV_K - 1;
    int32_t lo = row == r0 ? (int32_t)row * CONV_S - CONV_PAD : hi - CONV_S + 1;

    if (lo < 0)
        lo = 0;
    if (hi > CONV_H - 1)
        hi = CONV_H - 1;
    pend_lo = lo;
    pend_hi = hi;
    pend_x0 = x0;
    for (int32_t iy = lo; iy <= hi; iy++) {
        int32_t c = iy * CONV_W + (int32_t)x0 * CONV_S - CONV_PAD; /* input pixel of patch row 0, kx = 0 */

        mg_idma_memcpy_2d_strided(WFE, 0, (uint32_t)x_inp + (uint32_t)(c * CONV_CIN * 2), slot_addr(ring, iy),
                                  CONV_KC * 2, CONV_S * CONV_CIN * 2, CONV_KC * 2, CONV_CP, &load_ev, NULL);
    }
}

/* Zeroes the taps kx of patch row p of the expanded row at e whose input column c + kx is outside the image. */
static inline void zero_outside(uint16_t *e, int32_t c)
{
    for (int32_t kx = 0; kx < CONV_K; kx++)
        if (c + kx < 0 || c + kx >= CONV_W)
            for (int32_t ci = 0; ci < CONV_CIN; ci++)
                e[kx * CONV_CIN + ci] = 0;
}

/* Waits for the loads started by l1_e_load, then zeroes the parts of their patch rows outside the image. */
void l1_e_load_wait(uint32_t ring)
{
    if (pend_lo > pend_hi)
        return;
    mg_idma_wait(0, WFE, &load_ev);
    for (int32_t iy = pend_lo; iy <= pend_hi; iy++) {
        uint16_t *e = (uint16_t *)slot_addr(ring, iy);
        int32_t p;

        /* left edge, then right edge (CONV_CP output columns from x0) */
        for (p = 0; p < CONV_CP && ((int32_t)pend_x0 + p) * CONV_S - CONV_PAD < 0; p++)
            zero_outside(e + p * CONV_KC, ((int32_t)pend_x0 + p) * CONV_S - CONV_PAD);
        for (p = CONV_CP - 1; p >= 0 && ((int32_t)pend_x0 + p) * CONV_S - CONV_PAD + CONV_K - 1 >= CONV_W; p--)
            zero_outside(e + p * CONV_KC, ((int32_t)pend_x0 + p) * CONV_S - CONV_PAD);
    }
}

/**
 * y [CONV_CP x tc] = the convolution of the block of output row `row` from the expanded input rows in `ring`, with the
 * weights `w` [CONV_KKC x tc]: for each kernel row ky whose input row is inside the image, in ky order,
 * y += E_iy [CONV_CP x CONV_KC] @ w[ky rows] [CONV_KC x tc], the first job a MATMUL. Issues the jobs (two in the
 * RedMulE queue) and waits for the last one.
 */
void l1_e_gemm(uint32_t y, uint32_t ring, uint32_t w, uint32_t row, uint32_t tc)
{
    int first = 1;
    mg_event_t ev;

    for (int32_t ky = 0; ky < CONV_K; ky++) {
        int32_t iy = (int32_t)row * CONV_S - CONV_PAD + ky;
        uint32_t wk = w + (uint32_t)ky * CONV_KC * tc * 2;

        if (iy < 0 || iy >= CONV_H)
            continue; /* zeros: nothing to accumulate */
        if (first)
            mg_redmule_matmul(WFE, slot_addr(ring, iy), wk, y, CONV_CP, CONV_KC, tc, &ev, NULL);
        else
            mg_redmule_gemm(WFE, slot_addr(ring, iy), wk, y, CONV_CP, CONV_KC, tc, &ev, NULL);
        first = 0;
    }
    /* the jobs complete in order: waiting for the last one waits for all of them */
    mg_redmule_wait(WFE, &ev);
}
