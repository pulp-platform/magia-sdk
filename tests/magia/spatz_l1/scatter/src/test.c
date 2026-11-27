// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include "l1check.h"
#include "scatter_fp16_spatz.h"

/*
 * ScatterElements over the middle axis: data [13, 9, 7], indices / updates [13, 4, 7]. The
 * indices of one (outer, inner) column are distinct, one in four is negative. The full API takes
 * int64 indices, the L1 API int32.
 */
#define OUTER 13
#define DAX   9
#define IAX   4
#define INNER 7
#define ND    (OUTER * DAX * INNER)
#define NI    (OUTER * IAX * INNER)

uint16_t d[ND], u[NI], y_ref[ND];
int64_t idx64[NI];
int32_t idx32[NI];

int main(void)
{
    l1c_init();
    if (get_hartid() == 0) {
        l1c_fill(d, ND, 1, 10, 17, 1);
        l1c_fill(u, NI, 2, 10, 17, 1);
        for (uint32_t o = 0; o < OUTER; o++)
            for (uint32_t a = 0; a < IAX; a++)
                for (uint32_t i = 0; i < INNER; i++) {
                    uint32_t k = (o * IAX + a) * INNER + i;
                    int32_t t  = (int32_t)((2 * a + o + i) % DAX);
                    idx32[k]   = (a == IAX - 1) ? t - DAX : t;
                    idx64[k]   = idx32[k];
                }
    }
    l1c_sync();

    L1C_FULL(MAGIA_scatter_fp16_spatz((const float16 *)d, idx64, (const float16 *)u, (float16 *)y_ref, OUTER, INNER, 1, DAX, IAX));
    l1c_sync();

    l1_alloc_init();
    uint16_t *d1 = l1c_alloc(ND * 2), *u1 = l1c_alloc(NI * 2), *y1 = l1c_alloc(ND * 2);
    int32_t *i1 = l1c_alloc(NI * 4);
    l1c_copy_in(d1, d, ND * 2);
    l1c_copy_in(u1, u, NI * 2);
    l1c_copy_in(i1, idx32, NI * 4);
    L1C_L1(MAGIA_scatter_fp16_spatz_l1((const float16 *)d1, i1, (const float16 *)u1, (float16 *)y1, OUTER, INNER, 1, DAX, IAX));

    uint32_t mx, err = l1c_compare(y1, y_ref, ND, 0, &mx);
    return l1c_report("scatter", err, mx);
}
