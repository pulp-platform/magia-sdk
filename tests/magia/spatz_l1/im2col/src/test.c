// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include "l1check.h"
#include "im2col_fp16_spatz.h"

/* cin even (4-byte aligned patch rows); 13 output rows over the tiles */
#define H   13
#define W   7
#define CIN 4
#define K   5
#define ROWSZ (W * K * K * CIN)

uint16_t x[H * W * CIN], y_ref[H * ROWSZ];

int main(void)
{
    l1c_init();
    if (get_hartid() == 0)
        l1c_fill(x, H * W * CIN, 1, 10, 17, 1);
    l1c_sync();

    L1C_FULL(MAGIA_im2col_fp16_spatz((const float16 *)x, (float16 *)y_ref, H, W, CIN, K));
    l1c_sync();

    l1_alloc_init();
    im2col_fp16_spatz_params_t *p = l1c_alloc(sizeof(*p));
    uint16_t *x1 = l1c_alloc(H * W * CIN * 2), *y1 = l1c_alloc(H * ROWSZ * 2);
    l1c_copy_in(x1, x, H * W * CIN * 2);
    /* the L1 API builds one output row: the whole image, one call per row */
    p->src    = (uintptr_t)x1;
    p->height = H;
    p->width  = W;
    p->cin    = CIN;
    p->ksize  = K;
    p->pad    = (K - 1) / 2;
    p->lo     = 0;
    p->x0     = 0;
    p->cp     = W;
    p->src_w  = W;
    p->src_x  = 0;
    for (uint32_t r = 0; r < H; r++) {
        p->row = r;
        p->dst = (uintptr_t)(y1 + r * ROWSZ);
        L1C_L1(MAGIA_im2col_fp16_spatz_l1(p));
    }

    uint32_t mx, err = l1c_compare(y1, y_ref, H * ROWSZ, 0, &mx);
    return l1c_report("im2col", err, mx);
}
