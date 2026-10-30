// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include <stdint.h>
#include "test.h"

#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"
#include "mg_idma.h"

/**
 * Nearest x2 upsample of a HWC tensor (UP_C fp16 channels per pixel) by DMA replication, no compute: the input
 * pixels are split over the tiles, in segments cut at the row ends. A segment is loaded to L1 in one transfer, then
 * written to L2 with four 2D transfers, one per (row, column) phase of the output, each putting the segment's
 * pixels on every other output pixel (stride 2 pixels in L2). Double buffered: the next segment loads while this
 * one is written. The data is a pattern of 16-bit words, written by all the tiles before the run and checked after
 * it; tile 0 prints the cycles of the run, between two global barriers.
 */

#define HID get_hartid()

#define L2EXT __attribute__((section(".l2ext"), aligned(64)))

static uint16_t x_l2[UP_H * UP_W * UP_C] L2EXT;
static uint16_t y_l2[4 * UP_H * UP_W * UP_C] L2EXT;

static void sync(void)
{
    fsync_sync_global();
    eu_fsync_wait(WFE);
}

static uint32_t seg_end(uint32_t p, uint32_t p1, uint32_t w)
{
    uint32_t e = (p / w + 1) * w;
    return e < p1 ? e : p1;
}

static void upsample2x_dma(const uint16_t *X, uint16_t *Y, uint32_t c, uint32_t h, uint32_t w)
{
    uint32_t px = c * sizeof(uint16_t), n = h * w / NUM_HARTS, left = h * w % NUM_HARTS;
    uint32_t p0 = HID * n + (HID < left ? HID : left), p1 = p0 + n + (HID < left ? 1 : 0);
    uint32_t buf[2], b = 0, p, q;
    mg_event_t ld, st;

    if (p0 == p1)
        return;
    l1_alloc_init();
    buf[0] = (uint32_t)l1_alloc(w * px);
    buf[1] = (uint32_t)l1_alloc(w * px);
    mg_idma_memcpy_1d(WFE, 0, (uint32_t)(X + p0 * c), buf[0], (seg_end(p0, p1, w) - p0) * px, &ld, NULL);
    for (p = p0; p < p1; p = q, b ^= 1) {
        uint32_t y = p / w, x = p % w;
        q = seg_end(p, p1, w);
        mg_idma_wait(0, WFE, &ld);
        if (q < p1)
            mg_idma_memcpy_1d(WFE, 0, (uint32_t)(X + q * c), buf[b ^ 1], (seg_end(q, p1, w) - q) * px, &ld, NULL);
        for (uint32_t a = 0; a < 2; a++)
            for (uint32_t d = 0; d < 2; d++)
                mg_idma_memcpy_2d(WFE, 1, (uint32_t)(Y + ((2 * y + a) * 2 * w + 2 * x + d) * c), buf[b], px, 2 * px,
                                  q - p, &st, NULL);
        /* the buffer is loaded again two segments later */
        mg_idma_wait(1, WFE, &st);
    }
}

static uint16_t pattern(uint32_t i)
{
    return (uint16_t)(i * 40503u + (i >> 16));
}

int main(void)
{
    uint32_t n = UP_H * UP_W * UP_C, errors = 0, t0, t1;

    fsync_init();
    idma_init();
    eu_init();
    eu_fsync_init(0);
    eu_idma_init(0);
    perf_init();

    for (uint32_t i = HID; i < n; i += NUM_HARTS)
        x_l2[i] = pattern(i);

    sync();
    t0 = perf_get_cycles();
    upsample2x_dma(x_l2, y_l2, UP_C, UP_H, UP_W);
    sync();
    t1 = perf_get_cycles();

    for (uint32_t o = HID; o < 4 * n; o += NUM_HARTS) {
        uint32_t p = o / UP_C, oy = p / (2 * UP_W), ox = p % (2 * UP_W);
        if (y_l2[o] != pattern(((oy / 2) * UP_W + ox / 2) * UP_C + o % UP_C))
            errors++;
    }
    if (errors)
        printf("[upsample_dma] tile %u errors=%u\n", HID, errors);
    if (HID == 0)
        printf("[upsample_dma] %ux%u x%u ch -> %ux%u cycles=%u\n", UP_H, UP_W, UP_C, 2 * UP_H, 2 * UP_W, t1 - t0);
    sync();

    return errors != 0;
}
