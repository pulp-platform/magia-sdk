// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "eventunit.h"
#include "fsync.h"
#include "idma.h"
#include "tile.h"

#include "kernels_compare_utils.h"
#include "rope_fp16_spatz.h"

#include "data.h"

/* Only for SPATZ_BINARY_START: the binary array itself is defined by the kernel source. */
#define SPATZ_BINARY_NO_DEFINE
#include "onnx_rope_task_bin.h"
#undef SPATZ_BINARY_NO_DEFINE

/* Output tensor in L2: every tile writes back its own rows. */
static float16 Y[ROWS * HEADS * HEAD_DIM];

static void sync(void)
{
    fsync_sync_global();
    eu_fsync_wait(WFE);
}

int main(void)
{
    uint32_t cycle_start;
    uint32_t cycle_stop;
    int ret = 0;

    if (HID == 0)
        printf("############################## ONNX_ROPE TEST on Tiles "
               "#############################\n");

    fsync_init();
    idma_init();
    eu_init();
    eu_spatz_init(0);
    eu_fsync_init(0);
    eu_idma_init(0);
    spatz_init(SPATZ_BINARY_START);
    perf_init();

    sync();

    cycle_start = perf_get_cycles();
    MAGIA_rope_fp16_spatz(X, cos_tab, sin_tab, ROWS, HEADS, HEAD_DIM, Y);
    cycle_stop = perf_get_cycles();

    printf("[CV32 (%d)] rope completed in %d cycles\n", HID, cycle_stop - cycle_start);

    sync();

    spatz_clk_dis();

    if (HID == 0) {
        ret = compare_fp16_bitwise((uintptr_t)Y, (uintptr_t)G, ROWS * HEADS * HEAD_DIM);
        printf("[CV32] Test %s with %d mismatches\n", ret ? "FAILED" : "SUCCESS", ret);
    }

    sync();

    return ret;
}
