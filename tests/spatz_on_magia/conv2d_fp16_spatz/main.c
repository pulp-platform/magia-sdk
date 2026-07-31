// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/* Test for the FP16 conv2d Spatz kernel in kernels/spatz_fp16/conv2d. */

#include "tile.h"

#include "eventunit.h"

#include "data.h"
#include "kernel_test_utils.h"
#include "conv2d_fp16_spatz.h"

#define HID get_hartid()

/*
 * One output buffer per tile: each tile pre-fills its own with NaN and only
 * checks what it wrote. The non-zero initializer keeps the object in .data,
 * since crt0 clears .bss from hart 0 only, while the other tiles already run.
 */
static float16 Y[NUM_HARTS][OUT_LEN] = {{1.0f}};

/* Defined by the CV32 linker script; the Spatz blob is embedded by the kernel */
extern uint32_t _spatz_binary_start;

int main(void)
{
    uint32_t in_shape[4]  = {IN_N, IN_C, IN_H, IN_W};
    uint32_t out_shape[4] = {IN_N, OUT_C, OUT_H, OUT_W};
    bool ok;

    if (HID == 0)
        printf("\n########## CONV2D_FP16_SPATZ KERNEL TEST (%dx%dx%dx%d to %d channels, kernel "
               "%dx%d) on %d Tiles ##########\n\n",
               IN_N,
               IN_C,
               IN_H,
               IN_W,
               OUT_C,
               KERNEL_H,
               KERNEL_W,
               NUM_HARTS);

    kt_mark_unwritten(Y[HID], OUT_LEN);
    kt_spatz_init((uint32_t)&_spatz_binary_start);

    MAGIA_conv2d_fp16_spatz(X,
                            WEIGHTS,
                            BIASES,
                            Y[HID],
                            in_shape,
                            out_shape,
                            KERNEL_H,
                            KERNEL_W,
                            STRIDE_H,
                            STRIDE_W,
                            PAD_H,
                            PAD_W,
                            GROUPS,
                            HAS_BIAS);

    spatz_clk_dis();

    ok = kt_check(Y[HID], G, OUT_LEN, ULP_TOLL);
    printf("[CV32 (%d)] Test %s\n", HID, ok ? "SUCCESS" : "FAILED");

    if (HID == 0)
        printf("\n##########################################################################\n\n");

    return ok ? 0 : -1;
}
