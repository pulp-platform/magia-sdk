// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/*
 * Test for the FP16 maxpool2d Spatz kernel in kernels/spatz_fp16/maxpool2d.
 *
 * Two pools over the same input: 2x2/stride 2/pad 0, where every window is 4-byte
 * aligned and the vector path runs, and 3x3/stride 2/pad 1 - ResNet18's pool - where
 * the window base is odd from the second output column on and the kernel has to fall
 * back to a scalar window max, because the Spatz VLSU corrupts misaligned vector
 * accesses. A maximum returns an input value unchanged, so both are bit-exact.
 */

#include "tile.h"

#include "eventunit.h"

#include "data.h"
#include "kernel_test_utils.h"
#include "maxpool2d_fp16_spatz.h"

#define HID get_hartid()

/*
 * One output buffer per tile, reused by both cases: each tile pre-fills its own with
 * NaN and only checks what it wrote. The non-zero initializer keeps the object in
 * .data, since crt0 clears .bss from hart 0 only, while the other tiles already run.
 */
static float16 Y[NUM_HARTS][OUT_LEN_MAX] = {{1.0f}};

/* Defined by the CV32 linker script; the Spatz blob is embedded by the kernel */
extern uint32_t _spatz_binary_start;

static bool run_case(const char *name,
                     uint32_t kernel,
                     uint32_t stride,
                     uint32_t pad,
                     uint32_t out_h,
                     uint32_t out_w,
                     uint32_t out_len,
                     const float16 *golden)
{
    uint32_t in_shape[4]  = {IN_N, IN_C, IN_H, IN_W};
    uint32_t out_shape[4] = {IN_N, IN_C, out_h, out_w};
    bool ok;

    kt_mark_unwritten(Y[HID], out_len);

    MAGIA_maxpool2d_fp16_spatz(
        X, Y[HID], kernel, kernel, stride, stride, pad, pad, in_shape, out_shape);

    ok = kt_check(Y[HID], golden, out_len, ULP_TOLL);
    printf("[CV32 (%d)] %s %s\n", HID, name, ok ? "SUCCESS" : "FAILED");

    return ok;
}

int main(void)
{
    bool ok;

    if (HID == 0)
        printf("\n########## MAXPOOL2D_FP16_SPATZ KERNEL TEST (%dx%dx%dx%d) "
               "on %d Tiles ##########\n\n",
               IN_N,
               IN_C,
               IN_H,
               IN_W,
               NUM_HARTS);

    kt_spatz_init((uint32_t)&_spatz_binary_start);

    ok = run_case(
        "2x2 stride 2 pad 0 (vector)", KERNEL_A, STRIDE_A, PAD_A, OUT_H_A, OUT_W_A, OUT_LEN_A, G_A);

    ok &= run_case("3x3 stride 2 pad 1 (misaligned windows)",
                   KERNEL_B,
                   STRIDE_B,
                   PAD_B,
                   OUT_H_B,
                   OUT_W_B,
                   OUT_LEN_B,
                   G_B);

    spatz_clk_dis();

    printf("[CV32 (%d)] Test %s\n", HID, ok ? "SUCCESS" : "FAILED");

    if (HID == 0)
        printf("\n##########################################################################\n\n");

    return ok ? 0 : -1;
}
