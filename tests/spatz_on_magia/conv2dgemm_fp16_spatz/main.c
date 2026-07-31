// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/*
 * Exercises kernels/spatz_fp16/conv2dgemm on the whole mesh.
 *
 * The kernel shards the output channels, blocks the im2col over the output rows, and
 * takes a different path per stride and per grouping, so this runs five convolutions over
 * the same input: 3x3/stride 1/pad 1 goes through the iDMA im2col and 3x3/stride 2/pad 1
 * through the scalar fallback, each ungrouped and again depthwise (group = C_in, the case
 * the ViT models are full of), plus one group = 4 convolution where a group is eight
 * output channels wide. The shapes are chosen so the first one needs three blocks and all
 * of them end up with an odd number of columns per block - see test_data/generator.py for
 * why that matters. All five are bit-exact against the golden (ULP_TOLL 0).
 *
 * The output buffer is per-tile and pre-filled with FP16 NaN, so kt_check only compares
 * what this tile actually wrote and reports the count; summing the counts over the tiles
 * is what shows the tiles together covered the whole tensor.
 */

#include "tile.h"

#include "eventunit.h"

#include "conv2dgemm_fp16_spatz.h"
#include "data.h"
#include "kernel_test_utils.h"

#define HID get_hartid()

/* One output buffer per tile, reused by both cases. The non-zero initializer keeps it in
 * .data: crt0 clears .bss from hart 0 only, while the other tiles are already running. */
static float16 Y[NUM_HARTS][OUT_LEN_MAX] = {{1.0f}};

extern uint32_t _spatz_binary_start; /* CV32 linker script */

static bool run_case(const char *name,
                     uint32_t stride,
                     uint32_t pad,
                     uint32_t groups,
                     uint32_t out_c,
                     uint32_t out_h,
                     uint32_t out_w,
                     uint32_t out_len,
                     const float16 *weights,
                     const float16 *biases,
                     const float16 *golden)
{
    uint32_t in_shape[4]  = {IN_N, IN_C, IN_H, IN_W};
    uint32_t out_shape[4] = {IN_N, out_c, out_h, out_w};
    bool ok;

    kt_mark_unwritten(Y[HID], out_len);

    MAGIA_conv2dgemm_fp16_spatz(X,
                                weights,
                                biases,
                                Y[HID],
                                in_shape,
                                out_shape,
                                KERNEL_H,
                                KERNEL_W,
                                stride,
                                stride,
                                pad,
                                pad,
                                groups,
                                HAS_BIAS);

    ok = kt_check(Y[HID], golden, out_len, ULP_TOLL);
    printf("[CV32 (%d)] %s %s\n", HID, name, ok ? "SUCCESS" : "FAILED");

    return ok;
}

int main(void)
{
    bool ok;

    if (HID == 0)
        printf("\n########## CONV2DGEMM_FP16_SPATZ KERNEL TEST "
               "(%dx%dx%dx%d, %dx%d kernel, %d cases) on %d Tiles ##########\n\n",
               IN_N,
               IN_C,
               IN_H,
               IN_W,
               KERNEL_H,
               KERNEL_W,
               NUM_CASES,
               NUM_HARTS);

    kt_spatz_init((uint32_t)&_spatz_binary_start);

#define RUN_CASE(name, tag, w, b)                                                                  \
    run_case(name,                                                                                 \
             STRIDE_##tag,                                                                         \
             PAD_##tag,                                                                            \
             GROUPS_##tag,                                                                         \
             OUT_C_##tag,                                                                          \
             OUT_H_##tag,                                                                          \
             OUT_W_##tag,                                                                          \
             OUT_LEN_##tag,                                                                        \
             w,                                                                                    \
             b,                                                                                    \
             G_##tag)

    ok  = RUN_CASE("stride 1 pad 1 (dma im2col)", 1, WEIGHTS, BIASES);
    ok &= RUN_CASE("stride 2 pad 1 (scalar im2col)", 2, WEIGHTS, BIASES);
    ok &= RUN_CASE("stride 1 pad 1 depthwise (dma im2col)", 3, WEIGHTS_DW, BIASES_DW);
    ok &= RUN_CASE("stride 2 pad 1 depthwise (scalar im2col)", 4, WEIGHTS_DW, BIASES_DW);
    ok &= RUN_CASE("stride 1 pad 1 group 4 (dma im2col)", 5, WEIGHTS_G4, BIASES_G4);

#undef RUN_CASE

    spatz_clk_dis();

    printf("[CV32 (%d)] Test %s\n", HID, ok ? "SUCCESS" : "FAILED");

    if (HID == 0)
        printf("\n#####################################################\n\n");

    return ok ? 0 : -1;
}
