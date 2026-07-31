// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/*
 * Test for the FP16 softmax Spatz kernel in kernels/spatz_fp16/softmax.
 *
 * Two cases, see test_data/generator.py: an odd row length, which puts every
 * other row on a 2-byte address and so takes the kernel's scalar fallback, and
 * an even one longer than VLMAX, which takes the vector path and folds its sum
 * through the lane accumulator.
 */

#include "tile.h"

#include "eventunit.h"

#include "data.h"
#include "kernel_test_utils.h"
#include "softmax_fp16_spatz.h"

#define HID get_hartid()

/*
 * One output buffer per tile: each tile pre-fills its own with NaN and only
 * checks what it wrote. The non-zero initializer keeps the object in .data,
 * since crt0 clears .bss from hart 0 only, while the other tiles already run.
 */
static float16 Y[NUM_HARTS][OUT_LEN_MAX] = {{1.0f}};

/* Defined by the CV32 linker script; the Spatz blob is embedded by the kernel */
extern uint32_t _spatz_binary_start;

static bool run_case(const char *name,
                     uint32_t n,
                     uint32_t c,
                     uint32_t h,
                     uint32_t w,
                     uint32_t out_len,
                     const float16 *X,
                     const float16 *golden)
{
    uint32_t in_shape[4] = {n, c, h, w};
    bool ok;

    kt_mark_unwritten(Y[HID], out_len);

    MAGIA_softmax_fp16_spatz(X, Y[HID], in_shape);

    ok = kt_check(Y[HID], golden, out_len, ULP_TOLL);
    printf("[CV32 (%d)] %s %s\n", HID, name, ok ? "SUCCESS" : "FAILED");

    return ok;
}

int main(void)
{
    bool ok;

    if (HID == 0)
        printf("\n########## SOFTMAX_FP16_SPATZ KERNEL TEST (%d cases) on %d Tiles "
               "##########\n\n",
               NUM_CASES,
               NUM_HARTS);

    kt_spatz_init((uint32_t)&_spatz_binary_start);

    ok = run_case("17 per row, odd (scalar fallback)",
                  IN_N_A,
                  IN_C_A,
                  IN_H_A,
                  IN_W_A,
                  OUT_LEN_A,
                  X_A,
                  G_A);

    ok &= run_case("300 per row, 2 vector chunks",
                   IN_N_B,
                   IN_C_B,
                   IN_H_B,
                   IN_W_B,
                   OUT_LEN_B,
                   X_B,
                   G_B);

    spatz_clk_dis();

    printf("[CV32 (%d)] Test %s\n", HID, ok ? "SUCCESS" : "FAILED");

    if (HID == 0)
        printf("\n##########################################################################\n\n");

    return ok ? 0 : -1;
}
