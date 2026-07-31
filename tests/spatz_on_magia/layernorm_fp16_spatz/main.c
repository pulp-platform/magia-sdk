// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/*
 * Test for the FP16 layernorm Spatz kernel in kernels/spatz_fp16/layernorm.
 *
 * Three row lengths in one run: one inside a single vector chunk, one longer than VLMAX
 * so the lane accumulator decides the result, and one long enough that the statistics
 * would leave FP16 range without the kernel's power-of-two scaling. See
 * test_data/generator.py.
 */

#include "tile.h"

#include "eventunit.h"

#include "data.h"
#include "kernel_test_utils.h"
#include "layernorm_fp16_spatz.h"

#define HID get_hartid()

/*
 * One output buffer per tile: each tile pre-fills its own with NaN and only
 * checks what it wrote. The non-zero initializer keeps the object in .data,
 * since crt0 clears .bss from hart 0 only, while the other tiles already run.
 */
static float16 Y[NUM_HARTS][OUT_LEN_MAX] = {{1.0f}};

/* Defined by the CV32 linker script; the Spatz blob is embedded by the kernel */
extern uint32_t _spatz_binary_start;

/* Not inlined: folding every case into main() puts the old GCC PULP backend into an
 * ICE during LTO register allocation. */
static bool __attribute__((noinline)) run_case(const char *tag,
                                               const float16 *X,
                                               const float16 *scale,
                                               const float16 *bias,
                                               const float16 *golden,
                                               uint32_t rows,
                                               uint32_t w_len)
{
    uint32_t in_shape[RANK] = {rows, w_len};
    bool ok;

    kt_mark_unwritten(Y[HID], rows * w_len);

    MAGIA_layernorm_fp16_spatz(X, scale, bias, (float16)EPSILON, in_shape, RANK, Y[HID]);

    ok = kt_check(Y[HID], golden, rows * w_len, ULP_TOLL);

    printf("[CV32 (%d)] case %s (%dx%d): %s\n",
           HID,
           tag,
           (int)rows,
           (int)w_len,
           ok ? "ok" : "MISMATCH");

    return ok;
}

int main(void)
{
    bool ok;

    if (HID == 0)
        printf("\n########## LAYERNORM_FP16_SPATZ KERNEL TEST (%d cases) on %d Tiles "
               "##########\n\n",
               NUM_CASES,
               NUM_HARTS);

    kt_spatz_init((uint32_t)&_spatz_binary_start);

    ok = run_case("S", X_S, SCALE_S, BIAS_S, G_S, DIM_0_S, DIM_1_S);
    ok &= run_case("L", X_L, SCALE_L, BIAS_L, G_L, DIM_0_L, DIM_1_L);
    ok &= run_case("W", X_W, SCALE_W, BIAS_W, G_W, DIM_0_W, DIM_1_W);

    spatz_clk_dis();

    printf("[CV32 (%d)] Test %s\n", HID, ok ? "SUCCESS" : "FAILED");

    if (HID == 0)
        printf("\n##########################################################################\n\n");

    return ok ? 0 : -1;
}
