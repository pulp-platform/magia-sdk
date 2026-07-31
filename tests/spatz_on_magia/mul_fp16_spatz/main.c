// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/*
 * Test for the FP16 mul Spatz kernel in kernels/spatz_fp16/mul.
 *
 * One case per entry point: the flat elementwise multiply, and the two broadcast
 * forms ONNX Mul needs - a shared row (vfmul.vv) and a per-row scalar
 * (vfmul.vf). See test_data/generator.py for the shapes.
 */

#include "tile.h"

#include "eventunit.h"

#include "data.h"
#include "kernel_test_utils.h"
#include "mul_bcast_fp16_spatz_params.h"
#include "mul_fp16_spatz.h"

#define HID get_hartid()

/*
 * One output buffer per tile: each tile pre-fills its own with NaN and only
 * checks what it wrote. The non-zero initializer keeps the object in .data,
 * since crt0 clears .bss from hart 0 only, while the other tiles already run.
 */
static float16 Y[NUM_HARTS][OUT_LEN_MAX] = {{1.0f}};

/* Defined by the CV32 linker script; the Spatz blob is embedded by the kernel */
extern uint32_t _spatz_binary_start;

static bool report(const char *name, uint32_t out_len, const float16 *golden)
{
    bool ok = kt_check(Y[HID], golden, out_len, ULP_TOLL);

    printf("[CV32 (%d)] %s %s\n", HID, name, ok ? "SUCCESS" : "FAILED");

    return ok;
}

int main(void)
{
    bool ok;

    if (HID == 0)
        printf("\n########## MUL_FP16_SPATZ KERNEL TEST (%d cases) on %d Tiles ##########\n\n",
               NUM_CASES,
               NUM_HARTS);

    kt_spatz_init((uint32_t)&_spatz_binary_start);

    kt_mark_unwritten(Y[HID], OUT_LEN_E);
    MAGIA_mul_fp16_spatz(A, B, Y[HID], VEC_LEN);
    ok = report("elementwise", OUT_LEN_E, G);

    kt_mark_unwritten(Y[HID], OUT_LEN_B);
    MAGIA_mul_bcast_fp16_spatz(A2, B_ROW, Y[HID], ROWS, ROW_LEN, MUL_BCAST_ROW);
    ok &= report("broadcast row", OUT_LEN_B, G_ROW);

    kt_mark_unwritten(Y[HID], OUT_LEN_B);
    MAGIA_mul_bcast_fp16_spatz(A2, B_SCALAR, Y[HID], ROWS, ROW_LEN, MUL_BCAST_SCALAR);
    ok &= report("broadcast scalar", OUT_LEN_B, G_SCALAR);

    spatz_clk_dis();

    printf("[CV32 (%d)] Test %s\n", HID, ok ? "SUCCESS" : "FAILED");

    if (HID == 0)
        printf("\n##########################################################################\n\n");

    return ok ? 0 : -1;
}
