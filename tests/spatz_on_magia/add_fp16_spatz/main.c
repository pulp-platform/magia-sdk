// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/*
 * Test for the FP16 add Spatz kernel in kernels/spatz_fp16/add.
 *
 * One case per entry point: the flat elementwise add, and the two broadcast forms ONNX
 * Add needs - a shared row (vfadd.vv, which is what a MatMul bias and a shared attention
 * bias both are) and a per-row scalar (vfadd.vf). See test_data/generator.py for the
 * shapes.
 *
 * Every tile runs the very same call and only computes - and checks - its own shard, so
 * each tile gets a private output buffer pre-filled with NaN and compares whatever is no
 * longer NaN. The kernel does not bring up the accelerator itself: the event unit and
 * Spatz are initialized here, before calling it.
 */

#include "tile.h"

#include "eventunit.h"

#include "add_bcast_fp16_spatz_params.h"
#include "add_fp16_spatz.h"
#include "data.h"
#include "kernel_test_utils.h"

#define HID get_hartid()

/*
 * One output buffer per tile. The non-zero initializer keeps the object in .data,
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
        printf("\n########## ADD_FP16_SPATZ KERNEL TEST (%d cases) on %d Tiles ##########\n\n",
               NUM_CASES,
               NUM_HARTS);

    kt_spatz_init((uint32_t)&_spatz_binary_start);

    kt_mark_unwritten(Y[HID], OUT_LEN_E);
    MAGIA_add_fp16_spatz(A, B, Y[HID], VEC_LEN);
    ok = report("elementwise", OUT_LEN_E, G);

    kt_mark_unwritten(Y[HID], OUT_LEN_B);
    MAGIA_add_bcast_fp16_spatz(A2, B_ROW, Y[HID], ROWS, ROW_LEN, ADD_BCAST_ROW);
    ok &= report("broadcast row", OUT_LEN_B, G_ROW);

    kt_mark_unwritten(Y[HID], OUT_LEN_B);
    MAGIA_add_bcast_fp16_spatz(A2, B_SCALAR, Y[HID], ROWS, ROW_LEN, ADD_BCAST_SCALAR);
    ok &= report("broadcast scalar", OUT_LEN_B, G_SCALAR);

    spatz_clk_dis();

    printf("[CV32 (%d)] Test %s\n", HID, ok ? "SUCCESS" : "FAILED");

    if (HID == 0)
        printf("\n##########################################################################\n\n");

    return ok ? 0 : -1;
}
