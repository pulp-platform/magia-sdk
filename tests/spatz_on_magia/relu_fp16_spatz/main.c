// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/* Test for the FP16 relu Spatz kernel in kernels/spatz_fp16/relu. */

#include "tile.h"

#include "eventunit.h"

#include "data.h"
#include "kernel_test_utils.h"
#include "relu_fp16_spatz.h"

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
    bool ok;

    if (HID == 0)
        printf("\n########## RELU_FP16_SPATZ KERNEL TEST (%d elements) on %d Tiles ##########\n\n",
               VEC_LEN,
               NUM_HARTS);

    kt_mark_unwritten(Y[HID], OUT_LEN);
    kt_spatz_init((uint32_t)&_spatz_binary_start);

    MAGIA_relu_fp16_spatz(X, Y[HID], VEC_LEN);

    spatz_clk_dis();

    ok = kt_check(Y[HID], G, OUT_LEN, ULP_TOLL);
    printf("[CV32 (%d)] Test %s\n", HID, ok ? "SUCCESS" : "FAILED");

    if (HID == 0)
        printf("\n##########################################################################\n\n");

    return ok ? 0 : -1;
}
