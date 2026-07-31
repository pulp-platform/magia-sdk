// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/*
 * Test for the FP16 matmul Spatz kernel in kernels/spatz_fp16/matmul.
 *
 * Several shapes in one run: the vector path, the odd-O shape that has to take the
 * alignment guard's scalar fallback, a shared (non-batched) B, and a single batch.
 * See test_data/generator.py for what each case is there to cover.
 */

#include "tile.h"

#include "eventunit.h"

#include "data.h"
#include "kernel_test_utils.h"
#include "matmul_fp16_spatz.h"

#define HID get_hartid()

/*
 * One output buffer per tile: each tile pre-fills its own with NaN and only
 * checks what it wrote. The non-zero initializer keeps the object in .data,
 * since crt0 clears .bss from hart 0 only, while the other tiles already run.
 */
static float16 Y[NUM_HARTS][MAX_LEN] = {{1.0f}};

/* Defined by the CV32 linker script; the Spatz blob is embedded by the kernel */
extern uint32_t _spatz_binary_start;

/* Not inlined: folding every case into main() puts the old GCC PULP backend into an
 * ICE during LTO register allocation. */
static bool __attribute__((noinline)) run_case(uint32_t c)
{
    kt_mark_unwritten(Y[HID], CASE_Y_LEN[c]);

    MAGIA_matmul_fp16_spatz(&A[CASE_A_OFF[c]],
                            &B[CASE_B_OFF[c]],
                            Y[HID],
                            CASE_M[c],
                            CASE_K[c],
                            CASE_O[c],
                            CASE_BATCHES[c],
                            CASE_A_BATCHED[c],
                            CASE_B_BATCHED[c]);

    return kt_check(Y[HID], &G[CASE_Y_OFF[c]], CASE_Y_LEN[c], ULP_TOLL);
}

int main(void)
{
    bool all_ok = true;

    if (HID == 0)
        printf("\n########## MATMUL_FP16_SPATZ KERNEL TEST (%d cases) on %d Tiles ##########\n\n",
               NUM_CASES,
               NUM_HARTS);

    kt_spatz_init((uint32_t)&_spatz_binary_start);

    for (uint32_t c = 0; c < NUM_CASES; c++) {
        bool ok = run_case(c);

        all_ok = all_ok && ok;

        printf("[CV32 (%d)] case %d: %s\n", HID, (int)c, ok ? "ok" : "MISMATCH");
    }

    spatz_clk_dis();

    printf("[CV32 (%d)] Test %s\n", HID, all_ok ? "SUCCESS" : "FAILED");

    if (HID == 0)
        printf("\n##########################################################################\n\n");

    return all_ok ? 0 : -1;
}
