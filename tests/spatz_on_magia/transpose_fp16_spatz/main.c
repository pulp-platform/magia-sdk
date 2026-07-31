// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/*
 * Test for the FP16 transpose Spatz kernel in kernels/spatz_fp16/transpose.
 *
 * Several shapes in one run, because which axis the kernel shards - and therefore which
 * of its two transfers is the strided one - depends on the perm and on the two leading
 * extents. See test_data/generator.py for what each case is there to cover.
 */

#include "tile.h"

#include "eventunit.h"

#include "data.h"
#include "kernel_test_utils.h"
#include "transpose_fp16_spatz.h"

#define HID get_hartid()

/*
 * One output buffer per tile: each tile pre-fills its own with NaN and only
 * checks what it wrote. The non-zero initializer keeps the object in .data,
 * since crt0 clears .bss from hart 0 only, while the other tiles already run.
 */
static float16 Y[NUM_HARTS][MAX_LEN] = {{1.0f}};

/* Defined by the CV32 linker script; the Spatz blob is embedded by the kernel */
extern uint32_t _spatz_binary_start;

/* Not inlined: folding six kernel calls plus their checks into main() puts the old GCC
 * PULP backend into an ICE during LTO register allocation. */
static bool __attribute__((noinline)) run_case(uint32_t c)
{
    uint32_t rank = CASE_RANK[c];
    uint32_t in_shape[MAX_RANK];
    uint32_t out_shape[MAX_RANK];
    uint32_t perm[MAX_RANK];

    for (uint32_t i = 0; i < rank; i++) {
        in_shape[i]  = CASE_IN_SHAPE[c][i];
        out_shape[i] = CASE_OUT_SHAPE[c][i];
        perm[i]      = CASE_PERM[c][i];
    }

    kt_mark_unwritten(Y[HID], CASE_LEN[c]);

    MAGIA_transpose_fp16_spatz(&X[CASE_OFF[c]], Y[HID], perm, in_shape, out_shape, rank,
                               in_shape[0]);

    return kt_check(Y[HID], &G[CASE_OFF[c]], CASE_LEN[c], ULP_TOLL);
}

int main(void)
{
    bool all_ok = true;

    if (HID == 0)
        printf("\n########## TRANSPOSE_FP16_SPATZ KERNEL TEST (%d cases) on %d Tiles "
               "##########\n\n",
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
