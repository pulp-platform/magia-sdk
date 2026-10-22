// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>
//
// L1 kernels called (synchronously, on the CV32 control core) by the
// mgc-generated test.c. All pointer arguments are L1 addresses.

#include <stdint.h>

#include "test.h"
#include "tile.h"
#include "add_fp16_spatz.h"
#include "exp_fp16_spatz.h"
#include "expminus_fp16_spatz.h"
#include "max_fp16_spatz.h"
#include "memset_fp16_spatz.h"
#include "mul_broadcast_fp16_spatz.h"
#include "mul_fp16_spatz.h"
#include "reducemax_fp16_spatz.h"
#include "reducesum_fp16_spatz.h"
#include "rowdiv_fp16_spatz.h"
#include "rowscale_fp16_spatz.h"
#include "sub_fp16_spatz.h"
#include "eventunit.h"
#include "flat_attn_naive_spatz_task_bin.h"

/* Per-tile flag: Spatz is set up on each tile before its first task. */
static volatile uint8_t spatz_ready __attribute__((section(".tile_bss")));

/**
 * Lazy Spatz setup on this tile (event unit line, binary and clock): the mgc-generated main
 * does not know about Spatz.
 */
static void spatz_setup(void)
{
    if (spatz_ready)
        return;
    eu_spatz_init(0);
    spatz_init(SPATZ_BINARY_START);
    spatz_ready = 1;
}

/**
 * y[k] = 0 for `n` fp16 elements, on this tile's Spatz.
 */
void l1_memset_zero_fp16_spatz(uint32_t y, uint32_t n)
{
    spatz_setup();
    MAGIA_memset_zero_fp16_spatz_l1((float16 *)y, n);
}

/**
 * y[k] = x[k] for `n` fp16 elements, on the CV32 core (32-bit loads/stores,
 * plus a 16-bit one for an odd tail).
 */
void l1_copy_fp16(uint32_t y, uint32_t x, uint32_t n)
{
    volatile uint32_t *y32 = (volatile uint32_t *)y;
    volatile uint32_t *x32 = (volatile uint32_t *)x;
    for (uint32_t k = 0; k < n / 2; k++)
        y32[k] = x32[k];
    if (n & 1)
        ((volatile uint16_t *)y)[n - 1] = ((volatile uint16_t *)x)[n - 1];
}

/**
 * y[k] = x[k] * 1/sqrt(PROJ_SIZE) for `n` fp16 elements, on this tile's Spatz.
 * The scalar lives on the CV32 stack (in L1), like the Spatz task params.
 */
void l1_scale_rsqrt_p_fp16_spatz(uint32_t y, uint32_t x, uint32_t n)
{
    spatz_setup();
    volatile float16 r = (float16)(1.0f / __builtin_sqrtf((float)PROJ_SIZE));
    MAGIA_mul_broadcast_fp16_spatz_l1((const float16 *)x, (const float16 *)&r, (float16 *)y, n);
}

/**
 * m[r] = max_c x[r, c] for a `rows` x `cols` fp16 matrix, on this tile's Spatz.
 */
void l1_rowmax_fp16_spatz(uint32_t m, uint32_t x, uint32_t rows, uint32_t cols)
{
    spatz_setup();
    MAGIA_reducemax_fp16_spatz_l1((const float16 *)x, (float16 *)m, rows, cols, 1);
}

/**
 * y[k] = max(a[k], b[k]) for `n` fp16 elements, on this tile's Spatz.
 */
void l1_max_fp16_spatz(uint32_t y, uint32_t a, uint32_t b, uint32_t n)
{
    spatz_setup();
    MAGIA_max_fp16_spatz_l1((const float16 *)a, (const float16 *)b, (float16 *)y, n);
}

/**
 * m[c] = max_r x[r, c] for a `rows` x `cols` fp16 matrix, on this tile's Spatz
 * (vectorized along the columns).
 */
void l1_colmax_fp16_spatz(uint32_t m, uint32_t x, uint32_t rows, uint32_t cols)
{
    spatz_setup();
    MAGIA_reducemax_fp16_spatz_l1((const float16 *)x, (float16 *)m, 1, rows, cols);
}

/**
 * s[r] = sum_c x[r, c] for a `rows` x `cols` fp16 matrix, on this tile's Spatz.
 */
void l1_rowsum_fp16_spatz(uint32_t s, uint32_t x, uint32_t rows, uint32_t cols)
{
    spatz_setup();
    MAGIA_reducesum_fp16_spatz_l1((const float16 *)x, (float16 *)s, rows, cols, 1);
}

/**
 * s[c] = sum_r x[r, c] for a `rows` x `cols` fp16 matrix, on this tile's Spatz
 * (vectorized along the columns).
 */
void l1_colsum_fp16_spatz(uint32_t s, uint32_t x, uint32_t rows, uint32_t cols)
{
    spatz_setup();
    MAGIA_reducesum_fp16_spatz_l1((const float16 *)x, (float16 *)s, 1, rows, cols);
}

/**
 * y[r, c] = exp(x[r, c] - m[r]) for a `rows` x `cols` fp16 matrix, on this tile's Spatz.
 */
void l1_expminus_fp16_spatz(uint32_t y, uint32_t x, uint32_t m, uint32_t rows, uint32_t cols)
{
    spatz_setup();
    MAGIA_expminus_fp16_spatz_l1((const float16 *)x, (const float16 *)m, (float16 *)y, rows, cols);
}

/**
 * y[k] = a[k] - b[k] for `n` fp16 elements, on this tile's Spatz.
 */
void l1_sub_fp16_spatz(uint32_t y, uint32_t a, uint32_t b, uint32_t n)
{
    spatz_setup();
    MAGIA_sub_fp16_spatz_l1((const float16 *)a, (const float16 *)b, (float16 *)y, n);
}

/**
 * y[k] = exp(x[k]) for `n` fp16 elements, on this tile's Spatz.
 */
void l1_exp_fp16_spatz(uint32_t y, uint32_t x, uint32_t n)
{
    spatz_setup();
    MAGIA_exp_fp16_spatz_l1((const float16 *)x, (float16 *)y, n);
}

/**
 * y[k] = a[k] * b[k] for `n` fp16 elements, on this tile's Spatz.
 */
void l1_mul_fp16_spatz(uint32_t y, uint32_t a, uint32_t b, uint32_t n)
{
    spatz_setup();
    MAGIA_mul_fp16_spatz_l1((const float16 *)a, (const float16 *)b, (float16 *)y, n);
}

/**
 * y[k] = a[k] + b[k] for `n` fp16 elements, on this tile's Spatz.
 */
void l1_add_fp16_spatz(uint32_t y, uint32_t a, uint32_t b, uint32_t n)
{
    spatz_setup();
    MAGIA_add_fp16_spatz_l1((const float16 *)a, (const float16 *)b, (float16 *)y, n);
}

/**
 * y[r, c] = x[r, c] * s[r] for a `rows` x `cols` fp16 matrix, i.e. y = diag(s) x, on this
 * tile's Spatz.
 */
void l1_rowscale_fp16_spatz(uint32_t y, uint32_t x, uint32_t s, uint32_t rows, uint32_t cols)
{
    spatz_setup();
    MAGIA_rowscale_fp16_spatz_l1((const float16 *)x, (const float16 *)s, (float16 *)y, rows, cols);
}

/**
 * y[r, c] = x[r, c] / s[r] for a `rows` x `cols` fp16 matrix, i.e. y = diag(s)^-1 x, on this
 * tile's Spatz.
 */
void l1_rowdiv_fp16_spatz(uint32_t y, uint32_t x, uint32_t s, uint32_t rows, uint32_t cols)
{
    spatz_setup();
    MAGIA_rowdiv_fp16_spatz_l1((const float16 *)x, (const float16 *)s, (float16 *)y, rows, cols);
}
