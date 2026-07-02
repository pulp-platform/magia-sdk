// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// RedMulE-backed kernels extracted from the maps test.

#ifndef KERNELS_REDMULE_KERNELS_H
#define KERNELS_REDMULE_KERNELS_H

#include <stdint.h>

#include "eventunit.h"
#include "redmule.h"

/**
 * Launch an FP16 GEMM on the RedMulE accelerator and block until it completes.
 * Computes Y = X * W + Y with X:[m x n], W:[n x k], Y:[m x k], mirroring the
 * behaviour previously inlined in the maps test.
 *
 * @param redmule   RedMulE controller handle.
 * @param eu        Event-unit controller used to wait for completion.
 * @param x         L1 (OBI) address of input matrix X.
 * @param w         L1 (OBI) address of weight matrix W.
 * @param y         L1 (OBI) address of the in-place accumulate output Y.
 * @param m         Rows of X and Y.
 * @param n         Inner dimension (cols of X / rows of W).
 * @param k         Cols of W and Y.
 * @param wait_mode Event-unit wait mode (e.g. WFE).
 */
void maps_matmul_redmule(redmule_controller_t *redmule,
                         eu_controller_t *eu,
                         uint32_t x,
                         uint32_t w,
                         uint32_t y,
                         uint16_t m,
                         uint16_t n,
                         uint16_t k,
                         eu_wait_mode_t wait_mode);

#endif /* KERNELS_REDMULE_KERNELS_H */
