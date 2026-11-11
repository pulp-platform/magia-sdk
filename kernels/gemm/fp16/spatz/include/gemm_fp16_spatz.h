// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef GEMM_FP16_SPATZ_H_
#define GEMM_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_gemm_fp16_spatz(const float16 *A,
                           const float16 *B,
                           const float16 *C,
                           float16 alpha,
                           float16 beta,
                           int transA,
                           int transB,
                           uint32_t A_shape[2],
                           uint32_t B_shape[2],
                           uint32_t C_shape[2],
                           uint32_t Y_shape[2],
                           float16 *Y);
/* Same operation on this tile only, with A, B, C and Y already in this tile's L1. */
void MAGIA_gemm_fp16_spatz_l1(const float16 *A,
                              const float16 *B,
                              const float16 *C,
                              float16 alpha,
                              float16 beta,
                              int transA,
                              int transB,
                              uint32_t A_shape[2],
                              uint32_t B_shape[2],
                              uint32_t C_shape[2],
                              uint32_t Y_shape[2],
                              float16 *Y);

#endif /* GEMM_FP16_SPATZ_H_ */
