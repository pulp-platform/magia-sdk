// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef MATMUL_FP16_SPATZ_H_
#define MATMUL_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_matmul_fp16_spatz(const float16 *A,
                             const float16 *B,
                             float16 *Y,
                             uint32_t M,
                             uint32_t K,
                             uint32_t O,
                             uint32_t total_batches,
                             uint32_t a_batched,
                             uint32_t b_batched);
/* Same operation on this tile only, with A, B, Y already in this tile's L1. */
void MAGIA_matmul_fp16_spatz_l1(const float16 *A,
                                const float16 *B,
                                float16 *Y,
                                uint32_t M,
                                uint32_t K,
                                uint32_t O,
                                uint32_t total_batches,
                                uint32_t a_batched,
                                uint32_t b_batched);

#endif /* MATMUL_FP16_SPATZ_H_ */
