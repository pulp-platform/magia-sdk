// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef GROUPNORM_FP16_SPATZ_H_
#define GROUPNORM_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_groupnorm_fp16_spatz(const float16 *X,
                                const float16 *Y,
                                const float16 *scale,
                                const float16 *B,
                                uint32_t input_shape[4],
                                uint32_t num_groups,
                                float16 epsilon);
/* Same operation on this tile only, with X, Y, scale and B already in this tile's L1. */
void MAGIA_groupnorm_fp16_spatz_l1(const float16 *X,
                                   float16 *Y,
                                   const float16 *scale,
                                   const float16 *B,
                                   uint32_t input_shape[4],
                                   uint32_t num_groups,
                                   float16 epsilon);

#endif /* GROUPNORM_FP16_SPATZ_H_ */
