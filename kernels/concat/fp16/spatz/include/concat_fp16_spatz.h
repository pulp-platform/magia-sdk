// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef CONCAT_FP16_SPATZ_H_
#define CONCAT_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_concat_fp16_spatz(const float16 **inputs,
                             const uint32_t *lens,
                             uint32_t num_inputs,
                             float16 *concat_result,
                             uint32_t iterations);
/* Same operation on this tile only, with the inputs and concat_result already in this tile's L1. */
void MAGIA_concat_fp16_spatz_l1(const float16 **inputs,
                                const uint32_t *lens,
                                uint32_t num_inputs,
                                float16 *concat_result,
                                uint32_t iterations);

#endif /* CONCAT_FP16_SPATZ_H_ */
