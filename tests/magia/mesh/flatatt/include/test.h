// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Alberto Dequino <alberto.dequino@unibo.it>
// Victor Isachi <victor.isachi@unibo.it>

#ifndef _TEST_FLATATT_INCLUDE_GUARD_
#define _TEST_FLATATT_INCLUDE_GUARD_

#define S_SIZE (512)
#define D_SIZE (128)
#define D_SIZE_SQRT (11.3137)

extern uint16_t q_inp [S_SIZE * D_SIZE] = {};
extern uint16_t k_inp [D_SIZE * S_SIZE] = {};
extern uint16_t v_inp [S_SIZE * D_SIZE] = {};

int max_compare(uint32_t prev, uint32_t curr, uint32_t dim);
int rowmax(uint32_t s, uint32_t maxes, uint32_t dim_h, uint32_t dim_w);
int rowdiff(uint32_t s, uint32_t m, uint32_t h, uint32_t w);
int rowsum(uint32_t s, uint32_t l, uint32_t h, uint32_t w);
int vect_sum(uint32_t v1, uint32_t v2, uint32_t dim);
#endif //_TEST_FLATATT_INCLUDE_GUARD_

