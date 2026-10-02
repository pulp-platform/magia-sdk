// Copyright 2025-2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Alberto Dequino <alberto.dequino@unibo.it>

#include <stdint.h>
#include "test.h"
#include "tile.h"
#include "idma.h"
#include "eventunit.h"

#define N_BUF     1              // Number of KBs to be transfered
#define BUF_SIZE  (N_BUF * 1024) // Buffer size

#define WAIT_MODE WFE

/**
 * This test allocates two buffers of a certain dimension,
 * adds a block of data to both of them,
 * then checks the values are the same.
 */
int main(void)
{
    uint32_t hartid = get_hartid();

    idma_init();

#if STALLING == 0
    eu_init();
    eu_idma_init(0);
#endif

    l1_alloc_init();
    uint32_t pointer_A = l1_alloc(BUF_SIZE);
    uint32_t pointer_B = l1_alloc(BUF_SIZE);

    // printf("[HARTID %d] Pointer A is 0X%x Pointer B is 0X%x\n", hartid, pointer_A, pointer_B);

    // IDMA Transfer
    idma_memcpy_1d(0, (uint32_t)l2_vect, pointer_A, (uint32_t)BUF_SIZE);
#if STALLING == 0
    eu_idma_wait_a2o(WAIT_MODE);
#endif
    idma_memcpy_1d(0, (uint32_t)l2_vect, pointer_B, (uint32_t)BUF_SIZE);
#if STALLING == 0
    eu_idma_wait_a2o(WAIT_MODE);
#endif

    uint32_t n_errors = 0;
    for (uint32_t i = 0; i < (BUF_SIZE); i++) {
        if (*(uint8_t *)(pointer_A + i) != *(uint8_t *)(pointer_B + i)) {
#if EVAL == 1
            printf("[HARTID %d]ERROR: pointer_A[%d] = %b, pointer_B[%d] = %b\n",
                   hartid,
                   i,
                   *(uint8_t *)(pointer_A + i),
                   i,
                   *(uint8_t *)(pointer_B + i));
#endif
            n_errors++;
        }
    }

    printf("[HARTID %d] N. Errors: %d\n", hartid, n_errors);
    return n_errors;
}