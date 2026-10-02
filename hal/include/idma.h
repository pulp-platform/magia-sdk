// Copyright 2024-2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Viviane Potocnik <vivianep@iis.ee.ethz.ch>
// Alberto Dequino <alberto.dequino@unibo.it>

#ifndef HAL_IDMA_H
#define HAL_IDMA_H

/**
 * Opens and initializes the IDMA interface.
 */
extern int idma_init();

/**
 * Waits for the IDMA IRQ_DONE.
 */
/* extern void idma_wait(); */

/**
 * Start 1-dimensional memory copy.
 */
extern int idma_memcpy_1d(uint8_t dir, uint32_t axi_addr, uint32_t obi_addr, uint32_t len);

/**
 * Start 2-dimensional memory copy.
 * Copies `reps` blocks of `len` bytes each. After each block, the source address
 * is advanced by `std` bytes (stride), while the destination is advanced by `len`.
 * The caller must wait for completion (e.g. via eu_idma_wait).
 *
 * @param dir   Copy direction. 0 = AXI to OBI (L2 to L1), !0 = OBI to AXI (L1 to L2).
 * @param axi_addr AXI (L2) memory address of first element.
 * @param obi_addr OBI (L1) memory address of first element.
 * @param len   Byte length of each row to transfer.
 * @param std   Stride in bytes between row starts in the strided side (source for L2->L1, dest for
 * L1->L2).
 * @param reps  Number of rows (repetitions).
 *
 * @return 0 on successful dispatch.
 */
extern int idma_memcpy_2d(
    uint8_t dir, uint32_t axi_addr, uint32_t obi_addr, uint32_t len, uint32_t std, uint32_t reps);

#endif // HAL_IDMA_H
