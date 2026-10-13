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

extern int idma_collective_1d(
    uint32_t axi_addr, uint32_t obi_addr, uint32_t len, uint32_t mask, uint32_t coll_op);

/**
 * Identifier of a submitted asynchronous transfer (0 = none / not available).
 */
typedef uint32_t idma_ticket_t;

/**
 * Submit a 1-dimensional memory copy without waiting.
 *
 * @return The transfer ID, to be checked with idma_ticket_is_done, or 0 if the
 * hardware queue rejected the descriptor (retry) or IDs are unavailable.
 */
extern idma_ticket_t idma_submit_1d(uint8_t dir, uint32_t axi_addr, uint32_t obi_addr, uint32_t len);

/**
 * Submit a 3-dimensional memory copy without waiting.
 * Row strides (dimension 2) and page strides (dimension 3) are given
 * independently for each side; page strides are measured between page starts.
 *
 * @return The transfer ID, or 0 as for idma_submit_1d.
 */
extern idma_ticket_t idma_submit_3d(uint8_t dir,
                                    uint32_t axi_addr,
                                    uint32_t obi_addr,
                                    uint32_t row_bytes,
                                    uint32_t axi_stride_2,
                                    uint32_t obi_stride_2,
                                    uint32_t reps_2,
                                    uint32_t axi_stride_3,
                                    uint32_t obi_stride_3,
                                    uint32_t reps_3);

/**
 * Whether the transfer `ticket` submitted on direction `dir` has completed.
 */
extern uint32_t idma_ticket_is_done(uint8_t dir, idma_ticket_t ticket);

/**
 * Start a 2-dimensional memory copy with an independent row stride on each side.
 * The caller must wait for completion (e.g. via eu_idma_wait).
 */
extern int idma_memcpy_2d_ex(uint8_t dir,
                             uint32_t axi_addr,
                             uint32_t obi_addr,
                             uint32_t len,
                             uint32_t axi_std,
                             uint32_t obi_std,
                             uint32_t reps);

/**
 * Start a 3-dimensional memory copy (strides as for idma_submit_3d).
 * The caller must wait for completion (e.g. via eu_idma_wait).
 */
extern int idma_memcpy_3d(uint8_t dir,
                          uint32_t axi_addr,
                          uint32_t obi_addr,
                          uint32_t row_bytes,
                          uint32_t axi_stride_2,
                          uint32_t obi_stride_2,
                          uint32_t reps_2,
                          uint32_t axi_stride_3,
                          uint32_t obi_stride_3,
                          uint32_t reps_3);

#endif // HAL_IDMA_H
