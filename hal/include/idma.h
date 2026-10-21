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
 * The memcpy/collective calls below launch one job on the channel of `dir` by
 * reading its NEXT_ID register, and return the job id issued by the hardware
 * (0 when built with IDMA_MM=0, which has no job ids). Each channel queues the
 * launched jobs in hardware (iDMA_JobFifoDepth = 16); with the queue full, the
 * NEXT_ID read - and so the call - stalls until a slot frees.
 */

/**
 * Start 1-dimensional memory copy.
 *
 * @return the hardware job id of the transfer.
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
 * @return the hardware job id of the transfer.
 */
extern int idma_memcpy_2d(
    uint8_t dir, uint32_t axi_addr, uint32_t obi_addr, uint32_t len, uint32_t std, uint32_t reps);

/**
 * Start 2-dimensional memory copy with independent strides on both sides.
 * Copies `reps` blocks of `len` bytes each; after each block the AXI address
 * advances by `axi_std` and the OBI address by `obi_std` bytes. A zero stride
 * re-reads (or re-writes) the same block, e.g. obi_std = 0 with dir = 1 pushes
 * one L1 block to `reps` equally spaced AXI destinations.
 */
extern int idma_memcpy_2d_strided(uint8_t dir,
                                  uint32_t axi_addr,
                                  uint32_t obi_addr,
                                  uint32_t len,
                                  uint32_t axi_std,
                                  uint32_t obi_std,
                                  uint32_t reps);

extern int idma_collective_1d(
    uint32_t axi_addr, uint32_t obi_addr, uint32_t len, uint32_t mask, uint32_t coll_op);

/**
 * Reads the DONE_ID register of the channel of `dir`: the id of the last job
 * the channel has completed. Jobs on one channel complete in issue order, so
 * job `id` is done once DONE_ID has reached it.
 */
extern uint32_t idma_done_id(uint8_t dir);

#endif // HAL_IDMA_H
