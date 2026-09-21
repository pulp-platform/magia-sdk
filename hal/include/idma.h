// Copyright 2024-2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Viviane Potocnik <vivianep@iis.ee.ethz.ch>
// Alberto Dequino <alberto.dequino@unibo.it>

#ifndef HAL_IDMA_H
#define HAL_IDMA_H

/** Forward declaration of the idma controller instance and API structure. */
typedef struct idma_controller idma_controller_t;
typedef struct idma_controller_api idma_controller_api_t;

/**
 * WIP
 * Holds the API function pointers, base address and controller-specific configuration.
 */
struct idma_controller {
    idma_controller_api_t *api; /**< Function pointers for this interface. */
    uint32_t base;              /**< MMIO base address (if applicable). */
    void *cfg;                  /**< Driver‑specific configuration. */
};

/**
 * IDMA configuration structure.
 * This structure holds the configuration settings for IDMA initialization.
 */
typedef struct {
    uint32_t hartid; /**< Mesh tile ID*/
} idma_config_t;

/**
 * Opens and initializes the IDMA interface.
 */
extern int idma_init(idma_controller_t *ctrl);

/**
 * Waits for the IDMA IRQ_DONE.
 */
/* extern void idma_wait(); */

/**
 * Start 1-dimensional memory copy.
 */
extern int idma_memcpy_1d(
    idma_controller_t *ctrl, uint8_t dir, uint32_t axi_addr, uint32_t obi_addr, uint32_t len);

/**
 * Start 2-dimensional memory copy.
 * Copies `reps` blocks of `len` bytes each. After each block, the source address
 * is advanced by `std` bytes (stride), while the destination is advanced by `len`.
 * The caller must wait for completion (e.g. via eu_idma_wait).
 *
 * @param ctrl  IDMA controller handle.
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
extern int idma_memcpy_2d(idma_controller_t *ctrl,
                          uint8_t dir,
                          uint32_t axi_addr,
                          uint32_t obi_addr,
                          uint32_t len,
                          uint32_t std,
                          uint32_t reps);

/**
 * Start 2-dimensional memory copy with independent strides on both sides.
 *
 * Same as idma_memcpy_2d except that the contiguous side is not forced to advance by
 * `len`: both the AXI (L2) and the OBI (L1) address advance by their own stride after
 * each row. That is what lets a caller drop a narrow rectangle into the middle of a
 * wider L1 buffer - idma_memcpy_2d is the special case obi_std == len.
 *
 * @param ctrl     IDMA controller handle.
 * @param dir      Copy direction. 0 = AXI to OBI (L2 to L1), !0 = OBI to AXI (L1 to L2).
 * @param axi_addr AXI (L2) address of the first element.
 * @param obi_addr OBI (L1) address of the first element.
 * @param len      Byte length of each row.
 * @param axi_std  Byte stride between row starts on the AXI (L2) side.
 * @param obi_std  Byte stride between row starts on the OBI (L1) side.
 * @param reps     Number of rows.
 *
 * @return 0 on successful dispatch.
 */
extern int idma_memcpy_2d_ex(idma_controller_t *ctrl,
                             uint8_t dir,
                             uint32_t axi_addr,
                             uint32_t obi_addr,
                             uint32_t len,
                             uint32_t axi_std,
                             uint32_t obi_std,
                             uint32_t reps);

/**
 * Start a 3-dimensional memory copy with independent strides on both sides.
 *
 * Dimension 1 is a contiguous row of `row_bytes`. Dimension 2 repeats the row
 * `reps_2` times, advancing by the corresponding dimension-2 stride. Dimension
 * 3 repeats the resulting page `reps_3` times. Dimension-3 strides are measured
 * between the starts of consecutive pages.
 *
 * @param ctrl         IDMA controller handle.
 * @param dir          Copy direction. 0 = AXI to OBI, !0 = OBI to AXI.
 * @param axi_addr     AXI address of the first element.
 * @param obi_addr     OBI address of the first element.
 * @param row_bytes    Number of contiguous bytes per row.
 * @param axi_stride_2 AXI byte stride between rows.
 * @param obi_stride_2 OBI byte stride between rows.
 * @param reps_2       Number of rows per page.
 * @param axi_stride_3 AXI byte stride between page starts.
 * @param obi_stride_3 OBI byte stride between page starts.
 * @param reps_3       Number of pages.
 *
 * @return 0 on successful dispatch.
 */
extern int idma_memcpy_3d(idma_controller_t *ctrl,
                          uint8_t dir,
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
 * WIP
 * IDMA API
 */
struct idma_controller_api {
    int (*init)(idma_controller_t *ctrl);
    /*     void (*wait)(); */
    int (*memcpy_1d)(idma_controller_t *ctrl,
                     uint8_t dir,
                     uint32_t axi_addr,
                     uint32_t obi_addr,
                     volatile uint32_t len);

    int (*memcpy_2d)(idma_controller_t *ctrl,
                     uint8_t dir,
                     uint32_t axi_addr,
                     uint32_t obi_addr,
                     uint32_t len,
                     uint32_t std,
                     uint32_t reps);

    int (*memcpy_2d_ex)(idma_controller_t *ctrl,
                        uint8_t dir,
                        uint32_t axi_addr,
                        uint32_t obi_addr,
                        uint32_t len,
                        uint32_t axi_std,
                        uint32_t obi_std,
                        uint32_t reps);

    int (*memcpy_3d)(idma_controller_t *ctrl,
                     uint8_t dir,
                     uint32_t axi_addr,
                     uint32_t obi_addr,
                     uint32_t row_bytes,
                     uint32_t axi_stride_2,
                     uint32_t obi_stride_2,
                     uint32_t reps_2,
                     uint32_t axi_stride_3,
                     uint32_t obi_stride_3,
                     uint32_t reps_3);
};

/*
 * Generic implementation of the IDMA controller.
 */
extern idma_controller_api_t idma_api;

#endif // HAL_IDMA_H
