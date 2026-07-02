// Copyright 2024-2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Viviane Potocnik <vivianep@iis.ee.ethz.ch>
// Alberto Dequino <alberto.dequino@unibo.it>

#ifndef HAL_IDMA_H
#define HAL_IDMA_H

#include "eventunit.h"

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
 * Start 3-dimensional memory copy.
 * Copies `reps3` planes, each plane being `reps2` rows of `len` bytes. On the strided
 * (L2/AXI) side, rows advance by `std2` bytes and planes advance by `std3` bytes. On the
 * contiguous (L1/OBI) side, rows advance by `len` and planes by `len*reps2`.
 * The caller must wait for completion (e.g. via eu_idma_wait).
 *
 * @param ctrl  IDMA controller handle.
 * @param dir   Copy direction. 0 = AXI to OBI (L2 to L1), !0 = OBI to AXI (L1 to L2).
 * @param axi_addr AXI (L2) memory address of first element.
 * @param obi_addr OBI (L1) memory address of first element.
 * @param len   Byte length of each row to transfer.
 * @param std2  Stride in bytes between row starts on the strided side (2nd dimension).
 * @param reps2 Number of rows per plane (repetitions of the 2nd dimension).
 * @param std3  Stride in bytes between plane starts on the strided side (3rd dimension).
 * @param reps3 Number of planes (repetitions of the 3rd dimension).
 *
 * @return 0 on successful dispatch.
 */
extern int idma_memcpy_3d(idma_controller_t *ctrl,
                          uint8_t dir,
                          uint32_t axi_addr,
                          uint32_t obi_addr,
                          uint32_t len,
                          uint32_t std2,
                          uint32_t reps2,
                          uint32_t std3,
                          uint32_t reps3);

#define IDMA_ND_MAX_RANK 4u

#ifndef IDMA_ND_WAIT_MODE
#define IDMA_ND_WAIT_MODE WFE
#endif

/**
 * Descriptor for a multi-dimensional memory copy (see idma_memcpy_md_to_nd).
 */
typedef struct {
    uint32_t rank;
    uint32_t shape[IDMA_ND_MAX_RANK];
    uint32_t elem_bytes;
    uint32_t src_strides_bytes[IDMA_ND_MAX_RANK];
    uint32_t dst_strides_bytes[IDMA_ND_MAX_RANK];
} copy_desc_t;

/**
 * Start a multi-dimensional memory copy, dispatched at runtime to the narrowest
 * native IDMA burst that fits the transfer (1D, 2D, or 3D), based on copy's
 * shape/strides after coalescing contiguous innermost dimensions. Shapes without
 * a native path yet fall back to a software loop of 1D bursts. Since the
 * underlying controller supports only one in-flight transfer, each burst is waited
 * on via eu_ctrl (if non-NULL, using IDMA_ND_WAIT_MODE) before the next is issued.
 *
 * @param ctrl     IDMA controller handle.
 * @param dir      Copy direction. 0 = AXI to OBI (L2 to L1), !0 = OBI to AXI (L1 to L2).
 * @param dst_addr Destination address of first element.
 * @param src_addr Source address of first element.
 * @param copy     Copy descriptor (rank, shape, element size, per-side strides).
 * @param eu_ctrl  Event-unit controller used to wait for each burst; pass NULL to skip waiting.
 *
 * @return 0 on success, -1 if copy->rank exceeds IDMA_ND_MAX_RANK.
 */
extern int idma_memcpy_md_to_nd(idma_controller_t *ctrl,
                                uint8_t dir,
                                uint32_t dst_addr,
                                uint32_t src_addr,
                                const copy_desc_t *copy,
                                eu_controller_t *eu_ctrl);

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

    int (*memcpy_3d)(idma_controller_t *ctrl,
                     uint8_t dir,
                     uint32_t axi_addr,
                     uint32_t obi_addr,
                     uint32_t len,
                     uint32_t std2,
                     uint32_t reps2,
                     uint32_t std3,
                     uint32_t reps3);

    int (*memcpy_md_to_nd)(idma_controller_t *ctrl,
                           uint8_t dir,
                           uint32_t dst_addr,
                           uint32_t src_addr,
                           const copy_desc_t *copy,
                           eu_controller_t *eu_ctrl);
};

/*
 * Generic implementation of the IDMA controller.
 */
extern idma_controller_api_t idma_api;

#endif // HAL_IDMA_H
