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
 * A single dimension of a tensor sub-slice: a strided range of elements.
 *
 * The address of index i along this dimension is `base + (start + i) * stride`,
 * for i in [0, length). `stride` is a byte gap; `start` is in element/index units.
 */
typedef struct {
    uint32_t start;  /**< Starting index in this dimension (element units). */
    uint32_t length; /**< Number of elements along this dimension. */
    uint32_t stride; /**< Byte gap between consecutive indices in this dimension. */
} TensorRange;

/**
 * Descriptor for one side (source or destination) of a multi-dimensional copy.
 *
 * Dimensions are stored outermost-first (row-major): dims[0] varies slowest,
 * dims[rank-1] varies fastest. `num_elems` must equal the product of dims[].length.
 * The two sides of a copy may have different rank/shape as long as their
 * `num_elems` match; element k in row-major order maps to element k on the other
 * side (see idma_memcpy_md_to_nd).
 */
typedef struct {
    uint32_t rank;                      /**< Number of active dims. */
    uint32_t num_elems;                 /**< Product of dims[].length. */
    TensorRange dims[IDMA_ND_MAX_RANK]; /**< Per-dimension ranges, outermost first. */
} tensor_sub_slice_t;

/**
 * Start a multi-dimensional memory copy between two independently-shaped tensor
 * sub-slices, dispatched at runtime to the widest native IDMA burst that fits
 * (2D or 1D). Source and destination may differ in rank and shape as long as they
 * hold the same number of elements; element k of `src` (row-major) is copied to
 * element k of `dst`.
 *
 * The AXI endpoint (given by dst_addr/src_addr on the non-OBI side) may be L2 or
 * another tile's L1 over the NoC, so this covers both L2<->L1 and tile-to-tile
 * L1->L1 transfers; only the OBI port (this tile's own L1) can never be strided.
 *
 * After folding per-dim starts into the base addresses and coalescing contiguous
 * dimensions, the copy is issued as native 2D bursts whenever the local (OBI)
 * side is contiguous (the fast, preferred path), collapsing to a single 1D burst
 * when both sides are fully contiguous. When the OBI side is not contiguous it
 * falls back to a software loop of 1D bursts over the largest common contiguous
 * run. Since the underlying controller supports only one in-flight transfer, each
 * burst is waited on via eu_ctrl (if non-NULL, using IDMA_ND_WAIT_MODE) before the
 * next is issued.
 *
 * @param ctrl       IDMA controller handle.
 * @param dir        Copy direction. 0 = AXI to OBI (remote/L2 -> local L1),
 *                   !0 = OBI to AXI (local L1 -> remote/L2).
 * @param dst_addr   Destination address of first element.
 * @param src_addr   Source address of first element.
 * @param src        Source sub-slice descriptor (rank, num_elems, per-dim ranges).
 * @param dst        Destination sub-slice descriptor.
 * @param elem_bytes Size of a single element in bytes (shared by both sides).
 * @param eu_ctrl    Event-unit controller used to wait for each burst; pass NULL to skip waiting.
 *
 * @return 0 on success, -1 if either rank exceeds IDMA_ND_MAX_RANK or the two
 *         sides' num_elems differ.
 */
extern int idma_memcpy_md_to_nd(idma_controller_t *ctrl,
                                uint8_t dir,
                                uint32_t dst_addr,
                                uint32_t src_addr,
                                const tensor_sub_slice_t *src,
                                const tensor_sub_slice_t *dst,
                                uint32_t elem_bytes,
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
                           const tensor_sub_slice_t *src,
                           const tensor_sub_slice_t *dst,
                           uint32_t elem_bytes,
                           eu_controller_t *eu_ctrl);
};

/*
 * Generic implementation of the IDMA controller.
 */
extern idma_controller_api_t idma_api;

#endif // HAL_IDMA_H
