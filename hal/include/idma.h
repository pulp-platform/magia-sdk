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
    uint32_t hartid;    /**< Mesh tile ID*/
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
extern int idma_memcpy_1d(idma_controller_t *ctrl, uint8_t dir, uint32_t axi_addr, uint32_t obi_addr, uint32_t len);

/**
 * Start 2-dimensional memory copy.
 */
extern int idma_memcpy_2d(idma_controller_t *ctrl, uint8_t dir, uint32_t axi_addr, uint32_t obi_addr, uint32_t len, uint32_t std, uint32_t reps);

/**
 * WIP
 * IDMA API
 */
struct idma_controller_api {
    int (*init)(idma_controller_t *ctrl);
/*     void (*wait)(); */
    int (*memcpy_1d)(idma_controller_t *ctrl, uint8_t dir, uint32_t axi_addr, uint32_t obi_addr, volatile uint32_t len);
    int (*memcpy_2d)(idma_controller_t *ctrl, uint8_t dir, uint32_t axi_addr, uint32_t obi_addr, uint32_t len, uint32_t std, uint32_t reps);
};

/*
 * Generic implementation of the IDMA controller.
 */
extern idma_controller_api_t idma_api;

#endif //HAL_IDMA_H