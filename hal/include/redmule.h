// Copyright 2024-2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Viviane Potocnik <vivianep@iis.ee.ethz.ch>
// Alberto Dequino <alberto.dequino@unibo.it>

#ifndef HAL_REDMULE_H
#define HAL_REDMULE_H

/** Forward declaration of the Redmule controller instance and API structure. */
typedef struct redmule_controller redmule_controller_t;
typedef struct redmule_controller_api redmule_controller_api_t;

/**
 * WIP
 * Holds the API function pointers, base address and controller-specific configuration.
 */
struct redmule_controller {
    redmule_controller_api_t *api;  /**< Function pointers for this interface. */
    uint32_t base;                  /**< MMIO base address (if applicable). */
    void *cfg;                      /**< Driver‑specific configuration. */
};

/**
 * Redmule configuration structure.
 *
 * This structure holds the configuration settings for Redmule initialization.
 */
typedef struct {
    uint32_t hartid;   /**< Mesh Tile ID. */
} redmule_config_t;

extern int redmule_init(redmule_controller_t *ctrl);

/* extern void redmule_wait(); */

/**
 * This function prepares and execute an accelerated generic matrix multiplication.
 * (N x M * M x K) + (N x K) = (N x K)
 */
extern int redmule_gemm(redmule_controller_t *ctrl, uint32_t x, uint32_t w, uint32_t y, uint16_t m, uint16_t n, uint16_t k);

/**
 * WIP
 * Redmule API
 */
struct redmule_controller_api {
    int (*init)(redmule_controller_t *ctrl);
/*     void (*wait)(); */
    int (*gemm)(redmule_controller_t *ctrl, uint32_t x, uint32_t w, uint32_t y, uint16_t m, uint16_t n, uint16_t k);
};

/*
 * Generic implementation of the Redmule controller.
 */
extern redmule_controller_api_t redmule_api;

#endif //HAL_REDMULE_H