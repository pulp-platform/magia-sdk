// Copyright 2024-2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Viviane Potocnik <vivianep@iis.ee.ethz.ch>
// Alberto Dequino <alberto.dequino@unibo.it>


#ifndef HAL_FSYNC_H
#define HAL_FSYNC_H

/** Forward declaration of the fsync controller instance and API structure. */
typedef struct fsync_controller fsync_controller_t;
typedef struct fsync_controller_api fsync_controller_api_t;

/**
 * WIP
 * Holds the API function pointers, base address and controller-specific configuration.
 */
struct fsync_controller {
    fsync_controller_api_t *api;    /**< Function pointers for this interface. */
    uint32_t base;                  /**< MMIO base address (if applicable). */
    void *cfg;                      /**< Driver‑specific configuration. */
};

/**
 * Fsync configuration structure.
 * This structure holds the configuration settings for Fsync initialization.
 */
typedef struct {
    uint32_t hartid;    /**< Mesh tile ID*/
} fsync_config_t;

/**
 * Default Fsync configuration settings.
 */
extern fsync_config_t default_cfg;

/**
 * Opens and initializes the fsync interface.
 */
extern int fsync_init(fsync_controller_t *ctrl);

/**
 * Synchronizes the current tile with the ones of the same synchronization level.
 */
extern int fsync_sync(fsync_controller_t *ctrl, uint32_t level);

/**
 * Gets the current tile's group ID for the selected synchronization level.
 */
extern int fsync_getgroup(fsync_controller_t *ctrl, uint32_t level);


/**
 * WIP
 * Fsync API
 */
struct fsync_controller_api {
    int (*init)(fsync_controller_t *ctrl);
    int (*sync)(fsync_controller_t *ctrl, uint32_t level);
    int (*getgroup)(fsync_controller_t *ctrl, uint32_t level);
};

/*
 * Generic implementation of the Fsync controller.
 */
extern fsync_controller_api_t fsync_api;

#endif