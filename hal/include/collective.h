// Copyright 2025-2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0


#ifndef HAL_COLLECTIVE_H
#define HAL_COLLECTIVE_H

#include <stdint.h>

/** Forward declaration of the collective controller instance and API structure. */
typedef struct floo_collective floo_collective_t;
typedef struct floo_collective_api floo_collective_api_t;

/**
 * WIP
 * Holds the API function pointers, base address and controller-specific configuration.
 */
struct floo_collective {
    floo_collective_api_t *api;     /**< Function pointers for this interface. */
    uint32_t base;                  /**< MMIO base address (if applicable). */
    void *cfg;                      /**< Driver‑specific configuration. */
};

/**
 * Collective configuration structure.
 * This structure holds the configuration settings for the collective initialization.
 */
typedef struct {
    uint32_t hartid;    /**< Mesh tile ID*/
} floo_collective_config_t;

/**
 * Builds the collective mask for the given group of communicators.
 */
extern int gen_collective_mask(floo_collective_t *ctrl, uint32_t group);

/**
 * Multicasts a word to the tiles of the same row over the narrow channel.
 */
extern int narrow_mcast_row(floo_collective_t *ctrl, uint32_t collective_word, uint32_t addr_offset);

/**
 * Multicasts a word to the tiles of the same column over the narrow channel.
 */
extern int narrow_mcast_column(floo_collective_t *ctrl, uint32_t collective_word, uint32_t addr_offset);

/**
 * Multicasts a word to the entire mesh over the narrow channel.
 */
extern int narrow_mcast_mesh(floo_collective_t *ctrl, uint32_t collective_word, uint32_t addr_offset);

/**
 * Synchronizes the current tile with the ones of the same row.
 */
extern int narrow_sync_row(floo_collective_t *ctrl, uint32_t collective_word, uint32_t addr_offset);

/**
 * Synchronizes the current tile with the ones of the same column.
 */
extern int narrow_sync_column(floo_collective_t *ctrl, uint32_t collective_word, uint32_t addr_offset);

/**
 * Synchronizes the entire mesh.
 */
extern int narrow_sync_mesh(floo_collective_t *ctrl, uint32_t collective_word, uint32_t addr_offset);


/**
 * WIP
 * Collective API
 */
struct floo_collective_api {
    int (*gen_collective_mask)(floo_collective_t *ctrl, uint32_t group);
    int (*mcast_row)(floo_collective_t *ctrl, uint32_t collective_word, uint32_t addr_offset);
    int (*mcast_column)(floo_collective_t *ctrl, uint32_t collective_word, uint32_t addr_offset);
    int (*mcast_mesh)(floo_collective_t *ctrl, uint32_t collective_word, uint32_t addr_offset);
    int (*sync_row)(floo_collective_t *ctrl, uint32_t collective_word, uint32_t addr_offset);
    int (*sync_column)(floo_collective_t *ctrl, uint32_t collective_word, uint32_t addr_offset);
    int (*sync_mesh)(floo_collective_t *ctrl, uint32_t collective_word, uint32_t addr_offset);
};

/*
 * FlooNoC implementation of the collective controller.
 */
extern floo_collective_api_t floonoc_collective_api;

#endif
