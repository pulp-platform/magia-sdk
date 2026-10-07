// Copyright 2025-2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "collective.h"


/*-----------------------------------------------------------------------*/
/* Collective weak stubs (can be overridden by platform implementations) */
/*-----------------------------------------------------------------------*/

/*
__attribute__((weak)) int gen_collective_mask(floo_collective_t *ctrl, uint32_t group){
    (void) ctrl;
    (void) group;
    return 0;
}*/

/*
__attribute__((weak)) int narrow_mcast_row(floo_collective_t *ctrl, uint32_t comm_group, uint32_t word, uint32_t addr_offset){
    (void) ctrl;
    (void) comm_group;
    (void) word;
    (void) addr_offset;
    return 1;
}*/

/*
__attribute__((weak)) int narrow_mcast_column(floo_collective_t *ctrl, uint32_t comm_group, uint32_t word, uint32_t addr_offset){
    (void) ctrl;
    (void) comm_group;
    (void) word;
    (void) addr_offset;
    return 1;
}*/

/*
__attribute__((weak)) int narrow_mcast_mesh(floo_collective_t *ctrl, uint32_t comm_group, uint32_t word, uint32_t addr_offset){
    (void) ctrl;
    (void) comm_group;
    (void) word;
    (void) addr_offset;
    return 1;
}*/

/*
__attribute__((weak)) int narrow_sync_row(floo_collective_t *ctrl, uint32_t comm_group, uint32_t word, uint32_t addr_offset){
    (void) ctrl;
    (void) comm_group;
    (void) word;
    (void) addr_offset;
    return 1;
}*/

/*
__attribute__((weak)) int narrow_sync_column(floo_collective_t *ctrl, uint32_t comm_group, uint32_t word, uint32_t addr_offset){
    (void) ctrl;
    (void) comm_group;
    (void) word;
    (void) addr_offset;
    return 1;
}*/

/*
__attribute__((weak)) int narrow_sync_mesh(floo_collective_t *ctrl, uint32_t comm_group, uint32_t word, uint32_t addr_offset){
    (void) ctrl;
    (void) comm_group;
    (void) word;
    (void) addr_offset;
    return 1;
}*/

/*----------------------------------------------*/
/* Export the controller API for the Collective */
/*----------------------------------------------*/
__attribute__((weak)) floo_collective_api_t floonoc_collective_api = {
    .gen_collective_mask = gen_collective_mask,
    .mcast_row           = narrow_mcast_row,
    .mcast_column        = narrow_mcast_column,
    .mcast_mesh          = narrow_mcast_mesh,
    .sync_row            = narrow_sync_row,
    .sync_column         = narrow_sync_column,
    .sync_mesh           = narrow_sync_mesh,
};
