// Copyright 2024-2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Viviane Potocnik <vivianep@iis.ee.ethz.ch>
// Alberto Dequino <alberto.dequino@unibo.it>

#include <stdint.h>
#include "fsync.h"
#include "utils/tinyprintf.h"


/*-----------------------------------------------------------------*/
/* Fsync weak stubs (can be overridden by platform implementations) */
/*-----------------------------------------------------------------*/

/*
__attribute__((weak)) int fsync_init(fsync_controller_t *ctrl){
    (void) ctrl;
    return 1;
}
*/

/*
__attribute__((weak)) int fsync_sync_level(fsync_controller_t *ctrl, uint32_t level, uint8_t dir){
    (void) ctrl;
    (void) level;
    (void) dir;
    return 1;
}*/

/*
__attribute__((weak)) int fsync_getgroup_level(fsync_controller_t *ctrl, uint32_t level, uint32_t id, uint8_t dir){
    (void) ctrl;
    (void) level;
    (void) id;
    (void) dir;
    return 1;
}*/


/*
__attribute__((weak)) int fsync_sync_row(fsync_controller_t *ctrl){
    (void) ctrl;
    (void) level;
    return 1;
}*/

/*
__attribute__((weak)) int fsync_sync_col(fsync_controller_t *ctrl){
    (void) ctrl;
    return 1;
}*/

/*
__attribute__((weak)) int fsync_sync_diag(fsync_controller_t *ctrl){
    (void) ctrl;
    return 1;
}*/

/*
__attribute__((weak)) int fsync_sync(fsync_controller_t *ctrl, uint32_t *ids, uint8_t n_tiles, uint8_t dir){
    (void) ctrl;
    (void) ids;
    (void) n_tiles;
    (void) dir;
    return 1;
}*/

/*----------------------------------------*/
/* Export the controller API for the Fsync */
/*----------------------------------------*/
__attribute__((weak)) fsync_controller_api_t fsync_api = {
    .init = fsync_init,
    .sync_level = fsync_sync_level,
    .getgroup_level = fsync_getgroup_level,
    .sync_col = fsync_sync_col,
    .sync_row = fsync_sync_row,
    .sync_diag = fsync_sync_diag,
    .sync = fsync_sync,
};