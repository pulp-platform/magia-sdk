// Copyright 2024-2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Viviane Potocnik <vivianep@iis.ee.ethz.ch>
// Alberto Dequino <alberto.dequino@unibo.it>

#include <stdint.h>
#include "redmule.h"

/*--------------------------------------------------------------------*/
/* Redmule weak stubs (can be overridden by platform implementations) */
/*--------------------------------------------------------------------*/

/*__attribute__((weak)) int redmule_init(redmule_controller_t *ctrl){
    (void) ctrl;
    return 1;
}*/


/*__attribute__((weak)) int redmule_wait(redmule_controller_t *ctrl){
    (void) ctrl;
    return 1;
}*/

/**
 * This function prepares and execute an accelerated generic matrix multiplication.
 * (N x M * M x K) + (N x K) = (N x K)
 *
 * @param x First matrix initial pointer.
 * @param w Second matrix initial pointer.
 * @param y Bias Matrix initial pointer. The GEMM output is also stored here.
 * @param m Number of rows of x and y.
 * @param n Number of columns of x, and number of rows of w.
 * @param k Number of columns of w and y.
 * 
 * @return 1 if successful
 * 
 */
/*__attribute__((weak)) int redmule_gemm(redmule_controller_t *ctrl, uint32_t x, uint32_t w, uint32_t y, uint16_t m, uint16_t n, uint16_t k){
    (void) ctrl;
    (void) x;
    (void) w;
    (void) y;
    (void) m;
    (void) n;
    (void) k;
    return 1;
}*/

/*-------------------------------------------*/
/* Export the controller API for the Redmule */
/*-------------------------------------------*/
__attribute__((weak)) redmule_controller_api_t redmule_api = {
    .init = redmule_init,
    .wait = redmule_wait,
    .gemm = redmule_gemm,
};