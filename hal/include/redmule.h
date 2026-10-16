// Copyright 2024-2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Viviane Potocnik <vivianep@iis.ee.ethz.ch>
// Alberto Dequino <alberto.dequino@unibo.it>

#ifndef HAL_REDMULE_H
#define HAL_REDMULE_H

#include <stdint.h>

extern int redmule_init();

/* extern void redmule_wait(); */

/**
 * Reads the hardware ACQUIRE register, which locks the controller and
 * returns a fresh job id. Returns -1 if the hardware job queue (depth 2)
 * is already full.
 */
extern int32_t redmule_acquire();

/**
 * Reads the hardware RUNNING_JOB register, which returns the job ID
 * of the currently running job, or the last run job if Redmule is currently idle.
 */
extern int32_t redmule_running_job();

/**
 * This function prepares and execute an accelerated generic matrix multiplication.
 * (N x M * M x K) + (N x K) = (N x K)
 */
extern int redmule_gemm(uint32_t x, uint32_t w, uint32_t y, uint16_t m, uint16_t n, uint16_t k);

/**
 * WIP
 * Redmule API
 */
struct redmule_controller_api {
    int (*init)();
    /*     void (*wait)(); */
    int32_t (*acquire)();
    int (*gemm)(uint32_t x, uint32_t w, uint32_t y, uint16_t m, uint16_t n, uint16_t k);
    int32_t (*running_job)();
};

#endif // HAL_REDMULE_H