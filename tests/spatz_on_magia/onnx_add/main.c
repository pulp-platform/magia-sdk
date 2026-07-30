// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "tile.h"
#include "eventunit.h"
#include "idma.h"

#include "compare_utils.h"
#include "data.h"
#include "onnx_add_mem_layout.h"
#include "onnx_add_params.h"
#include "onnx_add_task_bin.h"

#define HID get_hartid()

static int init_data(void *params)
{
    volatile onnx_add_params_t *add_params;
    uint32_t start;
    uint32_t chunk;
    uint32_t left;
    uint32_t len;
    uint32_t end;

    idma_config_t idma_cfg      = {.hartid = get_hartid()};
    idma_controller_t idma_ctrl = {
        .base = NULL,
        .cfg  = &idma_cfg,
        .api  = &idma_api,
    };
    idma_init(&idma_ctrl);

    eu_config_t eu_cfg      = {.hartid = get_hartid()};
    eu_controller_t eu_ctrl = {
        .base = NULL,
        .cfg  = &eu_cfg,
        .api  = &eu_api,
    };

    eu_init(&eu_ctrl);
    eu_idma_init(&eu_ctrl, 0);

    add_params = (volatile onnx_add_params_t *)params;

    chunk = TENSOR_LEN / NUM_HARTS;
    left  = TENSOR_LEN % NUM_HARTS;
    start = HID * chunk + (HID < left ? HID : left);
    end   = start + chunk + (HID < left ? 1 : 0);
    len   = end - start;

    // for (unsigned int i = 0; i < len; i++) {
    //     int global_idx;
    //     uint32_t offset;

    //     global_idx = start + i;
    //     offset     = i * sizeof(float16);

    //     mmio_fp16(CHUNK_A_BASE + offset) = A[global_idx];
    //     mmio_fp16(CHUNK_B_BASE + offset) = B[global_idx];
    //     mmio_fp16(CHUNK_G_BASE + offset) = G[global_idx];
    //     mmio_fp16(CHUNK_C_BASE + offset) = 0;
    // }

    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)A, CHUNK_A_BASE, (len * 2));
    eu_idma_wait_a2o(&eu_ctrl, WFE);
    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)B, CHUNK_B_BASE, (len * 2));
    eu_idma_wait_a2o(&eu_ctrl, WFE);
    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)G, CHUNK_G_BASE, (len * 2));
    eu_idma_wait_a2o(&eu_ctrl, WFE);

    add_params->chunk_A = CHUNK_A_BASE;
    add_params->chunk_B = CHUNK_B_BASE;
    add_params->chunk_C = CHUNK_C_BASE;
    add_params->chunk_G = CHUNK_G_BASE;
    add_params->start   = start;
    add_params->len     = len;
    add_params->end     = end;

    return 0;
}

static int run_spatz_task()
{
    int ret;
    eu_config_t eu_cfg;
    eu_controller_t eu_ctrl;

    eu_cfg.hartid = get_hartid();
    eu_ctrl.base = NULL, eu_ctrl.cfg = &eu_cfg, eu_ctrl.api = &eu_api,

    eu_init(&eu_ctrl);
    eu_spatz_init(&eu_ctrl, 0);

    printf("[CV32] Initializing spatz binary start...\n");
    spatz_init(SPATZ_BINARY_START);

    printf("[CV32] Running spatz task...\n");
    spatz_run_task_with_params(ONNX_ADD_TASK, ONNX_ADD_PARAMS_BASE);

    printf("[CV32] Waiting spatz task...\n");
    eu_spatz_wait(&eu_ctrl, WFE);

    ret = spatz_get_exit_code();

    spatz_clk_dis();

    return ret;
}

static bool check_result(void *params)
{
    volatile onnx_add_params_t *add_params;
    add_params = (volatile onnx_add_params_t *)params;
    return chunk_compare_fp16_bitwise(
        add_params->chunk_C, add_params->chunk_G, add_params->start, add_params->len);
}

static bool run_test()
{
    int ret;
    bool check;
    volatile onnx_add_params_t *params;

    params = (volatile onnx_add_params_t *)ONNX_ADD_PARAMS_BASE;

    printf("[CV32] Initializing parameters...\n");
    ret = init_data((void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] Params initialization failed with error: %d\n", HID, ret);
        return ret;
    }

    printf("[CV32] Runnin task...\n");
    ret = run_spatz_task();
    if (ret != 0) {
        printf("[CV32 (%d)] Spatz task FAILED with error: %d", HID, ret);
        return ret;
    }

    printf("[CV32] Checking results...\n");
    check = check_result((void *)params);
    if (check) {
        printf("[CV32 (%d)] Test SUCCESS\n", HID);
    } else {
        printf("[CV32 (%d)] Test FAILED\n", HID);
        ret = -1;
    }

    return ret;
}

int main(void)
{
    int ret;
    int hartid = HID;

    if (HID == 0)
        printf("############################### ONNX_ADD TEST on Tiles "
               "################################\n");

    ret = run_test();

    if (HID == 0)
        printf("\n#################################################################################"
               "#########\n\n");

    return ret;
}
