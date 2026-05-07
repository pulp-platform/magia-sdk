/*
Dense -> CSR -> Parallel SpMV on MAGIA

Copyright (c) 2026
*/

#include <stdint.h>

#include "tile.h"
#include "fsync.h"
#include "eventunit.h"
#include "test.h"

#define WAIT_MODE WFE
#define NUM_CORES 4

/* Runtime CSR storage */
static uint16_t values[MAX_NNZ];
static uint16_t col_idx[MAX_NNZ];
static uint16_t row_ptr[M + 1];
static uint16_t nnz_count[M];

int main(void)
{
    perf_start();
    int start = perf_get_cycles();

    uint32_t hartid = get_hartid();

    /* Only first 2 cores participate */
    if (hartid >= NUM_CORES) {
        return 0;
    }

    /*
    --------------------------------------------------
    Initialize MAGIA synchronization
    --------------------------------------------------
    */
    fsync_config_t fsync_cfg = {
        .hartid = hartid
    };

    fsync_controller_t fsync_ctrl = {
        .base = NULL,
        .cfg  = &fsync_cfg,
        .api  = &fsync_api,
    };

    fsync_init(&fsync_ctrl);

    eu_config_t eu_cfg = {
        .hartid = hartid
    };

    eu_controller_t eu_ctrl = {
        .base = NULL,
        .cfg  = &eu_cfg,
        .api  = &eu_api,
    };

    eu_init(&eu_ctrl);
    eu_clear_events(0xFFFFFFFF);
    eu_fsync_init(&eu_ctrl, 0);

    /*
    --------------------------------------------------
    Partition rows
    --------------------------------------------------
    */
    uint32_t rows_per_core = (M + NUM_CORES - 1) / NUM_CORES;

    uint32_t start_row = hartid * rows_per_core;
    uint32_t end_row   = start_row + rows_per_core;

    if (end_row > M) {
        end_row = M;
    }
    

    /*
    ==================================================
    PHASE 1: Count nonzeros
    ==================================================
    */
    for (uint32_t i = start_row; i < end_row; i++) {

        uint16_t count = 0;

        for (uint32_t j = 0; j < N; j++) {

            if (A_dense[i * N + j] != 0) {
                count++;
            }
        }

        nnz_count[i] = count;
    }

    /* Barrier */
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /*
    ==================================================
    PHASE 2: Prefix sum (core 0)
    ==================================================
    */
    if (hartid == 0) {

        row_ptr[0] = 0;

        for (uint32_t i = 0; i < M; i++) {
            row_ptr[i + 1] = row_ptr[i] + nnz_count[i];
        }
    }

    /* Barrier */
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /*
    ==================================================
    PHASE 3: Fill CSR arrays
    ==================================================
    */
    for (uint32_t i = start_row; i < end_row; i++) {

        uint32_t idx = row_ptr[i];

        for (uint32_t j = 0; j < N; j++) {

            uint16_t val = A_dense[i * N + j];

            if (val != 0) {

                values[idx]  = val;
                col_idx[idx] = j;

                idx++;
            }
        }
    }

    /* Barrier */
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /*
    ==================================================
    PHASE 4: CSR SpMV
    ==================================================
    */
    for (uint32_t i = start_row; i < end_row; i++) {

        uint32_t sum = 0;

        for (uint32_t j = row_ptr[i]; j < row_ptr[i + 1]; j++) {

            sum += values[j] * x[col_idx[j]];
        }

        y[i] = (uint16_t)sum;
    }

    /* Barrier */
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    int end = perf_get_cycles();
    printf("Core[%d] Cycles: %d\n", hartid, end - start);

    /*
    ==================================================
    PHASE 5: Verify
    ==================================================
    */
    if (hartid == 0) {

        int errors = 0;

        for (uint32_t i = 0; i < M; i++) {

            if (y[i] != y_expected[i]) {
                errors++;

#if EVAL == 1
                printf("Mismatch at y[%d] = %u (expected %u)\n",
                       i,
                       y[i],
                       y_expected[i]);
#endif
            }
        }

        printf("Errors: %d\n", errors);

        return errors;
    }

    return 0;
}