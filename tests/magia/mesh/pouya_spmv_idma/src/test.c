/*
    This code is the implementation of SpMV multiplication on MAGIA architecture by using IDMA. Data in this code is given in normal format and 
    during the first step of the code they convert to compressed CSR format. Then they are transfered to L2 memory. Now everything is ready for
    starting SpMV multiplication in CSR format. the number of cycles are counted only for bringing data into L1 and applying the SpMV operation.
    
    Using this code is free by citing the name author:
    Pouya Shirinshahrakfard
    Email: pouyashirinfard@gmail.com
    
    for more info, please contact the author.
*/   


#include <stdint.h>

#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"
#include "test.h"

#define WAIT_MODE WFE
#define NUM_CORES 4

#define ROWS_PER_CORE ((M + NUM_CORES - 1) / NUM_CORES)

/*
--------------------------------------------------
L2 buffers for compressed data
(shared across cores)
--------------------------------------------------
*/
static uint16_t values_l2[M * N] __attribute__((section(".l2")));
static uint16_t colidx_l2[M * N] __attribute__((section(".l2")));
static uint16_t rowptr_l2[NUM_CORES * (ROWS_PER_CORE + 1)] __attribute__((section(".l2")));

int main(void)
{
    uint32_t hartid = get_hartid();

    if (hartid >= NUM_CORES) {
        return 0;
    }

    /*
    --------------------------------------------------
    Init IDMA
    --------------------------------------------------
    */
    idma_config_t idma_cfg = {
        .hartid = hartid
    };

    idma_controller_t idma_ctrl = {
        .base = NULL,
        .cfg  = &idma_cfg,
        .api  = &idma_api,
    };

    idma_init(&idma_ctrl);

    /*
    --------------------------------------------------
    Init FSYNC
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

    /*
    --------------------------------------------------
    Init Event Unit
    --------------------------------------------------
    */
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
    eu_idma_init(&eu_ctrl, 0);
    eu_fsync_init(&eu_ctrl, 0);

    /*
    --------------------------------------------------
    Row partition
    --------------------------------------------------
    */
    uint32_t rows_per_core = (M + NUM_CORES - 1) / NUM_CORES;

    uint32_t start_row = hartid * rows_per_core;
    uint32_t end_row   = start_row + rows_per_core;

    if (end_row > M) {
        end_row = M;
    }

    uint32_t local_rows = end_row - start_row;

    /*
    --------------------------------------------------
    L1 memory layout
    --------------------------------------------------
    */
    uint32_t l1 = get_l1_base(hartid);

    uint32_t addr_A       = l1;
    uint32_t addr_x       = addr_A + local_rows * N * 2;
    uint32_t addr_values  = addr_x + N * 2;
    uint32_t addr_colidx  = addr_values + local_rows * N * 2;
    uint32_t addr_rowptr  = addr_colidx + local_rows * N * 2;
    uint32_t addr_ylocal  = addr_rowptr + (local_rows + 1) * 2;

    /*
    --------------------------------------------------
    DMA dense block into L1
    --------------------------------------------------
    */
    uint32_t axi_A =
        (uint32_t)A_dense +
        start_row * N * 2;

    idma_memcpy_2d(
        &idma_ctrl,
        0,
        axi_A,
        addr_A,
        N * 2,
        N * 2,
        local_rows
    );

    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

    /*
    --------------------------------------------------
    Local pointers
    --------------------------------------------------
    */
    volatile uint16_t *local_A      = (uint16_t*)addr_A;
    volatile uint16_t *local_x      = (uint16_t*)addr_x;
    volatile uint16_t *local_values = (uint16_t*)addr_values;
    volatile uint16_t *local_colidx = (uint16_t*)addr_colidx;
    volatile uint16_t *local_rowptr = (uint16_t*)addr_rowptr;
    volatile uint16_t *local_y      = (uint16_t*)addr_ylocal;

    /*
    ==================================================
    Sparse Matrix -> CSR locally
    ==================================================
    */
    uint32_t nnz = 0;
    local_rowptr[0] = 0;

    for (uint32_t i = 0; i < local_rows; i++) {

        uint16_t *row = (uint16_t*)&local_A[i * N];

        for (uint32_t j = 0; j < N; j++) {

            uint16_t v = row[j];

            if (v != 0) {
                local_values[nnz] = v;
                local_colidx[nnz] = j;
                nnz++;
            }
        }

        local_rowptr[i + 1] = nnz;
    }

    /*
    --------------------------------------------------
    Store compressed CSR in L2
    --------------------------------------------------
    */
    uint32_t l2_offset = start_row * N;

    uint32_t addr_values_l2 =
        (uint32_t)&values_l2[l2_offset];

    uint32_t addr_colidx_l2 =
        (uint32_t)&colidx_l2[l2_offset];

    uint32_t addr_rowptr_l2 =
        (uint32_t)&rowptr_l2[hartid * (ROWS_PER_CORE + 1)];

    idma_memcpy_1d(
        &idma_ctrl,
        1,
        addr_values_l2,
        addr_values,
        nnz * 2
    );
    eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);

    idma_memcpy_1d(
        &idma_ctrl,
        1,
        addr_colidx_l2,
        addr_colidx,
        nnz * 2
    );
    eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);

    idma_memcpy_1d(
        &idma_ctrl,
        1,
        addr_rowptr_l2,
        addr_rowptr,
        (local_rows + 1) * 2
    );
    eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);

    /*
    Barrier
    */
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /*
    ==================================================
    Start measurement:
    compressed invoke -> x invoke -> SpMV
    ==================================================
    */
    perf_start();
    int start = perf_get_cycles();

    /*
    --------------------------------------------------
    Bring compressed CSR from L2 -> L1
    --------------------------------------------------
    */
    idma_memcpy_1d(
        &idma_ctrl,
        0,
        addr_values_l2,
        addr_values,
        nnz * 2
    );
    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

    idma_memcpy_1d(
        &idma_ctrl,
        0,
        addr_colidx_l2,
        addr_colidx,
        nnz * 2
    );
    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

    idma_memcpy_1d(
        &idma_ctrl,
        0,
        addr_rowptr_l2,
        addr_rowptr,
        (local_rows + 1) * 2
    );
    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

    /*
    --------------------------------------------------
    Invoke vector x into L1
    --------------------------------------------------
    */
    idma_memcpy_1d(
        &idma_ctrl,
        0,
        (uint32_t)x,
        addr_x,
        N * 2
    );

    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

    /*
    ==================================================
    CSR SpMV locally
    ==================================================
    */
    for (uint32_t i = 0; i < local_rows; i++) {

        uint32_t sum = 0;

        for (uint32_t j = local_rowptr[i];
             j < local_rowptr[i + 1];
             j++) {

            sum += local_values[j] *
                   local_x[local_colidx[j]];
        }

        local_y[i] = (uint16_t)sum;
    }

    int end = perf_get_cycles();
    printf("Core[%d] Cycles: %d\n", hartid, end - start);

    /*
    --------------------------------------------------
    DMA local y -> global y
    --------------------------------------------------
    */
    idma_memcpy_1d(
        &idma_ctrl,
        1,
        (uint32_t)(y + start_row),
        addr_ylocal,
        local_rows * 2
    );

    eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);

    /*
    Barrier
    */
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /*
    --------------------------------------------------
    Verify
    --------------------------------------------------
    */
    if (hartid == 0) {

        int errors = 0;

        for (int i = 0; i < M; i++) {
            printf("y[%d] = %d | y_expected[%d] = %d\n", i, y[i], i, y_expected[i]);
            if (y[i] != y_expected[i]) {
                printf("Mismatch at index %d: got %d, expected %d\n", i, y[i], y_expected[i]);
                errors++;
            }
        }

        printf("Errors: %d\n", errors);
        return errors;
    }

    return 0;
}