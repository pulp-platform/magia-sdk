#include <stdint.h>

#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"
#include "test.h"

#define WAIT_MODE WFE
#define NUM_CORES 4

int main(void)
{
    perf_start();
    int start = perf_get_cycles();

    uint32_t hartid = get_hartid();

    /* use cores */
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
    uint32_t rows_per_core = M / NUM_CORES;

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
    DMA vector x into L1
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

        uint16_t *row = &local_A[i * N];
        
        for (int32_t j = 0; j < N; j++) {
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
    Barrier
    */
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

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

    int end = perf_get_cycles();
    printf("Core[%d] Cycles: %d\n", hartid, end - start);

    /*
    --------------------------------------------------
    Verify
    --------------------------------------------------
    */
    if (hartid == 0) {

        int errors = 0;

        for (int i = 0; i < M; i++) {
            if (y[i] != y_expected[i]) {
                errors++;
            }
        }

        printf("Errors: %d\n", errors);

        return errors;
    }

    return 0;
}