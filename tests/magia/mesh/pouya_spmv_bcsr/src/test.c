#include <stdint.h>

#include "tile.h"
#include "idma.h"
#include "redmule.h"
#include "fsync.h"
#include "eventunit.h"
#include "test.h"

#define WAIT_MODE WFE
#define NUM_CORES 4

#define TK 8   // panel width

int main(void)
{


    uint32_t hartid = get_hartid();

    if (hartid >= NUM_CORES) return 0;

    /*
    --------------------------------------------------
    Init
    --------------------------------------------------
    */
    idma_config_t idma_cfg = {.hartid = hartid};
    idma_controller_t idma_ctrl = {
        .base = NULL,
        .cfg  = &idma_cfg,
        .api  = &idma_api,
    };
    idma_init(&idma_ctrl);

    redmule_config_t redmule_cfg = {.hartid = hartid};
    redmule_controller_t redmule_ctrl = {
        .base = NULL,
        .cfg  = &redmule_cfg,
        .api  = &redmule_api,
    };
    redmule_init(&redmule_ctrl);

    fsync_config_t fsync_cfg = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {
        .base = NULL,
        .cfg  = &fsync_cfg,
        .api  = &fsync_api,
    };
    fsync_init(&fsync_ctrl);

    eu_config_t eu_cfg = {.hartid = hartid};
    eu_controller_t eu_ctrl = {
        .base = NULL,
        .cfg  = &eu_cfg,
        .api  = &eu_api,
    };
    eu_init(&eu_ctrl);
    eu_idma_init(&eu_ctrl, 0);
    eu_redmule_init(&eu_ctrl, 0);
    eu_fsync_init(&eu_ctrl, 0);

    /*
    --------------------------------------------------
    Row partition
    --------------------------------------------------
    */
    uint32_t rows_per_core = (M + NUM_CORES - 1) / NUM_CORES;

    uint32_t start_row = hartid * rows_per_core;
    uint32_t end_row   = (start_row + rows_per_core > M) ? M : start_row + rows_per_core;

    uint32_t local_rows = end_row - start_row;

    /*
    --------------------------------------------------
    L1 layout
    --------------------------------------------------
    */
    uint32_t l1 = get_l1_base(hartid);

    uint32_t addr_A      = l1;
    uint32_t addr_x      = addr_A + local_rows * N * 2;

    uint32_t addr_panel_ptr  = addr_x + N * 2;
    uint32_t addr_panel_base = addr_panel_ptr + (local_rows + 1) * 4;
    uint32_t addr_panel_vals = addr_panel_base + (local_rows * (N/TK)) * 2;

    uint32_t addr_xbuf   = addr_panel_vals + (local_rows * (N/TK) * TK * 2);
    uint32_t addr_wbuf   = addr_xbuf + TK * 2;
    uint32_t addr_ybuf   = addr_wbuf + TK * 2;

    uint32_t addr_y      = addr_ybuf + 2;

    /*
    --------------------------------------------------
    DMA A and x
    --------------------------------------------------
    */
    idma_memcpy_2d(&idma_ctrl, 0,
        (uint32_t)A_dense + start_row * N * 2,
        addr_A,
        N * 2,
        N * 2,
        local_rows);
    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

    idma_memcpy_1d(&idma_ctrl, 0,
        (uint32_t)x,
        addr_x,
        N * 2);
    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

    /*
    --------------------------------------------------
    Local pointers
    --------------------------------------------------
    */
    uint16_t *local_A = (uint16_t*)addr_A;
    uint16_t *local_x = (uint16_t*)addr_x;

    uint32_t *panel_ptr  = (uint32_t*)addr_panel_ptr;
    uint16_t *panel_base = (uint16_t*)addr_panel_base;
    uint16_t *panel_vals = (uint16_t*)addr_panel_vals;

    uint16_t *xbuf = (uint16_t*)addr_xbuf;
    uint16_t *wbuf = (uint16_t*)addr_wbuf;
    uint16_t *ybuf = (uint16_t*)addr_ybuf;

    uint16_t *local_y = (uint16_t*)addr_y;

    /*
    ==================================================
    PANEL COMPRESSION
    ==================================================
    */
    uint32_t pcount = 0;
    panel_ptr[0] = 0;

    for (uint32_t i = 0; i < local_rows; i++) {

        for (uint32_t b = 0; b < N; b += TK) {

            int nonzero = 0;

            for (uint32_t k = 0; k < TK; k++) {
                if (local_A[i*N + b + k] != 0) {
                    nonzero = 1;
                    break;
                }
            }

            if (nonzero) {
                panel_base[pcount] = b;

                for (uint32_t k = 0; k < TK; k++) {
                    panel_vals[pcount*TK + k] =
                        local_A[i*N + b + k];
                }

                pcount++;
            }
        }

        panel_ptr[i+1] = pcount;
    }

    /*
    Barrier
    */
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /*
    ==================================================
    RedMulE SpMV
    ==================================================
    */
    perf_start();
    int start = perf_get_cycles();

    for (uint32_t i = 0; i < local_rows; i++) {

        // y = 0
        ybuf[0] = 0;

        for (uint32_t p = panel_ptr[i];
             p < panel_ptr[i+1];
             p++) {

            uint32_t base = panel_base[p];

            // load x slice
            for (uint32_t k = 0; k < TK; k++) {
                xbuf[k] = local_x[base + k];
                wbuf[k] = panel_vals[p*TK + k];
            }

            // GEMV: y += xbuf (1xTK) * wbuf (TKx1)
            redmule_gemm(&redmule_ctrl,
                         addr_xbuf,
                         addr_wbuf,
                         addr_ybuf,
                         1, TK, 1);

            eu_redmule_wait(&eu_ctrl, WAIT_MODE);
        }

        local_y[i] = ybuf[0];
    }

    int end = perf_get_cycles();
    printf("Core[%d] Cycles: %d\n", hartid, end - start);

    /*
    --------------------------------------------------
    DMA back
    --------------------------------------------------
    */
    idma_memcpy_1d(&idma_ctrl, 1,
        (uint32_t)(y + start_row),
        addr_y,
        local_rows * 2);

    eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);

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
            if (y[i] != y_expected[i]) {
                errors++;
            }
        }

        printf("Errors: %d\n", errors);
        return errors;
    }

    return 0;
}