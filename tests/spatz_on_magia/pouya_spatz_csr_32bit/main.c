#include <stdint.h>
#include <stdbool.h>

#include "compare_utils.h"
#include "tile.h"              /* mmio*, mmio_fp32, get_hartid, L1_BASE, printf */
#include "eventunit.h"         /* eu_* controller API                        */
#include "data.h"              /* DIM_M, DIM_K, NNZ, row_ptr/col_idx/values/x/G */
#include "onnx_spmv_mem_layout.h"        /* *_BASE addresses in this tile's L1         */
#include "onnx_spmv_params.h"            /* spatz_csr_params_t                         */
#include "spatz_csr_task_bin.h"/* generated: SPATZ_CSR_TASK, SPATZ_BINARY_START */

/* magia_tile_utils.h's mmio_fp32(x) macro expands to
 * "*(volatile float32 *)(x)", but this SDK's include tree never typedefs
 * "float32" (unlike "float16", which tile.h does provide) -- so the macro
 * doesn't compile as shipped. Supply the missing typedef locally. */
typedef float float32;



/* Write row_ptr, col_idx (pre-scaled to byte offsets), values, x, Y, G and
 * the params block into this tile's L1, ready for the Spatz task. */
static void init_data(void) {
    uint32_t i;

    /* row_ptr: uint32_t[M+1], copied as-is */
    for (i = 0; i <= DIM_M; i++) {
        mmio32(ROW_PTR_BASE + i * sizeof(uint32_t)) = row_ptr[i];
    }

    /* col_idx: data.h stores plain column indices; the task needs BYTE
     * offsets for vloxei32.v, so pre-scale here on the host. */
    for (i = 0; i < NNZ; i++) {
        mmio32(COL_IDX_BASE + i * sizeof(uint32_t)) = col_idx[i] * (uint32_t)sizeof(float);
    }

    /* values: float[NNZ] */
    for (i = 0; i < NNZ; i++) {
        mmio_fp32(VALUES_BASE + i * sizeof(float)) = values[i];
    }

    /* x: float[K] */
    for (i = 0; i < DIM_K; i++) {
        mmio_fp32(X_BASE + i * sizeof(float)) = x[i];
    }

    /* G: float[M], golden reference kept in L1 for the on-tile compare */
    for (i = 0; i < DIM_M; i++) {
        mmio_fp32(G_BASE + i * sizeof(float)) = G[i];
    }

    /* Y: float[M], zero the output buffer before the task runs */
    for (i = 0; i < DIM_M; i++) {
        mmio_fp32(Y_BASE + i * sizeof(float)) = 0.0f;
    }

    /* params block */
    volatile spatz_csr_params_t *p =
        (volatile spatz_csr_params_t *)SPATZ_CSR_PARAMS_BASE;
    p->addr_row_ptr = ROW_PTR_BASE;
    p->addr_col_idx = COL_IDX_BASE;
    p->addr_values  = VALUES_BASE;
    p->addr_x       = X_BASE;
    p->addr_Y       = Y_BASE;
    p->addr_G       = G_BASE;
    p->M            = DIM_M;
    p->K            = DIM_K;
    p->nnz          = NNZ;
}

/* Canonical host sequence from the guide, section 4.3. */
static int run_spatz_task(void) {
    eu_config_t     eu_cfg;
    eu_controller_t eu_ctrl;

    eu_cfg.hartid = get_hartid();
    eu_ctrl.base  = NULL;
    eu_ctrl.cfg   = &eu_cfg;
    eu_ctrl.api   = &eu_api;

    eu_init(&eu_ctrl);
    eu_spatz_init(&eu_ctrl, 0);

    spatz_init(SPATZ_BINARY_START);
    spatz_run_task_with_params(SPATZ_CSR_TASK, SPATZ_CSR_PARAMS_BASE);
    eu_spatz_wait(&eu_ctrl, WFE);

    int ret = spatz_get_exit_code();
    spatz_clk_dis();
    return ret;
}

int main(void) {
    init_data();

    int ret = run_spatz_task();
    if (ret != 0) {
        if (get_hartid() == 0) {
            printf("spatz_csr: task FAILED, exit code 0x%x\n", ret);
        }
        return ret;
    }

    bool ok = vector_compare_fp32_bitwise(Y_BASE, G_BASE, DIM_M);

    if (get_hartid() == 0) {
        if (ok) {
            printf("spatz_csr: SpMV (CSR, fp32) SUCCESS\n");
        } else {
            printf("spatz_csr: SpMV (CSR, fp32) FAILED (result mismatch)\n");
        }
    }

    return ok ? 0 : 1;
}