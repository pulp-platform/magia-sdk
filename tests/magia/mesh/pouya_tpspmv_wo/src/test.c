#include <stdint.h>

#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"

//#include "data.h"      /* made by tpspmv_preprocess.py */

//#include "data_poisson3Da.h"
//#include "data_raefsky5.h"
//#include "data_ex6.h"
//#include "data_cavity05.h"
//#include "data_g7jac140.h"
#include "data_fxm4_6.h"
//#include "data_scsd8-2r.h"
//#include "data_e18.h"
//#include "data_scfxm1-2b.h"
//#include "data_testbig.h"

/*
=====================================================================
tpSpMV on MAGIA (concept only, NO optimization)
All preprocessing is done OFFLINE (python). data.h gives:

  Pc[DELTA+1]            column split points of the DELTA tiles
  tile_ptr[DELTA*(M+1)]  row pointers of each tile (local to the tile)
  tile_base[DELTA+1]     first entry of each tile in tile_valcol
  tile_max_slice_nnz[]   biggest slice (nnz) of each tile
  tile_valcol[2*NNZ]     (value, LOCAL column) pairs, tile by tile
  x, y, y_expected
  M, N, NNZ, DELTA, SLICE_ROWS

  Phase 1 : core c works on column tile c
            x_seg -> L1, then for each slice of SLICE_ROWS rows:
            DMA rowptr + entries -> L1, compute, DMA y' -> partial_y
  --- global barrier ---
  Phase 2 : core c owns a block of rows
            for each slice: DMA the slice of all DELTA partial
            vectors -> L1, add, DMA y -> L2
=====================================================================
*/

#define WAIT_MODE       WFE
#define clock_freq_MHz  1450

#define L2 __attribute__((section(".l2"), aligned(64)))

static uint32_t HARTIDS[128]       L2 = {0};
static uint32_t NUM_CORES          L2 = 0;
static uint32_t run_cycle[128]     L2 = {0};
static uint32_t phase1_cycle[128]  L2 = {0};
static uint32_t phase2_cycle[128]  L2 = {0};

/* phase-1 results: partial_y[tile * M + row]  (runtime buffer) */
static int32_t partial_y[DELTA * M] L2 = {0};

static inline uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

uint32_t find_max(uint32_t input[], uint32_t length)
{
    uint32_t max = 0;
    for (uint32_t i = 0; i < length; i++)
        if (input[i] > max) max = input[i];
    return max;
}

int main(void)
{
    uint32_t hartid = get_hartid();

    /* ---------------- init IDMA / FSYNC / Event Unit ---------------- */
    idma_config_t idma_cfg = { .hartid = hartid };
    idma_controller_t idma_ctrl = { .base = NULL, .cfg = &idma_cfg, .api = &idma_api };
    idma_init(&idma_ctrl);

    fsync_config_t fsync_cfg = { .hartid = hartid };
    fsync_controller_t fsync_ctrl = { .base = NULL, .cfg = &fsync_cfg, .api = &fsync_api };
    fsync_init(&fsync_ctrl);

    eu_config_t eu_cfg = { .hartid = hartid };
    eu_controller_t eu_ctrl = { .base = NULL, .cfg = &eu_cfg, .api = &eu_api };
    eu_init(&eu_ctrl);
    eu_clear_events(0xFFFFFFFF);
    eu_idma_init(&eu_ctrl, 0);
    eu_fsync_init(&eu_ctrl, 0);

    /* ---------------- count cores ---------------- */
    HARTIDS[hartid] = 1;
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    if (hartid == 0) {
        for (int i = 0; i < 128; i++)
            if (HARTIDS[i] == 1) NUM_CORES++;
    }
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* data.h was made for DELTA cores */
    if (NUM_CORES != DELTA) {
        if (hartid == 0)
            printf("ERROR: data.h is for %d cores, but %u cores are running\n",
                   DELTA, NUM_CORES);
        return -1;
    }
    if (hartid >= NUM_CORES) return 0;

    /* ================= start timing ================= */
    perf_start();
    uint32_t t_start = perf_get_cycles();

    uint32_t l1 = get_l1_base(hartid);

    /* ==============================================================
       PHASE 1 : partial CSR-based SpMV on my column tile
       ============================================================== */
    uint32_t c0      = Pc[hartid];
    uint32_t ncols   = Pc[hartid + 1] - c0;          /* n' */
    uint32_t max_nnz = tile_max_slice_nnz[hartid];

    /* L1 layout, phase 1 */
    uint32_t addr_x  = l1;                                            /* x_seg           */
    uint32_t addr_rp = addr_x  + ncols * sizeof(int32_t);             /* rowptr slice    */
    uint32_t addr_vc = addr_rp + (SLICE_ROWS + 1) * sizeof(uint32_t); /* slice entries   */
    uint32_t addr_yb = addr_vc + 2 * max_nnz * sizeof(int32_t);       /* y' slice        */
    uint32_t l1_phase1_end = addr_yb + SLICE_ROWS * sizeof(int32_t);

    volatile int32_t  *local_x  = (int32_t  *)addr_x;
    volatile uint32_t *local_rp = (uint32_t *)addr_rp;
    volatile int32_t  *local_vc = (int32_t  *)addr_vc;
    volatile int32_t  *local_yb = (int32_t  *)addr_yb;

    /* cache x_seg in L1 */
    if (ncols > 0) {
        idma_memcpy_1d(&idma_ctrl, 0,
                       (uint32_t)&x[c0], addr_x, ncols * sizeof(int32_t));
        eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    }

    for (uint32_t r0 = 0; r0 < M; r0 += SLICE_ROWS) {
        uint32_t rows = min_u32(SLICE_ROWS, M - r0);

        /* Step 1a: rowptr slice -> L1 */
        idma_memcpy_1d(&idma_ctrl, 0,
                       (uint32_t)&tile_ptr[hartid * (M + 1) + r0],
                       addr_rp, (rows + 1) * sizeof(uint32_t));
        eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

        uint32_t s0  = local_rp[0];
        uint32_t nnz = local_rp[rows] - s0;

        /* Step 1b: entries of the slice -> L1 */
        if (nnz > 0) {
            idma_memcpy_1d(&idma_ctrl, 0,
                           (uint32_t)&tile_valcol[2 * (tile_base[hartid] + s0)],
                           addr_vc, 2 * nnz * sizeof(int32_t));
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
        }

        /* Step 2: compute y' slice */
        for (uint32_t i = 0; i < rows; i++) {
            int32_t  sum = 0;
            uint32_t js  = local_rp[i]     - s0;
            uint32_t je  = local_rp[i + 1] - s0;
            for (uint32_t j = js; j < je; j++) {
                int32_t  v = local_vc[2 * j];
                uint32_t c = (uint32_t)local_vc[2 * j + 1];
                sum += v * local_x[c];
            }
            local_yb[i] = sum;
        }

        /* Step 3: y' slice -> L2 */
        idma_memcpy_1d(&idma_ctrl, 1,
                       (uint32_t)&partial_y[hartid * M + r0],
                       addr_yb, rows * sizeof(int32_t));
        eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
    }

    uint32_t t_phase1_end = perf_get_cycles();

    /* all partial vectors must be in L2 before phase 2 */
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* ==============================================================
       PHASE 2 : accumulate the DELTA partial vectors (my rows only)
       ============================================================== */
    uint32_t rows_per_core = (M + DELTA - 1) / DELTA;
    uint32_t row_begin = min_u32(hartid * rows_per_core, M);
    uint32_t row_end   = min_u32(row_begin + rows_per_core, M);

    uint32_t addr_in  = l1;                                              /* DELTA*SLICE_ROWS */
    uint32_t addr_out = addr_in + DELTA * SLICE_ROWS * sizeof(int32_t);  /* SLICE_ROWS       */
    uint32_t l1_phase2_end = addr_out + SLICE_ROWS * sizeof(int32_t);

    volatile int32_t *local_in  = (int32_t *)addr_in;
    volatile int32_t *local_out = (int32_t *)addr_out;

    for (uint32_t r0 = row_begin; r0 < row_end; r0 += SLICE_ROWS) {
        uint32_t rows = min_u32(SLICE_ROWS, row_end - r0);

        /* Step 1: this row-slice of every partial vector -> L1 */
        for (uint32_t c = 0; c < DELTA; c++) {
            idma_memcpy_1d(&idma_ctrl, 0,
                           (uint32_t)&partial_y[c * M + r0],
                           addr_in + c * SLICE_ROWS * sizeof(int32_t),
                           rows * sizeof(int32_t));
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
        }

        /* Step 2: accumulate */
        for (uint32_t i = 0; i < rows; i++) {
            int32_t sum = 0;
            for (uint32_t c = 0; c < DELTA; c++)
                sum += local_in[c * SLICE_ROWS + i];
            local_out[i] = sum;
        }

        /* Step 3: y slice -> L2 */
        idma_memcpy_1d(&idma_ctrl, 1,
                       (uint32_t)&y[r0], addr_out, rows * sizeof(int32_t));
        eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
    }

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    uint32_t t_end = perf_get_cycles();
    /* ================= stop timing ================= */

    run_cycle[hartid]    = t_end - t_start;
    phase1_cycle[hartid] = t_phase1_end - t_start;
    phase2_cycle[hartid] = t_end - t_phase1_end;   /* includes barrier wait */

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* ==============================================================
       Check + metrics (core 0)
       ============================================================== */
    if (hartid == 0) {
        volatile int32_t *y_res = (volatile int32_t *)y;   /* y is written by DMA */
        int errors = 0;
        for (int i = 0; i < M; i++) {
            if (y_res[i] != y_expected[i]) {
                if (errors < 20)
                    printf("Mismatch at row %d: got %d expected %d\n",
                           i, y_res[i], y_expected[i]);
                errors++;
            }
        }
        printf("Errors: %d\n", errors);

        uint32_t max_run    = find_max(run_cycle, 128);
        uint32_t max_phase1 = find_max(phase1_cycle, 128);
        uint32_t max_phase2 = find_max(phase2_cycle, 128);

        uint32_t total_flops  = 2 * NNZ;
        uint32_t gflops_x1000 = (total_flops * clock_freq_MHz) / max_run;
        uint32_t l1_used = (l1_phase1_end > l1_phase2_end ? l1_phase1_end : l1_phase2_end) - l1;

        printf("cores                    : %u\n", NUM_CORES);
        printf("run_time_cycles          : %u\n", max_run);
        printf("phase1_cycles (max)      : %u\n", max_phase1);
        printf("phase2_cycles (max)      : %u\n", max_phase2);
        printf("NNZ                      : %u\n", NNZ);
        printf("Total FLOPs              : %u\n", total_flops);
        printf("GFLOPS x1000             : %u\n", gflops_x1000);
        printf("L1 used (core 0) bytes   : %u\n", l1_used);

        return errors;
    }
    return 0;
}