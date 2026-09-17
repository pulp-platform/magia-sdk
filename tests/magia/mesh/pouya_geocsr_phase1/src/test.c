/*
==================================================================
 GeoCSR - Phase 1: Baseline CSR SpMV for MAGIA
==================================================================

 Goal (per project_description.txt, Phase 1 - Baseline CSR):
   - Implement a CSR parser
   - Execute standard (contiguous) row-wise partitioning
   - Measure execution time
   - Measure communication (DMA bytes)
   - Measure synchronization (fsync wait cycles)
   - Record DMA traffic

 This file intentionally contains NO optimization:
   - No tiling / double buffering (that belongs to Phase 7)
   - No geometry-aware x-vector placement (Phase 6)
   - No communication-aware clustering / mapping (Phases 3-5)

 Every core replicates the full x vector, then DMAs in exactly the
 CSR slice (row_ptr / col_idx / values) that corresponds to the
 contiguous block of rows it owns, computes locally, and DMAs the
 result back to L2. All of this is timestamped so the numbers
 produced here become the reproducible baseline that later phases
 are compared against.
==================================================================
*/

#include <stdint.h>

#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"

#include "data.h"

#define WAIT_MODE      WFE
#define clock_freq_MHz 1000

/* Map the generic CSR dimensions from data.h onto the names used below */
#define M DIM_M   /* number of matrix rows / length of y            */
#define N DIM_K   /* number of matrix columns / length of x          */

/*
==================================================================
 L2 profiling storage - one slot per possible hart
==================================================================
*/
static uint32_t HARTIDS[128]        __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t NUM_CORES           __attribute__((section(".l2"), aligned(64))) = 0;

static uint32_t run_time_cycle[128]   __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t compute_cycle[128]    __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t dma_wait_cycle[128]   __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t fsync_wait_cycle[128] __attribute__((section(".l2"), aligned(64))) = {0};
static uint64_t dma_bytes_core[128]   __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t local_rows_core[128]  __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t local_nnz_core[128]   __attribute__((section(".l2"), aligned(64))) = {0};

/* Result vector, written by every core into its own row range */
static int32_t y[M] __attribute__((section(".l2"), aligned(64))) = {0};

/*
==================================================================
 Helpers
==================================================================
*/
static uint32_t find_max(uint32_t input[], uint32_t length)
{
    uint32_t max = 0;
    for (uint32_t i = 0; i < length; i++) {
        if (input[i] > max) {
            max = input[i];
        }
    }
    return max;
}

static int32_t iabs(int32_t v)
{
    return (v < 0) ? -v : v;
}

int main(void)
{
    uint32_t hartid = get_hartid();

    /*
    ==============================================================
    Initialize IDMA / FSYNC / Event Unit
    ==============================================================
    */
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

    /*
    ==============================================================
    NUM_CORES discovery
    ==============================================================
    */
    HARTIDS[hartid] = 1;
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    if (hartid == 0) {
        for (int i = 0; i < 128; i++) {
            if (HARTIDS[i] == 1) {
                NUM_CORES++;
            }
        }
    }

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    if (hartid >= NUM_CORES) {
        return 0;
    }

    /*
    ==============================================================
    Start performance measurement
    ==============================================================
    */
    perf_start();
    uint32_t run_time_start = perf_get_cycles();

    /*
    ==============================================================
    Phase 1 - Task 1: Standard row-wise partitioning
    ------------------------------------------------------------
    Rows are split into NUM_CORES contiguous blocks. Any remainder
    (M % NUM_CORES) is spread one extra row at a time across the
    first `remainder` cores, so blocks differ in size by at most 1
    row. This is the "generic" partitioning baseline - it knows
    nothing about nnz distribution or communication cost.
    ==============================================================
    */
    uint32_t rows_per_core = M / NUM_CORES;
    uint32_t remainder     = M % NUM_CORES;

    uint32_t start_row, local_rows;
    if (hartid < remainder) {
        local_rows = rows_per_core + 1;
        start_row  = hartid * local_rows;
    } else {
        local_rows = rows_per_core;
        start_row  = remainder * (rows_per_core + 1) + (hartid - remainder) * rows_per_core;
    }
    uint32_t end_row = start_row + local_rows;

    /*
    ==============================================================
    Phase 1 - Task 2: CSR parsing
    ------------------------------------------------------------
    Parsing here means deriving, from the global row_ptr array,
    exactly which slice of col_idx / values this core needs to DMA
    in, and where its local row boundaries fall once relocated to
    L1. No offline-generated / pre-packed layout is used - row_ptr,
    col_idx and values are read straight out of data.h.
    ==============================================================
    */
    uint32_t start_nnz = row_ptr[start_row];
    uint32_t end_nnz    = row_ptr[end_row];
    uint32_t local_nnz  = end_nnz - start_nnz;

    /*
    ==============================================================
    L1 memory layout
    ==============================================================
    */
    uint32_t l1 = get_l1_base(hartid);

    uint32_t addr_x       = l1;
    uint32_t addr_colidx  = addr_x      + N * sizeof(int16_t);
    uint32_t addr_values  = addr_colidx + local_nnz * sizeof(uint32_t);
    uint32_t addr_rowptr  = addr_values + local_nnz * sizeof(int16_t);
    uint32_t addr_ylocal  = addr_rowptr + (local_rows + 1) * sizeof(uint32_t);

    volatile int16_t  *local_x      = (int16_t  *)addr_x;
    volatile uint32_t *local_colidx = (uint32_t *)addr_colidx;
    volatile int16_t  *local_values = (int16_t  *)addr_values;
    volatile uint32_t *local_rowptr = (uint32_t *)addr_rowptr;
    volatile int32_t  *local_y      = (int32_t  *)addr_ylocal;

    uint64_t dma_bytes     = 0;
    uint32_t dma_wait_time = 0;
    uint32_t t0;

    /*
    ==============================================================
    Communication: bring x, col_idx, values, row_ptr slice -> L1
    ------------------------------------------------------------
    Baseline has no x-vector placement optimization: every core
    pulls in the entire x vector, replicated. This is deliberately
    wasteful - Phase 6 is where this gets fixed.
    ==============================================================
    */
    t0 = perf_get_cycles();
    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)x, addr_x, N * sizeof(int16_t));
    dma_bytes += N * sizeof(int16_t);
    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    dma_wait_time += perf_get_cycles() - t0;

    t0 = perf_get_cycles();
    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)&col_idx[start_nnz], addr_colidx,
                   local_nnz * sizeof(uint32_t));
    dma_bytes += local_nnz * sizeof(uint32_t);
    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    dma_wait_time += perf_get_cycles() - t0;

    t0 = perf_get_cycles();
    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)&values[start_nnz], addr_values,
                   local_nnz * sizeof(int16_t));
    dma_bytes += local_nnz * sizeof(int16_t);
    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    dma_wait_time += perf_get_cycles() - t0;

    t0 = perf_get_cycles();
    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)&row_ptr[start_row], addr_rowptr,
                   (local_rows + 1) * sizeof(uint32_t));
    dma_bytes += (local_rows + 1) * sizeof(uint32_t);
    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    dma_wait_time += perf_get_cycles() - t0;

    /*
    ==============================================================
    Synchronization: barrier before compute
    ==============================================================
    */
    uint32_t f0 = perf_get_cycles();
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
    uint32_t fsync_wait_time = perf_get_cycles() - f0;

    /*
    ==============================================================
    Phase 1 - Task 3/6: Plain CSR SpMV compute
    ------------------------------------------------------------
    Straightforward row-major CSR MAC loop, one row at a time,
    accumulating in int32_t (values/x are int16_t) to avoid
    overflow, stored directly with no narrowing. No unrolling,
    no packing, no SIMD.
    ==============================================================
    */
    uint32_t c0 = perf_get_cycles();
    for (uint32_t i = 0; i < local_rows; i++) {
        int32_t sum = 0;

        uint32_t row_start = local_rowptr[i]     - start_nnz;
        uint32_t row_end   = local_rowptr[i + 1] - start_nnz;

        for (uint32_t j = row_start; j < row_end; j++) {
            uint32_t col = local_colidx[j];
            sum += (int32_t)local_values[j] * (int32_t)local_x[col];
        }

        local_y[i] = sum;
    }
    uint32_t compute_time = perf_get_cycles() - c0;

    /*
    ==============================================================
    Communication: local_y -> L2 y
    ==============================================================
    */
    t0 = perf_get_cycles();
    idma_memcpy_1d(&idma_ctrl, 1, (uint32_t)&y[start_row], addr_ylocal,
                   local_rows * sizeof(int32_t));
    dma_bytes += local_rows * sizeof(int32_t);
    eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
    dma_wait_time += perf_get_cycles() - t0;

    /*
    ==============================================================
    Final synchronization
    ==============================================================
    */
    f0 = perf_get_cycles();
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
    fsync_wait_time += perf_get_cycles() - f0;

    uint32_t run_time = perf_get_cycles() - run_time_start;

    run_time_cycle[hartid]   = run_time;
    compute_cycle[hartid]    = compute_time;
    dma_wait_cycle[hartid]   = dma_wait_time;
    fsync_wait_cycle[hartid] = fsync_wait_time;
    dma_bytes_core[hartid]   = dma_bytes;
    local_rows_core[hartid]  = local_rows;
    local_nnz_core[hartid]   = local_nnz;

    printf(
        "core %u rows=%u nnz=%u runtime=%u compute=%u dmaWait=%u fsyncWait=%u dmaBytes=%u\n",
        hartid,
        local_rows,
        local_nnz,
        run_time,
        compute_time,
        dma_wait_time,
        fsync_wait_time,
        (uint32_t)dma_bytes
    );

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /*
    ==============================================================
    Verification + aggregate metrics (core 0 only)
    ==============================================================
    */
    if (hartid == 0) {

        int errors = 0;

        for (int i = 0; i < M; i++) {
            int32_t got      = y[i];
            int32_t expected = G[i];

            if (iabs(got - expected) > 0) {
                printf("Mismatch at index %d: got %d expected %d\n", i, got, expected);
                errors++;
            }
        }

        printf("Errors: %d\n", errors);

        /*
        ----------------------------------------------------------
        Performance metrics
        ----------------------------------------------------------
        */
        uint32_t max_run_time   = find_max(run_time_cycle, 128);
        uint32_t max_dma_wait   = find_max(dma_wait_cycle, 128);
        uint32_t max_fsync_wait = find_max(fsync_wait_cycle, 128);
        uint32_t max_compute    = find_max(compute_cycle, 128);

        uint32_t total_macs  = NNZ;
        uint32_t total_flops = 2 * NNZ;

        uint32_t gflops_x1000     = (total_flops * clock_freq_MHz) / max_run_time;
        uint32_t mac_per_cycle_x1000 = (total_macs * 1000) / max_run_time;

        /*
        ----------------------------------------------------------
        Total DMA traffic across all cores (communication volume)
        ----------------------------------------------------------
        */
        uint64_t total_dma_bytes = 0;
        for (uint32_t i = 0; i < NUM_CORES; i++) {
            total_dma_bytes += dma_bytes_core[i];
        }

        uint32_t BW = (uint32_t)((total_dma_bytes * clock_freq_MHz) / max_run_time);

        /*
        ----------------------------------------------------------
        L1 / L2 memory occupancy
        ----------------------------------------------------------
        */
        uint32_t l1_per_core_bytes =
            N * sizeof(int16_t) +                        /* local_x       */
            local_nnz_core[0] * sizeof(uint32_t) +       /* local_colidx (core 0's share, illustrative) */
            local_nnz_core[0] * sizeof(int16_t) +        /* local_values                                */
            (local_rows_core[0] + 1) * sizeof(uint32_t) +/* local_rowptr                                */
            local_rows_core[0] * sizeof(int32_t);        /* local_y                                     */

        uint32_t total_l1_bytes = l1_per_core_bytes * NUM_CORES;

        uint32_t profiling_bytes =
            sizeof(HARTIDS) + sizeof(NUM_CORES) +
            sizeof(run_time_cycle) + sizeof(dma_wait_cycle) +
            sizeof(fsync_wait_cycle) + sizeof(compute_cycle) +
            sizeof(dma_bytes_core) + sizeof(local_rows_core) + sizeof(local_nnz_core);

        uint32_t l2_bytes =
            NNZ * sizeof(uint32_t) +      /* col_idx  */
            NNZ * sizeof(int16_t) +       /* values   */
            (M + 1) * sizeof(uint32_t) +  /* row_ptr  */
            N * sizeof(int16_t) +         /* x        */
            M * sizeof(int32_t) +         /* y        */
            M * sizeof(int32_t) +         /* G (expected) */
            profiling_bytes;

        uint32_t total_memory_footprint = total_l1_bytes + l2_bytes;

        /*
        ----------------------------------------------------------
        Print metrics
        ----------------------------------------------------------
        */
        printf("=== GeoCSR Phase 1 Baseline ===\n");
        printf("NUM_CORES                 : %u\n", NUM_CORES);
        printf("run_time_cycles           : %u\n", max_run_time);
        printf("dma_wait_cycles           : %u\n", max_dma_wait);
        printf("fsync_wait_cycles         : %u\n", max_fsync_wait);
        printf("compute_cycles            : %u\n", max_compute);
        printf("NNZ                       : %u\n", NNZ);
        printf("Total MACs                : %u\n", total_macs);
        printf("Total FLOPs               : %u\n", total_flops);
        printf("GFLOPS x1000              : %u\n", gflops_x1000);
        printf("MAC/Cycle x1000           : %u\n", mac_per_cycle_x1000);
        printf("Total DMA bytes           : %u\n", (uint32_t)total_dma_bytes);
        //printf("Bandwidth                 : %u B/cycle-equiv (MB/s @%uMHz)\n", BW, clock_freq_MHz);
        printf("L1 per core bytes         : %u\n", l1_per_core_bytes);
        printf("Total L1 bytes            : %u\n", total_l1_bytes);
        printf("Total L2 bytes            : %u\n", l2_bytes);
        printf("Total memory footprint    : %u\n", total_memory_footprint);
        printf("avg nnz/row               : %.2f\n", (double)NNZ / M);
        printf("reuse factor              : %.2f\n", (double)NNZ / N);

        return errors;
    }

    return 0;
}