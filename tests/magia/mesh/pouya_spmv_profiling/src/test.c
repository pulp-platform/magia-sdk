#include <stdint.h>

#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"
#include "test.h"

#define WAIT_MODE WFE
#define clock_freq_MHz 1000


/*
--------------------------------------------------
Packed CSR entry format

bits[15:0]  = signed int16 value
bits[31:16] = uint16 column
--------------------------------------------------
*/
typedef uint32_t csr_entry_t;

/*
--------------------------------------------------
Packing / unpacking
--------------------------------------------------
*/
#define GET_VALUE(x) \
    ((int16_t)((x) & 0xFFFF))

#define GET_COL(x) \
    ((uint16_t)((x) >> 16))

/*
--------------------------------------------------
L2 CSR storage

Generated offline by Python.
--------------------------------------------------
*/
static uint32_t HARTIDS[128] __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t NUM_CORES __attribute__((section(".l2"), aligned(64))) = 0;
static uint32_t run_time_cycle[128] __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t DMA_wait_cycle[128] __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t fsync_wait_cycle[128] __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t compute_cycle[128] __attribute__((section(".l2"), aligned(64))) = {0};

/*
=====================================================
Functions
=====================================================
*/

uint32_t find_max (uint32_t input[], uint32_t length)
{
    uint32_t max = 0;

    for (uint32_t i = 0; i < length; i++) {
        if (input[i] > max) {
            max = input[i];
        }
    }
    return max;
};


int main(void)
{
    uint32_t hartid = get_hartid();

    /*
    ==============================================================
    Initialize IDMA
    ==============================================================
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
    ==============================================================
    Initialize FSYNC
    ==============================================================
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
    ==============================================================
    Initialize Event Unit
    ==============================================================
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
    ===============================================================
    NUM_CORES Computation
    ================================================================
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
    int32_t run_time = perf_get_cycles();
    /*
    ==============================================================
    Row partitioning across cores
    ==============================================================
    */
    uint32_t rows_per_core =
        (M + NUM_CORES - 1) / NUM_CORES;

    uint32_t start_row =
        hartid * rows_per_core;

    uint32_t end_row =
        start_row + rows_per_core;

    if (end_row > M) {
        end_row = M;
    }

    uint32_t local_rows =
        end_row - start_row;

    /*
    ==============================================================
    L1 memory layout
    ==============================================================
    */
    uint32_t l1 = get_l1_base(hartid);

    /*
    --------------------------------------------------
    x vector buffer
    --------------------------------------------------
    */
    uint32_t addr_x = l1;

    /*
    --------------------------------------------------
    Double buffers for streamed CSR entries
    --------------------------------------------------
    */
    uint32_t tile_buffer_bytes =
        MAX_TILE_NNZ * sizeof(csr_entry_t);

    uint32_t addr_valcol_buf0 =
        addr_x + N * sizeof(int32_t);

    uint32_t addr_valcol_buf1 =
        addr_valcol_buf0 +
        tile_buffer_bytes;

    /*
    --------------------------------------------------
    y local buffer
    --------------------------------------------------
    */
    uint32_t addr_ylocal =
        addr_valcol_buf1 +
        tile_buffer_bytes;

    /*
    ==============================================================
    Local pointers
    ==============================================================
    */
    volatile int16_t *local_x =
        (int16_t*)addr_x;

    volatile csr_entry_t *valcol_buf[2];

    valcol_buf[0] =
        (csr_entry_t*)addr_valcol_buf0;

    valcol_buf[1] =
        (csr_entry_t*)addr_valcol_buf1;

    volatile int32_t *local_y =
        (int32_t*)addr_ylocal;

    /*
    ==============================================================
    Bring x vector -> L1
    ==============================================================
    */
    uint32_t dma_wait_start = perf_get_cycles();
    idma_memcpy_1d(
        &idma_ctrl,
        0,
        (uint32_t)x,
        addr_x,
        N * sizeof(int32_t)
    );
    
    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    uint32_t dma_wait_end = perf_get_cycles();
    uint32_t dma_wait_time = dma_wait_end - dma_wait_start;

    /*
    ==============================================================
    Synchronization
    ==============================================================
    */
    uint32_t fsync_wait_start = perf_get_cycles();
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
    uint32_t fsync_wait_end = perf_get_cycles();
    uint32_t fsync_wait_time = fsync_wait_end - fsync_wait_start;
    /*
    ==============================================================
    Double-buffered tiled SpMV
    ==============================================================
    */

    uint32_t current_buf = 0;
    uint32_t next_buf    = 1;

    /*
    --------------------------------------------------
    Total number of tiles
    --------------------------------------------------
    */
    uint32_t num_tiles =
        (local_rows + TILE_ROWS - 1)
        / TILE_ROWS;

    /*
    ==============================================================
    Prefetch first tile
    ==============================================================
    */
    uint32_t first_tile_rows =
        (local_rows > TILE_ROWS)
        ? TILE_ROWS
        : local_rows;

    uint32_t first_global_row_end =
        start_row + first_tile_rows;

    uint32_t first_start_nnz =
        rowptr_l2[start_row];

    uint32_t first_end_nnz =
        rowptr_l2[first_global_row_end];

    uint32_t first_tile_nnz =
        first_end_nnz - first_start_nnz;

    /*
    --------------------------------------------------
    DMA first tile
    --------------------------------------------------
    */
    dma_wait_start = perf_get_cycles();
    idma_memcpy_1d(
        &idma_ctrl,
        0,
        (uint32_t)&valcol_l2[first_start_nnz],
        (uint32_t)valcol_buf[current_buf],
        first_tile_nnz * sizeof(csr_entry_t)
    );
    
    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    dma_wait_end = perf_get_cycles();
    dma_wait_time += dma_wait_end - dma_wait_start;

    /*
    ==============================================================
    Main tile loop
    ==============================================================
    */
    uint32_t computed_time = 0;
    uint32_t compute_start = perf_get_cycles();

    for (uint32_t tile = 0;
         tile < num_tiles;
         tile++) {

        /*
        --------------------------------------------------
        Current tile row range (local)
        --------------------------------------------------
        */
        uint32_t tile_row_start =
            tile * TILE_ROWS;

        uint32_t tile_row_end =
            tile_row_start + TILE_ROWS;

        if (tile_row_end > local_rows) {
            tile_row_end = local_rows;
        }

        /*
        --------------------------------------------------
        Convert to global rows
        --------------------------------------------------
        */
        uint32_t global_tile_start =
            start_row + tile_row_start;

        uint32_t global_tile_end =
            start_row + tile_row_end;

        /*
        --------------------------------------------------
        Current tile nnz range
        --------------------------------------------------
        */
        uint32_t start_nnz =
            rowptr_l2[global_tile_start];

        uint32_t end_nnz =
            rowptr_l2[global_tile_end];

        uint32_t tile_nnz =
            end_nnz - start_nnz;
        /*
        ==========================================================
        Launch DMA for NEXT tile
        ==========================================================
        */
        if (tile + 1 < num_tiles) {

            uint32_t next_row_start =
                (tile + 1) * TILE_ROWS;

            uint32_t next_row_end =
                next_row_start + TILE_ROWS;

            if (next_row_end > local_rows) {
                next_row_end = local_rows;
            }

            uint32_t next_global_start =
                start_row + next_row_start;

            uint32_t next_global_end =
                start_row + next_row_end;

            uint32_t next_start_nnz =
                rowptr_l2[next_global_start];

            uint32_t next_end_nnz =
                rowptr_l2[next_global_end];

            uint32_t next_tile_nnz =
                next_end_nnz - next_start_nnz;

            /*
            ------------------------------------------------------
            Non-blocking DMA launch
            ------------------------------------------------------
            */
            idma_memcpy_1d(
                &idma_ctrl,
                0,
                (uint32_t)&valcol_l2[next_start_nnz],
                (uint32_t)valcol_buf[next_buf],
                next_tile_nnz * sizeof(csr_entry_t)
            );
        }

        /*
        ==========================================================
        Compute current tile
        ==========================================================
        */

        for (uint32_t i = tile_row_start;
             i < tile_row_end;
             i++) {

            uint32_t global_row =
                start_row + i;

            int32_t sum = 0;

            /*
            ------------------------------------------------------
            Convert global CSR offsets into local tile offsets
            ------------------------------------------------------
            */
            uint32_t local_start =
                rowptr_l2[global_row] - start_nnz;

            uint32_t local_end =
                rowptr_l2[global_row + 1] - start_nnz;

            for (uint32_t j = local_start;
                 j < local_end;
                 j++) {

                /*
                --------------------------------------------------
                Single 32-bit load
                --------------------------------------------------
                */
                csr_entry_t packed =
                    valcol_buf[current_buf][j];

                /*
                --------------------------------------------------
                Unpack value and column
                --------------------------------------------------
                */
                int16_t value =
                    GET_VALUE(packed);

                uint16_t col =
                    GET_COL(packed);

                /*
                --------------------------------------------------
                SpMV MAC
                --------------------------------------------------
                */
                sum +=
                    ((int32_t)value) *
                    ((int32_t)local_x[col]);
            }

            local_y[i] = sum;
        }

        /*
        ==========================================================
        Wait for next tile DMA completion
        ==========================================================
        */
        if (tile + 1 < num_tiles) {
            dma_wait_start = perf_get_cycles();
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
            dma_wait_end = perf_get_cycles();
            dma_wait_time += dma_wait_end - dma_wait_start;
        }

        /*
        ==========================================================
        Swap ping-pong buffers
        ==========================================================
        */
        current_buf ^= 1;
        next_buf    ^= 1;
    }
    
    /*
    ==============================================================
    DMA local y -> global y
    ==============================================================
    */
    idma_memcpy_1d(
        &idma_ctrl,
        1,
        (uint32_t)(y + start_row),
        addr_ylocal,
        local_rows * sizeof(int32_t)
    );
    dma_wait_start = perf_get_cycles();
    eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
    dma_wait_end = perf_get_cycles();
    dma_wait_time += dma_wait_end - dma_wait_start;

    uint32_t compute_end = perf_get_cycles();
    computed_time = compute_end - compute_start;

    /*
    ==============================================================
    Final synchronization
    ==============================================================
    */
    fsync_wait_start = perf_get_cycles();
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
    fsync_wait_end = perf_get_cycles();
    fsync_wait_time += fsync_wait_end - fsync_wait_start;

    run_time_cycle[hartid] = perf_get_cycles() - run_time;
    DMA_wait_cycle[hartid] = dma_wait_time;
    fsync_wait_cycle[hartid] = fsync_wait_time;
    compute_cycle[hartid] = computed_time;

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
    /*
    ==============================================================
    Verification + metrics
    ==============================================================
    */
    if (hartid == 0) {

        int errors = 0;

        for (int i = 0; i < M; i++) {

            if (y[i] != y_expected[i]) {

                printf(
                    "Mismatch at index %d: got %d expected %d\n",
                    i,
                    y[i],
                    y_expected[i]
                );

                errors++;
            }
        }

        printf("Errors: %d\n", errors);

        /*
        --------------------------------------------------------------
        Performance metrics
        --------------------------------------------------------------
        */

        uint32_t max_run_time = find_max(run_time_cycle, 128);
        uint32_t max_dma_wait = find_max(DMA_wait_cycle, 128);
        uint32_t max_fsync_wait = find_max(fsync_wait_cycle, 128);
        uint32_t max_compute = find_max(compute_cycle, 128);

        uint32_t total_macs  = NNZ;

        uint32_t total_flops = 2 * NNZ;

        uint32_t gflops_x1000 =
            (total_flops * clock_freq_MHz) /
            max_run_time;

        uint32_t mac_per_cycle_x1000 =
            (total_macs * 1000) /
            max_run_time;
        /*
        --------------------------------------------------------------
        Dense equivalent size
        --------------------------------------------------------------
        */
        uint32_t dense_matrix_bytes =
            M * N * sizeof(int16_t);

        /*
        --------------------------------------------------------------
        Bandwidth
        --------------------------------------------------------------
        */
        //              CSR entries              +      x vector       +        rowptr              +         y
        uint32_t BW = (NNZ * sizeof(csr_entry_t) + N * sizeof(int16_t) + (M + 1) * sizeof(uint32_t) + M * sizeof(int32_t)) * clock_freq_MHz / max_run_time;

        /*
        --------------------------------------------------------------
        L1 Memory Occupancy
        --------------------------------------------------------------
        */

        /* per core */
        uint32_t l1_per_core_bytes =
            N * sizeof(int16_t) +          // local_x
            2 * tile_buffer_bytes +        // ping-pong buffers
            local_rows * sizeof(int32_t);  // local_y

        /* whole chip */
        uint32_t total_l1_bytes =
            l1_per_core_bytes * NUM_CORES;

        /*
        --------------------------------------------------------------
        L2 Memory Occupancy
        --------------------------------------------------------------
        */

        uint32_t profiling_bytes =
            sizeof(HARTIDS) +
            sizeof(NUM_CORES) +
            sizeof(run_time_cycle) +
            sizeof(DMA_wait_cycle) +
            sizeof(fsync_wait_cycle) +
            sizeof(compute_cycle);

        uint32_t l2_bytes =
            NNZ * sizeof(csr_entry_t) +        // packed matrix
            (M + 1) * sizeof(uint32_t) +       // rowptr
            N * sizeof(int16_t) +              // x
            M * sizeof(int32_t) +              // y
            M * sizeof(int32_t) +              // y_expected
            profiling_bytes;

        /*
        --------------------------------------------------------------
        Total Occupancy
        --------------------------------------------------------------
        */

        uint32_t total_memory_footprint =
            total_l1_bytes +
            l2_bytes;

        /*
        --------------------------------------------------------------
        Print metrics
        --------------------------------------------------------------
        */
        printf("run_time_cycles          : %u\n", max_run_time);

        printf("dma_wait_cycles          : %u\n", max_dma_wait);

        printf("fsync_wait_cycles        : %u\n", max_fsync_wait);

        printf("compute_cycles           : %u\n", max_compute);

        printf("NNZ                       : %u\n", NNZ);

        printf("Total MACs                : %u\n", total_macs);

        printf("Total FLOPs               : %u\n", total_flops);

        printf("GFLOPS x1000              : %u\n", gflops_x1000);

        printf("MAC/Cycle x1000           : %u\n", mac_per_cycle_x1000);

        printf("Bandwidth                 : %u MB/second\n", BW);

        printf("L1 per core bytes         : %u KB\n", l1_per_core_bytes / 1024);

        printf("Total L1 bytes           : %u KB\n", total_l1_bytes / 1024);

        printf("Total L2 bytes           : %u KB\n", l2_bytes / 1024);

        printf("Total memory footprint    : %u KB\n", total_memory_footprint / 1024);

        return errors;
    }

    return 0;
}
