/*
    ==============================================================
    SpMV on MAGIA using IDMA + Double Buffering
    ==============================================================
    
    Optimized Version:
    ------------------
    1. CSR sparse format
    2. 32-bit packed entries:
    
        bits[15:0]  = value
        bits[31:16] = column index
    
    3. Double buffering on L1 for valcol
    4. Overlap IDMA transfers with SpMV computation
    5. Row-aligned tiling (CSR aware)
    6. Configurable tile size

    Dataflow:
    ---------
    L2 CSR
        ↓ IDMA
    L1 ping-pong buffers
        ↓
    SpMV compute

    Author:
    -------
    Pouya Shirinshahrakfard
*/

#include <stdint.h>

#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"
#include "test.h"

#define WAIT_MODE WFE
#define NUM_CORES 4

/*
--------------------------------------------------
Configurable tile size
--------------------------------------------------

Number of rows per streamed tile.

Tune this parameter according to:
- L1 size
- DMA bandwidth
- sparsity pattern
*/
#define TILE_ROWS 64

#define ROWS_PER_CORE ((M + NUM_CORES - 1) / NUM_CORES)

/*
--------------------------------------------------
Packed CSR entry format
--------------------------------------------------

32-bit packed layout:

bits[15:0]  = value
bits[31:16] = column
*/
typedef uint32_t csr_entry_t;

/*
--------------------------------------------------
Packing macros
--------------------------------------------------
*/
#define PACK_ENTRY(value, col) \
    ((((uint32_t)(col)) << 16) | ((uint32_t)(value)))

#define GET_VALUE(x) \
    ((uint16_t)((x) & 0xFFFF))

#define GET_COL(x) \
    ((uint16_t)((x) >> 16))

/*
--------------------------------------------------
L2 CSR storage
--------------------------------------------------

One 32-bit entry per nonzero.
*/
static csr_entry_t valcol_l2[M * N]
__attribute__((section(".l2"), aligned(64)));

static uint16_t rowptr_l2[
    NUM_CORES * (ROWS_PER_CORE + 1)
] __attribute__((section(".l2"), aligned(64)));

static inline uint32_t get_raw(const float val)
{
    uint32_t raw;
    memcpy(&raw, &val, sizeof(raw));
    return raw;
}

int main(void)
{
    uint32_t hartid = get_hartid();

    if (hartid >= NUM_CORES) {
        return 0;
    }

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
    Dense matrix buffer
    --------------------------------------------------
    */
    uint32_t addr_A = l1;

    /*
    --------------------------------------------------
    x vector buffer
    --------------------------------------------------
    */
    uint32_t addr_x =
        addr_A + local_rows * N * sizeof(uint16_t);

    /*
    --------------------------------------------------
    Double buffers for streamed CSR entries
    --------------------------------------------------

    Sparse-aware allocation:
    buffer sized by maximum tile NNZ
    computed offline by Python.
    --------------------------------------------------
    */
    uint32_t tile_buffer_bytes =
        TILE_ROWS * MAX_TILE_NNZ * sizeof(csr_entry_t);

    uint32_t addr_valcol_buf0 =
        addr_x + N * sizeof(uint16_t);

    uint32_t addr_valcol_buf1 =
        addr_valcol_buf0 +
        tile_buffer_bytes;

    /*
    --------------------------------------------------
    rowptr buffer
    --------------------------------------------------
    */
    uint32_t addr_rowptr =
        addr_valcol_buf1 +
        tile_buffer_bytes;

    /*
    --------------------------------------------------
    y local buffer
    --------------------------------------------------
    */
    uint32_t addr_ylocal =
        addr_rowptr +
        (local_rows + 1) * sizeof(uint16_t);

    /*
    ==============================================================
    DMA dense matrix -> L1
    ==============================================================
    */
    uint32_t axi_A =
        (uint32_t)A_dense +
        start_row * N * sizeof(uint16_t);

    idma_memcpy_2d(
        &idma_ctrl,
        0,
        axi_A,
        addr_A,
        N * sizeof(uint16_t),
        N * sizeof(uint16_t),
        local_rows
    );

    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

    /*
    ==============================================================
    Local pointers
    ==============================================================
    */
    volatile uint16_t *local_A =
        (uint16_t*)addr_A;

    volatile uint16_t *local_x =
        (uint16_t*)addr_x;

    volatile csr_entry_t *valcol_buf[2];

    valcol_buf[0] =
        (csr_entry_t*)addr_valcol_buf0;

    valcol_buf[1] =
        (csr_entry_t*)addr_valcol_buf1;

    volatile uint16_t *local_rowptr =
        (uint16_t*)addr_rowptr;

    volatile uint16_t *local_y =
        (uint16_t*)addr_ylocal;

    /*
    ==============================================================
    Dense -> CSR conversion
    ==============================================================
    */
    uint32_t nnz = 0;

    local_rowptr[0] = 0;

    for (uint32_t i = 0; i < local_rows; i++) {

        uint16_t *row =
            (uint16_t*)&local_A[i * N];

        for (uint32_t j = 0; j < N; j++) {

            uint16_t value = row[j];

            if (value != 0) {

                /*
                --------------------------------------------------
                Pack:
                bits[15:0]  = value
                bits[31:16] = column
                --------------------------------------------------
                */
                valcol_buf[0][nnz] =
                    PACK_ENTRY(value, j);

                nnz++;
            }
        }

        local_rowptr[i + 1] = nnz;
    }

    /*
    ==============================================================
    Store CSR into L2
    ==============================================================
    */
    uint32_t l2_offset =
        start_row * N;

    uint32_t addr_valcol_l2 =
        (uint32_t)&valcol_l2[l2_offset];

    uint32_t addr_rowptr_l2 =
        (uint32_t)&rowptr_l2[
            hartid * (ROWS_PER_CORE + 1)
        ];

    /*
    --------------------------------------------------
    Store packed CSR entries
    --------------------------------------------------
    */
    idma_memcpy_1d(
        &idma_ctrl,
        1,
        addr_valcol_l2,
        addr_valcol_buf0,
        nnz * sizeof(csr_entry_t)
    );

    eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);

    /*
    --------------------------------------------------
    Store rowptr
    --------------------------------------------------
    */
    idma_memcpy_1d(
        &idma_ctrl,
        1,
        addr_rowptr_l2,
        addr_rowptr,
        (local_rows + 1) * sizeof(uint16_t)
    );

    eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);

    /*
    ==============================================================
    Global synchronization
    ==============================================================
    */
    fsync_sync_global(&fsync_ctrl);

    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /*
    ==============================================================
    Start performance measurement
    ==============================================================
    */
    perf_start();

    int start = perf_get_cycles();

    /*
    ==============================================================
    Bring rowptr -> L1
    ==============================================================
    */
    idma_memcpy_1d(
        &idma_ctrl,
        0,
        addr_rowptr_l2,
        addr_rowptr,
        (local_rows + 1) * sizeof(uint16_t)
    );

    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

    /*
    ==============================================================
    Bring x vector -> L1
    ==============================================================
    */
    idma_memcpy_1d(
        &idma_ctrl,
        0,
        (uint32_t)x,
        addr_x,
        N * sizeof(uint16_t)
    );

    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

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

    uint32_t first_start_nnz =
        local_rowptr[0];

    uint32_t first_end_nnz =
        local_rowptr[first_tile_rows];

    uint32_t first_tile_nnz =
        first_end_nnz - first_start_nnz;

    /*
    --------------------------------------------------
    DMA first tile
    --------------------------------------------------
    */
    idma_memcpy_1d(
        &idma_ctrl,
        0,
        addr_valcol_l2 +
        first_start_nnz * sizeof(csr_entry_t),

        (uint32_t)valcol_buf[current_buf],

        first_tile_nnz * sizeof(csr_entry_t)
    );

    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

    /*
    ==============================================================
    Main tile loop
    ==============================================================
    */
    for (uint32_t tile = 0;
         tile < num_tiles;
         tile++) {

        /*
        --------------------------------------------------
        Current tile row range
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
        Current tile nnz range
        --------------------------------------------------
        */
        uint32_t start_nnz =
            local_rowptr[tile_row_start];

        uint32_t end_nnz =
            local_rowptr[tile_row_end];

        uint32_t tile_nnz =
            end_nnz - start_nnz;

        /*
        ==========================================================
        Launch DMA for NEXT tile (overlap region)
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

            uint32_t next_start_nnz =
                local_rowptr[next_row_start];

            uint32_t next_end_nnz =
                local_rowptr[next_row_end];

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
                addr_valcol_l2 +
                next_start_nnz * sizeof(csr_entry_t),

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

            uint32_t sum = 0;

            /*
            ------------------------------------------------------
            Convert global CSR offsets into local tile offsets
            ------------------------------------------------------
            */
            uint32_t local_start =
                local_rowptr[i] - start_nnz;

            uint32_t local_end =
                local_rowptr[i + 1] - start_nnz;

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
                uint16_t value =
                    GET_VALUE(packed);

                uint16_t col =
                    GET_COL(packed);

                /*
                --------------------------------------------------
                SpMV MAC
                --------------------------------------------------
                */
                sum += value * local_x[col];
            }

            local_y[i] = (uint16_t)sum;
        }

        /*
        ==========================================================
        Wait for next tile DMA completion
        ==========================================================
        */
        if (tile + 1 < num_tiles) {

            eu_idma_wait_a2o(
                &eu_ctrl,
                WAIT_MODE
            );
        }

        /*
        ==========================================================
        Swap ping-pong buffers
        ==========================================================
        */
        current_buf ^= 1;
        next_buf    ^= 1;
    }

    int end = perf_get_cycles();

    float cycles = end - start;

    printf("Core[%d] Cycles: %d\n", hartid, end - start);

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
        local_rows * sizeof(uint16_t)
    );

    eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);

    /*
    ==============================================================
    Final synchronization
    ==============================================================
    */
    fsync_sync_global(&fsync_ctrl);

    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /*
    ==============================================================
    Verification
    ==============================================================
    */
    if (hartid == 0) {

        int errors = 0;

        for (int i = 0; i < M; i++) {

            //printf("y[%d] = %d | y_expected[%d] = %d\n", i, y[i], i, y_expected[i]);

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
        Total MAC operations
        --------------------------------------------------------------
        One nonzero => one MAC:
            sum += value * x[col]

        MACs  = nnz
        FLOPs = 2 * nnz
        --------------------------------------------------------------
        */
        uint32_t total_macs  = nnz;

        uint32_t total_flops = 2 * nnz;

        /*
        --------------------------------------------------------------
        GFLOPS (fixed-point x1000)
        --------------------------------------------------------------

        GFLOPS =
            FLOPs * freq_GHz / cycles

        We use integer fixed-point arithmetic:
            value * 1000

        Example:
            1.234 GFLOPS
            stored as 1234
        --------------------------------------------------------------
        */
        uint32_t freq_MHz = 2000;

        uint32_t gflops_x1000 =
            (total_flops * freq_MHz) /
            cycles;

        /*
        --------------------------------------------------------------
        MACs per cycle (fixed-point x1000)
        --------------------------------------------------------------
        */
        uint32_t mac_per_cycle_x1000 =
            (total_macs * 1000) /
            cycles;

        /*
        --------------------------------------------------------------
        TRUE occupied memory footprint
        (after compression)
        --------------------------------------------------------------

        This reports ONLY useful occupied sparse data,
        not oversized static allocations.
        --------------------------------------------------------------
        */

        /*
        --------------------------------------------------------------
        Compressed CSR storage
        --------------------------------------------------------------

        Packed format:
        - one uint32_t per nonzero
        - one uint16_t rowptr entry per row
        --------------------------------------------------------------
        */
        uint32_t compressed_matrix_bytes =
            nnz * sizeof(csr_entry_t) +
            (M + 1) * sizeof(uint16_t);

        /*
        --------------------------------------------------------------
        Runtime vectors
        --------------------------------------------------------------
        */
        uint32_t vector_bytes =
            N * sizeof(uint16_t) +   /* x */
            M * sizeof(uint16_t);    /* y */

        /*
        --------------------------------------------------------------
        Double-buffered tile storage
        --------------------------------------------------------------
        */
        uint32_t buffering_bytes =
            2 * tile_buffer_bytes;

        /*
        --------------------------------------------------------------
        Total useful occupied memory
        --------------------------------------------------------------
        */
        uint32_t occupied_memory_bytes =
            compressed_matrix_bytes +
            vector_bytes +
            buffering_bytes;

        /*
        --------------------------------------------------------------
        Compression ratio vs dense
        --------------------------------------------------------------
        */
        uint32_t dense_matrix_bytes =
            M * N * sizeof(uint16_t);

        uint32_t compression_ratio_x100 =
            (dense_matrix_bytes * 100) /
            compressed_matrix_bytes;

        /*
        --------------------------------------------------------------
        Print metrics
        --------------------------------------------------------------
        */
        printf("NNZ                       : %u\n", nnz);

        printf("Total MACs                : %u\n", total_macs);

        printf("Total FLOPs               : %u\n", total_flops);

        printf("GFLOPS x1000              : %u\n", gflops_x1000);

        printf("MAC/Cycle x1000           : %u\n", mac_per_cycle_x1000);

        printf("Dense Matrix Size         : %u bytes (%u.%02u KB)\n",
            dense_matrix_bytes,
            dense_matrix_bytes / 1024,
            ((dense_matrix_bytes % 1024) * 100) / 1024);

        printf("Compressed Matrix Size    : %u bytes (%u.%02u KB)\n",
            compressed_matrix_bytes,
            compressed_matrix_bytes / 1024,
            ((compressed_matrix_bytes % 1024) * 100) / 1024);

        printf("Runtime Buffer Size       : %u bytes (%u.%02u KB)\n",
            buffering_bytes,
            buffering_bytes / 1024,
            ((buffering_bytes % 1024) * 100) / 1024);

        printf("Vector Storage Size       : %u bytes (%u.%02u KB)\n",
            vector_bytes,
            vector_bytes / 1024,
            ((vector_bytes % 1024) * 100) / 1024);

        printf("Total Occupied Memory     : %u bytes (%u.%02u KB)\n",
            occupied_memory_bytes,
            occupied_memory_bytes / 1024,
            ((occupied_memory_bytes % 1024) * 100) / 1024);

        printf("Compression Ratio         : %u.%02ux\n",
            compression_ratio_x100 / 100,
            compression_ratio_x100 % 100);
        
        return errors;
    }

    return 0;
}