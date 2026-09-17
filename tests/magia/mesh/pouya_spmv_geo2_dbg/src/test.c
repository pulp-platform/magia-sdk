#include <stdint.h>

#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"


//#include "sme3Da.h"
//#include "poisson3Da.h"
//#include "raefsky5.h"
#include "ex6.h"
//#include "cavity05.h"
//#include "g7jac140.h"
//#include "fxm4_6.h"
//#include "scsd8-2r.h"
//#include "e18.h"
//#include "scfxm1-2b.h"
//#include "sctap1-2b.h"
//#include "testbig.h"

//#include "test.h"


#define WAIT_MODE WFE
#define clock_freq_MHz 1000

/*
--------------------------------------------------
Debug printing for mismatched rows

Define HAVE_VALCOL_MERGE (before this point, or in the active
matrix header) only if that matrix's .h file provides:

    uint32_t valcol_l2_merge[NNZ];   // packed: [15:0]=col, [31:16]=val

If it is not defined, the packed/original-column columns are
simply skipped in the printout -- no build dependency either way.

Adjust UNPACK_COL / UNPACK_VAL if the actual bit layout differs.
--------------------------------------------------
*/
#define MAX_DETAILED_ERRORS 20

#define UNPACK_COL(w)  ((int16_t)((w) & 0xFFFF))
#define UNPACK_VAL(w)  ((int16_t)(((w) >> 16) & 0xFFFF))


/*
--------------------------------------------------
CSR value/column format

Each logical CSR entry occupies TWO uint32_t words:

    word[2*i]     = 32-bit value
    word[2*i + 1] = 32-bit column

Therefore:

    value0, column0,
    value1, column1,
    value2, column2,
    ...

Each CSR entry = 8 bytes.
--------------------------------------------------
*/

typedef uint32_t csr_word_t;

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
static uint32_t DMA_bytes[128] __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t computed_pure[128] __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t loc_nnz[128] __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t loc_row[128] __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t inner_loop[128] __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t sequential[128] __attribute__((section(".l2"), aligned(64))) = {0};

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

/*
--------------------------------------------------
debug_print_row

Reconstructs and prints, for a single mismatched row `row`:
  - which pack / which hartid owned (computed) that row
  - every (value, x_addr, x_val, partial) that fed into y[row]
  - if HAVE_VALCOL_MERGE is defined, also the original
    (pre-address-translation) column index and value for
    cross-checking against the CSR staging step

This reads only valcol_l2 / rowptr_l2 / x (via the translated
L1 address) -- all of which are either never written after
setup or written exactly once -- so the reconstruction after
the run is faithful to what the tile loop actually consumed.
--------------------------------------------------
*/
static void debug_print_row(uint32_t row,
                             uint32_t *pack_order,
                             uint32_t *pack_row_start,
                             uint32_t *pack_row_end)
{
    /* -------- find owning pack, then owning hartid -------- */
    uint32_t owner_pack = 0xFFFFFFFF;
    for (uint32_t p = 0; p < NUM_CORES; p++) {
        if (row >= pack_row_start[p] && row < pack_row_end[p]) {
            owner_pack = p;
            break;
        }
    }
    uint32_t owner_hart = 0xFFFFFFFF;
    for (uint32_t h = 0; h < NUM_CORES; h++) {
        if (pack_order[h] == owner_pack) {
            owner_hart = h;
            break;
        }
    }

    uint32_t start = rowptr_l2[row];
    uint32_t end   = rowptr_l2[row + 1];

    printf("---- row %u  (owner core = hartid %u, pack %u, nnz=%u) ----\n",
           row, owner_hart, owner_pack, end - start);

    /*

#ifdef HAVE_VALCOL_MERGE
    printf("  %5s  %10s  %12s  %10s  %8s  %10s  %12s\n",
           "idx", "value", "x_addr(L1)", "x_val", "orig_col", "orig_val", "partial");
#else
    printf("  %5s  %10s  %12s  %10s  %12s\n",
           "idx", "value", "x_addr(L1)", "x_val", "partial");
#endif
    */

    int32_t sum = 0;
    for (uint32_t j = start; j < end; j++) {
        int32_t  value   = (int32_t)valcol_l2[2 * j];
        uint32_t x_addr  = valcol_l2[2 * j + 1];
        int16_t  x_val   = *(volatile int16_t *)x_addr;
        int32_t  partial = value * (int32_t)x_val;
        printf("%u -> valued %d  x addr %x  x value %d result %d  core %u\n", j-start, value, x_addr, x_val, value * (int32_t)x_val, owner_hart);
        sum += partial;
    }

    /*

#ifdef HAVE_VALCOL_MERGE
        uint32_t packed   = valcol_l2_merge[j];
        int16_t  orig_col = UNPACK_COL(packed);
        int16_t  orig_val = UNPACK_VAL(packed);

        printf("  %5u  %10d  0x%08x  %10d  %8d  %10d  %12d\n",
               j - start, value, x_addr, x_val, orig_col, orig_val, partial);
#else
        printf("  %5u  %10d  0x%08x  %10d  %12d\n",
               j - start, value, x_addr, x_val, partial);
#endif
    }
    */

    printf("  recomputed sum = %d, y[%u] = %d, y_expected[%u] = %d\n\n",
           sum, row, y[row], row, y_expected[row]);
}


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


    uint32_t y_id = GET_Y_ID(hartid);
    uint32_t x_id = GET_X_ID(hartid);

    /*
    ==============================================================
    Start performance measurement
    ==============================================================
    */

    perf_start();
    int32_t run_time = perf_get_cycles();
    int32_t sequential_start = 0;
    /*
    ==============================================================
    Row partitioning across cores (SA-optimized pack layout)

    pack_order[hartid]  -> which pack this physical core owns
    pack_row_start[n]   -> first row owned by pack n
    pack_row_end[n]     -> one-past-last row owned by pack n
    ==============================================================
    */
    uint32_t *pack_order     = NULL;
    uint32_t *pack_row_start = NULL;
    uint32_t *pack_row_end   = NULL;
    uint32_t *pack_xid_ptr   = NULL;
    uint32_t *pack_xid_ids   = NULL;

    switch(NUM_CORES)
    {
        case 1:
            //pack_order     = pack_order_1;
            //pack_row_start = pack_row_start_1;
            //pack_row_end   = pack_row_end_1;
            //pack_xid_ptr   = pack_xid_ptr_1;
            //pack_xid_ids   = pack_xid_ids_1;
            break;

        case 4:
            //pack_order     = pack_order_4;
            //pack_row_start = pack_row_start_4;
            //pack_row_end   = pack_row_end_4;
            //pack_xid_ptr   = pack_xid_ptr_4;
            //pack_xid_ids   = pack_xid_ids_4;
            break;

        case 16:
            //pack_order     = pack_order_16;
            //pack_row_start = pack_row_start_16;
            //pack_row_end   = pack_row_end_16;
            //pack_xid_ptr   = pack_xid_ptr_16;
            //pack_xid_ids   = pack_xid_ids_16;
            break;

        case 64:
            pack_order     = pack_order_64;
            pack_row_start = pack_row_start_64;
            pack_row_end   = pack_row_end_64;
            pack_xid_ptr   = pack_xid_ptr_64;
            pack_xid_ids   = pack_xid_ids_64;
            break;

        case 256:
            //pack_order     = pack_order_256;
            //pack_row_start = pack_row_start_256;
            //pack_row_end   = pack_row_end_256;
            //pack_xid_ptr   = pack_xid_ptr_256;
            //pack_xid_ids   = pack_xid_ids_256;
            break;

        default:
            return -1;
    }

    uint32_t pack_id = pack_order[hartid];

    uint32_t start_row = pack_row_start[pack_id];
    uint32_t end_row   = pack_row_end[pack_id];
    uint32_t local_rows = end_row - start_row;

    /*
    ==============================================================
    L1 memory layout
    ==============================================================
    */

    uint32_t l1 = get_l1_base(hartid);

    /*
    ==============================================================
    x vector partitioning across cores (SA-optimized pack layout)

    Each pack owns a dedicated, offline-computed set of x ids
    (not necessarily contiguous -- it's whatever x elements the
    pack's own rows actually reference), stored CSR-style:

        pack_xid_ids[pack_xid_ptr[n] : pack_xid_ptr[n+1]]
    ==============================================================
    */
    uint32_t xid_begin = pack_xid_ptr[pack_id];
    uint32_t xid_end   = pack_xid_ptr[pack_id + 1];

    uint32_t local_x_count = xid_end - xid_begin;

    /*
    --------------------------------------------------
    x vector buffer

    Only this core's dedicated x elements are stored here now
    (local_x_count elements, not N). Rows on this core that
    reference x elements owned by other cores read them
    directly through the global L1 address encoded offline in
    valcol_l2 (transparent NoC access) -- no change needed to
    the compute loop below.
    --------------------------------------------------
    */
    uint32_t addr_x = l1;
    //printf("addr_x = 0x%x\n", addr_x);

    /*
    --------------------------------------------------
    rowptr buffer (L1, local copy)

    Each pack's rows are contiguous (start_row .. end_row), so
    the corresponding slice of rowptr_l2 -- (local_rows + 1)
    entries -- is brought into L1 as a single contiguous DMA
    block before the main tile loop starts. The main loop and
    the per-row inner loop then index this local copy instead
    of touching rowptr_l2 in L2 on every row, which is what was
    previously reintroducing an L2 access into the hot path.

    local_rowptr[k] == rowptr_l2[start_row + k],  0 <= k <= local_rows
    --------------------------------------------------
    */
    uint32_t addr_rowptr =
        addr_x + local_x_count * sizeof(int16_t);

    /*
    --------------------------------------------------
    Double buffers for streamed CSR entries
    --------------------------------------------------
    */
    uint32_t tile_buffer_bytes =
        2 * MAX_TILE_NNZ * sizeof(csr_word_t);

    uint32_t addr_valcol_buf0 =
        addr_rowptr + (local_rows + 1) * sizeof(uint32_t);

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

    volatile uint32_t *local_rowptr =
        (uint32_t*)addr_rowptr;

    volatile csr_word_t *valcol_buf[2];

    valcol_buf[0] =
        (csr_word_t*)addr_valcol_buf0;

    valcol_buf[1] =
        (csr_word_t*)addr_valcol_buf1;

    volatile int32_t *local_y =
        (int32_t*)addr_ylocal;

    /*
    ==============================================================
    Bring this core's dedicated x elements -> L1
    AND this core's rowptr slice -> L1

    The rowptr slice is contiguous, so it is launched as a
    single non-blocking DMA (channel 0) up front, then the CPU
    goes on to gather the (non-contiguous) dedicated x elements
    one-by-one while that DMA is in flight. We only block on the
    rowptr DMA completion right before the main loop needs it,
    so the two transfers overlap instead of serializing.

    pack_xid_ids[xid_begin : xid_end] is not guaranteed to be
    contiguous (it's whatever x ids this pack's rows reference),
    so a single block idma_memcpy_1d transfer doesn't apply to
    it -- each dedicated element is gathered individually
    straight from L2 into the local L1 buffer instead.
    ==============================================================
    */

    uint64_t dma_bytes = 0;
    uint32_t dma_wait_start = perf_get_cycles();

    uint32_t rowptr_bytes =
        (local_rows + 1) * sizeof(uint32_t);

    idma_memcpy_1d(
        &idma_ctrl,
        0,
        (uint32_t)&rowptr_l2[start_row],
        (uint32_t)local_rowptr,
        rowptr_bytes
    );

    dma_bytes += rowptr_bytes;

    for (uint32_t i = 0; i < local_x_count; i++) {
        uint32_t xid = pack_xid_ids[xid_begin + i];
        local_x[i] = x[xid];
    }

    dma_bytes += local_x_count * sizeof(int16_t);

    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    if (57 == get_hartid()){
        uint32_t x_addr = 0x3520028;
        int16_t x_value = *(volatile int16_t *)x_addr;
        printf("coe------------->[%u]  x value = %u\n", get_hartid(), x_value);
    }

    uint32_t dma_wait_end = perf_get_cycles();
    uint32_t dma_wait_time = dma_wait_end - dma_wait_start;
    //printf("value of L1 in local_x [0x%x] = %u\n",local_x, *local_x);

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
    dma_wait_start = perf_get_cycles();

    idma_memcpy_1d(
        &idma_ctrl,
        0,
        (uint32_t)&valcol_l2[2 * first_start_nnz],
        (uint32_t)valcol_buf[current_buf],
        2 * first_tile_nnz * sizeof(csr_word_t)
    );

    dma_bytes +=
        2 * first_tile_nnz * sizeof(csr_word_t);
    
    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    dma_wait_end = perf_get_cycles();
    dma_wait_time += (dma_wait_end - dma_wait_start);

    /*
    ==============================================================
    Main tile loop
    ==============================================================
    */
    sequential_start = perf_get_cycles() - run_time;

    uint32_t computed_time_pure = 0;
    uint32_t computed_time_pure_start = 0;
    uint32_t computed_time = 0;
    uint32_t inner_forloop_start = 0;
    uint32_t inner_forloop_time = 0;
    uint32_t compute_start = perf_get_cycles();

    uint32_t first_for_loop_iteration = num_tiles;
    uint32_t second_for_loop_iteration;
    uint32_t third_for_loop_iteration;

    for (uint32_t tile = 0;
         tile < num_tiles;
         tile++) {

        inner_forloop_start = perf_get_cycles();
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
        Current tile nnz range

        (indexed from the local rowptr copy -- tile_row_start
        and tile_row_end are already local row offsets, so no
        need to subtract start_row here)
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
                (uint32_t)&valcol_l2[2 * next_start_nnz],
                (uint32_t)valcol_buf[next_buf],
                2 * next_tile_nnz * sizeof(csr_word_t)
            );

            dma_bytes +=
                2 * next_tile_nnz * sizeof(csr_word_t);
        }

        inner_forloop_time += (perf_get_cycles() - inner_forloop_start);

        /*
        ==========================================================
        Compute current tile
        ==========================================================
        */
        second_for_loop_iteration += tile_row_end - tile_row_start;

        computed_time_pure_start = perf_get_cycles();

        for (uint32_t i = tile_row_start;
             i < tile_row_end;
             i++) {
            
            

            int32_t sum = 0;

            /*
            ------------------------------------------------------
            Convert local rowptr offsets into local tile offsets

            local_rowptr[i] and local_rowptr[i + 1] are read from
            L1 (no L2 access in this hot per-row loop anymore).
            ------------------------------------------------------
            */
            uint32_t local_start =
                local_rowptr[i] - start_nnz;

            uint32_t local_end =
                local_rowptr[i + 1] - start_nnz;

            

            third_for_loop_iteration += local_end - local_start;

            int32_t inner_log[local_end - local_start];
            int32_t inner_log2[(local_end - local_start)*4];
            

            for (uint32_t j = local_start;
                j < local_end;
                j++) {

                /*
                --------------------------------------------------
                One logical CSR entry occupies two uint32 words:

                    [2*j]     = value
                    [2*j + 1] = L1 address of x element
                --------------------------------------------------
                */

                int32_t value =
                    (int32_t)valcol_buf[current_buf][2 * j];

                uint32_t x_addr =
                    valcol_buf[current_buf][2 * j + 1];

                int16_t x_value =
                    *(volatile int16_t *)x_addr;

                //inner_log[j - local_start] = value * (int32_t)x_value;
                inner_log2[(j - local_start)*4] = value;
                inner_log2[(j - local_start)*4+1] = x_addr;
                inner_log2[(j - local_start)*4+2] = x_value;
                inner_log2[(j - local_start)*4+3] = value * (int32_t)x_value;

                /*
                --------------------------------------------------
                SpMV MAC
                --------------------------------------------------
                */
                sum +=
                    value * (int32_t)x_value;
            }

            if(sum == 200506){
                for (int i=0; i<(local_end - local_start); i++){
                    printf("[%u] ->  value[%d] x addr[%x] x value[%d] result[%d]   core[%u]\n", i, inner_log2[i*4], inner_log2[i*4+1], inner_log2[i*4+2], inner_log2[i*4+3], get_hartid());
                }
            }
            local_y[i] = sum;
        }
        computed_time_pure += (perf_get_cycles() - computed_time_pure_start);

        /*
        ==========================================================
        Wait for next tile DMA completion
        ==========================================================
        */
        if (tile + 1 < num_tiles) {
            dma_wait_start = perf_get_cycles();
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
            dma_wait_end = perf_get_cycles();
            dma_wait_time += (dma_wait_end - dma_wait_start);
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
    dma_bytes += (local_rows * sizeof(int32_t));

    dma_wait_start = perf_get_cycles();
    eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
    dma_wait_end = perf_get_cycles();
    dma_wait_time += (dma_wait_end - dma_wait_start);

    uint32_t compute_end = perf_get_cycles();
    computed_time = compute_end - compute_start;

    /*
    ==============================================================
    Final synchronization
    ==============================================================
    */
    uint32_t fsync_wait_start = perf_get_cycles();
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
    uint32_t fsync_wait_end = perf_get_cycles();
    uint32_t fsync_wait_time = fsync_wait_end - fsync_wait_start;

    uint32_t local_nnz =
        local_rowptr[local_rows] -
        local_rowptr[0];
    

    run_time_cycle[hartid] = perf_get_cycles() - run_time;
    DMA_wait_cycle[hartid] = dma_wait_time;
    fsync_wait_cycle[hartid] = fsync_wait_time;
    compute_cycle[hartid] = computed_time;
    DMA_bytes[hartid] = dma_bytes;
    computed_pure[hartid] = computed_time_pure;
    loc_nnz[hartid] = local_nnz;
    loc_row[hartid] = local_rows;
    inner_loop[hartid] = inner_forloop_time;
    sequential[hartid] = sequential_start;



    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);    



    /*

    printf(
        "core %u rows=%u nnz=%u runtime=%u compute=%u PureComp=%u fsync=%u dmaC=%u dmaB=%u dma/cycle=%u\n",
        hartid,
        local_rows,
        local_nnz,
        run_time_cycle[hartid],
        compute_cycle[hartid],
        computed_time_pure,
        fsync_wait_cycle[hartid],
        DMA_wait_cycle[hartid],
        DMA_bytes[hartid],
        DMA_bytes[hartid] / DMA_wait_cycle[hartid]
    );
    */




    /*
    ==============================================================
    Verification + metrics
    ==============================================================
    */
    if (hartid == 57) {

        /*

        printf("FIRST PART\n");
        for (int i=0; i<64; i++){
            printf("%u\n", run_time_cycle[i]);
                
        }

        printf("SECOND PART\n");
        for (int i=0; i<64; i++){
            printf("%u\n", compute_cycle[i]);
                
        }

        printf("THIRD PART\n");
        for (int i=0; i<64; i++){
            printf("%u\n", computed_pure[i]);
                
        }

        printf("FOURTH PART\n");
        for (int i=0; i<64; i++){
            printf("%u\n", fsync_wait_cycle[i]);
                
        }
            */

        int errors = 0;
        int detailed_errors = 0;

        for (int i = 0; i < M; i++) {

            if (y[i] != y_expected[i]) {
                
                printf(
                    "Mismatch at index %d: got %d expected %d\n",
                    i,
                    y[i],
                    y_expected[i]
                );

                if (detailed_errors < MAX_DETAILED_ERRORS) {
                    debug_print_row((uint32_t)i, pack_order, pack_row_start, pack_row_end);
                    detailed_errors++;
                }
                
                errors++;
            }
        }

        if (errors > MAX_DETAILED_ERRORS) {
            printf("... %d additional mismatches not printed in detail (capped at %d)\n",
                   errors - MAX_DETAILED_ERRORS, MAX_DETAILED_ERRORS);
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
        uint32_t BW =
                (
                    2 * NNZ * sizeof(csr_word_t) +
                    N * sizeof(int16_t) +
                    (M + 1) * sizeof(uint32_t) +
                    M * sizeof(int32_t)
                ) * clock_freq_MHz / max_run_time;

        /*
        --------------------------------------------------------------
        L1 Memory Occupancy
        --------------------------------------------------------------
        */

        /* per core (core 0's local_x_count used as representative sample;
           actual local x partition size varies by +/-1 element across
           cores depending on N % NUM_CORES) */
        uint32_t l1_per_core_bytes =
            local_x_count * sizeof(int16_t) +  // local_x
            (local_rows + 1) * sizeof(uint32_t) + // local_rowptr
            2 * tile_buffer_bytes +            // ping-pong buffers
            local_rows * sizeof(int32_t);      // local_y

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
            2 * NNZ * sizeof(csr_word_t) +   // value + column
            (M + 1) * sizeof(uint32_t) +     // rowptr
            N * sizeof(int16_t) +             // x
            M * sizeof(int32_t) +             // y
            M * sizeof(int32_t) +             // y_expected
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
        /*
        for (int i = 0; i < NUM_CORES; i++) {
            printf("wait to run ratio [core %d]  : %u%%\n", i, (fsync_wait_cycle[i]) * 100 / run_time_cycle[i]);
        }
        */
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

        printf("avg nnz/row = %.2f\n", (double)NNZ / M);
        printf("reuse factor = %.2f\n", (double)NNZ / N);

        printf("First loop iteration = %d\n", first_for_loop_iteration);
        printf("Second loop iteration = %d\n", second_for_loop_iteration);
        printf("Third loop iteration = %d\n", third_for_loop_iteration);

        return errors;
    }

    return 0;
}