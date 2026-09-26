#include <stdint.h>

#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"

#include "data.h"

#include "onnx_spmv_params.h"
#include "onnx_spmv_mem_layout.h"
#include "spatz_spmv_task_bin.h"

#define WAIT_MODE      WFE
#define clock_freq_MHz 1450

typedef uint32_t csr_word_t;

/* Profiling arrays in L2 */
static uint32_t HARTIDS[128]         __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t NUM_CORES            __attribute__((section(".l2"), aligned(64))) = 0;
static uint32_t run_time_cycle[128]  __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t DMA_wait_cycle[128]  __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t fsync_wait_cycle[128] __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t compute_cycle[128]   __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t spatz_wait_cycle[128] __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t spatz_calls[128]     __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t diff_finish[128]     __attribute__((section(".l2"), aligned(64))) = {0};

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

/* Launch SPATZ on the vectors already written in L1 and wait for it. */
static inline void spatz_dot(eu_controller_t *eu_ctrl,
                             volatile spmv_params_t *p,
                             uint32_t len)
{
    p->len = len;
    if (get_hartid() == 63) printf("63: run len=%u\n", len);
    
    if(get_hartid() == 63) printf("spatz goes for run\n");
    
    spatz_run_task_with_params(SPMV_TASK, (uint32_t)p);

    eu_spatz_wait(eu_ctrl, WAIT_MODE);

    //wait_nop(10);

    if(get_hartid() == 63) printf("spatz finished\n");
    
    if (get_hartid() == 63) printf("63: done res=%d\n", *(volatile int32_t *)p->addr_res);
}

int main(void)
{
    uint32_t hartid = get_hartid();

    /* ---------------- iDMA ---------------- */
    idma_config_t idma_cfg = {.hartid = hartid};
    idma_controller_t idma_ctrl = {
        .base = NULL, .cfg = &idma_cfg, .api = &idma_api,
    };
    idma_init(&idma_ctrl);

    /* ---------------- FSYNC ---------------- */
    fsync_config_t fsync_cfg = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {
        .base = NULL, .cfg = &fsync_cfg, .api = &fsync_api,
    };
    fsync_init(&fsync_ctrl);

    /* ---------------- Event Unit ---------------- */
    eu_config_t eu_cfg = {.hartid = hartid};
    eu_controller_t eu_ctrl = {
        .base = NULL, .cfg = &eu_cfg, .api = &eu_api,
    };
    eu_init(&eu_ctrl);
    eu_clear_events(0xFFFFFFFF);
    eu_idma_init(&eu_ctrl, 0);
    eu_fsync_init(&eu_ctrl, 0);

    /* ---------------- SPATZ ---------------- */
    eu_spatz_init(&eu_ctrl, 0);
    spatz_init(SPATZ_BINARY_START);

    /* ---------------- NUM_CORES ---------------- */
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
        spatz_clk_dis();
        return 0;
    }

    perf_start();
    int32_t run_time = perf_get_cycles();
    int32_t sequential_start = 0;
    (void)sequential_start;

    /* ---------------- Partitioning ---------------- */
    uint32_t *pack_order     = NULL;
    uint32_t *pack_row_start = NULL;
    uint32_t *pack_row_end   = NULL;
    uint32_t *pack_xid_ptr   = NULL;
    uint32_t *pack_xid_ids   = NULL;

    switch (NUM_CORES) {
        case 64:
            pack_order     = pack_order_64;
            pack_row_start = pack_row_start_64;
            pack_row_end   = pack_row_end_64;
            pack_xid_ptr   = pack_xid_ptr_64;
            pack_xid_ids   = pack_xid_ids_64;
            break;
        /* add other core counts here like before */
        default:
            spatz_clk_dis();
            return -1;
    }

    uint32_t pack_id    = pack_order[hartid];
    uint32_t start_row  = pack_row_start[pack_id];
    uint32_t end_row    = pack_row_end[pack_id];
    uint32_t local_rows = end_row - start_row;

    uint32_t xid_begin      = pack_xid_ptr[pack_id];
    uint32_t xid_end        = pack_xid_ptr[pack_id + 1];
    uint32_t local_x_count  = xid_end - xid_begin;

    /* ---------------- L1 layout ---------------- */
    uint32_t l1 = get_l1_base(hartid);

    uint32_t tile_buffer_bytes = 2 * MAX_TILE_NNZ * sizeof(csr_word_t);

    spmv_l1_layout_t L = spmv_l1_layout(l1, local_x_count, local_rows,
                                        tile_buffer_bytes);

    volatile int16_t    *local_x      = (int16_t *)L.x;
    volatile uint32_t   *local_rowptr = (uint32_t *)L.rowptr;
    volatile csr_word_t *valcol_buf[2];
    valcol_buf[0] = (csr_word_t *)L.valcol0;
    valcol_buf[1] = (csr_word_t *)L.valcol1;
    volatile int32_t *local_y = (int32_t *)L.y;

    /* SPATZ buffers */
    volatile spmv_params_t *sp_params = (volatile spmv_params_t *)L.params;
    volatile int32_t *vec_val = (int32_t *)L.vec_val;
    volatile int32_t *vec_x   = (int32_t *)L.vec_x;
    volatile int32_t *sp_res  = (int32_t *)L.result;

    sp_params->addr_val = L.vec_val;
    sp_params->addr_x   = L.vec_x;
    sp_params->addr_res = L.result;
    sp_params->len      = 0;

    /* ---------------- rowptr + x -> L1 ---------------- */
    uint32_t dma_wait_start = perf_get_cycles();

    uint32_t rowptr_bytes = (local_rows + 1) * sizeof(uint32_t);

    idma_memcpy_1d(&idma_ctrl, 0,
                   (uint32_t)&rowptr_l2[start_row],
                   (uint32_t)local_rowptr,
                   rowptr_bytes);

    for (uint32_t i = 0; i < local_x_count; i++) {
        uint32_t xid = pack_xid_ids[xid_begin + i];
        local_x[i] = x[xid];
    }

    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

    uint32_t dma_wait_time = perf_get_cycles() - dma_wait_start;

    /* ---------------- Prefetch first tile ---------------- */
    uint32_t current_buf = 0;
    uint32_t next_buf    = 1;

    uint32_t num_tiles = (local_rows + TILE_ROWS - 1) / TILE_ROWS;

    uint32_t first_tile_rows = (local_rows > TILE_ROWS) ? TILE_ROWS : local_rows;
    uint32_t first_start_nnz = local_rowptr[0];
    uint32_t first_end_nnz   = local_rowptr[first_tile_rows];
    uint32_t first_tile_nnz  = first_end_nnz - first_start_nnz;

    dma_wait_start = perf_get_cycles();

    idma_memcpy_1d(&idma_ctrl, 0,
                   (uint32_t)&valcol_l2[2 * first_start_nnz],
                   (uint32_t)valcol_buf[current_buf],
                   2 * first_tile_nnz * sizeof(csr_word_t));

    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    dma_wait_time += perf_get_cycles() - dma_wait_start;

    uint32_t fsync_wait_start = perf_get_cycles();
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
    uint32_t fsync_wait_time = perf_get_cycles() - fsync_wait_start;

    /* ---------------- Main tile loop ---------------- */
    uint32_t spatz_wait_time = 0;
    uint32_t n_spatz_calls   = 0;
    uint32_t compute_start   = perf_get_cycles();


    /*this if is for test, 
    you change its value 
    to test the performance of each core*/
    if(hartid == 63){

        for (uint32_t tile = 0; tile < num_tiles; tile++) {

            uint32_t tile_row_start = tile * TILE_ROWS;
            uint32_t tile_row_end   = tile_row_start + TILE_ROWS;
            if (tile_row_end > local_rows) {
                tile_row_end = local_rows;
            }

            uint32_t start_nnz = local_rowptr[tile_row_start];

            /* ---- launch DMA for NEXT tile (non-blocking) ---- */
            if (tile + 1 < num_tiles) {
                uint32_t next_row_start = (tile + 1) * TILE_ROWS;
                uint32_t next_row_end   = next_row_start + TILE_ROWS;
                if (next_row_end > local_rows) {
                    next_row_end = local_rows;
                }
                uint32_t next_start_nnz = local_rowptr[next_row_start];
                uint32_t next_end_nnz   = local_rowptr[next_row_end];
                uint32_t next_tile_nnz  = next_end_nnz - next_start_nnz;

                idma_memcpy_1d(&idma_ctrl, 0,
                            (uint32_t)&valcol_l2[2 * next_start_nnz],
                            (uint32_t)valcol_buf[next_buf],
                            2 * next_tile_nnz * sizeof(csr_word_t));
            }

            /* ---- compute current tile: rows -> vector pairs -> SPATZ ---- */
            volatile csr_word_t *buf = valcol_buf[current_buf];

            for (uint32_t i = tile_row_start; i < tile_row_end; i++) {

                int32_t  sum         = 0;
                uint32_t local_start = local_rowptr[i]     - start_nnz;
                uint32_t local_end   = local_rowptr[i + 1] - start_nnz;

                /* Split the row in chunks of at most SPATZ_MAX_VL elements */
                uint32_t j = local_start;
                while (j < local_end) {

                    uint32_t n = local_end - j;
                    if (n > SPATZ_MAX_VL) {
                        n = SPATZ_MAX_VL;
                    }

                    /* control core builds the two vectors: values and x */
                    for (uint32_t c = 0; c < n; c++) {
                        uint32_t x_addr = buf[2 * (j + c) + 1];
                        vec_val[c] = (int32_t)buf[2 * (j + c)];
                        vec_x[c]   = (int32_t)(*(volatile int16_t *)x_addr);
                    }

                    /* SPATZ: multiply + accumulate */
                    uint32_t t0 = perf_get_cycles();
                    spatz_dot(&eu_ctrl, sp_params, n);
                    spatz_wait_time += perf_get_cycles() - t0;
                    n_spatz_calls++;

                    /* control core accumulates the partial result */
                    sum += *sp_res;

                    j += n;
                }

                local_y[i] = sum;
            }

            /* ---- wait for next tile DMA ---- */
            if (tile + 1 < num_tiles) {
                dma_wait_start = perf_get_cycles();
                eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
                dma_wait_time += perf_get_cycles() - dma_wait_start;
            }

            current_buf ^= 1;
            next_buf    ^= 1;
        }

        /* ---------------- local y -> global y ---------------- */
        idma_memcpy_1d(&idma_ctrl, 1,
                    (uint32_t)(y + start_row),
                    L.y,
                    local_rows * sizeof(int32_t));

        dma_wait_start = perf_get_cycles();
        eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
        dma_wait_time += perf_get_cycles() - dma_wait_start;

        uint32_t computed_time = perf_get_cycles() - compute_start;


        run_time_cycle[hartid]   = perf_get_cycles() - run_time;
        DMA_wait_cycle[hartid]   = dma_wait_time;
        fsync_wait_cycle[hartid] = fsync_wait_time;
        compute_cycle[hartid]    = computed_time;
        spatz_wait_cycle[hartid] = spatz_wait_time;
        spatz_calls[hartid]      = n_spatz_calls;

        printf("core %u finished\n", hartid);

        int32_t finish_start = perf_get_cycles();
        fsync_sync_global(&fsync_ctrl);
        eu_fsync_wait(&eu_ctrl, WAIT_MODE);

        /* SPATZ no longer needed */
        spatz_clk_dis();

        diff_finish[hartid] = perf_get_cycles() - finish_start;

        /* ---------------- Verification + metrics ---------------- */
        if (hartid == 0) {

            printf("RUNTIME\n");
            for (int i = 0; i < 64; i++) printf("%u\n", run_time_cycle[i]);

            printf("COMPUTE TIME\n");
            for (int i = 0; i < 64; i++) printf("%u\n", compute_cycle[i]);

            printf("SPATZ WAIT TIME\n");
            for (int i = 0; i < 64; i++) printf("%u\n", spatz_wait_cycle[i]);

            printf("SPATZ CALLS\n");
            for (int i = 0; i < 64; i++) printf("%u\n", spatz_calls[i]);

            printf("FSYNCRONIZATION\n");
            for (int i = 0; i < 64; i++) printf("%u\n", fsync_wait_cycle[i]);

            printf("DIFFERENCE OF FINISH\n");
            for (int i = 0; i < 64; i++) printf("%u\n", diff_finish[i]);

            int errors = 0;
            for (int i = 0; i < M; i++) {
                if (y[i] != y_expected[i]) {
                    printf("Mismatch at index %d: got %d expected %d\n",
                        i, y[i], y_expected[i]);
                    errors++;
                }
            }
            printf("Errors: %d\n", errors);

            uint32_t max_run_time   = find_max(run_time_cycle, 128);
            uint32_t max_dma_wait   = find_max(DMA_wait_cycle, 128);
            uint32_t max_fsync_wait = find_max(fsync_wait_cycle, 128);
            uint32_t max_compute    = find_max(compute_cycle, 128);
            uint32_t max_spatz_wait = find_max(spatz_wait_cycle, 128);

            uint32_t total_macs  = NNZ;
            uint32_t total_flops = 2 * NNZ;

            uint32_t gflops_x1000 = (total_flops * clock_freq_MHz) / max_run_time;
            uint32_t mac_per_cycle_x1000 = (total_macs * 1000) / max_run_time;

            printf("run_time_cycles          : %u\n", max_run_time);
            printf("dma_wait_cycles          : %u\n", max_dma_wait);
            printf("fsync_wait_cycles        : %u\n", max_fsync_wait);
            printf("compute_cycles           : %u\n", max_compute);
            printf("spatz_wait_cycles        : %u\n", max_spatz_wait);
            printf("NNZ                      : %u\n", NNZ);
            printf("Total MACs               : %u\n", total_macs);
            printf("Total FLOPs              : %u\n", total_flops);
            printf("GFLOPS x1000             : %u\n", gflops_x1000);
            printf("MAC/Cycle x1000          : %u\n", mac_per_cycle_x1000);

            return errors;
        }

    }

    return 0;
}