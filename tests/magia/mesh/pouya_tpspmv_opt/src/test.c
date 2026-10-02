#include <stdint.h>

#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"

//#include "data.h"      /* made by tpspmv_preprocess.py */

//#include "data_opt_poisson3Da.h"
//#include "data_opt_raefsky5.h"
//#include "data_opt_ex6.h"
#include "data_opt_cavity05.h"
//#include "data_opt_g7jac140.h"
//#include "data_opt_fxm4_6.h"
//#include "data_opt_scsd8-2r.h"
//#include "data_opt_e18.h"
//#include "data_opt_scfxm1-2b.h"
//#include "data_opt_testbig.h"

/*
=====================================================================
tpSpMV on MAGIA WITH the 3 optimizations of the paper

  1) data reduction : done offline (empty rows removed -> MR rows,
                      empty columns of each tile removed). x_seg is
                      gathered from L2 with the id list xt_ids.
                      (DATA_REDUCTION in data.h)
  2) aligned access : done offline (blocks are padded, ALIGN_BYTES
                      in data.h; 4 = no alignment)
  3) pipelining     : here. ENABLE_PIPELINE = 1 -> DMA of the next
                      slice and DMA of the previous result overlap
                      with the compute of the current slice.

data.h gives, per tile t (= per core) and per slice s, ONE block:
      [ rowptr (RP_WORDS, relative to the slice) | (value,col) pairs ]
so one DMA loads a whole slice. block_off[t*(NUM_SLICES+1)+s] is the
start (in words) of the block in tile_blocks.

Phase 1 : core t, slice s : DMA block -> L1, compute, DMA y' ->
          partial_y[(s*DELTA + t)*SLICE_ROWS ...]
--- barrier ---
Phase 2 : core p owns a range of slices. Slice s of all DELTA partial
          vectors is contiguous -> ONE DMA, add, DMA y -> L2.
y is stored in the reduced (MR rows) order.
=====================================================================
*/

#ifndef ENABLE_PIPELINE
#define ENABLE_PIPELINE 1      /* 1 = pipelined, 0 = sequential */
#endif

#define WAIT_MODE       WFE
#define clock_freq_MHz  1450

#define Q         (ALIGN_BYTES / 4)                          /* words per align unit */
#define RP_WORDS  ((((SLICE_ROWS) + 1 + Q - 1) / Q) * Q)     /* rowptr words / block */

#define L2 __attribute__((section(".l2"), aligned(L2_ALIGN)))

static uint32_t HARTIDS[128]      L2 = {0};
static uint32_t NUM_CORES         L2 = 0;
static uint32_t run_cycle[128]    L2 = {0};
static uint32_t phase1_cycle[128] L2 = {0};
static uint32_t phase2_cycle[128] L2 = {0};

/* phase-1 results: partial_y[(slice*DELTA + tile)*SLICE_ROWS + i] */
static int32_t partial_y[NUM_SLICES * DELTA * SLICE_ROWS] L2 = {0};

static inline uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }
static inline uint32_t align_up(uint32_t v) { return (v + ALIGN_BYTES - 1) & ~(uint32_t)(ALIGN_BYTES - 1); }

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

    if (NUM_CORES != DELTA) {
        if (hartid == 0)
            printf("ERROR: data.h is for %d cores, but %u cores are running\n", DELTA, NUM_CORES);
        return -1;
    }
    if (hartid >= NUM_CORES) return 0;

    /* common start */
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* ================= start timing ================= */
    perf_start();
    uint32_t t_start = perf_get_cycles();

    /* ==============================================================
       PHASE 1 : partial SpMV on my column tile
       ============================================================== */
    uint32_t ncols = tile_ncols[hartid];
    uint32_t bw    = tile_max_block_words[hartid];     /* words of the biggest block */

    /* L1 layout (all aligned) */
    uint32_t addr_x   = align_up(get_l1_base(hartid));       /* x_seg (ncols)            */
    uint32_t addr_off = align_up(addr_x + ncols * 4);        /* block_off row (S+1)      */
    uint32_t addr_b0  = align_up(addr_off + (NUM_SLICES + 1) * 4);
    uint32_t addr_b1  = addr_b0 + bw * 4;                    /* 2 block buffers          */
    uint32_t addr_y0  = addr_b1 + bw * 4;
    uint32_t addr_y1  = addr_y0 + SLICE_ROWS * 4;            /* 2 y' buffers             */
    uint32_t l1_phase1_end = addr_y1 + SLICE_ROWS * 4;

    uint32_t blk_addr[2] = { addr_b0, addr_b1 };
    uint32_t yb_addr[2]  = { addr_y0, addr_y1 };

    volatile int32_t  *local_x   = (int32_t  *)addr_x;
    volatile uint32_t *local_off = (uint32_t *)addr_off;

    /* block offsets of my tile -> L1 */
    idma_memcpy_1d(&idma_ctrl, 0,
                   (uint32_t)&block_off[hartid * (NUM_SLICES + 1)],
                   addr_off, (NUM_SLICES + 1) * 4);
    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

    /* x_seg -> L1 */
#if DATA_REDUCTION
    {   /* gather only the used x elements (empty columns are removed) */
        uint32_t xb = xt_ptr[hartid];
        for (uint32_t i = 0; i < ncols; i++)
            local_x[i] = x[xt_ids[xb + i]];
    }
#else
    if (ncols > 0) {
        idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)&x[Pc[hartid]], addr_x, ncols * 4);
        eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    }
#endif

    /* first block -> L1 */
    idma_memcpy_1d(&idma_ctrl, 0,
                   (uint32_t)&tile_blocks[local_off[0]], blk_addr[0],
                   (local_off[1] - local_off[0]) * 4);
    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

    for (uint32_t s = 0; s < NUM_SLICES; s++) {
        uint32_t cur = s & 1;
        uint32_t nxt = cur ^ 1;

#if ENABLE_PIPELINE
        /* prefetch next block while computing this one */
        if (s + 1 < NUM_SLICES) {
            idma_memcpy_1d(&idma_ctrl, 0,
                           (uint32_t)&tile_blocks[local_off[s + 1]], blk_addr[nxt],
                           (local_off[s + 2] - local_off[s + 1]) * 4);
        }
#endif

        /* ---- compute slice s ---- */
        volatile uint32_t *blk = (uint32_t *)blk_addr[cur];
        volatile int32_t  *yb  = (int32_t  *)yb_addr[cur];
        uint32_t rows = min_u32(SLICE_ROWS, MR - s * SLICE_ROWS);

        for (uint32_t i = 0; i < rows; i++) {
            int32_t  sum = 0;
            uint32_t js = blk[i];
            uint32_t je = blk[i + 1];
            for (uint32_t j = js; j < je; j++) {
                int32_t  v = (int32_t)blk[RP_WORDS + 2 * j];
                uint32_t c = blk[RP_WORDS + 2 * j + 1];
                sum += v * local_x[c];
            }
            yb[i] = sum;
        }

        /* ---- store y' slice ---- */
#if ENABLE_PIPELINE
        if (s > 0) eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);       /* store of slice s-1 */
        idma_memcpy_1d(&idma_ctrl, 1,
                       (uint32_t)&partial_y[(s * DELTA + hartid) * SLICE_ROWS],
                       yb_addr[cur], SLICE_ROWS * 4);
        if (s + 1 < NUM_SLICES) eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);  /* block s+1 */
#else
        idma_memcpy_1d(&idma_ctrl, 1,
                       (uint32_t)&partial_y[(s * DELTA + hartid) * SLICE_ROWS],
                       yb_addr[cur], SLICE_ROWS * 4);
        eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
        if (s + 1 < NUM_SLICES) {
            idma_memcpy_1d(&idma_ctrl, 0,
                           (uint32_t)&tile_blocks[local_off[s + 1]], blk_addr[nxt],
                           (local_off[s + 2] - local_off[s + 1]) * 4);
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
        }
#endif
    }
#if ENABLE_PIPELINE
    eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);                      /* last store */
#endif

    uint32_t t_phase1_end = perf_get_cycles();

    /* all partial vectors must be in L2 before phase 2 */
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* ==============================================================
       PHASE 2 : accumulate. I own a range of slices.
       ============================================================== */
    uint32_t base    = NUM_SLICES / DELTA;
    uint32_t rem     = NUM_SLICES % DELTA;
    uint32_t b_begin = hartid * base + min_u32(hartid, rem);
    uint32_t b_cnt   = base + (hartid < rem ? 1 : 0);

    uint32_t in_bytes = DELTA * SLICE_ROWS * 4;
    uint32_t addr_in0  = align_up(get_l1_base(hartid));
    uint32_t addr_in1  = addr_in0 + in_bytes;
    uint32_t addr_out0 = addr_in1 + in_bytes;
    uint32_t addr_out1 = addr_out0 + SLICE_ROWS * 4;
    uint32_t l1_phase2_end = addr_out1 + SLICE_ROWS * 4;

    uint32_t in_addr[2]  = { addr_in0,  addr_in1 };
    uint32_t out_addr[2] = { addr_out0, addr_out1 };

    if (b_cnt > 0) {
        idma_memcpy_1d(&idma_ctrl, 0,
                       (uint32_t)&partial_y[b_begin * DELTA * SLICE_ROWS],
                       in_addr[0], in_bytes);
        eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    }

    for (uint32_t k = 0; k < b_cnt; k++) {
        uint32_t b   = b_begin + k;
        uint32_t cur = k & 1;
        uint32_t nxt = cur ^ 1;

#if ENABLE_PIPELINE
        if (k + 1 < b_cnt) {
            idma_memcpy_1d(&idma_ctrl, 0,
                           (uint32_t)&partial_y[(b + 1) * DELTA * SLICE_ROWS],
                           in_addr[nxt], in_bytes);
        }
#endif

        volatile int32_t *lin  = (int32_t *)in_addr[cur];
        volatile int32_t *lout = (int32_t *)out_addr[cur];
        uint32_t rows = min_u32(SLICE_ROWS, MR - b * SLICE_ROWS);

        for (uint32_t i = 0; i < rows; i++) {
            int32_t sum = 0;
            for (uint32_t c = 0; c < DELTA; c++)
                sum += lin[c * SLICE_ROWS + i];
            lout[i] = sum;
        }

#if ENABLE_PIPELINE
        if (k > 0) eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
        idma_memcpy_1d(&idma_ctrl, 1, (uint32_t)&y[b * SLICE_ROWS],
                       out_addr[cur], SLICE_ROWS * 4);
        if (k + 1 < b_cnt) eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
#else
        idma_memcpy_1d(&idma_ctrl, 1, (uint32_t)&y[b * SLICE_ROWS],
                       out_addr[cur], SLICE_ROWS * 4);
        eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
        if (k + 1 < b_cnt) {
            idma_memcpy_1d(&idma_ctrl, 0,
                           (uint32_t)&partial_y[(b + 1) * DELTA * SLICE_ROWS],
                           in_addr[nxt], in_bytes);
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
        }
#endif
    }
#if ENABLE_PIPELINE
    if (b_cnt > 0) eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
#endif

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    uint32_t t_end = perf_get_cycles();
    /* ================= stop timing ================= */

    run_cycle[hartid]    = t_end - t_start;
    phase1_cycle[hartid] = t_phase1_end - t_start;
    phase2_cycle[hartid] = t_end - t_phase1_end;   /* includes barrier wait */
    uint32_t l1_used = (l1_phase1_end > l1_phase2_end ? l1_phase1_end : l1_phase2_end)
                       - align_up(get_l1_base(hartid));
    (void)l1_used;

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* ==============================================================
       Check + metrics (core 0)
       ============================================================== */
    if (hartid == 0) {
        volatile int32_t *y_res = (volatile int32_t *)y;   /* y is written by DMA */
        int errors = 0;
        for (int i = 0; i < MR; i++) {
            if (y_res[i] != y_expected[i]) {
                if (errors < 20)
                    printf("Mismatch at row %d: got %d expected %d\n", i, y_res[i], y_expected[i]);
                errors++;
            }
        }
        printf("Errors: %d\n", errors);

        uint32_t max_run    = find_max(run_cycle, 128);
        uint32_t max_phase1 = find_max(phase1_cycle, 128);
        uint32_t max_phase2 = find_max(phase2_cycle, 128);

        uint32_t total_flops = 2 * NNZ;                     /* real nnz (no padding) */
        uint32_t   gflops_x1000      = total_flops * clock_freq_MHz / max_run;

        printf("config: DATA_REDUCTION=%d ALIGN_BYTES=%d PIPELINE=%d SLICE_ROWS=%d SLICES=%d\n",
               DATA_REDUCTION, ALIGN_BYTES, ENABLE_PIPELINE, SLICE_ROWS, NUM_SLICES);
        printf("rows M=%d -> MR=%d\n", M, MR);
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