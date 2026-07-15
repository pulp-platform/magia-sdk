#include <stdint.h>
#include "test.h"

#include "tile.h"
#include "gemm_utils.h"
#include "fsync.h"
#include "idma.h"
#include "redmule.h"
#include "eventunit.h"
#include "utils/l1_fifo.h"

#define WAIT_MODE            WFE

/**
 * Tile group definitions for 8x8 mesh (64 tiles).
 *
 * GEMM1 tiles: [0, 1, 8, 9]           (4 tiles,  top-left 2x2 block)
 * GEMM2 tiles: [16-19, 24-27, 32-35,
 *               40-43, 48-51, 56-59]   (24 tiles, rows 2-7 cols 0-3)
 * GEMM3 tiles: [2-7, 10-15]           (12 tiles, rows 0-1 cols 2-7)
 * GEMM4 tiles: [20-23, 28-31, 36-39,
 *               44-47, 52-55, 60-63]   (24 tiles, rows 2-7 cols 4-7)
 */
#define GEMM1_N_TILES        4
#define GEMM2_N_TILES        24
#define GEMM3_N_TILES        12
#define GEMM4_N_TILES        24

#define abs_threshold_millis 8 /* 0.008 expressed as integer millis */

/* FIFO message type tags */
#define MATRIX_R1            0u
#define MATRIX_R2            1u
#define MATRIX_R3            2u

/*
 * Compute the per-tile FIFO reservation size from test dimensions so it
 * scales automatically when the matrices grow.
 *
 * Worst case is a GEMM3 tile with:
 *   num_slots  = ceil(DIM_A / GEMM3_N_TILES) + DIM_C
 *   slot_stride = FIFO_SLOT_META_SIZE + ceil(DIM_C * 0.2) * DIM_E * 2  (r2 dominates)
 *
 * DIM_C/5+1 is a conservative integer approximation of ceil(DIM_C * 0.2).
 * FIFO_HEADER_SIZE / FIFO_SLOT_META_SIZE come from l1_fifo.h so this tracks the
 * slot metadata size automatically. Result is rounded up to the next 4 KB.
 */
#define _FIFO_R2_BATCH       (DIM_C / 5 + 1)
#define _FIFO_SLOT_STRIDE    (FIFO_SLOT_META_SIZE + (uint32_t)(_FIFO_R2_BATCH * DIM_E * 2))
/*
 * Per-sub-ring capacity (one producer's messages). Sized to the old whole-FIFO
 * total so a producer's sub-ring never wraps in this workload — required because
 * the GEMM3 consumer holds R1 payload pointers (r1_data_ptrs) across later pops,
 * so those slots must not be reused.
 */
#define _FIFO_NUM_SLOTS_MAX  (2u + (uint32_t)DIM_C)
/*
 * Widest producer fan-in: a GEMM3 consumer gets one sub-ring per R1 producer
 * (every GEMM1 tile) plus one per R2 producer (every GEMM2 tile).
 */
#define _FIFO_NUM_PRODUCERS_MAX (GEMM1_N_TILES + GEMM2_N_TILES)
#define FIFO_RESERVE_SIZE                                                                           \
    ((FIFO_HEADER_SIZE + _FIFO_NUM_PRODUCERS_MAX * FIFO_RING_STATE_SIZE +                           \
      _FIFO_NUM_PRODUCERS_MAX * _FIFO_NUM_SLOTS_MAX * _FIFO_SLOT_STRIDE + 0xFFFu) &                 \
     ~0xFFFu)

/*
 * Fraction of a tile's rows to push per fifo_push_row_block() call.
 * Set at build time via -DFIFO_BATCH_FRAC=F (default 0.2, i.e. 5 steps of 20%).
 * If <= 0, falls back to one row at a time.
 */
#ifndef FIFO_BATCH_FRAC
#define FIFO_BATCH_FRAC 0.2f
#endif

/*
 * Compute the batch size (rows per push) for a tile with `total` rows.
 * Returns ceil(total * frac), minimum 1.
 * If frac <= 0, returns 1 (one row at a time).
 */
static uint32_t compute_batch(uint32_t total, float frac)
{
    if (frac <= 0.0f || total == 0)
        return 1;
    uint32_t b = (uint32_t)(total * frac + 0.9999f); /* ceiling */
    return (b < 1) ? 1 : b;
}

/*
 * Compute chunk i out of n_chunks for splitting `total` row count.
 * Distributes the remainder across the first chunks (mirrors get_row_range).
 */
static void
compute_chunk(uint32_t total, uint32_t n_chunks, uint32_t idx, uint32_t *start, uint32_t *len)
{
    uint32_t base = total / n_chunks;
    uint32_t rem  = total % n_chunks;
    *start        = idx * base + (idx < rem ? idx : rem);
    *len          = base + (idx < rem ? 1 : 0);
}

static const uint32_t gemm1_tiles[GEMM1_N_TILES] = {0, 1, 8, 9};

static const uint32_t gemm2_tiles[GEMM2_N_TILES] = {16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35,
                                                    40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59};

static const uint32_t gemm3_tiles[GEMM3_N_TILES] = {2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15};

static const uint32_t gemm4_tiles[GEMM4_N_TILES] = {20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39,
                                                    44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63};

static int get_local_idx(uint32_t hartid, const uint32_t *tiles, uint32_t n_tiles)
{
    for (uint32_t i = 0; i < n_tiles; i++)
        if (tiles[i] == hartid)
            return (int)i;
    return -1;
}

static void get_row_range(uint32_t local_idx,
                          uint32_t n_tiles,
                          uint32_t total_rows,
                          uint32_t *start_row,
                          uint32_t *num_rows)
{
    uint32_t base = total_rows / n_tiles;
    uint32_t rem  = total_rows % n_tiles;
    *start_row    = local_idx * base + (local_idx < rem ? local_idx : rem);
    *num_rows     = base + (local_idx < rem ? 1 : 0);
}

static void mem_set_zero(uint32_t o, uint32_t dim)
{
    for (uint32_t i = 0; i < dim; i++)
        mmio16(o + i * 2) = 0x0000;
}

/*
 * Build the producer-local packed source layout (batch rows x cols, fp16) and
 * the logical descriptor (the same block positioned at global_row along dim 0)
 * for a row-block FIFO message.
 */
static void make_row_block(tensor_sub_slice_t *src,
                           tensor_sub_slice_t *desc,
                           uint32_t global_row,
                           uint32_t batch,
                           uint32_t cols)
{
    src->rank            = 2u;
    src->num_elems       = batch * cols;
    src->dims[0].start   = 0u;
    src->dims[0].length  = batch;
    src->dims[0].stride  = cols * 2u;
    src->dims[1].start   = 0u;
    src->dims[1].length  = cols;
    src->dims[1].stride  = 2u;

    desc->rank           = 2u;
    desc->num_elems      = batch * cols;
    desc->dims[0].start  = global_row;
    desc->dims[0].length = batch;
    desc->dims[0].stride = cols * 2u;
    desc->dims[1].start  = 0u;
    desc->dims[1].length = cols;
    desc->dims[1].stride = 2u;
}

/*
 * Push a packed row-block to a consumer FIFO slot via iDMA, timing the transfer.
 * The payload move (local L1 -> remote slot) and per-burst wait are handled by
 * fifo_push()/idma_memcpy_md_to_nd(); a packed source collapses to a single 1D
 * burst.
 */
static void fifo_push_row_block(uint32_t target_hartid,
                                uint32_t producer_idx,
                                uint32_t src_addr,
                                uint32_t global_row,
                                uint32_t batch,
                                uint32_t cols,
                                uint32_t tag,
                                idma_controller_t *idma_ctrl,
                                eu_controller_t *eu_ctrl,
                                uint32_t *cyc_push)
{
    tensor_sub_slice_t src, desc;
    make_row_block(&src, &desc, global_row, batch, cols);

    fifo_push_req_t req = {
        .target_hartid = target_hartid,
        .producer_idx  = producer_idx,
        .src_base_addr = src_addr,
        .src           = &src,
        .desc          = &desc,
        .tag           = tag,
        .elem_bytes    = 2u,
    };

    uint32_t t0 = perf_get_cycles();
    fifo_push(idma_ctrl, eu_ctrl, &req);
    PERF_DELTA(*cyc_push, t0);
}

/*
 * Push completed R3 rows to all overlapping GEMM4 tile FIFOs.
 */
static void push_r3_to_gemm4(uint32_t global_row,
                             uint32_t batch_rows,
                             uint32_t obi_r3_batch,
                             idma_controller_t *idma_ctrl,
                             eu_controller_t *eu_ctrl,
                             uint32_t *cyc_push)
{
    for (uint32_t k = 0; k < GEMM4_N_TILES; k++) {
        uint32_t g4_start, g4_nrows;
        get_row_range(k, GEMM4_N_TILES, DIM_A, &g4_start, &g4_nrows);
        if (g4_nrows == 0)
            continue;

        uint32_t ov_start = global_row > g4_start ? global_row : g4_start;
        uint32_t ov_end   = (global_row + batch_rows) < (g4_start + g4_nrows)
                                ? (global_row + batch_rows)
                                : (g4_start + g4_nrows);
        if (ov_start >= ov_end)
            continue;

        uint32_t ov_count = ov_end - ov_start;
        uint32_t src_off  = (ov_start - global_row) * DIM_E * 2;
        /* This producer's sub-ring in the consumer FIFO is keyed by its GEMM3 index. */
        uint32_t producer_idx = (uint32_t)get_local_idx(get_hartid(), gemm3_tiles, GEMM3_N_TILES);
        fifo_push_row_block(gemm4_tiles[k],
                            producer_idx,
                            obi_r3_batch + src_off,
                            ov_start,
                            ov_count,
                            DIM_E,
                            MATRIX_R3,
                            idma_ctrl,
                            eu_ctrl,
                            cyc_push);
    }
}

/*
 * Accumulate a partial R3 contribution for R1 batch at local row lr,
 * using R2 rows [k_start .. k_start + k_len).
 *
 * Returns the number of newly completed R3 rows (pushed to GEMM4), or 0.
 */
static uint32_t gemm3_partial_accum(uint32_t lr,
                                    uint32_t k_start,
                                    uint32_t k_len,
                                    uint32_t r1_batch,
                                    uint32_t r1_ptr,
                                    uint32_t obi_r2,
                                    uint32_t obi_r3,
                                    uint32_t obi_r1_tmp,
                                    uint32_t start_row,
                                    uint32_t *r3_k_done,
                                    uint8_t *r3_pushed,
                                    redmule_controller_t *redmule_ctrl,
                                    eu_controller_t *eu_ctrl,
                                    idma_controller_t *idma_ctrl,
                                    uint32_t *cyc_compute,
                                    uint32_t *cyc_push)
{
    uint32_t A_ptr;

    if (r1_batch == 1) {
        /* Single row: R1[0, k_start:k_start+k_len] is contiguous within the row */
        A_ptr = r1_ptr + k_start * 2;
    } else {
        /* Multiple rows: extract columns k_start..k_start+k_len into temp buffer */
        for (uint32_t r = 0; r < r1_batch; r++) {
            volatile uint16_t *src = (volatile uint16_t *)(r1_ptr + (r * DIM_C + k_start) * 2);
            volatile uint16_t *dst = (volatile uint16_t *)(obi_r1_tmp + r * k_len * 2);
            for (uint32_t c = 0; c < k_len; c++)
                dst[c] = src[c];
        }
        A_ptr = obi_r1_tmp;
    }

    uint32_t B_ptr        = obi_r2 + k_start * DIM_E * 2;
    uint32_t obi_r3_batch = obi_r3 + lr * DIM_E * 2;

    uint32_t t0 = perf_get_cycles();
    redmule_gemm(redmule_ctrl,
                 A_ptr,
                 B_ptr,
                 obi_r3_batch,
                 (uint16_t)r1_batch,
                 (uint16_t)k_len,
                 (uint16_t)DIM_E);
    eu_redmule_wait(eu_ctrl, WAIT_MODE);
    PERF_DELTA(*cyc_compute, t0);

    r3_k_done[lr] += k_len;

    if (r3_k_done[lr] == DIM_C && !r3_pushed[lr]) {
        r3_pushed[lr]       = 1;
        uint32_t global_row = start_row + lr;
        push_r3_to_gemm4(global_row, r1_batch, obi_r3_batch, idma_ctrl, eu_ctrl, cyc_push);
        return r1_batch;
    }

    return 0;
}

int main(void)
{
    /* 0. Initialization */
    uint32_t hartid       = get_hartid();
    uint32_t l1_tile_base = get_l1_base(hartid);

    /* Init iDMA */
    idma_config_t idma_cfg      = {.hartid = hartid};
    idma_controller_t idma_ctrl = {
        .base = NULL,
        .cfg  = &idma_cfg,
        .api  = &idma_api,
    };
    idma_init(&idma_ctrl);

    /* Init RedMulE */
    redmule_config_t redmule_cfg      = {.hartid = hartid};
    redmule_controller_t redmule_ctrl = {
        .base = NULL,
        .cfg  = &redmule_cfg,
        .api  = &redmule_api,
    };
    redmule_init(&redmule_ctrl);

    /* Init FractalSync */
    fsync_config_t fsync_cfg      = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {
        .base = NULL,
        .cfg  = &fsync_cfg,
        .api  = &fsync_api,
    };
    fsync_init(&fsync_ctrl);

/* Init Event Unit */
#if STALLING == 0
    eu_config_t eu_cfg      = {.hartid = hartid};
    eu_controller_t eu_ctrl = {
        .base = NULL,
        .cfg  = &eu_cfg,
        .api  = &eu_api,
    };

    eu_init(&eu_ctrl);
    eu_clear_events(0xFFFFFFFF);
    eu_fsync_init(&eu_ctrl, 0);
    eu_idma_init(&eu_ctrl, 0);
    eu_redmule_init(&eu_ctrl, 0);
#endif

    /* Per-tile cycle-count breakdown (accumulated across phases) */
    uint32_t cyc_barrier  = 0; /* fsync barrier wait */
    uint32_t cyc_preamble = 0; /* GEMM1 preamble (setup before first load), excluding prints */
    uint32_t cyc_cmi      = 0; /* input DMA: L2→L1 matrix loads + GEMM3 R2 workspace copy */
    uint32_t cyc_compute  = 0; /* RedMulE GEMM time */
    uint32_t cyc_push     = 0; /* fifo_push_row_block: DMA + publish */
    uint32_t cyc_spin     = 0; /* fifo_pop spin (consumer idle) */
    uint32_t cyc_cmo      = 0; /* output DMA: L1→L2 (GEMM4 only) */
    uint32_t _t0;

    /* Determine group membership for this tile (needed to size the FIFO) */
    int gemm1_idx = get_local_idx(hartid, gemm1_tiles, GEMM1_N_TILES);
    int gemm2_idx = get_local_idx(hartid, gemm2_tiles, GEMM2_N_TILES);
    int gemm3_idx = get_local_idx(hartid, gemm3_tiles, GEMM3_N_TILES);
    int gemm4_idx = get_local_idx(hartid, gemm4_tiles, GEMM4_N_TILES);

    /*
     * Initialize this tile's FIFO before the startup barrier.
     * Slot count and payload size depend on the tile's consumer role.
     * Only needed for GEMM 3 and GEMM 4 (which have as input the result
     * of other GEMMs, performed by other tiles).
     */
    uint32_t num_producers  = 0;
    uint32_t num_slots       = 0;
    uint32_t slot_data_size = 0;

    if (gemm3_idx >= 0) {
        uint32_t g3s, g3n;

        get_row_range((uint32_t)gemm3_idx, GEMM3_N_TILES, DIM_A, &g3s, &g3n);

        uint32_t r1_batch = compute_batch(g3n, FIFO_BATCH_FRAC);
        uint32_t r2_batch = compute_batch(DIM_C, FIFO_BATCH_FRAC);

        uint32_t r1_payload = r1_batch * DIM_C * 2;
        uint32_t r2_payload = r2_batch * DIM_E * 2;

        /* One sub-ring per R1 producer (GEMM1 tiles) and per R2 producer (GEMM2
         * tiles). Per-sub-ring depth holds the whole fan-in without wrapping so
         * held R1 payload pointers stay valid. */
        num_producers  = GEMM1_N_TILES + GEMM2_N_TILES;
        num_slots      = g3n + DIM_C;
        slot_data_size = r1_payload > r2_payload ? r1_payload : r2_payload;
    } else if (gemm4_idx >= 0) {
        uint32_t g4s, g4n;

        get_row_range((uint32_t)gemm4_idx, GEMM4_N_TILES, DIM_A, &g4s, &g4n);

        uint32_t r3_batch = compute_batch(g4n, FIFO_BATCH_FRAC);

        /* One sub-ring per R3 producer (GEMM3 tiles). */
        num_producers  = GEMM3_N_TILES;
        num_slots      = g4n;
        slot_data_size = r3_batch * DIM_E * 2;
    }

    fifo_init(hartid, num_producers, num_slots, slot_data_size);

    /*
     * Startup barrier: ensure all tiles finish crt0.S BSS zeroing (which
     * touches global output buffers) and FIFO initialization before any
     * tile begins pushing into another tile's FIFO.
     */
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    perf_reset();
    perf_start();

    /*
     * L1 workspace starts at FIFO_RESERVE_SIZE bytes past the tile base.
     * The first FIFO_RESERVE_SIZE bytes are reserved for the FIFO header
     * and slot array written by remote pushers.
     */
    uint32_t ws = l1_tile_base + FIFO_RESERVE_SIZE;

    _t0 = perf_get_cycles();

    /* ------------------------------------------------------------------ */
    /* GEMM1: R1 = M1 @ M2, push R1 rows to GEMM3 FIFOs                   */
    /*                                                                    */
    /* Double-buffered: split the local M1 row slice into chunks of       */
    /* compute_batch(num_rows, FIFO_BATCH_FRAC) rows. While RedMulE        */
    /* computes chunk i, iDMA prefetches chunk i+1 into the alternate     */
    /* ping-pong buffer. M2 is loaded once and shared across all chunks.  */
    /* ------------------------------------------------------------------ */
    if (gemm1_idx >= 0) {
        PERF_DELTA(cyc_preamble, _t0);
        _t0 = perf_get_cycles();

        uint32_t start_row, num_rows;
        get_row_range((uint32_t)gemm1_idx, GEMM1_N_TILES, DIM_A, &start_row, &num_rows);

        PERF_DELTA(cyc_preamble, _t0);
        _t0 = perf_get_cycles();

        if (num_rows > 0) {
            PERF_DELTA(cyc_preamble, _t0);
            _t0 = perf_get_cycles();

            uint32_t chunk_max = compute_batch(num_rows, FIFO_BATCH_FRAC);
            uint32_t n_chunks  = (num_rows + chunk_max - 1) / chunk_max;

            /* Workspace: [M2 | M1_pp[0] | M1_pp[1] | R1_pp[0] | R1_pp[1]] */
            uint32_t obi_m2    = ws;
            uint32_t obi_m1[2] = {obi_m2 + DIM_B * DIM_C * 2,
                                  obi_m2 + DIM_B * DIM_C * 2 + chunk_max * DIM_B * 2};
            uint32_t obi_r1[2] = {obi_m1[1] + chunk_max * DIM_B * 2,
                                  obi_m1[1] + chunk_max * DIM_B * 2 + chunk_max * DIM_C * 2};

            PERF_DELTA(cyc_preamble, _t0);

            /* Load full M2 from L2 */
            _t0 = perf_get_cycles();
            idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m2_inp, obi_m2, DIM_B * DIM_C * 2);
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
            PERF_DELTA(cyc_cmi, _t0);

            /* Prime the pipeline: load M1_pp[0] */
            uint32_t c0s, c0l;
            compute_chunk(num_rows, n_chunks, 0, &c0s, &c0l);

            uint32_t cyc_cmi_pre_prime = cyc_cmi;

            _t0 = perf_get_cycles();
            idma_memcpy_1d(&idma_ctrl,
                           0,
                           (uint32_t)m1_inp + (start_row + c0s) * DIM_B * 2,
                           obi_m1[0],
                           c0l * DIM_B * 2);
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
            PERF_DELTA(cyc_cmi, _t0);

            for (uint32_t i = 0; i < n_chunks; i++) {
                uint32_t cs, cl;
                compute_chunk(num_rows, n_chunks, i, &cs, &cl);
                uint32_t pp       = i & 1u;
                uint32_t has_next = (i + 1u < n_chunks);

                /* Kick off chunk i+1 load (overlaps with chunk i compute) */
                if (has_next) {
                    uint32_t ns, nl;
                    compute_chunk(num_rows, n_chunks, i + 1, &ns, &nl);
                    uint32_t cyc_cmi_before = cyc_cmi;
                    _t0                     = perf_get_cycles();
                    idma_memcpy_1d(&idma_ctrl,
                                   0,
                                   (uint32_t)m1_inp + (start_row + ns) * DIM_B * 2,
                                   obi_m1[(i + 1) & 1u],
                                   nl * DIM_B * 2);
                    PERF_DELTA(cyc_cmi, _t0);
                }

                /* Compute R1 chunk i (zero-init + GEMM issue + wait) */
                uint32_t cyc_compute_before = cyc_compute;
                _t0                         = perf_get_cycles();
                mem_set_zero(obi_r1[pp], cl * DIM_C);
                redmule_gemm(&redmule_ctrl,
                             obi_m1[pp],
                             obi_m2,
                             obi_r1[pp],
                             (uint16_t)cl,
                             (uint16_t)DIM_B,
                             (uint16_t)DIM_C);
                eu_redmule_wait(&eu_ctrl, WAIT_MODE);
                PERF_DELTA(cyc_compute, _t0);

                if (has_next) {
                    uint32_t cyc_cmi_wait_before = cyc_cmi;
                    _t0                          = perf_get_cycles();
                    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
                    PERF_DELTA(cyc_cmi, _t0);
                }

                /*
                 * Push this chunk's R1 rows to each GEMM3 tile that owns
                 * overlapping rows. Producer-side window for the chunk is
                 * [chunk_g_start, chunk_g_end).
                 */
                uint32_t chunk_g_start = start_row + cs;
                uint32_t chunk_g_end   = chunk_g_start + cl;
                for (uint32_t j = 0; j < GEMM3_N_TILES; j++) {
                    uint32_t g3_start, g3_nrows;
                    get_row_range(j, GEMM3_N_TILES, DIM_A, &g3_start, &g3_nrows);
                    if (g3_nrows == 0)
                        continue;

                    uint32_t ov_start = chunk_g_start > g3_start ? chunk_g_start : g3_start;
                    uint32_t ov_end =
                        chunk_g_end < (g3_start + g3_nrows) ? chunk_g_end : (g3_start + g3_nrows);
                    if (ov_start >= ov_end)
                        continue;

                    uint32_t g3_r1_batch = compute_batch(g3_nrows, FIFO_BATCH_FRAC);
                    for (uint32_t r = ov_start; r < ov_end; r += g3_r1_batch) {
                        uint32_t batch   = (r + g3_r1_batch <= ov_end) ? g3_r1_batch : (ov_end - r);
                        uint32_t src_off = (r - chunk_g_start) * DIM_C * 2;

                        uint32_t cyc_push_before = cyc_push;
                        /* R1 producers occupy sub-rings [0, GEMM1_N_TILES). */
                        fifo_push_row_block(gemm3_tiles[j],
                                            (uint32_t)gemm1_idx,
                                            obi_r1[pp] + src_off,
                                            r,
                                            batch,
                                            DIM_C,
                                            MATRIX_R1,
                                            &idma_ctrl,
                                            &eu_ctrl,
                                            &cyc_push);
                    }
                }
            }
        }
    }

    /* ------------------------------------------------------------------ */
    /* GEMM2: R2 = M3 @ M4, push R2 rows to all GEMM3 FIFOs               */
    /*                                                                    */
    /* Double-buffered using the same chunk strategy as GEMM1. M4 is      */
    /* loaded once; M3 chunks are ping-pong'd against RedMulE compute.    */
    /* ------------------------------------------------------------------ */
    if (gemm2_idx >= 0) {
        PERF_DELTA(cyc_preamble, _t0);
        _t0 = perf_get_cycles();

        uint32_t start_row, num_rows;
        get_row_range((uint32_t)gemm2_idx, GEMM2_N_TILES, DIM_C, &start_row, &num_rows);

        PERF_DELTA(cyc_preamble, _t0);
        _t0 = perf_get_cycles();

        if (num_rows > 0) {
            PERF_DELTA(cyc_preamble, _t0);
            _t0 = perf_get_cycles();

            uint32_t chunk_max = compute_batch(num_rows, FIFO_BATCH_FRAC);
            uint32_t n_chunks  = (num_rows + chunk_max - 1) / chunk_max;

            /* Workspace: [M4 | M3_pp[0] | M3_pp[1] | R2_pp[0] | R2_pp[1]] */
            uint32_t obi_m4    = ws;
            uint32_t obi_m3[2] = {obi_m4 + DIM_D * DIM_E * 2,
                                  obi_m4 + DIM_D * DIM_E * 2 + chunk_max * DIM_D * 2};
            uint32_t obi_r2[2] = {obi_m3[1] + chunk_max * DIM_D * 2,
                                  obi_m3[1] + chunk_max * DIM_D * 2 + chunk_max * DIM_E * 2};

            PERF_DELTA(cyc_preamble, _t0);

            /* Load full M4 from L2 */
            _t0 = perf_get_cycles();
            idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m4_inp, obi_m4, DIM_D * DIM_E * 2);
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
            PERF_DELTA(cyc_cmi, _t0);

            /* Prime: load chunk 0 of M3 */
            uint32_t c0s, c0l;
            compute_chunk(num_rows, n_chunks, 0, &c0s, &c0l);

            uint32_t cyc_cmi_pre_prime = cyc_cmi;

            _t0 = perf_get_cycles();
            idma_memcpy_1d(&idma_ctrl,
                           0,
                           (uint32_t)m3_inp + (start_row + c0s) * DIM_D * 2,
                           obi_m3[0],
                           c0l * DIM_D * 2);
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
            PERF_DELTA(cyc_cmi, _t0);

            uint32_t r2_batch = compute_batch(DIM_C, FIFO_BATCH_FRAC);

            for (uint32_t i = 0; i < n_chunks; i++) {
                uint32_t cs, cl;
                compute_chunk(num_rows, n_chunks, i, &cs, &cl);
                uint32_t pp       = i & 1u;
                uint32_t has_next = (i + 1u < n_chunks);

                if (has_next) {
                    uint32_t ns, nl;
                    compute_chunk(num_rows, n_chunks, i + 1, &ns, &nl);
                    uint32_t cyc_cmi_before = cyc_cmi;
                    _t0                     = perf_get_cycles();
                    idma_memcpy_1d(&idma_ctrl,
                                   0,
                                   (uint32_t)m3_inp + (start_row + ns) * DIM_D * 2,
                                   obi_m3[(i + 1) & 1u],
                                   nl * DIM_D * 2);
                    PERF_DELTA(cyc_cmi, _t0);
                }

                /* Compute R2 chunk i (zero-init + GEMM issue + wait) */
                uint32_t cyc_compute_before = cyc_compute;
                _t0                         = perf_get_cycles();
                mem_set_zero(obi_r2[pp], cl * DIM_E);
                redmule_gemm(&redmule_ctrl,
                             obi_m3[pp],
                             obi_m4,
                             obi_r2[pp],
                             (uint16_t)cl,
                             (uint16_t)DIM_D,
                             (uint16_t)DIM_E);
                eu_redmule_wait(&eu_ctrl, WAIT_MODE);
                PERF_DELTA(cyc_compute, _t0);

                if (has_next) {
                    uint32_t cyc_cmi_wait_before = cyc_cmi;
                    _t0                          = perf_get_cycles();
                    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
                    PERF_DELTA(cyc_cmi, _t0);
                }

                /*
                 * Push this chunk's R2 rows to every GEMM3 tile (all need
                 * the full R2). row_index = global row in the full R2
                 * matrix (0..DIM_C-1).
                 */
                for (uint32_t r = 0; r < cl; r += r2_batch) {
                    uint32_t batch    = (r + r2_batch <= cl) ? r2_batch : (cl - r);
                    uint32_t src_off  = r * DIM_E * 2;
                    uint32_t global_r = start_row + cs + r;
                    for (uint32_t j = 0; j < GEMM3_N_TILES; j++) {
                        uint32_t cyc_push_before = cyc_push;
                        /* R2 producers occupy sub-rings [GEMM1_N_TILES, GEMM1+GEMM2),
                         * disjoint from the R1 producers' sub-rings. */
                        fifo_push_row_block(gemm3_tiles[j],
                                            GEMM1_N_TILES + (uint32_t)gemm2_idx,
                                            obi_r2[pp] + src_off,
                                            global_r,
                                            batch,
                                            DIM_E,
                                            MATRIX_R2,
                                            &idma_ctrl,
                                            &eu_ctrl,
                                            &cyc_push);
                    }
                }
            }
        }
    }

    /* ------------------------------------------------------------------ */
    /* GEMM3: incremental partial GEMM with K-dimension accumulation      */
    /* ------------------------------------------------------------------ */
    if (gemm3_idx >= 0) {
        PERF_DELTA(cyc_preamble, _t0);
        _t0 = perf_get_cycles();

        uint32_t start_row, num_rows;
        get_row_range((uint32_t)gemm3_idx, GEMM3_N_TILES, DIM_A, &start_row, &num_rows);

        PERF_DELTA(cyc_preamble, _t0);

        _t0 = perf_get_cycles();

        /*
         * Workspace layout:
         *   [R2 (full DIM_C x DIM_E) | R3_slice (num_rows x DIM_E) | R1_tmp]
         *
         * R1 data is used directly from FIFO payload pointers.
         * R1_tmp is a scratch buffer for column extraction when batch_rows > 1.
         */
        uint32_t obi_r2 = ws;
        uint32_t obi_r3 = obi_r2 + DIM_C * DIM_E * 2;

        /* Batch size for R1/R3 rows owned by this tile */
        uint32_t my_r1_batch = compute_batch(num_rows, FIFO_BATCH_FRAC);
        uint32_t obi_r1_tmp  = obi_r3 + num_rows * DIM_E * 2;

        /*
         * Tracking state for incremental processing.
         * r1_received[lr]  : 1 once R1 batch at local row lr has arrived
         * r1_data_ptrs[lr] : FIFO payload pointer for the R1 batch at lr
         * r2_received[k]   : 1 once R2 global row k has been DMA'd to workspace
         * r3_k_done[lr]    : number of K-columns accumulated into R3 for batch lr
         * r3_pushed[lr]    : 1 once R3 for batch lr has been pushed to GEMM4
         */
        uint8_t r1_received[DIM_A];
        uint32_t r1_data_ptrs[DIM_A];
        uint8_t r2_received[DIM_C];
        uint32_t r3_k_done[DIM_A];
        uint8_t r3_pushed[DIM_A];

        for (uint32_t i = 0; i < DIM_A; i++) {
            r1_received[i]  = 0;
            r1_data_ptrs[i] = 0;
            r3_k_done[i]    = 0;
            r3_pushed[i]    = 0;
        }
        for (uint32_t i = 0; i < DIM_C; i++)
            r2_received[i] = 0;

        PERF_DELTA(cyc_preamble, _t0);

        uint32_t total_r3_done = 0;

        /* Zero R3 once; redmule_gemm accumulates (Y = X*W + Y) */
        uint32_t cyc_compute_zero_before = cyc_compute;
        _t0                              = perf_get_cycles();
        mem_set_zero(obi_r3, num_rows * DIM_E);
        PERF_DELTA(cyc_compute, _t0);

        /*
         * Spin until all R3 rows for this tile have been computed and pushed.
         * GEMM3 tiles with num_rows == 0 skip this loop entirely.
         */
        while (total_r3_done < num_rows) {
            fifo_msg_t msg;

            uint32_t cyc_spin_before = cyc_spin;

            /* Spin until a FIFO message arrives */
            _t0 = perf_get_cycles();
            while (!fifo_pop(hartid, &msg))
                ;
            PERF_DELTA(cyc_spin, _t0);

            uint32_t data_ptr   = msg.data_ptr;
            uint32_t data_size  = msg.data_size;
            uint32_t matrix_id  = msg.tag;
            uint32_t row_index  = msg.desc.dims[0].start;
            uint32_t batch_rows = msg.desc.dims[0].length;

            if (matrix_id == MATRIX_R1) {
                uint32_t cyc_compute_r1_before = cyc_compute;
                uint32_t cyc_push_r1_before    = cyc_push;

                /*
                 * Received a batch of R1 rows.
                 * Accumulate partial R3 for every contiguous group of R2
                 * rows that have already arrived.
                 */
                uint32_t local_row      = row_index - start_row;
                r1_data_ptrs[local_row] = data_ptr;
                r1_received[local_row]  = 1;

                /* Scan r2_received[] for contiguous groups */
                uint32_t k = 0;
                while (k < DIM_C) {
                    if (!r2_received[k]) {
                        k++;
                        continue;
                    }
                    uint32_t k_start = k;
                    while (k < DIM_C && r2_received[k])
                        k++;

                    total_r3_done += gemm3_partial_accum(local_row,
                                                         k_start,
                                                         k - k_start,
                                                         batch_rows,
                                                         data_ptr,
                                                         obi_r2,
                                                         obi_r3,
                                                         obi_r1_tmp,
                                                         start_row,
                                                         r3_k_done,
                                                         r3_pushed,
                                                         &redmule_ctrl,
                                                         &eu_ctrl,
                                                         &idma_ctrl,
                                                         &cyc_compute,
                                                         &cyc_push);
                }

            } else if (matrix_id == MATRIX_R2) {
                uint32_t cyc_cmi_r2_before     = cyc_cmi;
                uint32_t cyc_compute_r2_before = cyc_compute;
                uint32_t cyc_push_r2_before    = cyc_push;

                /*
                 * Received a batch of R2 rows. Copy to workspace and
                 * immediately accumulate against every R1 batch already present.
                 */
                _t0 = perf_get_cycles();
                idma_memcpy_1d(&idma_ctrl, 1, obi_r2 + row_index * DIM_E * 2, data_ptr, data_size);
                eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
                PERF_DELTA(cyc_cmi, _t0);

                for (uint32_t i = 0; i < batch_rows; i++)
                    r2_received[row_index + i] = 1;

                /* Accumulate this R2 batch against all received R1 batches */
                for (uint32_t lr = 0; lr < num_rows; lr += my_r1_batch) {
                    if (!r1_received[lr] || r3_pushed[lr])
                        continue;

                    uint32_t remaining = num_rows - lr;
                    uint32_t batch_r1  = (remaining < my_r1_batch) ? remaining : my_r1_batch;

                    total_r3_done += gemm3_partial_accum(lr,
                                                         row_index,
                                                         batch_rows,
                                                         batch_r1,
                                                         r1_data_ptrs[lr],
                                                         obi_r2,
                                                         obi_r3,
                                                         obi_r1_tmp,
                                                         start_row,
                                                         r3_k_done,
                                                         r3_pushed,
                                                         &redmule_ctrl,
                                                         &eu_ctrl,
                                                         &idma_ctrl,
                                                         &cyc_compute,
                                                         &cyc_push);
                }
            }
        }
    }

    /* ------------------------------------------------------------------ */
    /* GEMM4: prefetch M5 into L1, then enter FIFO consumer               */
    /*                                                                    */
    /* Output is double-buffered: each batch writes to its own slice of   */
    /* obi_o (indexed by local_row), so successive O DMAs never alias.    */
    /* We issue the L1→L2 write non-blocking and overlap it with the      */
    /* next batch's pop+compute, then wait before issuing the next write. */
    /* ------------------------------------------------------------------ */
    if (gemm4_idx >= 0) {
        PERF_DELTA(cyc_preamble, _t0);
        _t0 = perf_get_cycles();

        uint32_t start_row, num_rows;
        get_row_range((uint32_t)gemm4_idx, GEMM4_N_TILES, DIM_A, &start_row, &num_rows);

        PERF_DELTA(cyc_preamble, _t0);
        _t0 = perf_get_cycles();

        if (num_rows > 0) {
            PERF_DELTA(cyc_preamble, _t0);
            _t0 = perf_get_cycles();
            /*
             * Workspace layout: [M5 | O_slice]
             * M5 is placed first so Phase 3 only needs to know ws to find it.
             */
            uint32_t obi_m5 = ws;
            uint32_t obi_o  = obi_m5 + DIM_E * DIM_F * 2;

            uint32_t cyc_cmi_m5_before = cyc_cmi;
            /* Prefetch full M5 from L2 */
            _t0 = perf_get_cycles();
            idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m5_inp, obi_m5, DIM_E * DIM_F * 2);
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
            PERF_DELTA(cyc_cmi, _t0);

            /*
             * FIFO consumer loop (Phase 3, data-driven, no explicit barrier).
             * Spin until all expected R3 rows (== num_rows) have been processed.
             */
            uint32_t total_o_done = 0;
            uint32_t out_inflight = 0; /* 1 once an O write has been issued and not yet waited */

            while (total_o_done < num_rows) {
                fifo_msg_t msg;

                uint32_t cyc_spin_before = cyc_spin;
                /* Spin until a message arrives */
                _t0 = perf_get_cycles();
                while (!fifo_pop(hartid, &msg))
                    ;
                PERF_DELTA(cyc_spin, _t0);

                uint32_t data_ptr    = msg.data_ptr;
                uint32_t row_index   = msg.desc.dims[0].start;
                uint32_t batch_rows  = msg.desc.dims[0].length;
                uint32_t local_row   = row_index - start_row;
                uint32_t obi_o_batch = obi_o + local_row * DIM_F * 2;

                /* Compute O_batch = R3_batch @ M5; use payload directly as R3 input */
                uint32_t cyc_compute_before = cyc_compute;
                _t0                         = perf_get_cycles();
                mem_set_zero(obi_o_batch, batch_rows * DIM_F);
                redmule_gemm(&redmule_ctrl,
                             data_ptr,
                             obi_m5,
                             obi_o_batch,
                             (uint16_t)batch_rows,
                             (uint16_t)DIM_E,
                             (uint16_t)DIM_F);
                eu_redmule_wait(&eu_ctrl, WAIT_MODE);
                PERF_DELTA(cyc_compute, _t0);

                /* Wait for previous O write to complete before issuing the next */
                uint32_t cyc_cmo_wait_before = cyc_cmo;
                if (out_inflight) {
                    _t0 = perf_get_cycles();
                    eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
                    PERF_DELTA(cyc_cmo, _t0);
                    out_inflight = 0;
                }

                uint32_t cyc_cmo_issue_before = cyc_cmo;
                /* Issue O batch write to L2 (non-blocking; overlaps with next compute) */
                _t0 = perf_get_cycles();
                idma_memcpy_1d(&idma_ctrl,
                               1,
                               (uint32_t)o_out + row_index * DIM_F * 2,
                               obi_o_batch,
                               batch_rows * DIM_F * 2);
                PERF_DELTA(cyc_cmo, _t0);
                out_inflight = 1;

                total_o_done += batch_rows;
            }

            uint32_t cyc_cmo_drain_before = cyc_cmo;
            /* Drain the final outstanding O write before the closing barrier */
            if (out_inflight) {
                _t0 = perf_get_cycles();
                eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
                PERF_DELTA(cyc_cmo, _t0);
            }
        }
    }

    /* ------------------------------------------------------------------ */
    /* Final barrier: wait for all tiles to finish before validation       */
    /* ------------------------------------------------------------------ */
    uint32_t cyc_barrier_final_before = cyc_barrier;
    _t0                               = perf_get_cycles();
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
    PERF_DELTA(cyc_barrier, _t0);

    perf_stop();

    if (hartid == 0) {
        printf("====================== TOTAL CYCLES: %u ======================\n",
               perf_get_cycles());
    }

    /* Validation: tile 0 checks O against golden */
    uint32_t errors = 0;

    if (hartid == 0) {
        for (uint32_t i = 0; i < DIM_A; i++) {
            for (uint32_t j = 0; j < DIM_F; j++) {
                float16 computed = *(volatile float16 *)(&o_out[i * DIM_F + j]);
                float16 expected = o_golden[i * DIM_F + j];

                uint16_t uc = *(uint16_t *)&computed;
                uint16_t ue = *(uint16_t *)&expected;

                int32_t vc = fp16_to_millis(uc);
                int32_t ve = fp16_to_millis(ue);

                int32_t abs_diff = vc - ve;
                if (abs_diff < 0)
                    abs_diff = -abs_diff;
                if (abs_diff > abs_threshold_millis) {
                    errors++;
                    printf("O[%d][%d]: got=%f (0x%x) exp=%f (0x%x) (abs_diff=%ld)\n",
                           i,
                           j,
                           fp16_to_f64(uc),
                           uc,
                           fp16_to_f64(ue),
                           ue,
                           (long)abs_diff);
                }
            }
        }

        printf("\nTest complete. Errors: %d / %d\n\n", errors, DIM_A * DIM_F);
    }

    return errors;
}
