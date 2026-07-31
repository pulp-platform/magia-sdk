// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// nn_is_bench -- one inference of a fused NN block across the whole MAGIA v3
// mesh, exercising every accelerator of every tile:
//
//   Z = X * W + bias              RedMulE, input-stationary, iDMA + FractalSync
//   P = hardswish(Z*scale + b)    Spatz, fp16 vector
//   P = layernorm_rowwise(P)      Spatz, fp16 vector
//   Q = softmax_rowwise_q15(P)    PULP cluster, 8 cores, int16 packed SIMD
//
// WHAT IT ACTUALLY COMPUTES
//
// One linear layer applied to a batch, followed by an activation, a
// normalisation and a grouped softmax. With the default geometry (M=N=K=512):
//
//   X       [512 x 512] fp16   a batch of 512 activation vectors, 512 features
//   W       [512 x 512] fp16   one fully-connected layer, 512 in -> 512 out
//   bias    [512]       fp16   broadcast over the batch
//   -> 262 656 parameters, and 2*M*N*K = 268.4 MFLOP per inference.
//
// That parameter count is deliberately a realistic unit of work: it is exactly
// the size of one attention projection (W_q, W_k or W_v) in a d_model=512
// transformer, or about a quarter of one FFN sub-layer.
//
// It is NOT a classifier, and certainly not a binary one. Two things are driven
// by the machine rather than by any model:
//   * LayerNorm and softmax run over NN_PB = 128 columns, not over all 512,
//     because 128 columns is what one mesh column owns. So each sample gets four
//     independent 128-way softmaxes, not one coherent 512-way distribution.
//   * gamma and beta are scalar constants (1.0 and 0.0), not learned per-feature
//     vectors, and the weights are pseudo-random -- there is no trained model
//     behind this.
//
// So: the right shape, the right sizes, the right dtypes and the right dataflow
// for a transformer projection plus head, with arbitrary contents. It is a
// benchmark, not an inference engine.
//
// HOW THE ENGINES ARE SCHEDULED
//
// The GEMM is cut into NN_ROUNDS rounds of NN_BLOCK_TS timeslots; round r
// finalises a PB-wide column block of Z in L2, and each tile of a mesh row then
// post-processes a quarter of that block (NN_CH rows by NN_PB columns) while a
// later round's GEMM is still running -- NN_PIPE_LAG rounds later, which is when
// the neighbour syncs alone already guarantee the block is final (derivation in
// nn_is_params.h). While it waits for RedMulE the CV32 advances the
// post-processing state machine instead of sleeping, and falls back to WFE when
// that track has nothing left to do.
//
// Measured: 394 697 cycles per inference against 461 658 for the same work run
// phase by phase, i.e. -14.5%. Two things bound that gain, and both are worth
// knowing before trying to improve it:
//
//   * Only NN_ROUNDS - NN_PIPE_LAG - 1 of the NN_ROUNDS chunks have a later GEMM
//     round to hide under; with a 4-wide mesh that is 2 of 4, so half the
//     post-processing is STRUCTURALLY exposed. Splitting the chunks finer does
//     not help -- sub-chunks of the same block all become available at the same
//     instant. Only the next inference's GEMM could cover the tail.
//   * The row pipeline still drains at every round boundary, because the tile
//     stops advancing the GEMM while it finishes a chunk -- and that holds even
//     though the CV32 is asleep inside post_drain(). Dropping the explicit
//     per-round barrier (which would have been much worse) does not avoid this;
//     it is the tile not making GEMM progress that costs, not the CV32 being
//     busy. Confirmed by moving the pump around: restricting it to the RedMulE
//     wait, away from the latency-critical iDMA and FractalSync waits, changed
//     sync by less than 6% and total not at all. The pump sits only on the
//     RedMulE wait now because that is the least invasive place, not because the
//     others were the problem.
//
// Overlapping properly would need engine-to-engine event chaining (RedMulE's
// done triggering Spatz without the CV32 in the loop) or a second control
// thread. Neither exists on magia_v3.
//
// This is safe because the event unit latches events and eu_clear_events()
// clears only the bits it is given, so events for the three engines never
// shadow each other: Spatz is bit 8 (0x100), RedMulE bit 10 (0x400), PULP bit
// 12 (0x1000), iDMA bits 2/3, FractalSync bit 24 -- and every eu32_*_wait()
// helper uses its own precise single-bit mask. (EU_REDMULE_ALL_MASK, 0xF00,
// would have swallowed Spatz's bit 8; it is never used here.)
//
// One run of the binary is exactly one iteration of the kernel. The 1000x
// benchmark loop lives on the VFIO host, which re-pushes the whole image and
// the payload on every iteration -- a cold start each time, which is the
// worst-case overhead we want to measure.
//
// Two ways to feed the kernel, selected at runtime:
//   * standalone (gvrun --param binary=...): no host, the matrices travel
//     inside the ELF (.data, see scripts/gen_nn_is_data.py).
//   * VFIO host: an nn_hdr_t with NN_MAGIC staged at NN_HDR_ADDR redirects the
//     iDMA to the payload the host DMAed into L2.

#include <stdint.h>

#include "tile.h"
#include "fsync.h"
#include "idma.h"
#include "redmule.h"
#include "eventunit.h"
#include "utils/gemm_utils.h"

#include "nn_is_params.h"
#include "nn_is_data.h"
#include "nn_spatz_task_bin.h"
#include "nn_pulp_task_bin.h"

#define WAIT_MODE      WFE

/* fp16 bit patterns of the block's hyper-parameters. */
#define F16_SCALE      (0x3000u) /* 0.125   -- tames the GEMM output range   */
#define F16_BIAS       (0x3400u) /* 0.25                                     */
#define F16_GAMMA      (0x3C00u) /* 1.0                                      */
#define F16_BETA       (0x0000u) /* 0.0                                      */
#define F16_EPS        (0x1400u) /* 2^-10                                    */

#define PULP_CORE_MASK ((1u << PULP_CORE_COUNT) - 1u)

#define NN_MESH_TILES  (MESH_X_TILES * MESH_Y_TILES)

/* Which tiles run with the hardware cycle counters enabled.
 *
 * Not free: writing mcountinhibit makes the GVSoC CV32E40P model call
 * exec.switch_to_full_mode() for good, so that core leaves the fast instruction
 * handlers for the full ones that drive the event lines. Fine for profiling,
 * but it costs simulated-instruction throughput on every enabled core -- which
 * matters once the host is looping the kernel a thousand times.
 *
 * Default: everyone, so a single run gives the full per-tile picture. For
 * throughput runs, narrow it to e.g. 0x21 (tile 0, the leftmost of row 0 which
 * sources Z from L2, plus tile 5, an interior tile with a neighbour handover on
 * both sides) or to 0 to turn profiling off entirely. */
#ifndef NN_PERF_TILE_MASK
#define NN_PERF_TILE_MASK (0xFFFFFFFFu)
#endif

/* Flags used when no VFIO host has staged a header. The golden check is on by
 * default because it is what validates the barrier-free pipelining, but it is
 * ~36k cycles of CV32 work sitting between GEMM rounds, where it plausibly costs
 * more than itself by holding the tile back from the next round. It is already
 * excluded from cyc_total; whatever sync time it induces is not. Build with
 * -DNN_STANDALONE_FLAGS=0 to measure the kernel without it. */
#ifndef NN_STANDALONE_FLAGS
#define NN_STANDALONE_FLAGS (NN_FLAG_VERIFY)
#endif

/* ==========================================================================
 * Small helpers
 * ========================================================================== */

/* Mesh-wide rendezvous.
 *
 * Guarded on purpose: on a 1x1 mesh MAX_SYNC_LVL is 0, and fsync32_sync_global()
 * then returns without ever poking the FractalSync unit -- an unconditional
 * eu_fsync_wait() behind it would sleep forever. The neighbour syncs in the GEMM
 * phase need no such guard: they are only reachable when MESH_X_TILES > 1, and
 * then MESH_2_POWER > 0 so they always fire. */
static inline void mesh_barrier(fsync_controller_t *fs, eu_controller_t *eu)
{
#if NN_MESH_TILES > 1
    fsync_sync_global(fs);
    eu_fsync_wait(eu, WAIT_MODE);
#else
    (void)fs;
    (void)eu;
#endif
}

/* Release the hardware performance counters.
 *
 * MANDATORY before perf_get_cycles() means anything on this core. The CV32E40P
 * resets mcountinhibit with every implemented bit SET, i.e. all counters
 * disabled out of reset (RTL behaviour, faithfully modelled: see
 * Cv32e40pCsr::mcountinhibit reset_val in gvsoc). While mcountinhibit.CY is
 * set, mcycle stays frozen at its reset value and CSR 0xB00 reads back 0 --
 * which is exactly what a whole run of zero-cycle measurements looks like.
 * Clearing CY re-anchors the counter to the current clock, so the count starts
 * from zero here. */
static inline void perf_counters_enable(uint32_t hartid)
{
    if ((NN_PERF_TILE_MASK >> (hartid & 31)) & 1u) {
        asm volatile("csrw 0x320, zero" ::: "memory"); /* mcountinhibit = 0 */
    }
}

/* Boot Spatz and the PULP cluster, quietly.
 *
 * Same MMIO sequence as spatz_init() / pulp_init(), minus the printf those two
 * carry inside their polling loops. On a 4x4 mesh that chatter is ~450 printf
 * calls through the stdout model, and with the host looping the kernel it is
 * paid on every single iteration. The shared SDK headers are left untouched;
 * mmio32() is a volatile access, so the empty wait loops stand. */
static void spatz_boot(uint32_t binary_start)
{
    mmio32(SPATZ_TASKBIN) = binary_start;
    mmio32(SPATZ_CLK_EN)  = 1;
    while (mmio32(SPATZ_READY) == 0) {
    }
}

static void pulp_boot(uint32_t binary_start)
{
    mmio32(PULP_BINARY) = binary_start;
    mmio32(PULP_CLK_EN) = 1;
    while (mmio32(PULP_READY) == 0) {
    }
}

/* fp16 bit pattern of 1/n, for n a power of two (all our block widths are). */
static uint16_t fp16_recip_pow2(uint32_t n)
{
    int32_t e = 15;

    while (n > 1) {
        n >>= 1;
        e--;
    }
    return (uint16_t)((uint32_t)(e & 0x1F) << 10);
}

/* Does the VFIO host have a payload staged in L2 for us? */
static uint32_t host_payload_present(void)
{
    volatile nn_hdr_t *h = (volatile nn_hdr_t *)NN_HDR_ADDR;

    return h->magic == NN_MAGIC && h->m == NN_M && h->n == NN_N && h->k == NN_K &&
           h->timeslots == NN_TIMESLOTS;
}

/* Fingerprint of a tile-local block: deterministic, and enough to tell two runs
 * apart. Read as 32-bit words -- two fp16 elements per load -- because over
 * 16384 elements the load count is anything but negligible: having this inside
 * the timed region, and spread across the round boundaries where it also held
 * the GEMM back, cost 67k cycles of the 394k total. Hence both the wider loads
 * and, more importantly, the caller pulling the whole band back from L2 and
 * fingerprinting it only after the clock has stopped. */
static uint32_t block_checksum(uint32_t addr, uint32_t elems)
{
    uint32_t sum = 0;
    uint32_t i;

    for (i = 0; i < elems / 2; i++) {
        sum += *(volatile uint32_t *)(addr + i * 4);
    }
    return sum;
}

/* ==========================================================================
 * Post-processing track
 *
 * A three-step chain on one chunk -- Spatz activation, Spatz LayerNorm, PULP
 * softmax -- advanced one step per call, never blocking. Each step is launched
 * and left running; the next call picks it up when its event has landed. That is
 * what lets the chain span several GEMM timeslots: the PULP softmax alone runs
 * far longer than a single RedMulE job.
 *
 * The parameter blocks are constant across rounds (same L1 addresses, same
 * shapes), so they are configured once and only the L2 source/destination move,
 * which the gather and store outside the track take care of.
 * ========================================================================== */

enum { POST_IDLE = 0, POST_ACT, POST_LN, POST_PULP };

typedef struct {
    uint32_t state;
    uint32_t t_step; /* cycle at which the in-flight step was launched */

    volatile nn_spatz_act_params_t *act_p;
    volatile nn_spatz_ln_params_t *ln_p;
    volatile nn_pulp_params_t *pulp_p;

    uint32_t cyc_spatz; /* Spatz busy time, launch -> observed done */
    uint32_t cyc_pulp;  /* PULP cluster busy time                   */
    uint32_t errors;
} post_ctx_t;

static void post_launch(post_ctx_t *pc)
{
    eu_clear_events(EU_SPATZ_DONE_MASK);
    pc->t_step = perf_get_cycles();
    spatz_run_task_with_params(NN_ACT_SPATZ_TASK, (uint32_t)pc->act_p);
    pc->state = POST_ACT;
}

/* Advance the chain by at most one step. Returns non-zero while busy. */
static uint32_t post_pump(post_ctx_t *pc)
{
    uint32_t now;

    switch (pc->state) {
    case POST_ACT:
        if (eu_check_events(EU_SPATZ_DONE_MASK) == 0) {
            return 1;
        }
        eu_clear_events(EU_SPATZ_DONE_MASK);
        now = perf_get_cycles();
        pc->cyc_spatz += now - pc->t_step;
        if (spatz_get_exit_code() != 0) {
            pc->errors++;
        }
        pc->t_step = now;
        spatz_run_task_with_params(NN_LN_SPATZ_TASK, (uint32_t)pc->ln_p);
        pc->state = POST_LN;
        return 1;

    case POST_LN:
        if (eu_check_events(EU_SPATZ_DONE_MASK) == 0) {
            return 1;
        }
        eu_clear_events(EU_SPATZ_DONE_MASK);
        now = perf_get_cycles();
        pc->cyc_spatz += now - pc->t_step;
        if (spatz_get_exit_code() != 0) {
            pc->errors++;
        }
        pc->t_step = now;
        pulp_run_task_with_params(NN_SOFTMAX_PULP_TASK, (uint32_t)pc->pulp_p, PULP_CORE_MASK);
        pc->state = POST_PULP;
        return 1;

    case POST_PULP:
        if (eu_check_events(EU_PULP_EVT_MASK) == 0) {
            return 1;
        }
        eu_clear_events(EU_PULP_EVT_MASK);
        pc->cyc_pulp += perf_get_cycles() - pc->t_step;
        pc->state = POST_IDLE;
        return 0;

    default:
        return 0;
    }
}

/* Sleep until an event lands, WITHOUT consuming it.
 *
 * eu_wait_events(..., WFE, ...) clears the mask on the way out, which would rob
 * post_pump() of the event it is about to act on. Same loop, no clear. */
static inline void eu_sleep_until(uint32_t mask)
{
    while (eu_check_events(mask) == 0) {
        evt_read32(EU_CORE_EVENT_WAIT);
    }
}

/* Run the chain to completion, sleeping between steps rather than spinning. */
static void post_drain(post_ctx_t *pc)
{
    while (pc->state != POST_IDLE) {
        eu_sleep_until((pc->state == POST_PULP) ? EU_PULP_EVT_MASK : EU_SPATZ_DONE_MASK);
        post_pump(pc);
    }
}

/* Wait for an event, spending the wait on the post track when there is one.
 *
 * With nothing to overlap we sleep in WFE instead of spinning, so an idle CV32
 * costs no simulated instructions. */
static void wait_evt(uint32_t mask, post_ctx_t *pc)
{
    if (pc == NULL || pc->state == POST_IDLE) {
        eu_wait_events(mask, WAIT_MODE, 0);
        return;
    }
    while (eu_check_events(mask) == 0) {
        if (post_pump(pc) == 0) {
            eu_wait_events(mask, WAIT_MODE, 0);
            return;
        }
    }
    eu_clear_events(mask);
}

/* ==========================================================================
 * GEMM track
 *
 * Tile (y, x) owns X[y*TH .. +TH][x*TW .. +TW] and keeps it resident in L1 for
 * the whole phase. Per timeslot it pulls its W slice, receives the running Z
 * accumulator from its left neighbour (or from L2 if it is leftmost), folds its
 * own contribution in with RedMulE, and hands Z to its right neighbour (or
 * writes it to L2 if it is rightmost).
 *
 * Z is double-buffered by timeslot parity. That is what makes the handover safe
 * with only pairwise neighbour syncs: while tile x reads buffer parity(i) from
 * tile x-1, tile x-1 can only have advanced to timeslot i+1, which writes the
 * other buffer. It cannot reach timeslot i+2 (which reuses parity(i)) because
 * its sync_right(i+1) is gated on tile x's sync_left(i+1), which in turn happens
 * only after tile x finished the copy.
 * ========================================================================== */

typedef struct {
    idma_controller_t *idma;
    redmule_controller_t *rm;
    fsync_controller_t *fs;
    eu_controller_t *eu;

    uint32_t hartid;
    uint32_t x_id;
    uint32_t y_id;
    uint32_t l1;
    uint32_t row_l2; /* first L2 row owned by this tile */

    uint32_t x_l2;
    uint32_t w_l2;
    uint32_t y_l2;

    uint32_t cyc_dma;  /* iDMA busy time  */
    uint32_t cyc_gemm; /* RedMulE busy time */
    uint32_t cyc_sync; /* FractalSync wait time */
} gemm_ctx_t;

/* X stays put for the whole phase: one 2D transfer, then never again. */
static void gemm_load_x(gemm_ctx_t *g)
{
    uint32_t t0 = perf_get_cycles();

    idma_memcpy_2d(g->idma,
                   0,
                   g->x_l2 + (g->row_l2 * NN_N + g->x_id * NN_TW) * 2,
                   g->l1 + NN_L1_OFF_X,
                   NN_TW * 2,
                   NN_N * 2,
                   NN_TH);
    wait_evt(EU_IDMA_A2O_DONE_MASK, NULL);
    g->cyc_dma += perf_get_cycles() - t0;
}

/* One round: NN_BLOCK_TS consecutive timeslots. pc may be NULL. */
static void gemm_round(gemm_ctx_t *g, uint32_t r, post_ctx_t *pc)
{
    const uint32_t a_x    = g->l1 + NN_L1_OFF_X;
    const uint32_t a_w    = g->l1 + NN_L1_OFF_W;
    const uint32_t a_z[2] = {g->l1 + NN_L1_OFF_Z0, g->l1 + NN_L1_OFF_Z1};
    uint32_t t0;
    uint32_t i;

    for (i = r * NN_BLOCK_TS; i < (r + 1) * NN_BLOCK_TS; i++) {
        const uint32_t a_z_cur = a_z[i & 1];

        /* W slice for this timeslot: W[x*TW .. +TW][i*TS .. +TS] */
        t0 = perf_get_cycles();
        idma_memcpy_2d(g->idma,
                       0,
                       g->w_l2 + ((g->x_id * NN_TW) * NN_K + i * NN_TS) * 2,
                       a_w,
                       NN_TS * 2,
                       NN_K * 2,
                       NN_TW);
        wait_evt(EU_IDMA_A2O_DONE_MASK, NULL);
        g->cyc_dma += perf_get_cycles() - t0;

        /* Inbound Z accumulator */
        if (g->x_id == 0) {
            t0 = perf_get_cycles();
            idma_memcpy_2d(g->idma,
                           0,
                           g->y_l2 + (g->row_l2 * NN_K + i * NN_TS) * 2,
                           a_z_cur,
                           NN_TS * 2,
                           NN_K * 2,
                           NN_TH);
            wait_evt(EU_IDMA_A2O_DONE_MASK, NULL);
            g->cyc_dma += perf_get_cycles() - t0;
        } else {
            t0 = perf_get_cycles();
            fsync_sync_left(g->fs);
            wait_evt(EU_FSYNC_DONE_MASK, NULL);
            g->cyc_sync += perf_get_cycles() - t0;

            /* Straight L1-to-L1 pull of the neighbour's buffer: same shape, so
             * source stride == row length and the transfer is contiguous. */
            t0 = perf_get_cycles();
            idma_memcpy_2d(g->idma,
                           0,
                           get_l1_base(g->hartid - 1) + ((i & 1) ? NN_L1_OFF_Z1 : NN_L1_OFF_Z0),
                           a_z_cur,
                           NN_TS * 2,
                           NN_TS * 2,
                           NN_TH);
            wait_evt(EU_IDMA_A2O_DONE_MASK, NULL);
            g->cyc_dma += perf_get_cycles() - t0;
        }

        /* Z += X * W. The longest single wait in the timeslot, and therefore the
         * main window in which the post track gets to advance. */
        t0 = perf_get_cycles();
        redmule_gemm(g->rm, a_x, a_w, a_z_cur, (uint16_t)NN_TH, (uint16_t)NN_TW, (uint16_t)NN_TS);
        wait_evt(EU_REDMULE_DONE_MASK, pc);
        g->cyc_gemm += perf_get_cycles() - t0;

        /* Outbound Z accumulator */
        if (g->x_id == MESH_X_TILES - 1) {
            t0 = perf_get_cycles();
            idma_memcpy_2d(g->idma,
                           1,
                           g->y_l2 + (g->row_l2 * NN_K + i * NN_TS) * 2,
                           a_z_cur,
                           NN_TS * 2,
                           NN_K * 2,
                           NN_TH);
            wait_evt(EU_IDMA_O2A_DONE_MASK, NULL);
            g->cyc_dma += perf_get_cycles() - t0;
        } else {
            t0 = perf_get_cycles();
            fsync_sync_right(g->fs);
            wait_evt(EU_FSYNC_DONE_MASK, NULL);
            g->cyc_sync += perf_get_cycles() - t0;
        }
    }
}

/* ==========================================================================
 * Chunk movement
 *
 * This tile's slice of column block b: NN_CH rows starting at
 * row_l2 + x_id*NN_CH, by the full NN_PB columns starting at b*NN_PB.
 *
 * Slicing by rows and not by columns is deliberate -- LayerNorm is row-wise over
 * NN_PB columns, so whole rows stay with one tile and the global result is
 * bit-identical to the non-pipelined version.
 * ========================================================================== */

static uint32_t chunk_row0(const gemm_ctx_t *g, uint32_t x_id)
{
    return g->row_l2 + x_id * NN_CH;
}

static void chunk_gather(gemm_ctx_t *g, uint32_t x_id, uint32_t b)
{
    uint32_t t0 = perf_get_cycles();

    idma_memcpy_2d(g->idma,
                   0,
                   g->y_l2 + (chunk_row0(g, x_id) * NN_K + b * NN_PB) * 2,
                   g->l1 + NN_L1_OFF_P,
                   NN_PB * 2,
                   NN_K * 2,
                   NN_CH);
    wait_evt(EU_IDMA_A2O_DONE_MASK, NULL);
    g->cyc_dma += perf_get_cycles() - t0;
}

static void chunk_store(gemm_ctx_t *g, uint32_t q_l2, uint32_t x_id, uint32_t b)
{
    uint32_t t0 = perf_get_cycles();

    idma_memcpy_2d(g->idma,
                   1,
                   q_l2 + (chunk_row0(g, x_id) * NN_K + b * NN_PB) * 2,
                   g->l1 + NN_L1_OFF_Q,
                   NN_PB * 2,
                   NN_K * 2,
                   NN_CH);
    wait_evt(EU_IDMA_O2A_DONE_MASK, NULL);
    g->cyc_dma += perf_get_cycles() - t0;
}

/* ==========================================================================
 * Golden check on one chunk of Z
 *
 * Each sample carries its own absolute tolerance (nn_z_tol, in millis), which
 * the generator derives from the fp16 accumulation error bound
 * eps*sqrt(N)*sum|x*w| -- see the comment in scripts/gen_nn_is_data.py for why a
 * tolerance relative to the *result* is wrong for a GEMM with cancellation.
 *
 * Alongside pass/fail we track the worst fraction of that error budget any
 * sample consumed, in per-mille: 1000 means "exactly at the bound". That is the
 * number worth watching, because the bound is conservative and a healthy run
 * should sit far below it.
 * ========================================================================== */
static uint32_t verify_chunk(uint32_t p_addr,
                             uint32_t row0,
                             uint32_t col0,
                             uint32_t hartid,
                             uint32_t *n_checked,
                             uint32_t *worst_permille)
{
    uint32_t errors = 0;
    uint32_t s;

    for (s = 0; s < NN_NSAMPLES; s++) {
        uint32_t idx = nn_z_idx[s];
        uint32_t row = idx / NN_K;
        uint32_t col = idx % NN_K;
        uint32_t off, used;
        int32_t got, want, diff, tol;

        if (row < row0 || row >= row0 + NN_CH) {
            continue;
        }
        if (col < col0 || col >= col0 + NN_PB) {
            continue;
        }

        off  = (row - row0) * NN_PB + (col - col0);
        got  = fp16_to_millis(*(volatile uint16_t *)(p_addr + off * 2));
        want = fp16_to_millis(nn_z_val[s]);
        diff = (got > want) ? (got - want) : (want - got);
        tol  = (int32_t)nn_z_tol[s];
        (*n_checked)++;

        /* Guard the *1000 against a wildly wrong value (a broken GEMM can hand
         * us an fp16 inf, which fp16_to_millis maps to INT32_MAX). */
        used = (diff > 1000000) ? 0xFFFFFFFFu : (uint32_t)((diff * 1000) / tol);
        if (used > *worst_permille) {
            *worst_permille = used;
        }

        if (diff > tol) {
            if (errors < 2) {
                printf("[t%u] MISMATCH Z[%u][%u]: got %d milli, want %d milli, tol %d\n",
                       (unsigned)hartid,
                       (unsigned)row,
                       (unsigned)col,
                       (int)got,
                       (int)want,
                       (int)tol);
            }
            errors++;
        }
    }

    return errors;
}

/* Golden check on the chunk sitting in L1 right now, timed apart from the
 * kernel: it is a debug feature, and every sample costs a div/mod.
 *
 * Runs in host mode too. The golden samples embedded in the ELF stay valid as
 * long as the host stages a payload generated from the same seed
 * (NN_DATA_SEED), which is what scripts/gen_nn_payload.py produces -- so the
 * first VFIO iterations are self-checking instead of blind. The host clears
 * NN_FLAG_VERIFY once it moves to its own data, or for throughput runs.
 *
 * It also validates the barrier-free pipelining: a block gathered too early
 * would hold a partially accumulated Z and fail here. */
static uint32_t verify_gathered(gemm_ctx_t *g,
                                uint32_t x_id,
                                uint32_t b,
                                uint32_t hartid,
                                uint32_t flags,
                                uint32_t *checked,
                                uint32_t *worst_dev,
                                uint32_t *cyc_verify)
{
    uint32_t errors, t0;

    if ((flags & NN_FLAG_VERIFY) == 0) {
        return 0;
    }

    t0     = perf_get_cycles();
    errors = verify_chunk(
        g->l1 + NN_L1_OFF_P, chunk_row0(g, x_id), b * NN_PB, hartid, checked, worst_dev);
    *cyc_verify += perf_get_cycles() - t0;

    return errors;
}

/* ==========================================================================
 * main
 * ========================================================================== */
int main(void)
{
    const uint32_t hartid = get_hartid();
    const uint32_t x_id   = GET_X_ID(hartid);
    const uint32_t y_id   = GET_Y_ID(hartid);
    const uint32_t l1     = get_l1_base(hartid);

    idma_config_t idma_cfg      = {.hartid = hartid};
    idma_controller_t idma_ctrl = {.base = NULL, .cfg = &idma_cfg, .api = &idma_api};

    redmule_config_t rm_cfg      = {.hartid = hartid};
    redmule_controller_t rm_ctrl = {.base = NULL, .cfg = &rm_cfg, .api = &redmule_api};

    fsync_config_t fs_cfg      = {.hartid = hartid};
    fsync_controller_t fs_ctrl = {.base = NULL, .cfg = &fs_cfg, .api = &fsync_api};

    eu_config_t eu_cfg      = {.hartid = hartid};
    eu_controller_t eu_ctrl = {.base = NULL, .cfg = &eu_cfg, .api = &eu_api};

    volatile nn_pulp_core_res_t *pulp_res =
        (volatile nn_pulp_core_res_t *)(l1 + NN_L1_OFF_CTL + NN_CTL_OFF_RES);

    gemm_ctx_t g;
    post_ctx_t pc;

    uint32_t host_mode, iter_id, flags;
    uint32_t t_run, t0;
    uint32_t errors = 0, checked = 0, worst_dev = 0;
    uint32_t cyc_verify = 0, cyc_post = 0;
    uint32_t chk = 0;
    uint32_t r, i;
    int32_t inflight; /* block index on the post track, -1 when the track is idle */

    /* ---- controllers ---- */
    perf_counters_enable(hartid);

    fsync_init(&fs_ctrl);
    idma_init(&idma_ctrl);
    redmule_init(&rm_ctrl);

    eu_init(&eu_ctrl);
    eu_fsync_init(&eu_ctrl, 0);
    eu_redmule_init(&eu_ctrl, 0);
    eu_idma_init(&eu_ctrl, 0);
    eu_pulp_init(&eu_ctrl, 0);
    eu_spatz_init(&eu_ctrl, 0);

    spatz_boot(SPATZ_BINARY_START);
    pulp_boot(PULP_BINARY_START);

    /* ---- where does the data come from? ---- */
    host_mode = host_payload_present();
    if (host_mode) {
        volatile nn_hdr_t *h = (volatile nn_hdr_t *)NN_HDR_ADDR;
        iter_id              = h->iter_id;
        flags                = h->flags;
        g.x_l2               = NN_X_ADDR;
        g.w_l2               = NN_W_ADDR;
        g.y_l2               = NN_Y_ADDR;
    } else {
        iter_id = 0;
        flags   = NN_STANDALONE_FLAGS;
        g.x_l2  = (uint32_t)nn_x_inp;
        g.w_l2  = (uint32_t)nn_w_inp;
        g.y_l2  = (uint32_t)nn_y_inp;
    }

    {
        const uint32_t q_l2 = host_mode ? NN_Q_ADDR : (uint32_t)nn_q_out;

        g.idma     = &idma_ctrl;
        g.rm       = &rm_ctrl;
        g.fs       = &fs_ctrl;
        g.eu       = &eu_ctrl;
        g.hartid   = hartid;
        g.x_id     = x_id;
        g.y_id     = y_id;
        g.l1       = l1;
        g.row_l2   = y_id * NN_TH;
        g.cyc_dma  = 0;
        g.cyc_gemm = 0;
        g.cyc_sync = 0;

        /* Post-track parameter blocks: shape is identical every round, so they
         * are written once here and never touched again. */
        pc.state     = POST_IDLE;
        pc.t_step    = 0;
        pc.cyc_spatz = 0;
        pc.cyc_pulp  = 0;
        pc.errors    = 0;
        pc.act_p     = (volatile nn_spatz_act_params_t *)(l1 + NN_L1_OFF_CTL + NN_CTL_OFF_ACT);
        pc.ln_p      = (volatile nn_spatz_ln_params_t *)(l1 + NN_L1_OFF_CTL + NN_CTL_OFF_LN);
        pc.pulp_p    = (volatile nn_pulp_params_t *)(l1 + NN_L1_OFF_CTL + NN_CTL_OFF_PULP);

        pc.act_p->src   = l1 + NN_L1_OFF_P;
        pc.act_p->dst   = l1 + NN_L1_OFF_P;
        pc.act_p->len   = NN_CH * NN_PB;
        pc.act_p->scale = F16_SCALE;
        pc.act_p->bias  = F16_BIAS;

        pc.ln_p->src      = l1 + NN_L1_OFF_P;
        pc.ln_p->rows     = NN_CH;
        pc.ln_p->cols     = NN_PB;
        pc.ln_p->gamma    = F16_GAMMA;
        pc.ln_p->beta     = F16_BETA;
        pc.ln_p->eps      = F16_EPS;
        pc.ln_p->inv_cols = fp16_recip_pow2(NN_PB);

        pc.pulp_p->src           = l1 + NN_L1_OFF_P;
        pc.pulp_p->dst           = l1 + NN_L1_OFF_Q;
        pc.pulp_p->res           = l1 + NN_L1_OFF_CTL + NN_CTL_OFF_RES;
        pc.pulp_p->rows          = NN_CH;
        pc.pulp_p->cols          = NN_PB;
        pc.pulp_p->rows_per_core = NN_PULP_ROWS_PER_CORE;

        for (i = 0; i < PULP_CORE_COUNT; i++) {
            pulp_res[i].argmax    = 0;
            pulp_res[i].maxval    = 0;
            pulp_res[i].sumsq     = 0;
            pulp_res[i].rows_done = 0;
        }

        if (hartid == 0) {
            printf("[nn_is_bench] mesh %ux%u, M=%u N=%u K=%u, timeslots=%u\n",
                   (unsigned)MESH_X_TILES,
                   (unsigned)MESH_Y_TILES,
                   (unsigned)NN_M,
                   (unsigned)NN_N,
                   (unsigned)NN_K,
                   (unsigned)NN_TIMESLOTS);
            printf("[nn_is_bench] per tile: TH=%u TW=%u TS=%u PB=%u\n",
                   (unsigned)NN_TH,
                   (unsigned)NN_TW,
                   (unsigned)NN_TS,
                   (unsigned)NN_PB);
            printf("[nn_is_bench] pipelined: %u rounds x %u timeslots, chunk %ux%u, "
                   "%u PULP cores x %u rows\n",
                   (unsigned)NN_ROUNDS,
                   (unsigned)NN_BLOCK_TS,
                   (unsigned)NN_CH,
                   (unsigned)NN_PB,
                   (unsigned)PULP_CORE_COUNT,
                   (unsigned)NN_PULP_ROWS_PER_CORE);
            printf("[nn_is_bench] data source: %s, iter_id=%u\n",
                   host_mode ? "VFIO host payload in L2" : "embedded in ELF",
                   (unsigned)iter_id);
        }

        /* Line every tile up before the clock starts, so cyc_total measures the
         * kernel and not the staggered boot. */
        mesh_barrier(&fs_ctrl, &eu_ctrl);

        t_run = perf_get_cycles();

        gemm_load_x(&g);

        /* ---- the pipeline ----
         *
         * Round r's GEMM runs with an earlier chunk in flight on Spatz and the
         * PULP cluster. Block b becomes readable NN_PIPE_LAG rounds after the
         * round that produced it (see the derivation in nn_is_params.h) -- no
         * per-round barrier, so the systolic row pipeline is never forced to
         * drain and the fill/drain overhead stays at (MESH_X-1)/NN_TIMESLOTS.
         *
         * inflight is the block index currently on the post track, -1 for none.
         * A single P/Q buffer pair is enough: a chunk is always drained and
         * stored before the next one is gathered. */
        inflight = -1;

        for (r = 0; r < NN_ROUNDS; r++) {
            gemm_round(&g, r, (inflight >= 0) ? &pc : NULL);

            if (r >= NN_PIPE_LAG) {
                uint32_t b = r - NN_PIPE_LAG;

                if (inflight >= 0) {
                    t0 = perf_get_cycles();
                    post_drain(&pc);
                    cyc_post += perf_get_cycles() - t0;
                    chunk_store(&g, q_l2, x_id, (uint32_t)inflight);
                }

                chunk_gather(&g, x_id, b);
                errors +=
                    verify_gathered(&g, x_id, b, hartid, flags, &checked, &worst_dev, &cyc_verify);
                post_launch(&pc);
                inflight = (int32_t)b;
            }
        }

        /* Every tile has finished its GEMM, so every block is final in L2. */
        t0 = perf_get_cycles();
        mesh_barrier(&fs_ctrl, &eu_ctrl);
        g.cyc_sync += perf_get_cycles() - t0;

        if (inflight >= 0) {
            t0 = perf_get_cycles();
            post_drain(&pc);
            cyc_post += perf_get_cycles() - t0;
            chunk_store(&g, q_l2, x_id, (uint32_t)inflight);
            inflight = -1;
        }

        /* Tail: these blocks only became readable once the GEMM was over, so
         * there is nothing left to hide them under. Together with the chunk
         * drained just above at the post-GEMM barrier, that is what makes up
         * post_exposed -- half the post-processing on a 4-wide mesh. */
        for (r = NN_ROUNDS - NN_PIPE_LAG; r < NN_ROUNDS; r++) {
            chunk_gather(&g, x_id, r);
            errors +=
                verify_gathered(&g, x_id, r, hartid, flags, &checked, &worst_dev, &cyc_verify);
            post_launch(&pc);

            t0 = perf_get_cycles();
            post_drain(&pc);
            cyc_post += perf_get_cycles() - t0;
            chunk_store(&g, q_l2, x_id, r);
        }

        t0 = perf_get_cycles();
        mesh_barrier(&fs_ctrl, &eu_ctrl);
        g.cyc_sync += perf_get_cycles() - t0;

        t_run = perf_get_cycles() - t_run - cyc_verify;

        /* Fingerprint, after the clock has stopped. The four chunks together
         * span rows [row0, row0+NN_CH) over every column, so a single 2D pull
         * brings this tile's whole share of Q back into L1. */
        idma_memcpy_2d(&idma_ctrl,
                       0,
                       q_l2 + chunk_row0(&g, x_id) * NN_K * 2,
                       l1 + NN_L1_OFF_P,
                       NN_K * 2,
                       NN_K * 2,
                       NN_CH);
        wait_evt(EU_IDMA_A2O_DONE_MASK, NULL);
        chk = block_checksum(l1 + NN_L1_OFF_P, NN_CH * NN_K);
    }

    errors += pc.errors;

    for (i = 0; i < PULP_CORE_COUNT; i++) {
        if (pulp_res[i].rows_done != NN_PULP_ROWS_PER_CORE) {
            printf("[t%u] PULP core %u stopped after %u/%u rows\n",
                   (unsigned)hartid,
                   (unsigned)i,
                   (unsigned)pulp_res[i].rows_done,
                   (unsigned)NN_PULP_ROWS_PER_CORE);
            errors++;
        }
    }

    /* ---- stats: L2 slot per tile, so the host can DMA them back ---- */
    {
        volatile nn_tile_stats_t *st = (volatile nn_tile_stats_t *)NN_STAT_ADDR + hartid;
        /* Engine busy times are measured launch -> observed-done, so with the
         * tracks overlapping they legitimately sum to more than the wall time.
         * That excess IS the overlap, and is the headline number here. */
        uint32_t sum = g.cyc_dma + g.cyc_gemm + pc.cyc_spatz + pc.cyc_pulp + g.cyc_sync;

        st->hartid          = hartid;
        st->iter_id         = iter_id;
        st->cyc_total       = t_run;
        st->cyc_dma         = g.cyc_dma;
        st->cyc_gemm        = g.cyc_gemm;
        st->cyc_spatz       = pc.cyc_spatz;
        st->cyc_pulp        = pc.cyc_pulp;
        st->cyc_sync        = g.cyc_sync;
        st->checksum        = chk;
        st->argmax          = pulp_res[0].argmax;
        st->errors          = errors;
        st->samples_checked = checked;
        st->worst_dev       = worst_dev;
        st->cyc_verify      = cyc_verify;
        st->cyc_post        = cyc_post;
        st->cyc_overlap     = (sum > t_run) ? (sum - t_run) : 0;

        if (hartid == 0 || errors != 0 || (flags & NN_FLAG_VERBOSE)) {
            printf("[t%u] total=%u dma=%u gemm=%u spatz=%u pulp=%u sync=%u "
                   "post_exposed=%u overlap=%u chk=0x%x samples=%u dev=%u/1000 "
                   "errors=%u vfy=%u\n",
                   (unsigned)hartid,
                   (unsigned)t_run,
                   (unsigned)g.cyc_dma,
                   (unsigned)g.cyc_gemm,
                   (unsigned)pc.cyc_spatz,
                   (unsigned)pc.cyc_pulp,
                   (unsigned)g.cyc_sync,
                   (unsigned)cyc_post,
                   (unsigned)st->cyc_overlap,
                   (unsigned)chk,
                   (unsigned)checked,
                   (unsigned)worst_dev,
                   (unsigned)errors,
                   (unsigned)cyc_verify);
        }
    }

    return (int)errors;
}
