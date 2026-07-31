// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/*
 * TinyViT-5M end to end on the tile mesh, one layer at a time.
 *
 * All 296 layers of the graph run collectively on every tile at the model's native
 * 224x224 input. Weights and the input image sit in L2 for the whole run and each tile
 * pulls in only the slice it needs; each layer's output goes back to L2 before the next
 * layer starts, and a mesh-wide barrier between layers is what makes the next layer's
 * reads safe. Same shape as onnx_resnet18 and onnx_mobilevit_v2, with four differences
 * worth knowing:
 *
 *   - the graph's 64 Reshapes, its Flatten and two degenerate Transposes are pure views,
 *     so the generator aliases them into the layer table instead of emitting layers.
 *     That is what takes 363 ONNX nodes down to 296 layers.
 *   - its 10 Splits are NOT views, unlike MobileViT's: they cut the innermost axis into
 *     three runs of 32, so each of q, k and v is strided. Each Split is one de-interleave
 *     Transpose instead, after which the three outputs are contiguous and are aliased.
 *   - the 40 MatMuls against a weight go to the gemm kernel with beta = 0 (it shards M
 *     and has an alignment guard); the 20 batched attention MatMuls go to the matmul
 *     kernel, which shards M within each batch. Four of those have an odd O = 49 and take
 *     matmul's scalar fallback.
 *   - 38 of the 72 Adds broadcast a bias over the last axis and go to add_bcast.
 *
 * Which dimension gets split is otherwise left to each kernel - they all derive their
 * shard from get_hartid() internally.
 *
 * Pass/fail is the 1000 logits, checked bit-exactly against a golden that replays the
 * kernels' own FP16 arithmetic - they are the product of 295 of the 296 layers, so this
 * is a check on the whole network. The softmax on top of them is reported but not failed
 * on. The predicted class is checked as well.
 *
 * Build TVIT_MAX_LAYERS=n to stop after n layers, for bringing the thing up against the
 * per-layer tensors the generator drops in test_data/layers/.
 *
 * Every tile reports the cycles it spent, split into compute and barrier. That is two CSR
 * reads per layer and one printf per tile at the end, so it is on unconditionally - unlike
 * CONV2DGEMM_PROFILE, which prints per layer per tile. The barrier share is mostly this
 * tile waiting for the slowest one, so it doubles as a load-imbalance readout, and hart
 * 0's total is the inference's cycle count.
 */

#include "tile.h"

#include "eventunit.h"
#include "fsync.h"

#include "kernel_test_utils.h"
#include "performance_utils.h"
#include "tinyvit_graph.h"

#include "add_bcast_fp16_spatz_params.h"
#include "add_fp16_spatz.h"
#include "conv2dgemm_fp16_spatz.h"
#include "gelu_fp16_spatz.h"
#include "gemm_fp16_spatz.h"
#include "globalaveragepool_fp16_spatz.h"
#include "layernorm_fp16_spatz.h"
#include "matmul_fp16_spatz.h"
#include "softmax_fp16_spatz.h"
#include "transpose_fp16_spatz.h"

#define HID get_hartid()

/* The blob, loaded from the ELF into L2 by tinyvit_data.S. */
extern const uint8_t tinyvit_data[];

/*
 * Every layer's output, all resident at once, so a residual branch can read a tensor
 * produced several layers earlier without any liveness analysis. NOLOAD, so it costs
 * nothing in the ELF; nothing zeroes it, which is fine because every element is written
 * before it is read.
 */
static float16 tinyvit_arena[TVIT_ARENA_BYTES / sizeof(float16)]
    __attribute__((section(".l2_arena"), aligned(4)));

extern uint32_t _spatz_binary_start; /* CV32 linker script */

static inline float16 *arena_at(uint32_t off)
{
    return (float16 *)((uintptr_t)tinyvit_arena + off);
}

static inline const float16 *blob_at(uint32_t off)
{
    return (const float16 *)((uintptr_t)tinyvit_data + off);
}

static inline const float16 *operand(uint8_t space, uint32_t off)
{
    return space == TVIT_SPACE_BLOB ? blob_at(off) : arena_at(off);
}

static void run_layer(const tvit_layer_t *L)
{
    const float16 *src0 = operand(L->src0_space, L->src0_off);
    float16 *dst        = arena_at(L->dst_off);

    uint32_t in_shape[4]  = {L->in_shape[0], L->in_shape[1], L->in_shape[2], L->in_shape[3]};
    uint32_t out_shape[4] = {L->out_shape[0], L->out_shape[1], L->out_shape[2], L->out_shape[3]};

    switch (L->op) {
    case TVIT_CONV:
        MAGIA_conv2dgemm_fp16_spatz(src0,
                                    blob_at(L->w_off),
                                    L->has_bias ? blob_at(L->b_off) : NULL,
                                    dst,
                                    in_shape,
                                    out_shape,
                                    L->kernel_h,
                                    L->kernel_w,
                                    L->stride_h,
                                    L->stride_w,
                                    L->pad_h,
                                    L->pad_w,
                                    L->group,
                                    (int)L->has_bias);
        break;

    case TVIT_GELU:
        MAGIA_gelu_fp16_spatz(src0, dst, L->numel);
        break;

    case TVIT_LAYERNORMALIZATION: {
        /* Normalizes over the last axis, so the tensor goes in flattened to
         * [rows, row_len]; 20 of the 21 have no bias and get a zero vector. */
        uint32_t ln_shape[2] = {L->ln_rows, L->ln_len};

        MAGIA_layernorm_fp16_spatz(src0,
                                   blob_at(L->w_off),
                                   blob_at(L->b_off),
                                   (float16)TVIT_LN_EPS,
                                   ln_shape,
                                   2,
                                   dst);
        break;
    }

    case TVIT_MATMUL2D: {
        /*
         * A MatMul against a weight is a plain GEMM once the leading axes are folded
         * into M. beta = 0, so C is never read - the destination buffer is passed for it
         * because gemm stages C unconditionally and wants a valid [M, N] address.
         */
        uint32_t a_shape[2] = {L->mm_m, L->mm_k};
        uint32_t b_shape[2] = {L->mm_k, L->mm_o};
        uint32_t c_shape[2] = {L->mm_m, L->mm_o};
        uint32_t y_shape[2] = {L->mm_m, L->mm_o};

        MAGIA_gemm_fp16_spatz(src0,
                              blob_at(L->w_off),
                              dst,
                              (float16)1.0f,
                              (float16)0.0f,
                              0,
                              0,
                              a_shape,
                              b_shape,
                              c_shape,
                              y_shape,
                              dst);
        break;
    }

    case TVIT_MATMUL:
        /* Both operands are activations and both are batched over the heads. */
        MAGIA_matmul_fp16_spatz(src0,
                                operand(L->src1_space, L->src1_off),
                                dst,
                                L->mm_m,
                                L->mm_k,
                                L->mm_o,
                                L->mm_batches,
                                1,
                                1);
        break;

    case TVIT_ADD:
        MAGIA_add_fp16_spatz(src0, operand(L->src1_space, L->src1_off), dst, L->numel);
        break;

    case TVIT_ADDBCAST:
        MAGIA_add_bcast_fp16_spatz(src0,
                                   operand(L->src1_space, L->src1_off),
                                   dst,
                                   L->rows,
                                   L->row_len,
                                   L->bcast_mode);
        break;

    case TVIT_SOFTMAX: {
        /* The kernel takes rows = shape[0]*shape[1]*shape[2] and a row length of
         * shape[3], so the axis-(-1) softmax is passed flattened that way. */
        uint32_t sm_shape[4] = {L->sm_rows, 1, 1, L->sm_len};

        MAGIA_softmax_fp16_spatz(src0, dst, sm_shape);
        break;
    }

    case TVIT_TRANSPOSE: {
        /* Shapes and perm come from the table with the leading axes already merged; the
         * kernel takes them by non-const pointer, so copy them out. */
        uint32_t perm[TVIT_MAX_RANK];
        uint32_t t_in[TVIT_MAX_RANK];
        uint32_t t_out[TVIT_MAX_RANK];

        for (uint32_t i = 0; i < L->rank; i++) {
            perm[i]  = L->perm[i];
            t_in[i]  = L->t_in_shape[i];
            t_out[i] = L->t_out_shape[i];
        }

        MAGIA_transpose_fp16_spatz(src0, dst, perm, t_in, t_out, L->rank, L->iterations);
        break;
    }

    case TVIT_GLOBALAVERAGEPOOL:
        MAGIA_globalaveragepool_fp16_spatz(src0, dst, in_shape);
        break;

    case TVIT_GEMM: {
        /*
         * Called transposed: A is the [1000, 320] fc weight, B the 320-long embedding as
         * a column, Y a [1000, 1] column - which is the same bytes as [1, 1000]. The
         * kernel shards the GEMM's M, so the ONNX orientation (M = 1) would leave the
         * whole layer on tile 0.
         */
        uint32_t a_shape[2] = {L->out_shape[1], L->in_shape[1]};
        uint32_t b_shape[2] = {L->in_shape[1], 1};
        uint32_t c_shape[2] = {L->out_shape[1], 1};
        uint32_t y_shape[2] = {L->out_shape[1], 1};

        MAGIA_gemm_fp16_spatz(blob_at(L->w_off),
                              src0,
                              blob_at(L->b_off),
                              (float16)1.0f,
                              (float16)1.0f,
                              0,
                              0,
                              a_shape,
                              b_shape,
                              c_shape,
                              y_shape,
                              dst);
        break;
    }

    default:
        if (HID == 0)
            printf("[CV32 (%d)] unhandled op %d in layer %s\n", HID, L->op, L->name);
        break;
    }
}

int main(void)
{
    fsync_config_t fsync_cfg;
    fsync_controller_t fsync_ctrl;
    eu_config_t eu_cfg;
    eu_controller_t eu_ctrl;
    unsigned int cyc_compute = 0;
    unsigned int cyc_barrier = 0;
    bool ok = true;

    if (HID == 0)
        printf("\n########## ONNX_TINYVIT5M (224x224, %d layers) on %d Tiles ##########\n\n",
               TVIT_MAX_LAYERS,
               NUM_HARTS);

    /* Brings up the event unit and Spatz. Must come before eu_fsync_init: it calls
     * eu_init, which clears the event mask. */
    kt_spatz_init((uint32_t)&_spatz_binary_start);

    eu_cfg.hartid = HID;
    eu_ctrl.base  = 0;
    eu_ctrl.cfg   = &eu_cfg;
    eu_ctrl.api   = &eu_api;
    eu_fsync_init(&eu_ctrl, 0);

    fsync_cfg.hartid = HID;
    fsync_ctrl.base  = 0;
    fsync_ctrl.cfg   = &fsync_cfg;
    fsync_ctrl.api   = &fsync_api;
    fsync_init(&fsync_ctrl);

    /* crt0 clears .bss from hart 0 only, so no tile may touch L2 until that is done. */
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WFE);

    for (uint32_t i = 0; i < TVIT_MAX_LAYERS; i++) {
        const tvit_layer_t *L = &tinyvit_layers[i];

        unsigned int t_layer;
        unsigned int t_barrier;

        if (HID == 0)
            printf("[CV32 (%d)] layer %s\n", HID, L->name);

        t_layer = perf_get_cycles();

        run_layer(L);

        t_barrier = perf_get_cycles();

        /* Every tile's shard of this layer is in L2 before any tile reads it back. */
        fsync_sync_global(&fsync_ctrl);
        eu_fsync_wait(&eu_ctrl, WFE);

        cyc_compute += t_barrier - t_layer;
        cyc_barrier += perf_get_cycles() - t_barrier;

#ifdef CONV2DGEMM_PROFILE
        printf("[CV32 (%d)] [prof] %s compute %u barrier %u\n",
               HID,
               L->name,
               t_barrier - t_layer,
               perf_get_cycles() - t_barrier);
#endif
    }

    spatz_clk_dis();

    /* Parsed by the run scripts. hart 0's total is the inference's cycle count; the
     * barrier share is how long this tile spent waiting for the slowest one. */
    printf("[CV32 (%d)] [cycles] total %u compute %u barrier %u\n",
           HID,
           cyc_compute + cyc_barrier,
           cyc_compute,
           cyc_barrier);

    /*
     * Partial run: there is no compiled-in golden for an intermediate layer, so dump the
     * head and tail of the last one computed. test_data/layers/<nnn>_<op>.npy holds the
     * same tensor, so a few values are enough to tell whether that layer is right.
     */
    if (HID == 0 && TVIT_MAX_LAYERS != TVIT_NUM_LAYERS) {
        const tvit_layer_t *L = &tinyvit_layers[TVIT_MAX_LAYERS - 1];
        const float16 *y      = arena_at(L->dst_off);

        printf("\n[CV32 (%d)] stopped after %s (%d elements)\n", HID, L->name, L->numel);
        printf("[CV32 (%d)] head:", HID);
        for (uint32_t i = 0; i < 8 && i < L->numel; i++)
            printf(" %04x", kt_bits(y[i]));
        printf("\n[CV32 (%d)] tail:", HID);
        for (uint32_t i = (L->numel > 8 ? L->numel - 8 : 0); i < L->numel; i++)
            printf(" %04x", kt_bits(y[i]));
        printf("\n\n");
    }

    if (HID == 0 && TVIT_MAX_LAYERS == TVIT_NUM_LAYERS) {
        const float16 *logits        = arena_at(TVIT_LOGITS_OFF);
        const float16 *golden_logits = blob_at(TVIT_GOLDEN_LOGITS_OFF);
        const float16 *output        = arena_at(TVIT_OUTPUT_OFF);
        const float16 *golden_output = blob_at(TVIT_GOLDEN_OUTPUT_OFF);
        uint32_t top1                = 0;

        printf("\n[CV32 (%d)] logits (bit-exact, pass/fail):\n", HID);
        ok = kt_check(logits, golden_logits, TVIT_LOGITS_LEN, TVIT_LOGITS_ULP_TOLL);

        for (uint32_t i = 1; i < TVIT_OUTPUT_LEN; i++)
            if (output[i] > output[top1])
                top1 = i;

        printf("[CV32 (%d)] softmax (informational):\n", HID);
        (void)kt_check(output, golden_output, TVIT_OUTPUT_LEN, 0);

        printf("[CV32 (%d)] top-1 class %d, expected %d\n", HID, top1, TVIT_TOP1);
        if (top1 != TVIT_TOP1)
            ok = false;

        printf("[CV32 (%d)] Test %s\n", HID, ok ? "SUCCESS" : "FAILED");
        printf("\n####################################################################\n\n");
    }

    return ok ? 0 : -1;
}
