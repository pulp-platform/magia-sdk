// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/*
 * MobileViT-v2 end to end on the tile mesh, one layer at a time.
 *
 * All 205 layers of the graph run collectively on every tile at the model's native
 * 256x256 input. Weights and the input image sit in L2 for the whole run and each tile
 * pulls in only the slice it needs; each layer's output goes back to L2 before the next
 * layer starts, and a mesh-wide barrier between layers is what makes the next layer's
 * reads safe. Same shape as onnx_resnet18, with three differences worth knowing:
 *
 *   - the graph's 12 Reshapes, 9 Splits and 1 Flatten are all pure NCHW views, so the
 *     generator aliases them into the layer table instead of emitting layers. That is
 *     why there is no Split kernel here and why 227 ONNX nodes come to 205 layers.
 *   - the six Transposes are passed with their leading axes merged, because the kernel
 *     shards axis 0 and that is the batch. See test_data/generator.py.
 *   - 18 of the 43 Muls broadcast, and go to MAGIA_mul_bcast_fp16_spatz.
 *
 * Which dimension gets split is otherwise left to each kernel - they all derive their
 * shard from get_hartid() internally.
 *
 * Pass/fail is the 1000 logits, checked bit-exactly against a golden that replays the
 * kernels' own FP16 arithmetic - they are the product of 204 of the 205 layers, so this
 * is a check on the whole network. The softmax on top of them is reported but not failed
 * on. The predicted class is checked as well.
 *
 * Build MVIT_MAX_LAYERS=n to stop after n layers, for bringing the thing up against the
 * per-layer tensors the generator drops in test_data/layers/.
 */

#include "tile.h"

#include "eventunit.h"
#include "fsync.h"

#include "kernel_test_utils.h"
#include "mobilevit_graph.h"

#ifdef CONV2DGEMM_PROFILE
#include "performance_utils.h"
#endif

#include "add_fp16_spatz.h"
#include "conv2dgemm_fp16_spatz.h"
#include "gemm_fp16_spatz.h"
#include "globalaveragepool_fp16_spatz.h"
#include "groupnorm_fp16_spatz.h"
#include "mul_bcast_fp16_spatz_params.h"
#include "mul_fp16_spatz.h"
#include "reducesum_fp16_spatz.h"
#include "relu_fp16_spatz.h"
#include "sigmoid_fp16_spatz.h"
#include "softmax_fp16_spatz.h"
#include "transpose_fp16_spatz.h"

#define HID get_hartid()

/* The blob, loaded from the ELF into L2 by mobilevit_data.S. */
extern const uint8_t mobilevit_data[];

/*
 * Every layer's output, all resident at once, so a residual branch can read a tensor
 * produced several layers earlier without any liveness analysis. NOLOAD, so it costs
 * nothing in the ELF; nothing zeroes it, which is fine because every element is written
 * before it is read.
 */
static float16 mobilevit_arena[MVIT_ARENA_BYTES / sizeof(float16)]
    __attribute__((section(".l2_arena"), aligned(4)));

extern uint32_t _spatz_binary_start; /* CV32 linker script */

static inline float16 *arena_at(uint32_t off)
{
    return (float16 *)((uintptr_t)mobilevit_arena + off);
}

static inline const float16 *blob_at(uint32_t off)
{
    return (const float16 *)((uintptr_t)mobilevit_data + off);
}

static inline const float16 *operand(uint8_t space, uint32_t off)
{
    return space == MVIT_SPACE_BLOB ? blob_at(off) : arena_at(off);
}

static void run_layer(const mvit_layer_t *L)
{
    const float16 *src0 = operand(L->src0_space, L->src0_off);
    float16 *dst        = arena_at(L->dst_off);

    uint32_t in_shape[4]  = {L->in_shape[0], L->in_shape[1], L->in_shape[2], L->in_shape[3]};
    uint32_t out_shape[4] = {L->out_shape[0], L->out_shape[1], L->out_shape[2], L->out_shape[3]};

    switch (L->op) {
    case MVIT_CONV:
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

    case MVIT_RELU:
        MAGIA_relu_fp16_spatz(src0, dst, L->numel);
        break;

    case MVIT_SIGMOID:
        MAGIA_sigmoid_fp16_spatz(src0, dst, L->numel);
        break;

    case MVIT_ADD:
        MAGIA_add_fp16_spatz(src0, operand(L->src1_space, L->src1_off), dst, L->numel);
        break;

    case MVIT_MUL:
        MAGIA_mul_fp16_spatz(src0, operand(L->src1_space, L->src1_off), dst, L->numel);
        break;

    case MVIT_MULBCAST:
        MAGIA_mul_bcast_fp16_spatz(src0,
                                   operand(L->src1_space, L->src1_off),
                                   dst,
                                   L->rows,
                                   L->row_len,
                                   L->bcast_mode);
        break;

    case MVIT_GROUPNORMALIZATION:
        MAGIA_groupnorm_fp16_spatz(src0,
                                   dst,
                                   blob_at(L->w_off),
                                   blob_at(L->b_off),
                                   in_shape,
                                   L->num_groups,
                                   (float16)MVIT_GN_EPS);
        break;

    case MVIT_SOFTMAX: {
        /* The kernel takes rows = shape[0]*shape[1]*shape[2] and a row length of
         * shape[3], so the axis-(-1) softmax is passed flattened that way. */
        uint32_t sm_shape[4] = {L->sm_rows, 1, 1, L->sm_len};

        MAGIA_softmax_fp16_spatz(src0, dst, sm_shape);
        break;
    }

    case MVIT_REDUCESUM:
        MAGIA_reducesum_fp16_spatz(src0, dst, L->outer, L->reduce, L->inner);
        break;

    case MVIT_TRANSPOSE: {
        /* Shapes and perm come from the table with the leading axes already merged; the
         * kernel takes them by non-const pointer, so copy them out. */
        uint32_t perm[MVIT_MAX_RANK];
        uint32_t t_in[MVIT_MAX_RANK];
        uint32_t t_out[MVIT_MAX_RANK];

        for (uint32_t i = 0; i < L->rank; i++) {
            perm[i]  = L->perm[i];
            t_in[i]  = L->t_in_shape[i];
            t_out[i] = L->t_out_shape[i];
        }

        MAGIA_transpose_fp16_spatz(src0, dst, perm, t_in, t_out, L->rank, L->iterations);
        break;
    }

    case MVIT_GLOBALAVERAGEPOOL:
        MAGIA_globalaveragepool_fp16_spatz(src0, dst, in_shape);
        break;

    case MVIT_GEMM: {
        /*
         * Called transposed: A is the [1000, 256] fc weight, B the 256-long embedding as
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
    bool ok = true;

    if (HID == 0)
        printf("\n########## ONNX_MOBILEVIT_V2 (256x256, %d layers) on %d Tiles ##########\n\n",
               MVIT_MAX_LAYERS,
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

    for (uint32_t i = 0; i < MVIT_MAX_LAYERS; i++) {
        const mvit_layer_t *L = &mobilevit_layers[i];

        if (HID == 0)
            printf("[CV32 (%d)] layer %s\n", HID, L->name);

#ifdef CONV2DGEMM_PROFILE
        unsigned int t_layer = perf_get_cycles();
#endif

        run_layer(L);

#ifdef CONV2DGEMM_PROFILE
        unsigned int t_barrier = perf_get_cycles();
#endif

        /* Every tile's shard of this layer is in L2 before any tile reads it back. */
        fsync_sync_global(&fsync_ctrl);
        eu_fsync_wait(&eu_ctrl, WFE);

#ifdef CONV2DGEMM_PROFILE
        /* Barrier time is mostly this tile waiting for the slowest one, so it doubles as
         * a load-imbalance readout. Printed from every tile, not just hart 0. */
        printf("[CV32 (%d)] [prof] %s compute %u barrier %u\n",
               HID,
               L->name,
               t_barrier - t_layer,
               perf_get_cycles() - t_barrier);
#endif
    }

    spatz_clk_dis();

    /*
     * Partial run: there is no compiled-in golden for an intermediate layer, so dump the
     * head and tail of the last one computed. test_data/layers/<nnn>_<op>.npy holds the
     * same tensor, so a few values are enough to tell whether that layer is right.
     */
    if (HID == 0 && MVIT_MAX_LAYERS != MVIT_NUM_LAYERS) {
        const mvit_layer_t *L = &mobilevit_layers[MVIT_MAX_LAYERS - 1];
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

    if (HID == 0 && MVIT_MAX_LAYERS == MVIT_NUM_LAYERS) {
        const float16 *logits        = arena_at(MVIT_LOGITS_OFF);
        const float16 *golden_logits = blob_at(MVIT_GOLDEN_LOGITS_OFF);
        const float16 *output        = arena_at(MVIT_OUTPUT_OFF);
        const float16 *golden_output = blob_at(MVIT_GOLDEN_OUTPUT_OFF);
        uint32_t top1                = 0;

        printf("\n[CV32 (%d)] logits (bit-exact, pass/fail):\n", HID);
        ok = kt_check(logits, golden_logits, MVIT_LOGITS_LEN, MVIT_LOGITS_ULP_TOLL);

        for (uint32_t i = 1; i < MVIT_OUTPUT_LEN; i++)
            if (output[i] > output[top1])
                top1 = i;

        printf("[CV32 (%d)] softmax (informational):\n", HID);
        (void)kt_check(output, golden_output, MVIT_OUTPUT_LEN, 0);

        printf("[CV32 (%d)] top-1 class %d, expected %d\n", HID, top1, MVIT_TOP1);
        if (top1 != MVIT_TOP1)
            ok = false;

        printf("[CV32 (%d)] Test %s\n", HID, ok ? "SUCCESS" : "FAILED");
        printf("\n####################################################################\n\n");
    }

    return ok ? 0 : -1;
}
