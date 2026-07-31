/*
 * Copyright (C) 2025 ETH Zurich and University of Bologna
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Convolution as im2col + GEMM, fp16, one shard of the output channels per tile.
 *
 * X, W, B and Y stay where the caller put them - typically L2 - and only what this
 * tile needs for the block it is working on is brought into its L1:
 *
 *   - the im2col matrix is built one column block at a time, because the whole thing
 *     does not fit in L1 for any realistic layer (a 64x112x112 output off a 3x224x224
 *     input needs 3.7 MB of it against 896 KB of L1);
 *   - the weight slice is loaded a channel block at a time, which for most layers means
 *     the whole of it, once;
 *   - each block's output is written straight back out.
 *
 * A column block is a whole number of output rows, and the number of rows per block is
 * chosen to divide H_out, so every block has the same width N and the GEMM's row strides
 * never change. N is rounded up to an even number of columns because the Spatz VLSU
 * corrupts vector accesses that are not 4-byte aligned and the task walks B, C and Y
 * with unit-stride vector accesses stepping by N; the extra column is computed and
 * discarded.
 *
 * Blocking the output channels as well as the columns is what makes the L1 footprint
 * independent of the mesh size: a tile's weight slice is c_out / NUM_HARTS rows deep, so
 * with column blocking alone a *smaller* mesh needs *more* L1 per tile, and ResNet18's
 * 512x512x3x3 convs stopped fitting below about six tiles. See plan_conv.
 *
 * The im2col itself is done by the iDMA rather than by the CV32, which matters a lot:
 * for ResNet18 at 224x224 the scalar version writes 14.7 M elements per tile per
 * inference against ~113 MMAC of actual Spatz work, so it, not the convolution,
 * would set the runtime. With stride_w == 1 an im2col row over a block of output rows
 * is a plain strided rectangle of the input and one 2D descriptor covers it. With
 * stride_w != 1 the innermost run is not contiguous and the hardware's third dimension
 * is not usable (the GVSoC magia_v3 iDMA model never issues it), so those layers keep
 * the scalar path - they are a minority of the im2col volume.
 */
#include <errno.h>
#include <stdint.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "kernels_dma_utils.h"

#include "conv2dgemm_fp16_spatz.h"
#include "conv2dgemm_fp16_spatz_params.h"
#include "conv2dgemm_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "conv2dgemm_fp16_spatz"

/*
 * Per-phase cycle accounting, off unless CONV2DGEMM_PROFILE is defined. What it answers:
 * whether a layer's time goes into moving the im2col, into the GEMM, or into the weight
 * and result transfers - which is not obvious, because the im2col is replicated per tile
 * while the GEMM is divided by it. perf_get_cycles() reads CSR 0xB00 and works on GVSoC
 * only (see performance_utils.h).
 */
#ifdef CONV2DGEMM_PROFILE
#include "performance_utils.h"

#define PROF_DECL()                                                                                \
    unsigned int prof_t0 = 0, prof_im2col = 0, prof_wload = 0, prof_spatz = 0, prof_store = 0
#define PROF_START()   (prof_t0 = perf_get_cycles())
#define PROF_STOP(acc) ((acc) += perf_get_cycles() - prof_t0)
#define PROF_REPORT(p)                                                                             \
    printf("[CV32 (%d)] [prof] %dx%d c_out %d g %d K_g %d | %dr x %dc rows %d ocblk %d | "         \
           "im2col %u wload %u spatz %u store %u\n",                                               \
           HID,                                                                                    \
           (p)->h_out,                                                                             \
           (p)->w_out,                                                                             \
           (p)->c_out,                                                                             \
           (p)->group,                                                                             \
           (p)->K_g,                                                                               \
           (p)->n_row_grp,                                                                         \
           (p)->n_col_grp,                                                                         \
           (p)->rows,                                                                              \
           (p)->oc_blk,                                                                            \
           prof_im2col,                                                                            \
           prof_wload,                                                                             \
           prof_spatz,                                                                             \
           prof_store)
#else
#define PROF_DECL()                                                                                \
    do {                                                                                           \
    } while (0)
#define PROF_START()                                                                               \
    do {                                                                                           \
    } while (0)
#define PROF_STOP(acc)                                                                             \
    do {                                                                                           \
    } while (0)
#define PROF_REPORT(p)                                                                             \
    do {                                                                                           \
    } while (0)
#endif

/* Room left in L1 for the params struct, the alpha/beta slots and alignment. */
#define L1_SLACK     (256)

#define DIV_UP(a, b) (((a) + (b) - 1) / (b))

/* Everything derived once from the layer's geometry and this tile's shard. */
typedef struct {
    uint32_t n_batches;
    uint32_t c_in;
    uint32_t c_out;
    uint32_t h_in;
    uint32_t w_in;
    uint32_t h_out;
    uint32_t w_out;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t group;
    int has_bias;

    uint32_t K;      /* c_in * kernel_h * kernel_w - the whole filter        */
    uint32_t hw_out; /* h_out * w_out - the full output plane                */

    uint32_t cin_g;  /* c_in / group - input channels one group reads        */
    uint32_t cout_g; /* c_out / group - output channels one group writes     */
    uint32_t K_g;    /* cin_g * kernel_h * kernel_w - the GEMM's K           */

    uint32_t n_row_grp; /* mesh split: row groups x channel groups          */
    uint32_t n_col_grp; /*   n_row_grp * n_col_grp == NUM_HARTS             */

    uint32_t r_start;  /* first output row owned by this tile                */
    uint32_t r_len;    /* how many rows it owns                              */
    uint32_t oc_start; /* first output channel owned by this tile            */
    uint32_t oc_len;   /* how many it owns - the GEMM's M                    */

    uint32_t rows;   /* output rows per block; divides h_out                 */
    uint32_t n_cols; /* rows * w_out - the columns a block actually produces  */
    uint32_t n_pad;  /* n_cols rounded up to even - the GEMM's N and the
                        row stride of shard_B, shard_C and shard_Y           */
    uint32_t oc_blk; /* output channels per offload; <= oc_len. Only smaller
                        than oc_len when the whole slice's weights would not
                        fit L1 alongside a column block.                     */
} conv_plan_t;

static inline void zero_fp16(uintptr_t addr, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++)
        mmio_fp16(addr + i * sizeof(float16)) = (float16)0.0f;
}

/*
 * Fills in the derived fields of the plan and picks how the work is blocked.
 *
 * L1 has to hold the weight slice for one offload plus one column block's B, C and Y:
 *
 *   oc_blk*K*2  +  (K + 2*oc_blk) * n_pad * 2  <=  L1_SIZE - slack
 *
 * Two knobs, and both are needed. Blocking only the columns (oc_blk = oc_len) leaves the
 * weight slice's size proportional to c_out / NUM_HARTS, so the *smaller* the mesh the
 * more L1 each tile needs - ResNet18's 512x512x3x3 convs want 288 KB of it at 16 tiles
 * but 1152 KB at 4, i.e. more than L1. Blocking the output channels too makes the
 * footprint independent of the tile count.
 *
 * Given a column block of n_pad columns, the largest affordable oc_blk falls out of the
 * inequality above:
 *
 *   oc_blk <= (budget - 2*K*n_pad) / (2*K + 4*n_pad)
 *
 * so the search is over the candidate block heights - the divisors of h_out, which keep
 * every block the same width and so keep the GEMM's N fixed - and picks the pair needing
 * the fewest offloads. Preferring few offloads is what keeps the weight traffic down:
 * with the column block as the outer loop, a tile reloads its weights once per column
 * block, so trading a taller block for a smaller oc_blk is usually worth it. Where the
 * whole slice already fits (oc_blk = oc_len, one offload per column block) this picks
 * exactly what the column-only version did.
 */
static int plan_conv(conv_plan_t *p,
                     uint32_t input_shape[4],
                     uint32_t output_shape[4],
                     uint32_t kernel_h,
                     uint32_t kernel_w,
                     uint32_t stride_h,
                     uint32_t stride_w,
                     uint32_t pad_h,
                     uint32_t pad_w,
                     uint32_t group,
                     int has_bias)
{
    uint32_t shard;
    uint32_t left;
    uint32_t budget;
    uint32_t best_cost;

    p->n_batches = input_shape[0];
    p->c_in      = input_shape[1];
    p->h_in      = input_shape[2];
    p->w_in      = input_shape[3];
    p->c_out     = output_shape[1];
    p->h_out     = output_shape[2];
    p->w_out     = output_shape[3];
    p->kernel_h  = kernel_h;
    p->kernel_w  = kernel_w;
    p->stride_h  = stride_h;
    p->stride_w  = stride_w;
    p->pad_h     = pad_h;
    p->pad_w     = pad_w;
    p->group     = group ? group : 1;
    p->has_bias  = has_bias;

    p->K      = p->c_in * kernel_h * kernel_w;
    p->hw_out = p->h_out * p->w_out;

    /*
     * A grouped convolution is `group` independent convolutions side by side: output
     * channel oc only ever reads input channels [g*cin_g, (g+1)*cin_g). Running them one
     * group at a time makes the GEMM's K the group's own K_g, instead of padding A out to
     * the full K with zeros - which for a depthwise layer (cin_g = cout_g = 1, K_g = 9)
     * means c_in times fewer multiply-adds for exactly the same result: the terms dropped
     * are `acc = fma(0, b, acc)`, and the ones that remain keep their ascending-k order.
     */
    p->cin_g  = p->c_in / p->group;
    p->cout_g = p->c_out / p->group;
    p->K_g    = p->cin_g * kernel_h * kernel_w;

    /*
     * Split the mesh over BOTH the output rows and the output channels.
     *
     * Sharding channels alone (the obvious reading of "one GEMM row set per tile") makes
     * every tile walk the whole output and so build the *entire* im2col for itself. That
     * work is then replicated NUM_HARTS times and does not shrink as the mesh grows,
     * while the GEMM does - which is why 64 tiles came out slower than 16. Sharding rows
     * alone has the mirror problem: every tile needs every weight.
     *
     * So pick the split that minimises what one tile has to move:
     *
     *   cost(Nr, Nc) = G_t*K_g*H_out*W_out / Nr  +  C_out*K_g / Nc,   Nr * Nc == NUM_HARTS
     *
     * where G_t is the number of groups the tile's channel slice touches - a tile only
     * builds im2col rows for the groups it actually has output channels in. For an
     * ungrouped convolution G_t is 1 and K_g is K, i.e. exactly the old expression.
     *
     * Early layers (large planes, few channels) come out heavily row-split, late ones
     * (small planes, many channels) fall back to the channel-only split, which is the
     * right answer there and is exactly what this used to do unconditionally.
     */
    {
        uint32_t wt_unit = p->c_out * p->K_g; /* weight elements, replicated per channel grp */
        uint32_t best    = 0;

        p->n_row_grp = 0;

        for (uint32_t nr = 1; nr <= (uint32_t)NUM_HARTS; nr++) {
            uint32_t nc = (uint32_t)NUM_HARTS / nr;
            uint32_t groups_owned;
            uint32_t cost;

            if ((uint32_t)NUM_HARTS % nr != 0)
                continue;
            /* Every group must have at least one row / one channel to work on. */
            if (nr > p->h_out || nc > p->c_out)
                continue;

            groups_owned = DIV_UP(p->c_out / nc, p->cout_g);

            cost = (groups_owned * p->K_g * p->hw_out) / nr + wt_unit / nc;

            if (p->n_row_grp == 0 || cost < best) {
                best         = cost;
                p->n_row_grp = nr;
                p->n_col_grp = nc;
            }
        }

        /* Fewer channels and rows than tiles: keep the old split and let the tiles with
         * nothing to do fall out below. */
        if (p->n_row_grp == 0) {
            p->n_row_grp = 1;
            p->n_col_grp = NUM_HARTS;
        }
    }

    {
        uint32_t ri = HID / p->n_col_grp; /* which row band this tile owns     */
        uint32_t ci = HID % p->n_col_grp; /* which channel group it owns       */

        shard      = p->h_out / p->n_row_grp;
        left       = p->h_out % p->n_row_grp;
        p->r_start = ri * shard + (ri < left ? ri : left);
        p->r_len   = shard + (ri < left ? 1 : 0);

        shard       = p->c_out / p->n_col_grp;
        left        = p->c_out % p->n_col_grp;
        p->oc_start = ci * shard + (ci < left ? ci : left);
        p->oc_len   = shard + (ci < left ? 1 : 0);
    }

    if (p->oc_len == 0 || p->r_len == 0) {
        p->oc_len = 0;
        p->rows = p->n_cols = p->n_pad = p->oc_blk = 0;
        return 0;
    }

    budget    = L1_SIZE - L1_SLACK;
    best_cost = 0;
    p->rows = p->oc_blk = 0;

    /* Block heights divide this tile's row band, so every block is the same width and
     * the GEMM's N never changes within the layer. */
    for (uint32_t rows = p->r_len; rows >= 1; rows--) {
        uint32_t n_cols;
        uint32_t n_pad;
        uint32_t b_bytes;
        uint32_t oc_blk;
        uint32_t cost;

        if (p->r_len % rows != 0)
            continue;

        n_cols = rows * p->w_out;
        n_pad  = n_cols + (n_cols & 1u);

        /* B alone is K_g*n_pad elements; bail before it can overflow the arithmetic. */
        if (n_pad > budget / (2u * p->K_g))
            continue;

        b_bytes = 2u * p->K_g * n_pad;
        oc_blk  = (budget - b_bytes) / (2u * p->K_g + 4u * n_pad);
        if (oc_blk == 0)
            continue;
        if (oc_blk > p->oc_len)
            oc_blk = p->oc_len;
        /* A channel block shares one im2col, so it may not straddle two groups. */
        if (oc_blk > p->cout_g)
            oc_blk = p->cout_g;

        /* One Spatz offload per (column block, channel block). */
        cost = DIV_UP(p->oc_len, oc_blk) * (p->r_len / rows);

        if (p->rows == 0 || cost < best_cost) {
            best_cost = cost;
            p->rows   = rows;
            p->oc_blk = oc_blk;
            p->n_cols = n_cols;
            p->n_pad  = n_pad;
        }
    }

    /* Nothing fits even one output row of one output channel: would need K blocking. */
    if (p->rows == 0)
        return ENOMEM;

    return 0;
}

static int alloc_l1(const conv_plan_t *p, volatile conv2dgemm_fp16_spatz_params_t **out)
{
    volatile conv2dgemm_fp16_spatz_params_t *cp;
    uintptr_t shard_A;
    uintptr_t shard_B;
    uintptr_t shard_C;
    uintptr_t shard_Y;
    uintptr_t alpha;
    uintptr_t beta;

    l1_alloc_init();

    cp = l1_alloc(sizeof(conv2dgemm_fp16_spatz_params_t));
    if (!cp)
        return ENOMEM;

    shard_A = (uintptr_t)l1_alloc(p->oc_blk * p->K_g * sizeof(float16));
    shard_B = (uintptr_t)l1_alloc(p->K_g * p->n_pad * sizeof(float16));
    /* Only read when beta != 0, but keep a valid aligned slot either way. */
    shard_C = (uintptr_t)l1_alloc((p->has_bias ? p->oc_blk * p->n_pad : 1) * sizeof(float16));
    shard_Y = (uintptr_t)l1_alloc(p->oc_blk * p->n_pad * sizeof(float16));
    alpha   = (uintptr_t)l1_alloc(sizeof(float16));
    beta    = (uintptr_t)l1_alloc(sizeof(float16));

    if (!shard_A || !shard_B || !shard_C || !shard_Y || !alpha || !beta)
        return ENOMEM;

    cp->shard_A = shard_A;
    cp->shard_B = shard_B;
    cp->shard_C = shard_C;
    cp->shard_Y = shard_Y;
    cp->alpha   = alpha;
    cp->beta    = beta;

    /* GEMM does Y = alpha * A @ B + beta * C; beta gates the bias term. */
    mmio_fp16(alpha) = (float16)1.0f;
    mmio_fp16(beta)  = p->has_bias ? (float16)1.0f : (float16)0.0f;

    /* One block per offload. M is set per offload (the last channel block is short);
     * N is the fixed block width. */
    cp->M         = p->oc_blk;
    cp->N         = p->n_pad;
    cp->K         = p->K_g;
    cp->n_batches = 1;
    cp->has_bias  = (uint32_t)p->has_bias;
    cp->c_out     = p->c_out;
    cp->group     = p->group;
    cp->oc_start  = p->oc_start;
    cp->c_in      = p->c_in;
    cp->h_in      = p->h_in;
    cp->w_in      = p->w_in;
    cp->h_out     = p->h_out;
    cp->w_out     = p->w_out;
    cp->kernel_h  = p->kernel_h;
    cp->kernel_w  = p->kernel_w;
    cp->stride_h  = p->stride_h;
    cp->stride_w  = p->stride_w;
    cp->pad_h     = p->pad_h;
    cp->pad_w     = p->pad_w;

    *out = cp;

    return 0;
}

/*
 * Loads the weight rows for the output channels [oc0, oc0 + oc_n), which always sit in
 * one group.
 *
 * ONNX W is [C_out, C_in/group, K_h, K_w] = [C_out, K_g] contiguous, so those rows are
 * one contiguous run whatever the grouping is and a single transfer moves them. (The
 * scalar per-row scatter this used to need was an artefact of padding A out to the full
 * K with zeros; running one group at a time removes both the scatter and the padding.)
 */
static void load_weights(const conv_plan_t *p,
                         volatile conv2dgemm_fp16_spatz_params_t *cp,
                         kdma_t *d,
                         const float16 *W,
                         uint32_t oc0,
                         uint32_t oc_n)
{
    kdma_in_2d(d,
               (uintptr_t)(W + (uintptr_t)oc0 * p->K_g),
               cp->shard_A,
               oc_n * p->K_g * sizeof(float16),
               0,
               0,
               1);
}

/*
 * Broadcasts the channel block's biases over the block's columns. The block width is
 * fixed for the layer, so this only has to be redone when the channel block changes.
 */
static void load_bias(const conv_plan_t *p,
                      volatile conv2dgemm_fp16_spatz_params_t *cp,
                      const float16 *B,
                      uint32_t oc0,
                      uint32_t oc_n)
{
    uintptr_t shard_C = cp->shard_C;

    if (!p->has_bias)
        return;

    for (uint32_t m = 0; m < oc_n; m++) {
        float16 b     = B[oc0 + m];
        uintptr_t row = shard_C + (uintptr_t)m * p->n_pad * sizeof(float16);

        for (uint32_t n = 0; n < p->n_pad; n++)
            mmio_fp16(row + n * sizeof(float16)) = b;
    }
}

/*
 * Builds the im2col block for output rows [r0, r0 + rows) of group g with the iDMA.
 *
 * B[k][n] = X[b, g*cin_g + ic, oh*stride_h - pad_h + ki, ow*stride_w - pad_w + kj], zero
 * where that falls outside the input, with k = (ic*K_h + ki)*K_w + kj over the group's own
 * cin_g input channels and n = (oh-r0)*W_out + ow. Ungrouped, g is 0 and ic runs over the
 * whole input.
 *
 * For stride_w == 1 the valid part of one such row is a rectangle of the input plane:
 * consecutive ow are consecutive input columns, and consecutive oh are input rows
 * stride_h apart. So per (ic, ki, kj) it is one 2D descriptor - which needs the
 * destination stride to stay W_out even when the transfer is narrower than that (the
 * padded columns), hence idma_memcpy_2d_ex rather than idma_memcpy_2d.
 *
 * Only the parts the padding actually invalidates are zeroed: at most the first/last
 * output rows of the block and, per row, the columns whose window hangs off the left or
 * right edge. That is O(rows + W_out) per row of B instead of O(rows * W_out).
 */
static void im2col_block_dma(const conv_plan_t *p,
                             volatile conv2dgemm_fp16_spatz_params_t *cp,
                             kdma_t *d,
                             const float16 *X_b,
                             uint32_t r0,
                             uint32_t g)
{
    uintptr_t shard_B    = cp->shard_B;
    uint32_t in_hw       = p->h_in * p->w_in;
    const float16 *X_g   = X_b + (uintptr_t)g * p->cin_g * in_hw;

    /*
     * 1x1 / stride 1 / no padding: im2col row ic *is* input plane ic, so the block is one
     * strided transfer - c_in rows of n_cols contiguous elements, in_hw apart in L2 and
     * n_pad apart in L1 - instead of one descriptor per input channel. Bytes moved are
     * the same either way; what this saves is c_in descriptors, each of which costs a
     * program-and-wait round trip because the DMA cannot be queued.
     *
     * Inert for ResNet18, whose three 1x1 convolutions are stride 2 and so take the
     * scalar path. It is here for mobilevit_v2 (51 of its 61 convolutions are exactly
     * this shape) and tinyvit5m (10).
     */
    if (p->kernel_h == 1 && p->kernel_w == 1 && p->stride_h == 1 && p->pad_h == 0 &&
        p->pad_w == 0) {
        if (p->n_pad != p->n_cols)
            for (uint32_t ic = 0; ic < p->cin_g; ic++)
                mmio_fp16(shard_B + ((uintptr_t)ic * p->n_pad + p->n_cols) * sizeof(float16)) =
                    (float16)0.0f;

        kdma_in_2d(d,
                   (uintptr_t)(X_g + (uintptr_t)r0 * p->w_in),
                   shard_B,
                   p->n_cols * sizeof(float16),
                   in_hw * sizeof(float16),
                   p->n_pad * sizeof(float16),
                   p->cin_g);
        return;
    }

    for (uint32_t ic = 0; ic < p->cin_g; ic++) {
        const float16 *plane = X_g + (uintptr_t)ic * in_hw;

        for (uint32_t ki = 0; ki < p->kernel_h; ki++) {
            /* Output rows whose window row lands inside the input. */
            int num_lo = (int)p->pad_h - (int)ki;
            int oh_lo  = num_lo <= 0 ? 0 : (num_lo + (int)p->stride_h - 1) / (int)p->stride_h;
            int num_hi = (int)p->h_in - 1 - (int)ki + (int)p->pad_h;
            int oh_hi  = num_hi < 0 ? -1 : num_hi / (int)p->stride_h;

            if (oh_lo < (int)r0)
                oh_lo = (int)r0;
            if (oh_hi > (int)(r0 + p->rows) - 1)
                oh_hi = (int)(r0 + p->rows) - 1;

            for (uint32_t kj = 0; kj < p->kernel_w; kj++) {
                uint32_t k    = (ic * p->kernel_h + ki) * p->kernel_w + kj;
                uintptr_t row = shard_B + (uintptr_t)k * p->n_pad * sizeof(float16);
                int ow_lo;
                int ow_hi;

                /* The even-rounding column is never stored back, but must be finite. */
                if (p->n_pad != p->n_cols)
                    mmio_fp16(row + p->n_cols * sizeof(float16)) = (float16)0.0f;

                /* Output columns whose window column lands inside the input
                 * (stride_w == 1, so iw = ow + kj - pad_w). */
                ow_lo = (int)p->pad_w - (int)kj;
                if (ow_lo < 0)
                    ow_lo = 0;
                ow_hi = (int)p->w_in - 1 - (int)kj + (int)p->pad_w;
                if (ow_hi > (int)p->w_out - 1)
                    ow_hi = (int)p->w_out - 1;

                if (oh_lo > oh_hi || ow_lo > ow_hi) {
                    zero_fp16(row, p->n_cols);
                    continue;
                }

                /* Output rows of this block that the window misses entirely. */
                for (int oh = (int)r0; oh < oh_lo; oh++)
                    zero_fp16(row + (uintptr_t)(oh - (int)r0) * p->w_out * sizeof(float16),
                              p->w_out);
                for (int oh = oh_hi + 1; oh < (int)(r0 + p->rows); oh++)
                    zero_fp16(row + (uintptr_t)(oh - (int)r0) * p->w_out * sizeof(float16),
                              p->w_out);

                /* Columns the window misses, in the rows it does reach. */
                if (ow_lo > 0 || ow_hi < (int)p->w_out - 1) {
                    for (int oh = oh_lo; oh <= oh_hi; oh++) {
                        uintptr_t r = row + (uintptr_t)(oh - (int)r0) * p->w_out * sizeof(float16);

                        zero_fp16(r, (uint32_t)ow_lo);
                        zero_fp16(r + (uintptr_t)(ow_hi + 1) * sizeof(float16),
                                  p->w_out - 1 - (uint32_t)ow_hi);
                    }
                }

                {
                    uint32_t ih   = (uint32_t)(oh_lo * (int)p->stride_h + (int)ki - (int)p->pad_h);
                    uint32_t iw   = (uint32_t)(ow_lo + (int)kj - (int)p->pad_w);
                    uint32_t nrow = (uint32_t)(oh_hi - oh_lo + 1);
                    uint32_t ncol = (uint32_t)(ow_hi - ow_lo + 1);

                    kdma_in_2d(d,
                               (uintptr_t)(plane + (uintptr_t)ih * p->w_in + iw),
                               row + ((uintptr_t)(oh_lo - (int)r0) * p->w_out + (uintptr_t)ow_lo) *
                                         sizeof(float16),
                               ncol * sizeof(float16),
                               p->stride_h * p->w_in * sizeof(float16),
                               p->w_out * sizeof(float16),
                               nrow);
                }
            }
        }
    }
}

/*
 * Same block, built by the CV32. Used when stride_w != 1: the innermost run of an im2col
 * row is then strided, which would need the iDMA's third dimension, and the GVSoC
 * magia_v3 model does not implement it (it programs one dmrep/dmstr pair and ignores the
 * *_STRIDE_3 / REPS_3 registers, so such a descriptor would quietly run as 2D).
 */
static void im2col_block_scalar(const conv_plan_t *p,
                                volatile conv2dgemm_fp16_spatz_params_t *cp,
                                const float16 *X_b,
                                uint32_t r0,
                                uint32_t g)
{
    uintptr_t shard_B  = cp->shard_B;
    uint32_t in_hw     = p->h_in * p->w_in;
    const float16 *X_g = X_b + (uintptr_t)g * p->cin_g * in_hw;

    for (uint32_t ic = 0; ic < p->cin_g; ic++) {
        const float16 *plane = X_g + (uintptr_t)ic * in_hw;

        for (uint32_t ki = 0; ki < p->kernel_h; ki++) {
            for (uint32_t kj = 0; kj < p->kernel_w; kj++) {
                uint32_t k    = (ic * p->kernel_h + ki) * p->kernel_w + kj;
                uintptr_t row = shard_B + (uintptr_t)k * p->n_pad * sizeof(float16);

                for (uint32_t r = 0; r < p->rows; r++) {
                    int ih = (int)((r0 + r) * p->stride_h) - (int)p->pad_h + (int)ki;

                    for (uint32_t ow = 0; ow < p->w_out; ow++) {
                        int iw    = (int)(ow * p->stride_w) - (int)p->pad_w + (int)kj;
                        float16 v = (float16)0.0f;

                        if (ih >= 0 && ih < (int)p->h_in && iw >= 0 && iw < (int)p->w_in)
                            v = plane[(uint32_t)ih * p->w_in + (uint32_t)iw];

                        mmio_fp16(row + (r * p->w_out + ow) * sizeof(float16)) = v;
                    }
                }

                if (p->n_pad != p->n_cols)
                    mmio_fp16(row + p->n_cols * sizeof(float16)) = (float16)0.0f;
            }
        }
    }
}

static int offload_spatz_task(kdma_t *d, volatile conv2dgemm_fp16_spatz_params_t *cp)
{
    int ret;

    spatz_run_task_with_params(CONV2DGEMM_FP16_SPATZ_TASK, (uint32_t)cp);

    ret = eu_spatz_wait(&d->eu, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return ret;
    }

    return spatz_get_exit_code();
}

/*
 * Writes one (channel block, column block) back. shard_Y is [oc_n, n_pad] in L1 and the
 * destination rows are hw_out apart in L2, so both sides are strided and one transfer
 * covers it. The even-rounding column is dropped here by transferring only n_cols of
 * each row.
 */
static void store_block(const conv_plan_t *p,
                        volatile conv2dgemm_fp16_spatz_params_t *cp,
                        kdma_t *d,
                        float16 *Y_b,
                        uint32_t r0,
                        uint32_t oc0,
                        uint32_t oc_n)
{
    float16 *dst = Y_b + (uintptr_t)oc0 * p->hw_out + (uintptr_t)r0 * p->w_out;

    kdma_out_2d(d,
                (uintptr_t)dst,
                cp->shard_Y,
                p->n_cols * sizeof(float16),
                p->hw_out * sizeof(float16),
                p->n_pad * sizeof(float16),
                oc_n);
}

void MAGIA_conv2dgemm_fp16_spatz(const float16 *X,
                                 const float16 *W,
                                 const float16 *B,
                                 float16 *Y,
                                 uint32_t input_shape[4],
                                 uint32_t output_shape[4],
                                 uint32_t kernel_h,
                                 uint32_t kernel_w,
                                 uint32_t stride_h,
                                 uint32_t stride_w,
                                 uint32_t pad_h,
                                 uint32_t pad_w,
                                 uint32_t group,
                                 int has_bias)
{
    volatile conv2dgemm_fp16_spatz_params_t *cp;
    conv_plan_t p;
    kdma_t d;
    int ret;

    PROF_DECL();

    ret = plan_conv(&p,
                    input_shape,
                    output_shape,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    group,
                    has_bias);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Layer does not fit in L1 (error %d)\n", HID, KERNEL_NAME, ret);
        return;
    }

    /* With more tiles than output channels some tiles have nothing to do. */
    if (p.oc_len == 0)
        return;

    kdma_open(&d);

    ret = alloc_l1(&p, &cp);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    /*
     * The column block is the outer loop so the im2col - much the most expensive part - is
     * built once and reused by every channel block over it. Inside it comes the group
     * loop, because a group's channels are the only ones its im2col serves; ungrouped,
     * there is exactly one group and this is the loop that was here before.
     *
     * The weights are the cheaper side and get reloaded per column block; when the whole
     * slice fits (oc_blk == oc_len, the common case) there is a single channel block,
     * loaded_oc stops it reloading, and this is exactly the single-loop version. oc0 is a
     * global output-channel index, so it alone identifies the weights in shard_A - which
     * is what makes that one-entry cache correct across groups.
     */
    for (uint32_t b = 0; b < p.n_batches; b++) {
        const float16 *X_b = X + (uintptr_t)b * p.c_in * p.h_in * p.w_in;
        float16 *Y_b       = Y + (uintptr_t)b * p.c_out * p.hw_out;
        uint32_t oc_end    = p.oc_start + p.oc_len;
        uint32_t g_first   = p.oc_start / p.cout_g;
        uint32_t g_last    = (oc_end - 1) / p.cout_g;
        uint32_t loaded_oc = UINT32_MAX;

        for (uint32_t r0 = p.r_start; r0 < p.r_start + p.r_len; r0 += p.rows) {
            for (uint32_t g = g_first; g <= g_last; g++) {
                uint32_t oc_lo = g * p.cout_g;
                uint32_t oc_hi = oc_lo + p.cout_g;

                if (oc_lo < p.oc_start)
                    oc_lo = p.oc_start;
                if (oc_hi > oc_end)
                    oc_hi = oc_end;

                PROF_START();
                if (p.stride_w == 1)
                    im2col_block_dma(&p, cp, &d, X_b, r0, g);
                else
                    im2col_block_scalar(&p, cp, X_b, r0, g);
                PROF_STOP(prof_im2col);

                for (uint32_t oc0 = oc_lo; oc0 < oc_hi; oc0 += p.oc_blk) {
                    uint32_t oc_n = oc_hi - oc0;

                    if (oc_n > p.oc_blk)
                        oc_n = p.oc_blk;

                    if (oc0 != loaded_oc) {
                        PROF_START();
                        load_weights(&p, cp, &d, W, oc0, oc_n);
                        load_bias(&p, cp, B, oc0, oc_n);
                        PROF_STOP(prof_wload);
                        loaded_oc = oc0;
                    }

                    cp->M = oc_n;

                    PROF_START();
                    ret = offload_spatz_task(&d, cp);
                    PROF_STOP(prof_spatz);
                    if (ret != 0) {
                        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n",
                               HID,
                               KERNEL_NAME,
                               ret);
                        return;
                    }

                    PROF_START();
                    store_block(&p, cp, &d, Y_b, r0, oc0, oc_n);
                    PROF_STOP(prof_store);
                }
            }
        }
    }

    PROF_REPORT(&p);
}
