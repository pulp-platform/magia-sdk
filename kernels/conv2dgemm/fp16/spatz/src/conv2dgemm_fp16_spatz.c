#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "kernel_idma_utils.h"
#include "kernels_profiling_utils.h"

#include "conv2dgemm_fp16_spatz.h"
#include "conv2dgemm_fp16_spatz_params.h"
#include "conv2dgemm_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "conv2dgemm_fp16_spatz"

static int alloc_l1(void **params,
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
    volatile conv2dgemm_fp16_spatz_params_t *conv_params;
    uintptr_t shard_X;
    uintptr_t shard_A;
    uintptr_t shard_B;
    uintptr_t shard_C;
    uintptr_t shard_Y;
    uintptr_t alpha;
    uintptr_t beta;

    size_t n_batches;
    size_t c_in;
    size_t c_out;
    size_t K;
    size_t K_g;
    size_t hw_in;
    size_t hw_out;
    size_t shard;
    size_t left;
    size_t oc_start;
    size_t oc_len;

    /* im2col over all input channels gives B = [K = C_in * K_h * K_w, HW_out]. Each output
       channel's weights are compact (K_g = K / group values); the grouped GEMM multiplies them
       by only that channel's group block of B (see the Spatz task). group == 1 -> K_g == K. */
    n_batches = input_shape[0];
    c_in      = input_shape[1];
    c_out     = output_shape[1];
    K         = c_in * kernel_h * kernel_w;
    K_g       = K / group;
    hw_in     = input_shape[2] * input_shape[3];
    hw_out    = output_shape[2] * output_shape[3];

    /* Shard the output channels across the tiles (GEMM M dimension). */
    shard    = c_out / NUM_HARTS;
    left     = c_out % NUM_HARTS;
    oc_start = HID * shard + (HID < left ? HID : left);
    oc_len   = shard + (HID < left ? 1 : 0);

    l1_alloc_init();

    conv_params = l1_alloc(sizeof(conv2dgemm_fp16_spatz_params_t));
    if (!conv_params)
        return ENOMEM;

    /* The whole input (X) is brought to L1, so im2col reads it locally instead of from L2. */
    shard_X = l1_alloc(n_batches * c_in * hw_in * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    /* Weights (A) and bias (C) are shared across batches; the im2col matrix (B) and
       the output (Y) are batched so the Spatz task can sweep all batches in one offload. */
    shard_A = l1_alloc(oc_len * K_g * sizeof(float16));
    if (!shard_A)
        return ENOMEM;

    /* A 1x1 convolution with unit stride and no padding has im2col(X) == X, byte for byte:
       B is the input itself, so it gets no buffer and im2col has nothing to do. */
    if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && pad_h == 0 &&
        pad_w == 0) {
        shard_B = shard_X;
    } else {
        shard_B = l1_alloc(n_batches * K * hw_out * sizeof(float16));
        if (!shard_B)
            return ENOMEM;
    }

    shard_C = l1_alloc(oc_len * hw_out * sizeof(float16));
    if (!shard_C)
        return ENOMEM;

    shard_Y = l1_alloc(n_batches * oc_len * hw_out * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    alpha = l1_alloc(sizeof(float16));
    if (!alpha)
        return ENOMEM;

    beta = l1_alloc(sizeof(float16));
    if (!beta)
        return ENOMEM;

    conv_params->shard_X   = shard_X;
    conv_params->shard_A   = shard_A;
    conv_params->shard_B   = shard_B;
    conv_params->shard_C   = shard_C;
    conv_params->shard_Y   = shard_Y;
    conv_params->alpha     = alpha;
    conv_params->beta      = beta;
    conv_params->M         = (uint32_t)oc_len;
    conv_params->N         = (uint32_t)hw_out;
    conv_params->K         = (uint32_t)K;
    conv_params->has_bias  = (uint32_t)has_bias;
    conv_params->n_batches = (uint32_t)n_batches;
    conv_params->c_out     = (uint32_t)c_out;
    conv_params->group     = group;
    conv_params->oc_start  = (uint32_t)oc_start;
    conv_params->c_in      = (uint32_t)c_in;
    conv_params->h_in      = input_shape[2];
    conv_params->w_in      = input_shape[3];
    conv_params->h_out     = output_shape[2];
    conv_params->w_out     = output_shape[3];
    conv_params->kernel_h  = kernel_h;
    conv_params->kernel_w  = kernel_w;
    conv_params->stride_h  = stride_h;
    conv_params->stride_w  = stride_w;
    conv_params->pad_h     = pad_h;
    conv_params->pad_w     = pad_w;

    *params = (void *)conv_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X, const float16 *W, const float16 *B)
{
    volatile conv2dgemm_fp16_spatz_params_t *conv_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t oc_start;
    uint32_t K_g;
    uint32_t M;
    uint32_t N;
    uint32_t total_in;

    conv_params = (volatile conv2dgemm_fp16_spatz_params_t *)params;
    K_g         = conv_params->K / conv_params->group; /* compact weights per output channel */
    oc_start    = conv_params->oc_start;
    M           = conv_params->M;
    N           = conv_params->N;
    total_in = conv_params->n_batches * conv_params->c_in * conv_params->h_in * conv_params->w_in;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* GEMM does Y = alpha * A @ B + beta * C; beta gates the bias term. */
    mmio_fp16(conv_params->alpha) = (float16)1.0f;
    mmio_fp16(conv_params->beta)  = conv_params->has_bias ? (float16)1.0f : (float16)0.0f;

    /* Every tile builds the whole im2col matrix, so it needs the whole input: contiguous in
       L2 -> a straight 1D transfer, after which im2col reads L1 only. */
    idma_memcpy_1d(
        &idma_ctrl, 0, (uint32_t)X, (uint32_t)conv_params->shard_X, total_in * sizeof(float16));
    eu_idma_wait_a2o(&eu_ctrl, WFE);

    if (M > 0) {
        /* Weights A = [M, K_g] are each output channel's compact weights, contiguous in W
           (which is [C_out, K_g]); this tile's rows W[oc_start : oc_start+M] are a plain 1D DMA.
           The grouped GEMM applies each row to its own group block of B (see the Spatz task). */
        idma_memcpy_1d(&idma_ctrl,
                       0,
                       (uint32_t)(W + oc_start * K_g),
                       (uint32_t)conv_params->shard_A,
                       M * K_g * sizeof(float16));
        eu_idma_wait_a2o(&eu_ctrl, WFE);

        /* Bias: broadcast the per-output-channel value over the N columns of C (a
           broadcast, not a plain copy -> scalar). */
        if (conv_params->has_bias) {
            for (uint32_t m = 0; m < M; m++) {
                float16 b = B[oc_start + m];
                for (uint32_t n = 0; n < N; n++)
                    mmio_fp16(conv_params->shard_C + (m * N + n) * sizeof(float16)) = b;
            }
        }
    }

    return 0;
}

static void im2col(void *params)
{
    volatile conv2dgemm_fp16_spatz_params_t *conv_params;
    uintptr_t shard_X;
    uintptr_t shard_B;
    uint32_t n_batches, c_in, h_in, w_in, h_out, w_out, kernel_h, kernel_w, stride_h, stride_w,
        pad_h, pad_w;
    uint32_t c_out, group, M, oc_start;
    uint32_t c_in_g, c_out_g, ic_start, ic_end;

    conv_params = (volatile conv2dgemm_fp16_spatz_params_t *)params;
    shard_X     = conv_params->shard_X;
    shard_B     = conv_params->shard_B;
    n_batches   = conv_params->n_batches;
    c_in        = conv_params->c_in;
    h_in        = conv_params->h_in;
    w_in        = conv_params->w_in;
    h_out       = conv_params->h_out;
    w_out       = conv_params->w_out;
    kernel_h    = conv_params->kernel_h;
    kernel_w    = conv_params->kernel_w;
    stride_h    = conv_params->stride_h;
    stride_w    = conv_params->stride_w;
    pad_h       = conv_params->pad_h;
    pad_w       = conv_params->pad_w;
    c_out       = conv_params->c_out;
    group       = conv_params->group;
    M           = conv_params->M;
    oc_start    = conv_params->oc_start;

    /* 1x1 with unit stride and no padding: B already is X (see alloc_l1). */
    if (shard_B == shard_X)
        return;

    /* Build only the rows of B this tile's GEMM reads, those of its output channels' groups:
       a standard convolution (group == 1) reads them all, a depthwise one only M of the C_in
       channels, and a tile with no output channels reads none. */
    if (M == 0)
        return;

    c_in_g   = c_in / group;
    c_out_g  = c_out / group;
    ic_start = (oc_start / c_out_g) * c_in_g;
    ic_end   = ((oc_start + M - 1) / c_out_g + 1) * c_in_g;

    uint32_t hw_out = h_out * w_out;
    uint32_t in_chw = c_in * h_in * w_in;

    /* im2col of every batch: B[b][k, n] = X[b, ic, oh*stride - pad + ki, ow*stride - pad + kj]
    (0 if padded), with k = (ic*K_h + ki)*K_w + kj and n = oh*W_out + ow. Each batch's
    [K, N] slice is laid out contiguously so the Spatz task sweeps them in one offload. */
    for (uint32_t b = 0; b < n_batches; b++) {
        const float16 *X_b = (const float16 *)shard_X + (uintptr_t)b * in_chw;
        float16 *B_b = (float16 *)shard_B + (uintptr_t)b * (c_in * kernel_h * kernel_w) * hw_out;

        for (uint32_t ic = ic_start; ic < ic_end; ic++) {
            const float16 *x_c = X_b + (uintptr_t)ic * h_in * w_in;
            for (uint32_t ki = 0; ki < kernel_h; ki++) {
                for (uint32_t kj = 0; kj < kernel_w; kj++) {
                    uint32_t k = (ic * kernel_h + ki) * kernel_w + kj;

                    /* Output columns whose input column iw = ow*stride - pad + kj lands inside the
                       image: [ow_start, ow_end). They depend on kj only, so the bounds checks
                       leave the per-element loop. */
                    int iw_off   = (int)kj - (int)pad_w;
                    int last     = (int)w_in - 1 - iw_off;
                    int ow_start = iw_off < 0 ? (-iw_off + (int)stride_w - 1) / (int)stride_w : 0;
                    int ow_end   = last < 0 ? 0 : last / (int)stride_w + 1;
                    if (ow_end > (int)w_out)
                        ow_end = (int)w_out;
                    if (ow_start > ow_end)
                        ow_start = ow_end;

                    for (uint32_t oh = 0; oh < h_out; oh++) {
                        int ih         = (int)(oh * stride_h) - (int)pad_h + (int)ki;
                        float16 *B_row = B_b + k * hw_out + oh * w_out;

                        if (ih < 0 || ih >= (int)h_in) {
                            for (uint32_t ow = 0; ow < w_out; ow++)
                                B_row[ow] = (float16)0.0f;
                            continue;
                        }

                        const float16 *x_row = x_c + (uint32_t)ih * w_in;
                        for (int ow = 0; ow < ow_start; ow++)
                            B_row[ow] = (float16)0.0f;
                        for (int ow = ow_start; ow < ow_end; ow++)
                            B_row[ow] = x_row[ow * (int)stride_w + iw_off];
                        for (int ow = ow_end; ow < (int)w_out; ow++)
                            B_row[ow] = (float16)0.0f;
                    }
                }
            }
        }
    }
}

static int offload_spatz_task(void *params)
{
    eu_controller_t eu_ctrl;
    int ret;

    eu_ctrl_init(&eu_ctrl);

    spatz_run_task_with_params(CONV2DGEMM_FP16_SPATZ_TASK, (uint32_t)params);

    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(void *params, float16 *Y)
{
    volatile conv2dgemm_fp16_spatz_params_t *conv_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t n_batches;
    uint32_t c_out;
    uint32_t M;
    uint32_t N;
    uint32_t oc_start;

    conv_params = (volatile conv2dgemm_fp16_spatz_params_t *)params;
    n_batches   = conv_params->n_batches;
    c_out       = conv_params->c_out;
    M           = conv_params->M;
    N           = conv_params->N;
    oc_start    = conv_params->oc_start;

    if (M == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* shard_Y is [n_batches, M, N] contiguous in L1; each batch's M*N block goes to
       Y[b, oc_start:oc_start+M, :] which is contiguous within a batch but c_out*N apart
       across batches -> one 2D transfer (reps = n_batches). */
    idma_memcpy_2d(&idma_ctrl,
                   1,
                   (uint32_t)(Y + oc_start * N),
                   (uint32_t)conv_params->shard_Y,
                   M * N * sizeof(float16),
                   c_out * N * sizeof(float16),
                   n_batches);
    eu_idma_wait_o2a(&eu_ctrl, WFE);

    return 0;
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
    int ret;
    volatile conv2dgemm_fp16_spatz_params_t *params;

    PROF_PHASE(prof_cyc_alloc,
               ret,
               alloc_l1((void **)&params,
                        input_shape,
                        output_shape,
                        kernel_h,
                        kernel_w,
                        stride_h,
                        stride_w,
                        pad_h,
                        pad_w,
                        group,
                        has_bias));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_in, ret, init_input_params((void *)params, X, W, B));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    PROF_PHASE_VOID(prof_cyc_prep, im2col((void *)params));

    PROF_PHASE(prof_cyc_compute, ret, offload_spatz_task((void *)params));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_out, ret, store_result((void *)params, Y));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
