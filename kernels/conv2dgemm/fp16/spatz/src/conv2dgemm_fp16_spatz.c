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

#define HID                  get_hartid()
#define KERNEL_NAME          "conv2dgemm_fp16_spatz"

/* Min output pixels per tile for the row split, below it the GEMM vectors are too short */
#define ROW_SPLIT_MIN_PIXELS 16

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
    size_t oh_start;
    size_t oh_len;
    int is_1x1;

    /* im2col over all input channels gives B = [K = C_in * K_h * K_w, HW_out]. Each output
       channel's weights are compact (K_g = K / group values); the grouped GEMM multiplies them
       by only that channel's group block of B (see the Spatz task). group == 1 -> K_g == K. */
    n_batches = input_shape[0];
    c_in      = input_shape[1];
    c_out     = output_shape[1];
    K         = c_in * kernel_h * kernel_w;
    K_g       = K / group;
    hw_in     = input_shape[2] * input_shape[3];

    /* 1x1, unit stride, no padding: im2col(X) == X */
    is_1x1 = (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && pad_h == 0 &&
              pad_w == 0);

    if (!is_1x1 && (output_shape[2] / NUM_HARTS) * output_shape[3] >= ROW_SPLIT_MIN_PIXELS) {
        /* Shard the output rows (GEMM N): each tile builds only its columns of B */
        shard    = output_shape[2] / NUM_HARTS;
        left     = output_shape[2] % NUM_HARTS;
        oh_start = HID * shard + (HID < left ? HID : left);
        oh_len   = shard + (HID < left ? 1 : 0);
        oc_start = 0;
        oc_len   = c_out;
    } else {
        /* Shard the output channels (GEMM M) */
        shard    = c_out / NUM_HARTS;
        left     = c_out % NUM_HARTS;
        oc_start = HID * shard + (HID < left ? HID : left);
        oc_len   = shard + (HID < left ? 1 : 0);
        oh_start = 0;
        oh_len   = output_shape[2];
    }

    /* This tile's output pixels */
    hw_out = oh_len * output_shape[3];

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

    /* 1x1: B is the input itself */
    if (is_1x1) {
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
    conv_params->oh_start  = (uint32_t)oh_start;
    conv_params->oh_len    = (uint32_t)oh_len;
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

    idma_init();
    eu_init();

    /* GEMM does Y = alpha * A @ B + beta * C; beta gates the bias term. */
    mmio_fp16(conv_params->alpha) = (float16)1.0f;
    mmio_fp16(conv_params->beta)  = conv_params->has_bias ? (float16)1.0f : (float16)0.0f;

    /* The whole input, contiguous in L2 -> a straight 1D transfer */
    idma_memcpy_1d(0, (uint32_t)X, (uint32_t)conv_params->shard_X, total_in * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    if (M > 0) {
        /* Weights A = [M, K_g] are each output channel's compact weights, contiguous in W
           (which is [C_out, K_g]); this tile's rows W[oc_start : oc_start+M] are a plain 1D DMA.
           The grouped GEMM applies each row to its own group block of B (see the Spatz task). */
        idma_memcpy_1d(0,
                       (uint32_t)(W + oc_start * K_g),
                       (uint32_t)conv_params->shard_A,
                       M * K_g * sizeof(float16));
        eu_idma_wait_a2o(WFE);

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

static int offload_spatz_task(void *params)
{
    int ret;

    eu_init();

    spatz_run_task_with_params(CONV2DGEMM_FP16_SPATZ_TASK, (uint32_t)params);

    ret = eu_spatz_wait(WFE);
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
    uint32_t n_batches;
    uint32_t c_out;
    uint32_t M;
    uint32_t N;
    uint32_t hw_out;
    uint32_t oc_start;
    uint32_t n_start;

    conv_params = (volatile conv2dgemm_fp16_spatz_params_t *)params;
    n_batches   = conv_params->n_batches;
    c_out       = conv_params->c_out;
    M           = conv_params->M;
    N           = conv_params->N;
    hw_out      = conv_params->h_out * conv_params->w_out;
    oc_start    = conv_params->oc_start;
    n_start     = conv_params->oh_start * conv_params->w_out;

    if (M == 0)
        return 0;

    idma_init();
    eu_init();

    /* One 2D transfer per batch: M rows of N pixels, HW_out apart in Y  */
    for (uint32_t b = 0; b < n_batches; b++) {
        idma_memcpy_2d(1,
                       (uint32_t)(Y + (b * c_out + oc_start) * hw_out + n_start),
                       (uint32_t)(conv_params->shard_Y + b * M * N * sizeof(float16)),
                       N * sizeof(float16),
                       hw_out * sizeof(float16),
                       M);
        eu_idma_wait_o2a(WFE);
    }

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
