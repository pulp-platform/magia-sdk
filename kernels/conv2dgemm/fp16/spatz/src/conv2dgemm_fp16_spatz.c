#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "conv2dgemm_fp16_spatz.h"
#include "conv2dgemm_fp16_spatz_params.h"
#include "conv2dgemm_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "conv2dgemm_fp16_spatz"

static int alloc_l1(void **params, uint32_t input_shape[4], uint32_t output_shape[4], uint32_t kernel_h, uint32_t kernel_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w, uint32_t group, int has_bias)
{
    volatile conv2dgemm_fp16_spatz_params_t *conv_params;
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
    size_t hw_out;
    size_t shard;
    size_t left;
    size_t oc_start;
    size_t oc_len;

    /* Full im2col over all input channels (K = C_in * K_h * K_w). Grouped convs keep
       this width and place each output channel's compact weights in its group block,
       so the GEMM stays a plain dense product (see init_input_params). */
    n_batches = input_shape[0];
    c_in = input_shape[1];
    c_out = output_shape[1];
    K = c_in * kernel_h * kernel_w;
    hw_out = output_shape[2] * output_shape[3];

    /* Shard the output channels across the tiles (GEMM M dimension). */
    shard = c_out / NUM_HARTS;
    left = c_out % NUM_HARTS;
    oc_start = HID * shard + (HID < left ? HID : left);
    oc_len = shard + (HID < left ? 1 : 0);

    l1_alloc_init();

    conv_params = l1_alloc(sizeof(conv2dgemm_fp16_spatz_params_t));
    if (!conv_params)
        return ENOMEM;

    /* Weights (A) and bias (C) are shared across batches; the im2col matrix (B) and
       the output (Y) are batched so the Spatz task can sweep all batches in one offload. */
    shard_A = l1_alloc(oc_len * K * sizeof(float16));
    if (!shard_A)
        return ENOMEM;

    shard_B = l1_alloc(n_batches * K * hw_out * sizeof(float16));
    if (!shard_B)
        return ENOMEM;

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

    conv_params->shard_A = shard_A;
    conv_params->shard_B = shard_B;
    conv_params->shard_C = shard_C;
    conv_params->shard_Y = shard_Y;
    conv_params->alpha = alpha;
    conv_params->beta = beta;
    conv_params->M = (uint32_t) oc_len;
    conv_params->N = (uint32_t) hw_out;
    conv_params->K = (uint32_t) K;
    conv_params->has_bias = (uint32_t) has_bias;
    conv_params->n_batches = (uint32_t) n_batches;
    conv_params->c_out = (uint32_t) c_out;
    conv_params->group = group;
    conv_params->oc_start = (uint32_t) oc_start;
    conv_params->c_in = (uint32_t) c_in;
    conv_params->h_in = input_shape[2];
    conv_params->w_in = input_shape[3];
    conv_params->h_out = output_shape[2];
    conv_params->w_out = output_shape[3];
    conv_params->kernel_h = kernel_h;
    conv_params->kernel_w = kernel_w;
    conv_params->stride_h = stride_h;
    conv_params->stride_w = stride_w;
    conv_params->pad_h = pad_h;
    conv_params->pad_w = pad_w;

    *params = (void *) conv_params;

    return 0;
}

static int init_input_params(void *params, const float16 *W, const float16 *B)
{
    volatile conv2dgemm_fp16_spatz_params_t *conv_params;
    uintptr_t shard_A;
    uintptr_t shard_C;
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t oc_start;
    uint32_t cout_g;
    uint32_t K_g;

    conv_params = (volatile conv2dgemm_fp16_spatz_params_t *) params;
    shard_A = conv_params->shard_A;
    shard_C = conv_params->shard_C;
    M = conv_params->M;
    N = conv_params->N;
    K = conv_params->K;
    oc_start = conv_params->oc_start;
    cout_g = conv_params->c_out / conv_params->group;   /* output channels per group */
    K_g = K / conv_params->group;                       /* (C_in / group) * K_h * K_w */

    /* GEMM does Y = alpha * A @ B + beta * C; beta gates the bias term. */
    mmio_fp16(conv_params->alpha) = (float16) 1.0f;
    mmio_fp16(conv_params->beta)  = conv_params->has_bias ? (float16) 1.0f : (float16) 0.0f;

    /* Weights: ONNX W is [C_out, C_in/group, K_h, K_w] = [C_out, K_g] contiguous. Output
       channel oc belongs to group g = oc / cout_g and only touches the im2col rows of that
       group, so its K_g compact weights go into A's block [g*K_g, (g+1)*K_g) and the rest
       of the row is zero (block-diagonal A -> plain dense GEMM). group == 1 fills the whole
       row (g = 0, K_g = K), matching the standard convolution with no zero padding. */
    for (uint32_t m = 0; m < M; m++) {
        uint32_t oc = oc_start + m;
        uint32_t g = oc / cout_g;
        const float16 *w_row = W + (uintptr_t) oc * K_g;
        for (uint32_t k = 0; k < K; k++)
            mmio_fp16(shard_A + (m * K + k) * sizeof(float16)) = (float16) 0.0f;
        for (uint32_t kk = 0; kk < K_g; kk++)
            mmio_fp16(shard_A + (m * K + g * K_g + kk) * sizeof(float16)) = w_row[kk];
    }

    /* Bias: broadcast the per-output-channel value over the N columns of C. */
    if (conv_params->has_bias) {
        for (uint32_t m = 0; m < M; m++) {
            float16 b = B[oc_start + m];
            for (uint32_t n = 0; n < N; n++)
                mmio_fp16(shard_C + (m * N + n) * sizeof(float16)) = b;
        }
    }

    return 0;
}

static void im2col(void *params, const float16 *X)
{
    volatile conv2dgemm_fp16_spatz_params_t *conv_params;
    uintptr_t shard_B;
    uint32_t n_batches, c_in, h_in, w_in, h_out, w_out, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w;

    conv_params = (volatile conv2dgemm_fp16_spatz_params_t *) params;
    shard_B = conv_params->shard_B;
    n_batches = conv_params->n_batches;
    c_in = conv_params->c_in;
    h_in = conv_params->h_in;
    w_in = conv_params->w_in;
    h_out = conv_params->h_out;
    w_out = conv_params->w_out;
    kernel_h = conv_params->kernel_h;
    kernel_w = conv_params->kernel_w;
    stride_h = conv_params->stride_h;
    stride_w = conv_params->stride_w;
    pad_h = conv_params->pad_h;
    pad_w = conv_params->pad_w;

    uint32_t hw_out = h_out * w_out;
    uint32_t in_chw = c_in * h_in * w_in;

    /* im2col of every batch: B[b][k, n] = X[b, ic, oh*stride - pad + ki, ow*stride - pad + kj]
    (0 if padded), with k = (ic*K_h + ki)*K_w + kj and n = oh*W_out + ow. Each batch's
    [K, N] slice is laid out contiguously so the Spatz task sweeps them in one offload. */
    for (uint32_t b = 0; b < n_batches; b++) {
        const float16 *X_b = X + (uintptr_t) b * in_chw;
        uintptr_t B_b = shard_B + (uintptr_t) b * (c_in * kernel_h * kernel_w) * hw_out * sizeof(float16);

        for (uint32_t ic = 0; ic < c_in; ic++) {
            const float16 *x_c = X_b + (uintptr_t) ic * h_in * w_in;
            for (uint32_t ki = 0; ki < kernel_h; ki++) {
                for (uint32_t kj = 0; kj < kernel_w; kj++) {
                    uint32_t k = (ic * kernel_h + ki) * kernel_w + kj;
                    for (uint32_t oh = 0; oh < h_out; oh++) {
                        int ih = (int) (oh * stride_h) - (int) pad_h + (int) ki;
                        for (uint32_t ow = 0; ow < w_out; ow++) {
                            int iw = (int) (ow * stride_w) - (int) pad_w + (int) kj;
                            uint32_t n = oh * w_out + ow;
                            float16 v = (float16) 0.0f;
                            if (ih >= 0 && ih < (int) h_in && iw >= 0 && iw < (int) w_in)
                                v = x_c[(uint32_t) ih * w_in + (uint32_t) iw];
                            mmio_fp16(B_b + (k * hw_out + n) * sizeof(float16)) = v;
                        }
                    }
                }
            }
        }
    }
}

static int offload_spatz_task(void *params)
{
    eu_controller_t eu_ctrl;
    eu_config_t eu_cfg;
    int ret;

    eu_cfg.hartid = HID;
    eu_ctrl.base = NULL;
    eu_ctrl.cfg = &eu_cfg;
    eu_ctrl.api = &eu_api;

    spatz_run_task_with_params(CONV2DGEMM_FP16_SPATZ_TASK, (uint32_t) params);

    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n", HID, KERNEL_NAME, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(void *params, float16 *Y)
{
    volatile conv2dgemm_fp16_spatz_params_t *conv_params;
    uintptr_t shard_Y;
    uint32_t n_batches;
    uint32_t c_out;
    uint32_t M;
    uint32_t N;
    uint32_t oc_start;

    conv_params = (volatile conv2dgemm_fp16_spatz_params_t *) params;
    shard_Y = conv_params->shard_Y;
    n_batches = conv_params->n_batches;
    c_out = conv_params->c_out;
    M = conv_params->M;
    N = conv_params->N;
    oc_start = conv_params->oc_start;

    for (uint32_t b = 0; b < n_batches; b++) {
        float16 *Y_b = Y + (uintptr_t) b * c_out * N;
        uintptr_t shard_Y_b = shard_Y + (uintptr_t) b * M * N * sizeof(float16);
        for (uint32_t m = 0; m < M; m++) {
            float16 *y_row = Y_b + (oc_start + m) * N;
            for (uint32_t n = 0; n < N; n++)
                y_row[n] = mmio_fp16(shard_Y_b + (m * N + n) * sizeof(float16));
        }
    }

    return 0;
}

void MAGIA_conv2dgemm_fp16_spatz(const float16* X, const float16 *W, const float16 *B, float16 *Y, uint32_t input_shape[4], uint32_t output_shape[4], uint32_t kernel_h, uint32_t kernel_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w, uint32_t group, int has_bias)
{
    int ret;
    volatile conv2dgemm_fp16_spatz_params_t *params;

    ret = alloc_l1((void **) &params, input_shape, output_shape, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, group, has_bias);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *) params, W, B);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    im2col((void *) params, X);

    ret = offload_spatz_task((void *) params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *) params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
