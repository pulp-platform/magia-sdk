#include "tile.h"
#include "conv2d_fp16_spatz_params.h"

static inline void compute_window_boundaries(const int out_idx, const uint32_t stride, const uint32_t pad, const uint32_t shape, const uint32_t in_len, int *win_start, int *win_len, int *ker_start)
{
    int logical_start = (out_idx * stride) - pad;
    int logical_end = logical_start + shape;

    int first = logical_start;
    if (first < 0) {
        *win_start = 0;
        *ker_start = -logical_start;
    } else {
        *win_start = first;
        *ker_start = 0;
    }

    int last = logical_end;
    if (last > in_len) {
        last = in_len;
    }

    if (last > *win_start)
        *win_len = last - *win_start;
    else
        *win_len = 0;
}

static inline void conv2d_scalar(const _Float16 *src, const _Float16 *weight, const _Float16 *bias, const uint32_t c_in_g, const uint32_t c_out_g, const uint32_t h_in, const uint32_t w_in, const uint32_t h_out, const uint32_t w_out, const uint32_t kernel_h, const uint32_t kernel_w, const uint32_t stride_h, const uint32_t stride_w, const uint32_t pad_h, const uint32_t pad_w, const uint32_t iter_start, const uint32_t iter_len, const uint32_t c_out, const uint32_t has_bias, _Float16 *dst)
{
    uint32_t in_hw  = h_in * w_in;
    uint32_t out_hw = h_out * w_out;
    uint32_t ker_hw = kernel_h * kernel_w;
    uint32_t full_in_size = (c_in_g * (c_out / c_out_g)) * in_hw;
    uintptr_t dst_base = (uintptr_t) dst;

    int h_win_start, h_win_len, h_ker_start;
    int w_win_start, w_win_len, w_ker_start;

    for (uint32_t i = 0; i < iter_len; i++) {
        uint32_t abs_iter = iter_start + i;
        uint32_t batch_idx = abs_iter / c_out;
        uint32_t oc_idx = abs_iter % c_out;
        uint32_t g = oc_idx / c_out_g;

        const _Float16 *batch_src = src + (batch_idx * full_in_size);
        const _Float16 *weight_c = weight + (oc_idx * c_in_g * ker_hw);
        _Float16 *dst_plane = dst + (i * out_hw);

        for (int oh = 0; oh < h_out; oh++) {
            compute_window_boundaries(oh, stride_h, pad_h, kernel_h, h_in, &h_win_start, &h_win_len, &h_ker_start);
            if (h_win_len == 0) continue;

            for (int ow = 0; ow < w_out; ow++) {
                compute_window_boundaries(ow, stride_w, pad_w, kernel_w, w_in, &w_win_start, &w_win_len, &w_ker_start);
                if (w_win_len == 0) continue;

                uint32_t ic_start = g * c_in_g;
                uint32_t ic_end   = ic_start + c_in_g;

                _Float16 acc_scaler = has_bias ? bias[oc_idx] : 0.0f;

                for (uint32_t ic = ic_start; ic < ic_end; ic++) {
                    uint32_t local_ic = ic - ic_start;

                    const _Float16 *src_c = batch_src + (ic * in_hw);
                    const _Float16 *weight_k = weight_c + (local_ic * ker_hw);

                    for (int kh = 0; kh < h_win_len; kh++) {
                        const _Float16 *p_src = src_c + ((h_win_start + kh) * w_in) + w_win_start;
                        const _Float16 *p_weight = weight_k + ((h_ker_start + kh) * kernel_w) + w_ker_start;

                        for (int kw = 0; kw < w_win_len; kw++) {
                            acc_scaler += p_src[kw] * p_weight[kw];
                        }
                    }
                }

                uint32_t dst_offset = ((oh * w_out) + ow);
                dst_plane[dst_offset] = acc_scaler;
            }
        }
    }
}

static inline void conv2d_rvv(const _Float16 *src, const _Float16 *weight, const _Float16 *bias, const uint32_t c_in_g, const uint32_t c_out_g, const uint32_t h_in, const uint32_t w_in, const uint32_t h_out, const uint32_t w_out, const uint32_t kernel_h, const uint32_t kernel_w, const uint32_t stride_h, const uint32_t stride_w, const uint32_t pad_h, const uint32_t pad_w, const uint32_t iter_start, const uint32_t iter_len, const uint32_t c_out, const uint32_t has_bias, _Float16 *dst)
{
    register _Float16 ZERO asm ("f10") = 0.0f;
    uint32_t in_hw  = h_in * w_in;
    uint32_t out_hw = h_out * w_out;
    uint32_t ker_hw = kernel_h * kernel_w;
    uint32_t full_in_size = (c_in_g * (c_out / c_out_g)) * in_hw;

    int h_win_start, h_win_len, h_ker_start;
    int w_win_start, w_win_len, w_ker_start;

    size_t vl;
    size_t avl;
    size_t vl_max;

    for (uint32_t i = 0; i < iter_len; i++) {
        uint32_t abs_iter = iter_start + i;
        uint32_t batch_idx = abs_iter / c_out;
        uint32_t oc_idx = abs_iter % c_out;
        uint32_t g = oc_idx / c_out_g;

        const _Float16 *batch_src = src + (batch_idx * full_in_size);
        const _Float16 *weight_c = weight + (oc_idx * c_in_g * ker_hw);
        _Float16 *dst_plane = dst + (i * out_hw);

        for (int oh = 0; oh < h_out; oh++) {
            compute_window_boundaries(oh, stride_h, pad_h, kernel_h, h_in, &h_win_start, &h_win_len, &h_ker_start);
            if (h_win_len == 0) continue;

            for (int ow = 0; ow < w_out; ow++) {
                compute_window_boundaries(ow, stride_w, pad_w, kernel_w, w_in, &w_win_start, &w_win_len, &w_ker_start);
                if (w_win_len == 0) continue;

                uint32_t ic_start = g * c_in_g;
                uint32_t ic_end   = ic_start + c_in_g;

                _Float16 acc_scaler = 0;
                _Float16 bias_val = has_bias ? bias[oc_idx] : 0.0f;

                asm volatile ("vsetvli %0, %1, e16, m8, tu, ma" : "=r"(vl_max) : "r"(w_win_len));
                asm volatile ("vfmv.v.f v0, %0" :: "f"(bias_val));
                asm volatile ("vfmv.v.f v8, %0" :: "f"(ZERO));

                for (uint32_t ic = ic_start; ic < ic_end; ic++) {
                    uint32_t local_ic = ic - ic_start;

                    const _Float16 *src_c = batch_src + (ic * in_hw);
                    const _Float16 *weight_k = weight_c + (local_ic * ker_hw);

                    for (int kh = 0; kh < h_win_len; kh++) {
                        const _Float16 *p_src = src_c + ((h_win_start + kh) * w_in) + w_win_start;
                        const _Float16 *p_weight = weight_k + ((h_ker_start + kh) * kernel_w) + w_ker_start;

                        avl = w_win_len;
                        for (; avl > 0; avl -= vl) {
                            asm volatile ("vsetvli %0, %1, e16, m8, tu, ma" : "=r"(vl) : "r"(avl));
                            asm volatile ("vle16.v v16, (%0)" :: "r"(p_src));
                            asm volatile ("vle16.v v24, (%0)" :: "r"(p_weight));
                            asm volatile ("vfmacc.vv v8, v16, v24");

                            p_src += vl;
                            p_weight += vl;
                        }
                    }
                }

                asm volatile ("vsetvli %0, %1, e16, m8, tu, ma" : "=r"(vl) : "r"(vl_max));
                asm volatile ("vfredosum.vs v0, v8, v0");
                asm volatile ("vfmv.f.s %0, v0" : "=f"(acc_scaler));

                uint32_t dst_offset = ((oh * w_out) + ow);
                dst_plane[dst_offset] = acc_scaler;
            }
        }
    }
}

int conv2d_fp16_spatz_task(void)
{
    volatile conv2d_fp16_spatz_params_t *params;
    uintptr_t params_addr;

    _Float16 *src;
    _Float16 *weight;
    _Float16 *bias;
    _Float16 *dst;

    uint32_t stride_h, stride_w;
    uint32_t kernel_h, kernel_w;
    uint32_t pad_h, pad_w;
    uint32_t h_in, w_in, h_out, w_out, c_out;
    uint32_t iter_start, iter_len, c_in_g, c_out_g;
    uint32_t has_bias;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile conv2d_fp16_spatz_params_t *) params_addr;

    src    = (_Float16 *) params->shard_X;
    weight = (_Float16 *) params->shard_W;
    bias   = (_Float16 *) params->shard_B;
    dst    = (_Float16 *) params->shard_Y;

    stride_h = params->stride_h;
    stride_w = params->stride_w;
    kernel_h = params->kernel_h;
    kernel_w = params->kernel_w;
    pad_h    = params->pad_h;
    pad_w    = params->pad_w;

    h_in     = params->h_in;
    w_in     = params->w_in;
    h_out    = params->h_out;
    w_out    = params->w_out;

    iter_start  = params->iter_start;
    iter_len    = params->iter_len;
    c_out       = params->c_out;
    c_in_g      = params->c_in_g;
    c_out_g     = params->c_out_g;
    has_bias    = params->has_bias;

    /* Notice: For the time being the scalar version is used because the reduction in fp16 introduces a significant difference wrt the golden model */
    // conv2d_rvv(src, weight, bias, c_in_g, c_out_g, h_in, w_in, h_out, w_out, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, iter_start, iter_len, c_out, has_bias, dst);
    conv2d_scalar(src, weight, bias, c_in_g, c_out_g, h_in, w_in, h_out, w_out, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, iter_start, iter_len, c_out, has_bias, dst);

    return 0;
}
