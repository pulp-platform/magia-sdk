#include "tile.h"
#include "convtranspose_fp16_spatz_params.h"

static inline void convtranspose(const _Float16 *src, const _Float16 *weight, const uint32_t n_batches, const uint32_t c_in, const uint32_t c_out, const uint32_t iter_start, const uint32_t iter_len, const uint32_t c_in_g, const uint32_t c_out_g, const uint32_t h_in, const uint32_t w_in, const uint32_t h_out, const uint32_t w_out, const uint32_t kernel_h, const uint32_t kernel_w, const uint32_t stride_h, const uint32_t stride_w, const uint32_t pad_h, const uint32_t pad_w, _Float16 *dst)
{
    uint32_t in_hw_len;
    uint32_t out_hw_len;
    uint32_t weight_kh_kw_len;

    int ho_start;
    int wo_start;
    int cur_ho;
    int local_wo_start;
    int kw_start;
    int kw_end;
    int v_len;

    _Float16 val_x;
    size_t avl;
    size_t vl;

    in_hw_len = h_in * w_in;
    out_hw_len = h_out * w_out;
    weight_kh_kw_len = kernel_h * kernel_w;

    for (uint32_t i = 0; i < iter_len; i++) {
        uint32_t abs_iter = iter_start + i;
        uint32_t batch_idx = abs_iter / c_out;
        uint32_t oc_idx = abs_iter % c_out;
        uint32_t g = oc_idx / c_out_g;

        const _Float16 *src_n = src + (batch_idx * c_in * in_hw_len);
        _Float16 *dst_c = dst + (i * out_hw_len);

        /* Loop 2: Tile's local group input channels */
        uint32_t ic_start = g * c_in_g;
        uint32_t ic_end   = ic_start + c_in_g;

        for (uint32_t ic = ic_start; ic < ic_end; ic++) {
            uint32_t local_ic = ic - ic_start;
            uint32_t oc_local = oc_idx % c_out_g;

            const _Float16 *src_c = src_n + (ic * in_hw_len);

            /* Weight layout: [C_in, C_out_g, kh, kw] */
            const _Float16 *weight_c = weight + (ic * c_out_g * weight_kh_kw_len) + (oc_local * weight_kh_kw_len);

            /* Loop 3 and 4: Tile's local input H and W  */
            for (int hi = 0; hi < (int)h_in; hi++) {
                ho_start = (hi * stride_h) - pad_h;

                for (int wi = 0; wi < (int)w_in; wi++) {
                    wo_start = (wi * stride_w) - pad_w;

                    val_x = src_c[(hi * w_in) + wi];

                    /* Optimization */
                    // if (val_x == 0.0f)
                    //     continue;

                    /* Loop 5: Kernel Height */
                    for (int kh = 0; kh < (int)kernel_h; kh++) {
                        cur_ho = ho_start + kh;

                        /* Check height window boundaries */
                        if ((cur_ho < 0) || (cur_ho >= (int)h_out))
                            continue;

                        /* Clip on window's width */
                        kw_start = 0;
                        local_wo_start = wo_start;

                        if (local_wo_start < 0) {
                            kw_start = -local_wo_start;
                            local_wo_start = 0;
                        }

                        kw_end = kernel_w;
                        if ((local_wo_start + (kw_end - kw_start)) > (int)w_out) {
                            kw_end = kw_start + ((int)w_out - local_wo_start);
                        }

                        /* Elements to compute */
                        v_len = kw_end - kw_start;
                        if (v_len <= 0)
                            continue;

                        const _Float16 *p_w = weight_c + (kh * kernel_w) + kw_start;
                        _Float16 *p_y = dst_c + (cur_ho * w_out) + local_wo_start;

                        avl = v_len;

                        for (; avl > 0; avl -= vl) {
                            asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

                            asm volatile ("vle16.v v0, (%0)" :: "r"(p_y));

                            asm volatile ("vle16.v v8, (%0)" :: "r"(p_w));

                            asm volatile ("vfmacc.vf v0, %0, v8" :: "f"(val_x));

                            asm volatile ("vse16.v v0, (%0)" :: "r"(p_y));

                            p_y += vl;
                            p_w += vl;
                        }
                    }
                }
            }
        }
    }
}

int convtranspose_fp16_spatz_task(void)
{
    volatile convtranspose_fp16_spatz_params_t *params;
    uintptr_t params_addr;

    _Float16 *src;
    _Float16 *weight;
    _Float16 *dst;

    uint32_t n_batches;
    uint32_t c_in;
    uint32_t c_out;
    uint32_t iter_start;
    uint32_t iter_len;
    uint32_t c_in_g;
    uint32_t c_out_g;
    uint32_t h_in;
    uint32_t h_out;
    uint32_t w_in;
    uint32_t w_out;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t pad_h;
    uint32_t pad_w;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile convtranspose_fp16_spatz_params_t *) params_addr;

    src      = (_Float16 *) params->shard_X;
    weight   = (_Float16 *) params->shard_W;
    dst      = (_Float16 *) params->shard_Y;

    n_batches  = params->n_batches;
    c_in       = params->c_in;
    c_out      = params->c_out;
    iter_start = params->iter_start;
    iter_len   = params->iter_len;
    c_in_g     = params->c_in_g;
    c_out_g    = params->c_out_g;
    h_in       = params->h_in;
    w_in       = params->w_in;
    h_out      = params->h_out;
    w_out      = params->w_out;
    kernel_h   = params->kernel_h;
    kernel_w   = params->kernel_w;
    stride_h   = params->stride_h;
    stride_w   = params->stride_w;
    pad_h      = params->pad_h;
    pad_w      = params->pad_w;

    convtranspose(src, weight, n_batches, c_in, c_out, iter_start, iter_len, c_in_g, c_out_g, h_in, w_in, h_out, w_out, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dst);

    return 0;
}
