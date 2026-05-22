#include "tile.h"
#include "averagepool2d_fp16_spatz_params.h"

static inline void compute_window_boundaries_2d(const int out_idx, const uint32_t stride, const uint32_t pad, const uint32_t shape, const uint32_t in_len, int *win_start, int *win_len)
{
    int logical_start;
    int logical_end;
    int first;
    int last;

    logical_start = (out_idx * stride) - pad;
    logical_end = logical_start + shape;

    first = logical_start;

    if (first < 0)
        first = 0;

    last = logical_end;

    if (last > in_len)
        last = in_len;

    *win_start = first;

    if (last > first)
        *win_len = last - first;
    else
        *win_len = 0;
}

static inline void averagepool2d(const _Float16 *src, const uint32_t c_len, const uint32_t h_in, const uint32_t w_in, const uint32_t h_out, const uint32_t w_out, const uint32_t kernel_h, const uint32_t kernel_w, const uint32_t stride_h, const uint32_t stride_w, const uint32_t pad_h, const uint32_t pad_w, _Float16 *dst)
{
    register _Float16 ZERO asm ("f10") = 0.0f;
    const _Float16 *p_src;
    uint32_t out_hw_len;
    uint32_t in_hw_len;
    _Float16 win_avg;
    int h_win_start;
    int w_win_start;
    int h_win_len;
    int w_win_len;
    size_t avl;
    size_t vl;

    in_hw_len = h_in * w_in;
    out_hw_len = h_out * w_out;

    for (int c = 0; c < c_len; c++) {
        const _Float16 *src_c;
        _Float16 *dst_c;

        src_c = src + (c * in_hw_len);
        dst_c = dst + (c * out_hw_len);

        for (int oh = 0; oh < h_out; oh++) {
            compute_window_boundaries_2d( oh, stride_h, pad_h, kernel_h, h_in, &h_win_start, &h_win_len);

            for (int ow = 0; ow < w_out; ow++) {
                compute_window_boundaries_2d( ow, stride_w, pad_w, kernel_w, w_in, &w_win_start, &w_win_len);

                if ((h_win_len == 0) || (w_win_len == 0)) {
                    dst_c[(oh * w_out) + ow] = 0;
                    continue;
                }

                win_avg = 0;

                asm volatile ("vfmv.v.f v8, %0" :: "f"(ZERO));

                for (int kh = 0; kh < h_win_len; kh++) {

                    p_src = src_c + ((h_win_start + kh) * w_in) + w_win_start;
                    avl = w_win_len;

                    asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(w_win_len));
                    asm volatile ("vfmv.v.f v0, %0" :: "f"(ZERO));

                    for (; avl > 0; avl -= vl) {
                        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
                        asm volatile ("vle16.v v16, (%0)" :: "r"(p_src));
                        asm volatile ("vfadd.vv v0, v16, v0");
                        p_src += vl;
                    }

                    asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(w_win_len));
                    asm volatile ("vfredosum.vs v8, v0, v8");
                }

                asm volatile ("vfmv.f.s %0, v8" : "=f"(win_avg));
                win_avg = win_avg / (_Float16)(h_win_len * w_win_len);
                dst_c[(oh * w_out) + ow] = win_avg;
            }
        }
    }
}

int averagepool2d_fp16_spatz_task(void)
{
    volatile averagepool2d_fp16_spatz_params_t *params;
    uintptr_t params_addr;

    _Float16 *src;
    _Float16 *dst;

    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t pad_h;
    uint32_t pad_w;

    uint32_t h_in;
    uint32_t w_in;
    uint32_t h_out;
    uint32_t w_out;
    uint32_t c_len;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile averagepool2d_fp16_spatz_params_t *) params_addr;

    src = (_Float16 *) params->shard_X;
    dst = (_Float16 *) params->shard_Y;

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
    c_len    = params->c_len;

    averagepool2d(src, c_len, h_in, w_in, h_out, w_out, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dst);

    return 0;
}
