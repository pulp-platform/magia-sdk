#include "tile.h"
#include "onnx_averagepool_params.h"

static inline void compute_window_boundaries( const int out_idx, const uint32_t stride, const uint32_t pad, const uint32_t shape, const uint32_t dilation, const uint32_t in_len, int *win_start, int *win_len)
{
    int logical_start;
    int logical_end;
    int offset;
    int first;
    int last;

    logical_start = (out_idx * stride) - pad;
    logical_end = logical_start + (shape - 1) * dilation + 1;

    first = logical_start;

    if (first < 0) {
        offset = (-first + dilation - 1) / dilation;
        first += offset * dilation;
    }

    last = logical_end;
    last = (last > in_len) ? in_len : last;

    *win_start = first;

    if (last > first)
        *win_len = (last - first + dilation - 1) / dilation;
    else
        *win_len = 0;
}

static inline void averagepool(const _Float16 *src, const uint32_t cnt_include_pad, const uint32_t dilation, const uint32_t stride, const uint32_t shape, const uint32_t pad, const uint32_t in_len, const uint32_t out_len, _Float16 *dst)
{
    register _Float16 ZERO asm ("f10") = 0.0f;
    const _Float16 *p_src;
    int strides_bytes;
    int win_start;
    int win_len;
    _Float16 win_avg;
    size_t avl;
    size_t vl;

    for (int out_idx = 0; out_idx < out_len; out_idx++) {
        compute_window_boundaries(out_idx, stride, pad, shape, dilation, in_len, &win_start, &win_len);
        strides_bytes = dilation * sizeof(_Float16);
        p_src = src + win_start;
        avl = win_len;
        win_avg = 0;

        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(win_len));
        asm volatile ("vfmv.v.f v0, %0" :: "f"(ZERO));
        asm volatile ("vfmv.v.f v8, %0" :: "f"(ZERO));

        for (; avl > 0; avl -=vl) {
            asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

            asm volatile ("vlse16.v v16, (%0), %1" :: "r"(p_src), "r"(strides_bytes));
            asm volatile ("vfadd.vv v0, v16, v0");

            p_src += vl * dilation;
        }

        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(win_len));
        asm volatile ("vfredosum.vs v8, v0, v8");
        asm volatile ("vfmv.f.s %0, v8" : "=f"(win_avg));

        if (cnt_include_pad)
            win_avg = win_avg / (_Float16) shape;
        else
            win_avg = win_avg / (_Float16) win_len;

        dst[out_idx] = win_avg;
    }
}

int onnx_averagepool_task(void)
{
    volatile onnx_averagepool_params_t *params;
    uintptr_t params_addr;

    _Float16 *src;
    _Float16 *dst;

    uint32_t cnt_include_pad;
    uint32_t dilation;
    uint32_t stride;
    uint32_t shape;
    uint32_t pad;

    size_t out_len;
    size_t in_len;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile onnx_averagepool_params_t *) params_addr;

    src = (_Float16 *) params->addr_input;
    dst = (_Float16 *) params->addr_res;

    cnt_include_pad = params->count_include_pad;
    dilation = params->dilation;
    stride = params->stride;
    shape = params->shape;
    pad = params->pad;

    out_len = params->len_out;
    in_len = params->len_in;

    averagepool(src, cnt_include_pad, dilation, stride, shape, pad, in_len, out_len, dst);

    return 0;
}
