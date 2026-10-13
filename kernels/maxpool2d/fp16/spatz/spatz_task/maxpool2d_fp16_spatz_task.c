#include "tile.h"
#include "maxpool2d_fp16_spatz_params.h"

static inline void compute_window_boundaries_2d(const int out_idx, const uint32_t stride, const uint32_t pad, const uint32_t shape, const uint32_t in_len, int *win_start, int *win_len)
{
    int logical_start;
    int logical_end;
    int first;
    int last;

    logical_start = (out_idx * stride) - pad;
    logical_end = logical_start + shape;

    first = (logical_start < 0) ? 0 : logical_start;
    last = (logical_end > in_len) ? in_len : logical_end;

    *win_start = first;

    if (last > first)
        *win_len = last - first;
    else
        *win_len = 0;
}

/* Max over the clamped window, element by element. A maximum picks an input value
 * unchanged, so this agrees with the vector path bit for bit.
 *
 * Needed because the Spatz VLSU corrupts vector accesses that are not 4-byte aligned.
 * A window starts at win_h_start * w_in + win_w_start, and with a padded odd-sized
 * kernel win_w_start is odd for most output columns - a 3x3 / stride 2 / pad 1 pool
 * (ResNet18's) puts every window from ow = 1 on a 2-byte address. */
static inline _Float16 maxpool2d_window_scalar(const _Float16 *p_src_row, const uint32_t w_in,
                                               const int win_h_len, const int win_w_len)
{
    _Float16 win_max = p_src_row[0];

    for (int kh = 0; kh < win_h_len; kh++) {
        const _Float16 *row = p_src_row + (kh * w_in);

        for (int kw = 0; kw < win_w_len; kw++)
            if (row[kw] > win_max)
                win_max = row[kw];
    }

    return win_max;
}

static inline void maxpool2d(const _Float16 *src, const uint32_t stride_h, const uint32_t stride_w, const uint32_t kernel_h, const uint32_t kernel_w, const uint32_t pad_h, const uint32_t pad_w, const uint32_t h_in, const uint32_t w_in, const uint32_t h_out, const uint32_t w_out, const uint32_t c_len, _Float16 *dst)
{
    const _Float16 *p_src_channel;
    const _Float16 *p_src_row;
    _Float16 *p_dst_channel;

    _Float16 win_max;
    int win_h_start, win_h_len;
    int win_w_start, win_w_len;

    size_t avl;
    size_t vl;

    uint32_t in_channel_stride = h_in * w_in;
    uint32_t out_channel_stride = h_out * w_out;

    for (int c = 0; c < c_len; c++) {
        p_src_channel = src + (c * in_channel_stride);
        p_dst_channel = dst + (c * out_channel_stride);

        for (int oh = 0; oh < h_out; oh++) {
            compute_window_boundaries_2d(oh, stride_h, pad_h, kernel_h, h_in, &win_h_start, &win_h_len);

            for (int ow = 0; ow < w_out; ow++) {
                compute_window_boundaries_2d(ow, stride_w, pad_w, kernel_w, w_in, &win_w_start, &win_w_len);

                if (win_h_len == 0 || win_w_len == 0) {
                    p_dst_channel[oh * w_out + ow] = 0;
                    continue;
                }

                p_src_row = p_src_channel + (win_h_start * w_in) + win_w_start;

                /* Successive window rows step by w_in, so the whole window is aligned
                 * only if its base is and w_in is even. */
                if ((((uintptr_t)p_src_row & 3u) != 0) || ((w_in & 1u) != 0)) {
                    p_dst_channel[oh * w_out + ow] =
                        maxpool2d_window_scalar(p_src_row, w_in, win_h_len, win_w_len);
                    continue;
                }

                win_max = p_src_row[0];

                asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(win_w_len));
                asm volatile ("vfmv.v.f v0, %0" :: "f"(win_max));
                asm volatile ("vfmv.v.f v8, %0" :: "f"(win_max));

                for (int kh = 0; kh < win_h_len; kh++) {
                    const _Float16 *p_src_v = p_src_row + (kh * w_in);
                    avl = win_w_len;

                    for (; avl > 0; avl -= vl) {
                        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
                        asm volatile ("vle16.v v16, (%0)" :: "r"(p_src_v));
                        asm volatile ("vfmax.vv v0, v16, v0");

                        p_src_v += vl;
                    }
                }

                asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(win_w_len));
                asm volatile ("vfredmax.vs v8, v0, v8");
                asm volatile ("vfmv.f.s %0, v8" : "=f"(win_max));

                p_dst_channel[oh * w_out + ow] = win_max;
            }
        }
    }
}

int maxpool2d_fp16_spatz_task(void)
{
    volatile maxpool2d_fp16_spatz_params_t *params;
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
    params = (volatile maxpool2d_fp16_spatz_params_t *) params_addr;

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

    maxpool2d(src, stride_h, stride_w, kernel_h, kernel_w, pad_h, pad_w, h_in, w_in, h_out, w_out, c_len, dst);

    return 0;
}
