#include "tile.h"
#include "resize_fp16_spatz_params.h"

static inline void resize_nearest(const _Float16 *src, _Float16 *dst, const uint32_t in_h, const uint32_t in_w, const uint32_t out_h, const uint32_t out_w, const uint32_t it_len)
{
    const _Float16 *p_src_chan;
    const _Float16 *p_src_row;
    _Float16 *p_dst_row;
    uint16_t *p_idx;

    float scale_h;
    float scale_w;
    float ih_f;
    float iw_f;

    uint32_t c;
    uint32_t oh;
    uint32_t ow;
    int32_t ih;
    int32_t iw;

    size_t avl;
    size_t vl;

    uint16_t w_idx[out_w];

    scale_h = (float)in_h / (float)out_h;
    scale_w = (float)in_w / (float)out_w;

    for (ow = 0; ow < out_w; ow++) {
        iw_f = ((float)ow + 0.5f) * scale_w - 0.5f;
        iw = (int32_t)(iw_f + 0.5f);

        if (iw_f == (float)iw - 0.5f)
            iw--;

        if (iw < 0)
            iw = 0;

        if (iw >= (int32_t)in_w)
            iw = in_w - 1;

        w_idx[ow] = (uint16_t)iw;
    }

    for (c = 0; c < it_len; c++) {
        p_src_chan = src + c * in_h * in_w;

        for (oh = 0; oh < out_h; oh++) {
            ih_f = ((float)oh + 0.5f) * scale_h - 0.5f;
            ih = (int32_t)(ih_f + 0.5f);

            if (ih_f == (float)ih - 0.5f)
                ih--;

            if (ih < 0)
                ih = 0;

            if (ih >= (int32_t)in_h)
                ih = in_h - 1;

            p_src_row = p_src_chan + ih * in_w;
            p_dst_row = dst + c * out_h * out_w + oh * out_w;
            p_idx = w_idx;
            avl = out_w;

            for (; avl > 0; avl -= vl) {
                asm volatile ("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(avl));

                asm volatile ("vle16.v v4, (%0)" :: "r"(p_idx));

                asm volatile ("vsll.vi v4, v4, 1");
                asm volatile ("vloxei16.v v0, (%0), v4" :: "r"(p_src_row));

                asm volatile ("vse16.v v0, (%0)" :: "r"(p_dst_row) : "memory");

                p_idx += vl;
                p_dst_row += vl;
            }
        }
    }
}

int resize_fp16_spatz_task(void)
{
    volatile resize_fp16_spatz_params_t *params;
    uintptr_t params_addr;

    const _Float16 *src;
    _Float16 *dst;

    uint32_t in_h;
    uint32_t in_w;
    uint32_t out_h;
    uint32_t out_w;
    uint32_t it_len;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile resize_fp16_spatz_params_t *) params_addr;

    src = (_Float16 *) params->shard_X;
    dst = (_Float16 *) params->shard_Y;
    in_h = params->in_h;
    in_w = params->in_w;
    out_h = params->out_h;
    out_w = params->out_w;
    it_len = params->iteration_len;

    resize_nearest(src, dst, in_h, in_w, out_h, out_w, it_len);

    return 0;
}
