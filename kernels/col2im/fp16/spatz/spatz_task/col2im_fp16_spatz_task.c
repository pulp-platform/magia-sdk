#include "tile.h"
#include "col2im_fp16_spatz_params.h"

static inline void col2im_row_acc(const _Float16 *src, _Float16 *dst, uint32_t count, uint32_t stride_w)
{
    size_t avl = count;
    size_t vl;

    if (stride_w == 1) {
        for (; avl > 0; avl -= vl) {
            asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
            asm volatile ("vle16.v v0, (%0)" :: "r"(src));
            asm volatile ("vle16.v v8, (%0)" :: "r"(dst));
            asm volatile ("vfadd.vv v0, v0, v8");
            asm volatile ("vse16.v v0, (%0)" :: "r"(dst) : "memory");
            src += vl;
            dst += vl;
        }
    } else {
        size_t stride_bytes = stride_w * sizeof(_Float16);
        for (; avl > 0; avl -= vl) {
            asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
            asm volatile ("vle16.v v0, (%0)" :: "r"(src));
            asm volatile ("vlse16.v v8, (%0), %1" :: "r"(dst), "r"(stride_bytes));
            asm volatile ("vfadd.vv v0, v0, v8");
            asm volatile ("vsse16.v v0, (%0), %1" :: "r"(dst), "r"(stride_bytes) : "memory");
            src += vl;
            dst += vl * stride_w;
        }
    }
}

int col2im_fp16_spatz_task(void)
{
    volatile col2im_fp16_spatz_params_t *params;
    uintptr_t params_addr;

    const _Float16 *src;
    _Float16 *dst;

    uint32_t batch;
    uint32_t c_len;
    uint32_t image_h;
    uint32_t image_w;
    uint32_t block_h;
    uint32_t block_w;
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t dilation_h;
    uint32_t dilation_w;
    uint32_t l_len;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile col2im_fp16_spatz_params_t *) params_addr;


    src = (_Float16 *) params->shard_X;
    dst = (_Float16 *) params->shard_Y;

    batch = params->batch;
    dilation_h = params->dilation_h;
    dilation_w = params->dilation_w;
    stride_h = params->stride_h;
    stride_w = params->stride_w;
    image_h = params->image_h;
    image_w = params->image_w;
    block_h = params->block_h;
    block_w = params->block_w;
    pad_h = params->pad_h;
    pad_w = params->pad_w;
    l_len = params->l_len;
    c_len = params->c_len;

    if (c_len == 0)
        return 0;

    uint32_t l_w = (image_w + 2 * pad_w - dilation_w * (block_w - 1) - 1) / stride_w + 1;
    uint32_t block_volume = block_h * block_w;
    uint32_t image_volume = image_h * image_w;
    uint32_t l_h = l_len / l_w;

    for (uint32_t n = 0; n < batch; n++) {
        for (uint32_t c = 0; c < c_len; c++) {

            uint32_t local_c_offset_in  = n * (c_len * block_volume * l_len) + (c * block_volume * l_len);
            uint32_t local_c_offset_out = n * (c_len * image_volume) + (c * image_volume);
            _Float16 *dst_img = dst + local_c_offset_out;

            for (uint32_t ky = 0; ky < block_h; ky++) {
                for (uint32_t kx = 0; kx < block_w; kx++) {

                    uint32_t k_idx = ky * block_w + kx;
                    const _Float16 *src_k = src + local_c_offset_in + (k_idx * l_len);
                    int base_w = (int) (kx * dilation_w) - (int) pad_w;

                    for (uint32_t lh = 0; lh < l_h; lh++) {
                        int im_h = (int) (lh * stride_h) + (int) (ky * dilation_h) - (int) pad_h;
                        if (im_h < 0 || im_h >= (int) image_h)
                            continue;

                        /* Input columns lw whose im_w = lw*stride_w + base_w falls in [0, image_w). */
                        int lw_start = (base_w >= 0) ? 0 : (((-base_w) + (int) stride_w - 1) / (int) stride_w);
                        int hi = (int) image_w - 1 - base_w;
                        if (hi < 0)
                            continue;
                        int lw_end = hi / (int) stride_w + 1;
                        if (lw_end > (int) l_w)
                            lw_end = (int) l_w;
                        if (lw_start >= lw_end)
                            continue;

                        const _Float16 *ps = src_k + lh * l_w + lw_start;
                        _Float16 *pd = dst_img + (uint32_t) im_h * image_w + (uint32_t) (lw_start * (int) stride_w + base_w);

                        col2im_row_acc(ps, pd, (uint32_t) (lw_end - lw_start), stride_w);
                    }
                }
            }
        }
    }

    return 0;
}
