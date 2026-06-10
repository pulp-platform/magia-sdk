#include "tile.h"
#include "col2im_fp16_spatz_params.h"

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

    for (uint32_t n = 0; n < batch; n++) {
        for (uint32_t c = 0; c < c_len; c++) {

            uint32_t local_c_offset_in  = n * (c_len * block_volume * l_len) + (c * block_volume * l_len);
            uint32_t local_c_offset_out = n * (c_len * image_volume) + (c * image_volume);
            _Float16 *dst_img = dst + local_c_offset_out;

            for (uint32_t ky = 0; ky < block_h; ky++) {
                for (uint32_t kx = 0; kx < block_w; kx++) {

                    uint32_t k_idx = ky * block_w + kx;
                    const _Float16 *p_src = src + local_c_offset_in + (k_idx * l_len);

                    size_t avl = l_len;
                    size_t vl;
                    uint32_t l_offset = 0;

                    for (; avl > 0; avl -= vl) {
                        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
                        asm volatile ("vle16.v v0, (%0)" :: "r"(p_src));

                        for (size_t j = 0; j < vl; j++) {
                            uint32_t curr_l = l_offset + j;
                            uint32_t curr_lh = curr_l / l_w;
                            uint32_t curr_lw = curr_l % l_w;

                            int32_t im_h = (int32_t)(curr_lh * stride_h) + (int32_t)(ky * dilation_h) - (int32_t)pad_h;
                            int32_t im_w = (int32_t)(curr_lw * stride_w) + (int32_t)(kx * dilation_w) - (int32_t)pad_w;

                            if (im_h >= 0 && im_h < (int32_t)image_h && im_w >= 0 && im_w < (int32_t)image_w) {
                                uint32_t out_idx = (im_h * image_w) + im_w;
                                _Float16 val;

                                if (j == 0) {
                                    asm volatile ("vfmv.f.s %0, v0" : "=f"(val));
                                } else {
                                    asm volatile ("vslidedown.vx v8, v0, %0" :: "r"(j));
                                    asm volatile ("vfmv.f.s %0, v8" : "=f"(val));
                                }

                                dst_img[out_idx] += val;
                            }
                        }
                        p_src += vl;
                        l_offset += vl;
                    }
                }
            }
        }
    }

    return 0;
}
