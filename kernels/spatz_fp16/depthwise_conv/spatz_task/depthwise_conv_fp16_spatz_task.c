#include "tile.h"
#include "depthwise_conv_fp16_spatz_params.h"

/* Products of two binary16 values are exact in binary32. Widening the
 * operands before vfmacc therefore preserves the existing FP32 accumulation
 * order, with one final FP16 rounding. Gather into aligned scratch to support
 * arbitrary shard strides, padding, and the VLSU's four-byte alignment rule. */
int depthwise_conv_fp16_spatz_task(void)
{
    const volatile depthwise_conv_fp16_spatz_params_t *p =
        (const volatile depthwise_conv_fp16_spatz_params_t *)mmio32(SPATZ_DATA);
    volatile float samples[64] __attribute__((aligned(16)));
    volatile float sums[64] __attribute__((aligned(16)));
    volatile uint8_t valid[8] __attribute__((aligned(16)));
    for (uint32_t n = 0; n < p->output_shape[0]; ++n)
        for (uint32_t c = 0; c < p->output_shape[1]; ++c) {
            uint32_t ic = (p->weight_origin_c + c) / p->channel_multiplier - p->input_origin_c;
            float bias = p->bias ? (float)*(const _Float16 *)(p->bias + c * p->bias_stride) : 0.0f;
            for (uint32_t y = 0; y < p->output_shape[2]; ++y)
                for (uint32_t x = 0; x < p->output_shape[3];) {
                    uint32_t remaining = p->output_shape[3] - x;
                    if (remaining > 64u) remaining = 64u;
                    uint32_t vl;
                    asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(remaining));
                    asm volatile("vfmv.v.f v8, %0" :: "f"(bias));
                    for (uint32_t ky = 0; ky < p->weight_shape[2]; ++ky)
                        for (uint32_t kx = 0; kx < p->weight_shape[3]; ++kx) {
                            int64_t iy = (int64_t)(p->output_origin_h + y) * p->stride_h - p->pad_top + ky * p->dilation_h - p->input_origin_h;
                            if (iy < 0 || iy >= p->input_shape[2]) continue;
                            float weight = (float)*(const _Float16 *)(p->weight + c * p->weight_strides[0] + ky * p->weight_strides[2] + kx * p->weight_strides[3]);
                            for (uint32_t byte = 0; byte < (vl + 7u) / 8u; ++byte) valid[byte] = 0;
                            for (uint32_t lane = 0; lane < vl; ++lane) {
                                int64_t ix = (int64_t)(p->output_origin_w + x + lane) * p->stride_w - p->pad_left + kx * p->dilation_w - p->input_origin_w;
                                uint32_t active = ix >= 0 && ix < p->input_shape[3];
                                if (active) valid[lane / 8u] |= 1u << (lane % 8u);
                                samples[lane] = active ? (float)*(const _Float16 *)(p->input + n * p->input_strides[0] + ic * p->input_strides[1] + (uint32_t)iy * p->input_strides[2] + (uint32_t)ix * p->input_strides[3]) : 0.0f;
                            }
                            uint32_t mask_bytes = (vl + 7u) / 8u;
                            asm volatile("vsetvli zero, %0, e8, m1, ta, ma" :: "r"(mask_bytes));
                            asm volatile("vle8.v v0, (%0)" :: "r"(valid) : "memory");
                            asm volatile("vsetvli zero, %0, e32, m1, ta, ma" :: "r"(vl));
                            asm volatile("vle32.v v4, (%0)" :: "r"(samples) : "memory");
                            asm volatile("vfmacc.vf v8, %0, v4, v0.t" :: "f"(weight));
                        }
                    asm volatile("vse32.v v8, (%0)" :: "r"(sums) : "memory");
                    for (uint32_t lane = 0; lane < vl; ++lane)
                        *(_Float16 *)(p->output + n * p->output_strides[0] + c * p->output_strides[1] + y * p->output_strides[2] + (x + lane) * p->output_strides[3]) = (_Float16)sums[lane];
                    x += vl;
                }
        }
    return 0;
}
