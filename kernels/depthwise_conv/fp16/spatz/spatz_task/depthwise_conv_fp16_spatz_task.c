#include "tile.h"
#include "depthwise_conv_fp16_spatz_params.h"

static inline uint32_t depthwise_cycle(void)
{
    uint32_t value;
    asm volatile("csrr %0, cycle" : "=r"(value) :: "memory");
    return value;
}

/* Flatten up to 64 output positions into one vector. Geometry tables are
 * computed once per spatial block, then reused by all channels. Even strides
 * use a uniform halfword shift; odd strides select the half independently
 * per lane, including when the vector spans output rows. */
static int depthwise_vector_plan(const volatile depthwise_conv_fp16_spatz_params_t *p)
{
    volatile uint32_t offsets[49][64] __attribute__((aligned(16)));
    volatile uint8_t masks[49][8] __attribute__((aligned(16)));
    volatile uint32_t shifts[49];
    volatile uint32_t lane_shifts[49][64] __attribute__((aligned(16)));
    volatile uint32_t packed[64] __attribute__((aligned(16)));
    volatile _Float16 output_scratch[64] __attribute__((aligned(16)));
    volatile uint32_t *profile=(volatile uint32_t *)p->vector_profile;
    uint32_t geometry_cycles=0u, input_cycles=0u, fma_cycles=0u, output_cycles=0u;
    const uint32_t height=p->output_shape[2], width=p->output_shape[3];
    const uint32_t kh=p->weight_shape[2], kw=p->weight_shape[3], kernel=kh*kw;
    const uint32_t ih=p->input_shape[2], iw=p->input_shape[3];
    const uint32_t sh=p->stride_h, sw=p->stride_w, dh=p->dilation_h, dw=p->dilation_w;
    const int64_t oy=(int64_t)p->output_origin_h*sh-p->pad_top-p->input_origin_h;
    const int64_t ox=(int64_t)p->output_origin_w*sw-p->pad_left-p->input_origin_w;
    const uint32_t input_row=p->input_strides[2];
    const uint32_t output_row=p->output_strides[2];
    uint32_t row_vl;
    asm volatile("vsetvli %0, %1, e16, m4, ta, mu" : "=r"(row_vl) : "r"(width));
    /* Wide rows share their horizontal gather geometry. Rebase the input
     * row for each output row instead of rebuilding all per-pixel offsets. */
    const int row_plan=width>=32u && width<=64u && row_vl==width;
    for (uint32_t block=0; block<height*width;) {
        uint32_t vl;
        uint32_t count=height*width-block;
        if (row_plan) count=width;
        else if (count>64u) count=64u;
        asm volatile("vsetvli %0, %1, e16, m4, ta, mu" : "=r"(vl) : "r"(count));
        uint32_t stamp=profile ? depthwise_cycle() : 0u;
        if (!row_plan || block==0u) {
            for (uint32_t k=0;k<kernel;++k) {
                for (uint32_t byte=0;byte<(vl+7u)/8u;++byte) masks[k][byte]=0;
                shifts[k]=(uint32_t)(ox+(int64_t)(k%kw)*dw)&1u ? 16u : 0u;
                for (uint32_t lane=0;lane<vl;++lane) {
                    const uint32_t pos=block+lane;
                    const int64_t y=row_plan ? (int64_t)(k/kw)*dh : oy+(int64_t)(pos/width)*sh+(int64_t)(k/kw)*dh;
                    const int64_t x=ox+(int64_t)(pos%width)*sw+(int64_t)(k%kw)*dw;
                    const int valid=(row_plan || (y>=0 && y<ih)) && x>=0 && x<iw;
                    offsets[k][lane]=valid ? ((uint32_t)y*input_row+(uint32_t)x*2u)&~3u : 0u;
                    if (sw&1u) lane_shifts[k][lane]=((uint32_t)x&1u)*16u;
                    if (valid) masks[k][lane/8u]|=1u<<(lane%8u);
                }
            }
        }
        if (profile) geometry_cycles+=depthwise_cycle()-stamp;
        for (uint32_t n=0;n<p->output_shape[0];++n)
            for (uint32_t c=0;c<p->output_shape[1];++c) {
                const uint32_t ic=(p->weight_origin_c+c)/p->channel_multiplier-p->input_origin_c;
                uintptr_t input=p->input+n*p->input_strides[0]+ic*p->input_strides[1];
                if (row_plan) input+=(uint32_t)(oy+(int64_t)(block/width)*sh)*input_row;
                const uintptr_t weight=p->weight+c*p->weight_strides[0];
                asm volatile("vsetvli zero, %0, e16, m4, ta, mu" :: "r"(vl));
                asm volatile("vfmv.v.f v8, %0" :: "f"((_Float16)0.0f));
                for (uint32_t k=0;k<kernel;++k) {
                    if (row_plan) {
                        const int64_t y=oy+(int64_t)(block/width)*sh+(int64_t)(k/kw)*dh;
                        if (y<0 || y>=ih) continue;
                    }
                    if (profile) stamp=depthwise_cycle();
                    const uint32_t bytes=(vl+7u)/8u;
                    asm volatile("vsetvli zero, %0, e8, m1, ta, ma" :: "r"(bytes));
                    asm volatile("vle8.v v0, (%0)" :: "r"(masks[k]) : "memory");
                    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" :: "r"(vl));
                    asm volatile("vle32.v v24, (%0)" :: "r"(offsets[k]) : "memory");
                    asm volatile("vluxei32.v v16, (%0), v24, v0.t" :: "r"(input) : "memory");
                    if (sw&1u) {
                        /* Adjacent output positions may select opposite halves
                         * of the loaded word. The offset registers are dead
                         * after the gather, so reuse them for per-lane shifts. */
                        asm volatile("vle32.v v24, (%0)" :: "r"(lane_shifts[k]) : "memory");
                        asm volatile("vsrl.vv v16, v16, v24");
                    } else {
                        asm volatile("vsrl.vx v16, v16, %0" :: "r"(shifts[k]));
                    }
                    asm volatile("vse32.v v16, (%0)" :: "r"(packed) : "memory");
                    asm volatile("vsetvli zero, %0, e16, m4, ta, mu" :: "r"(vl));
                    const uint32_t step=4u;
                    asm volatile("vlse16.v v4, (%0), %1" :: "r"(packed), "r"(step) : "memory");
                    if (profile) { input_cycles+=depthwise_cycle()-stamp; stamp=depthwise_cycle(); }
                    const _Float16 w=*(const _Float16 *)(weight+(k/kw)*p->weight_strides[2]+(k%kw)*p->weight_strides[3]);
                    asm volatile("vfmacc.vf v8, %0, v4, v0.t" :: "f"(w));
                    if (profile) fma_cycles+=depthwise_cycle()-stamp;
                }
                if (profile) stamp=depthwise_cycle();
                if (p->bias) {
                    const _Float16 bias=*(const _Float16 *)(p->bias+c*p->bias_stride);
                    asm volatile("vfadd.vf v8, v8, %0" :: "f"(bias));
                }
                const uintptr_t output=p->output+n*p->output_strides[0]+c*p->output_strides[1];
                if (output_row==width*2u && ((output+block*2u)&3u)==0u) {
                    asm volatile("vse16.v v8, (%0)" :: "r"(output+block*2u) : "memory");
                } else {
                    asm volatile("vse16.v v8, (%0)" :: "r"(output_scratch) : "memory");
                    for (uint32_t lane=0;lane<vl;++lane) {
                        const uint32_t pos=block+lane;
                        *(_Float16 *)(output+(pos/width)*output_row+(pos%width)*2u)=output_scratch[lane];
                    }
                }
                if (profile) output_cycles+=depthwise_cycle()-stamp;
            }
        block+=vl;
    }
    if (profile) { profile[0]=geometry_cycles; profile[1]=input_cycles; profile[2]=fma_cycles; profile[3]=output_cycles; }
    return 0;
}

/* Accumulate in FP16.
 * Gather into aligned scratch for arbitrary shard strides and padding, and
 * to satisfy the VLSU's four-byte alignment rule. Add bias after the reduction,
 * as the full-mesh GEMM does. */
int depthwise_conv_fp16_spatz_task(void)
{
    const volatile depthwise_conv_fp16_spatz_params_t *p =
        (const volatile depthwise_conv_fp16_spatz_params_t *)mmio32(SPATZ_DATA);
    if (p->vector_direct == 2u) return depthwise_vector_plan(p);
    if (p->im2col) {
        /* One group's [1,K] x [K,N] GEMM. Scratch rows and output are
         * four-byte aligned; m8 amortizes vector setup over more columns. */
        for (uint32_t x = 0; x < p->im2col_columns;) {
            uint32_t vl;
            asm volatile("vsetvli %0, %1, e16, m8, ta, ma"
                         : "=r"(vl) : "r"(p->im2col_columns - x));
            asm volatile("vfmv.v.f v0, %0" :: "f"((_Float16)0.0f));
            for (uint32_t ky = 0; ky < p->weight_shape[2]; ++ky)
                for (uint32_t kx = 0; kx < p->weight_shape[3]; ++kx) {
                    const _Float16 weight = *(const _Float16 *)(p->weight +
                        ky * p->weight_strides[2] + kx * p->weight_strides[3]);
                    const uintptr_t row = p->im2col +
                        ((ky * p->weight_shape[3] + kx) * p->im2col_stride + x) * 2u;
                    asm volatile("vle16.v v16, (%0)" :: "r"(row) : "memory");
                    asm volatile("vfmacc.vf v0, %0, v16" :: "f"(weight));
                }
            if (p->bias) {
                const _Float16 bias = *(const _Float16 *)p->bias;
                asm volatile("vfadd.vf v0, v0, %0" :: "f"(bias));
            }
            asm volatile("vse16.v v0, (%0)"
                         :: "r"(p->output + x * 2u) : "memory");
            x += vl;
        }
        return 0;
    }
    const uint32_t stride_h = p->stride_h;
    const uint32_t stride_w = p->stride_w;
    const uint32_t dilation_h = p->dilation_h;
    const uint32_t dilation_w = p->dilation_w;
    const uint32_t pad_top = p->pad_top;
    const uint32_t pad_left = p->pad_left;
    const uint32_t input_origin_h = p->input_origin_h;
    const uint32_t input_origin_w = p->input_origin_w;
    const uint32_t output_origin_h = p->output_origin_h;
    const uint32_t output_origin_w = p->output_origin_w;
    const uint32_t vector_direct = p->vector_direct;
    const uint32_t input_height = p->input_shape[2];
    const uint32_t input_width = p->input_shape[3];
    const uint32_t output_height = p->output_shape[2];
    const uint32_t output_width = p->output_shape[3];
    const uint32_t kernel_height = p->weight_shape[2];
    const uint32_t kernel_width = p->weight_shape[3];
    volatile uint32_t packed[64] __attribute__((aligned(16)));
    volatile _Float16 samples[64] __attribute__((aligned(16)));
    volatile _Float16 sums[64] __attribute__((aligned(16)));
    volatile uint8_t valid[8] __attribute__((aligned(16)));
    for (uint32_t n = 0; n < p->output_shape[0]; ++n)
        for (uint32_t c = 0; c < p->output_shape[1]; ++c) {
            uint32_t ic = (p->weight_origin_c + c) / p->channel_multiplier - p->input_origin_c;
            _Float16 bias = p->bias ? *(const _Float16 *)(p->bias + c * p->bias_stride) : (_Float16)0.0f;
            for (uint32_t y = 0; y < output_height; ++y)
                for (uint32_t x = 0; x < output_width;) {
                    uint32_t remaining = output_width - x;
                    if (remaining > 64u) remaining = 64u;
                    uint32_t vl;
                    asm volatile("vsetvli %0, %1, e16, m1, ta, mu" : "=r"(vl) : "r"(remaining));
                    asm volatile("vfmv.v.f v8, %0" :: "f"((_Float16)0.0f));
                    for (uint32_t ky = 0; ky < kernel_height; ++ky)
                        for (uint32_t kx = 0; kx < kernel_width; ++kx) {
                            int64_t iy = (int64_t)(output_origin_h + y) * stride_h - pad_top + ky * dilation_h - input_origin_h;
                            if (iy < 0 || iy >= input_height) continue;
                            _Float16 weight = *(const _Float16 *)(p->weight + c * p->weight_strides[0] + ky * p->weight_strides[2] + kx * p->weight_strides[3]);
                            /* For even horizontal strides, selected FP16 values
                             * occupy the same half of successive aligned words.
                             * Load words, select the half without FP conversion, then use
                             * the existing FP16 FMA. */
                            const int64_t first_x = (int64_t)(output_origin_w + x) * stride_w -
                                pad_left + kx * dilation_w - input_origin_w;
                            if (vector_direct) {
                                const int interior = first_x >= 0 &&
                                    first_x + (int64_t)(vl - 1u) * stride_w < input_width;
                                if (!interior) {
                                    for (uint32_t byte = 0; byte < (vl + 7u) / 8u; ++byte) valid[byte] = 0;
                                    for (uint32_t lane = 0; lane < vl; ++lane) {
                                        const int64_t ix = first_x + (int64_t)lane * stride_w;
                                        if (ix >= 0 && ix < input_width)
                                            valid[lane / 8u] |= 1u << (lane % 8u);
                                    }
                                    uint32_t mask_bytes = (vl + 7u) / 8u;
                                    asm volatile("vsetvli zero, %0, e8, m1, ta, ma" :: "r"(mask_bytes));
                                    asm volatile("vle8.v v0, (%0)" :: "r"(valid) : "memory");
                                }
                                const uintptr_t source = p->input + n * p->input_strides[0] +
                                    ic * p->input_strides[1] + (uint32_t)iy * p->input_strides[2] +
                                    (uint32_t)first_x * 2u;
                                const uintptr_t aligned = source & ~(uintptr_t)3u;
                                const uint32_t shift = (source & 2u) * 8u;
                                const uint32_t step = stride_w * 2u;
                                asm volatile("vsetvli zero, %0, e32, m2, ta, ma" :: "r"(vl));
                                if (interior)
                                    asm volatile("vlse32.v v16, (%0), %1" :: "r"(aligned), "r"(step) : "memory");
                                else
                                    asm volatile("vlse32.v v16, (%0), %1, v0.t" :: "r"(aligned), "r"(step) : "memory");
                                asm volatile("vsrl.vx v16, v16, %0" :: "r"(shift));
                                /* The simulator lacks integer narrowing. Use
                                 * aligned vector scratch, then select each low
                                 * halfword with a strided vector load. */
                                asm volatile("vse32.v v16, (%0)" :: "r"(packed) : "memory");
                                asm volatile("vsetvli zero, %0, e16, m1, ta, mu" :: "r"(vl));
                                const uint32_t half_step = 4u;
                                asm volatile("vlse16.v v4, (%0), %1" :: "r"(packed), "r"(half_step) : "memory");
                                if (interior)
                                    asm volatile("vfmacc.vf v8, %0, v4" :: "f"(weight));
                                else
                                    asm volatile("vfmacc.vf v8, %0, v4, v0.t" :: "f"(weight));
                                continue;
                            }
                            for (uint32_t byte = 0; byte < (vl + 7u) / 8u; ++byte) valid[byte] = 0;
                            for (uint32_t lane = 0; lane < vl; ++lane) {
                                int64_t ix = (int64_t)(output_origin_w + x + lane) * stride_w - pad_left + kx * dilation_w - input_origin_w;
                                uint32_t active = ix >= 0 && ix < input_width;
                                if (active) valid[lane / 8u] |= 1u << (lane % 8u);
                                samples[lane] = active ? *(const _Float16 *)(p->input + n * p->input_strides[0] + ic * p->input_strides[1] + (uint32_t)iy * p->input_strides[2] + (uint32_t)ix * p->input_strides[3]) : (_Float16)0.0f;
                            }
                            uint32_t mask_bytes = (vl + 7u) / 8u;
                            asm volatile("vsetvli zero, %0, e8, m1, ta, ma" :: "r"(mask_bytes));
                            asm volatile("vle8.v v0, (%0)" :: "r"(valid) : "memory");
                            asm volatile("vsetvli zero, %0, e16, m1, ta, mu" :: "r"(vl));
                            asm volatile("vle16.v v4, (%0)" :: "r"(samples) : "memory");
                            asm volatile("vfmacc.vf v8, %0, v4, v0.t" :: "f"(weight));
                        }
                    if (p->bias)
                        asm volatile("vfadd.vf v8, v8, %0" :: "f"(bias));
                    const uintptr_t output = p->output + n * p->output_strides[0] +
                        c * p->output_strides[1] + y * p->output_strides[2] + x * p->output_strides[3];
                    if (vector_direct && (output & 3u) == 0u) {
                        asm volatile("vse16.v v8, (%0)" :: "r"(output) : "memory");
                        x += vl;
                        continue;
                    }
                    asm volatile("vse16.v v8, (%0)" :: "r"(sums) : "memory");
                    for (uint32_t lane = 0; lane < vl; ++lane)
                        *(_Float16 *)(p->output + n * p->output_strides[0] + c * p->output_strides[1] + y * p->output_strides[2] + (x + lane) * p->output_strides[3]) = sums[lane];
                    x += vl;
                }
        }
    return 0;
}
