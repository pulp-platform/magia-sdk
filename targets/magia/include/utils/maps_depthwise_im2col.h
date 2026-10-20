#ifndef MAPS_DEPTHWISE_IM2COL_H
#define MAPS_DEPTHWISE_IM2COL_H

#ifndef MAPS_DEPTHWISE_IM2COL_GEMM
#define MAPS_DEPTHWISE_IM2COL_GEMM 1u
#endif

/* Rounded operation-dispatch calibration; mirrored by magia_v3's planner.
 * Use a 20% margin before replacing im2col, rather than chasing small wins. */
#ifndef MAPS_DEPTHWISE_VECTOR_AUTO
#define MAPS_DEPTHWISE_VECTOR_AUTO 1u
#endif
/* Freestanding RV32 has no libgcc 64-bit division helper. Dispatch runs once
 * per operation; bounded shift/subtract keeps shape arithmetic wide. */
static inline uint64_t maps_depthwise_dispatch_div(uint64_t n, uint64_t d)
{
    uint64_t q=0, r=0;
    for (int bit=63; bit>=0; --bit) {
        r=(r<<1)|(n>>63);
        n<<=1;
        q<<=1;
        if (r>=d) { r-=d; q|=1u; }
    }
    return q;
}
static inline int maps_depthwise_prefer_vector(
    const volatile depthwise_conv_fp16_spatz_params_t *p)
{
    const uint64_t h=p->output_shape[2], w=p->output_shape[3];
    const uint64_t k=(uint64_t)p->weight_shape[2]*p->weight_shape[3];
    const uint64_t c=(uint64_t)p->output_shape[0]*p->output_shape[1];
    if (!h || !w || !k || k>49u || !c ||
        p->input_strides[3]!=2u || p->output_strides[3]!=2u ||
        (p->input_shape[3]&1u) ||
        ((p->input | p->input_strides[0] | p->input_strides[1] | p->input_strides[2] |
          p->output | p->output_strides[0] | p->output_strides[1])&3u) ||
        p->output_strides[2]!=w*2u)
        return 0;
    const int row=w>=32u && w<=64u;
    const uint64_t blocks=row ? h : (h*w+63u)/64u;
    const uint64_t lanes=row ? w : (h*w<64u ? h*w : 64u);
    const uint64_t geometry=k*(row ? w : h*w);
    const uint64_t odd=p->stride_w&1u;
    const uint64_t vector=6000u+geometry*(76u+10u*odd)+
        c*blocks*(150u+k*(60u+lanes+12u*odd));
    const uint64_t band_rows=maps_depthwise_dispatch_div(65000u, 2u*(k+1u)*(w+1u));
    const uint64_t bands=maps_depthwise_dispatch_div(h+(band_rows ? band_rows : 1u)-1u, band_rows ? band_rows : 1u);
    const uint64_t prepare=p->stride_w==1u ? k*bands*900u : k*h*420u;
    const uint64_t data=p->stride_w==1u ? (h*w*(k+3u))/2u : maps_depthwise_dispatch_div(h*w*(k+3u)*4u, 5u);
    const uint64_t im2col=6000u+c*(prepare+data+bands*1050u);
    return vector*6u < im2col*5u;
}

/* Diagnostic-only counters: prepare, launch/wait, output-copy cycles,
 * input DMA submissions, launches, output DMA submissions. */
#if defined(MAPS_DEPTHWISE_PROFILE) && MAPS_DEPTHWISE_PROFILE
/* Diagnostics reserve this region in each tile's L1. A static C object may
 * reside in shared L2 and would mix counters from simultaneously active tiles. */
#define maps_depthwise_profile ((volatile uint32_t *)(get_l1_base(get_hartid()) + 0xB0000u))
#define MAPS_DW_STAMP(name) uint32_t name = maps_read_cycle()
#define MAPS_DW_TIME(index, name) (maps_depthwise_profile[index] += maps_read_cycle() - (name))
#define MAPS_DW_COUNT(index, count) (maps_depthwise_profile[index] += (count))
#else
#define MAPS_DW_STAMP(name) ((void)0)
#define MAPS_DW_TIME(index, name) ((void)0)
#define MAPS_DW_COUNT(index, count) ((void)0)
#endif

/* Bounded lower_bound avoids 64-bit division helpers unavailable in the
 * freestanding control-core runtime, while keeping origin arithmetic wide. */
static inline uint32_t maps_depthwise_row_bound(
    int64_t origin, uint32_t stride, uint32_t rows, uint32_t limit)
{
    uint32_t lo = 0u, hi = rows;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2u;
        if (origin + (int64_t)mid * stride < limit) lo = mid + 1u;
        else hi = mid;
    }
    return lo;
}

/* Runtime lowering of one depthwise shard to bounded, tile-local im2col
 * and per-channel GEMMs. The graph's input/output and FIFO layouts are
 * unchanged. Use the direct task for noncontiguous columns or insufficient scratch.
 * DMA transfers are ticketed because FIFO sends can share the same engine. */
static inline int maps_depthwise_im2col_gemm(
    maps_operation_runtime_t *runtime,
    volatile depthwise_conv_fp16_spatz_params_t *p,
    uint32_t block_rows)
{
    const depthwise_conv_fp16_spatz_params_t d = *p;
    const uint32_t width = d.output_shape[3];
    const uint32_t kernel = d.weight_shape[2] * d.weight_shape[3];
    const uintptr_t columns = ((uintptr_t)p + sizeof(*p) + 15u) & ~(uintptr_t)15u;
    for (uint32_t n = 0; n < d.output_shape[0]; ++n)
        for (uint32_t c = 0; c < d.output_shape[1]; ++c) {
            const uint32_t ic = (d.weight_origin_c + c) / d.channel_multiplier - d.input_origin_c;
            const uintptr_t input = d.input + n * d.input_strides[0] + ic * d.input_strides[1];
            p->weight = d.weight + c * d.weight_strides[0];
            p->bias = d.bias ? d.bias + c * d.bias_stride : 0u;
            for (uint32_t y = 0; y < d.output_shape[2]; y += block_rows) {
                uint32_t rows = d.output_shape[2] - y;
                if (rows > block_rows) rows = block_rows;
                MAPS_DW_STAMP(prepare_start);
                const uint32_t count = rows * width;
                const uint32_t pitch = (count + 1u) & ~1u;
                for (uint32_t ky = 0; ky < d.weight_shape[2]; ++ky)
                    for (uint32_t kx = 0; kx < d.weight_shape[3]; ++kx) {
                        const uintptr_t row = columns +
                            (ky * d.weight_shape[3] + kx) * pitch * 2u;
                        /* Origins are absolute tensor coordinates. Clip the
                         * valid rectangle to this shard's actual input halo. */
                        const int64_t ih0 = ((int64_t)d.output_origin_h + y) * d.stride_h -
                            d.pad_top + (int64_t)ky * d.dilation_h - d.input_origin_h;
                        const int64_t iw0 = (int64_t)d.output_origin_w * d.stride_w - d.pad_left +
                            (int64_t)kx * d.dilation_w - d.input_origin_w;
                        const uint32_t rlo = maps_depthwise_row_bound(ih0, d.stride_h, rows, 0u);
                        const uint32_t rhi = maps_depthwise_row_bound(ih0, d.stride_h, rows, d.input_shape[2]);
                        const uint32_t clo = maps_depthwise_row_bound(iw0, d.stride_w, width, 0u);
                        const uint32_t chi = maps_depthwise_row_bound(iw0, d.stride_w, width, d.input_shape[3]);
                        if (rlo >= rhi || clo >= chi) {
                            maps_operation_zero(row, count * 2u);
                        } else {
                            maps_operation_zero(row, rlo * width * 2u);
                            maps_operation_zero(row + rhi * width * 2u, (rows - rhi) * width * 2u);
                            if (clo || chi != width)
                                for (uint32_t r = rlo; r < rhi; ++r) {
                                    maps_operation_zero(row + r * width * 2u, clo * 2u);
                                    maps_operation_zero(row + (r * width + chi) * 2u, (width - chi) * 2u);
                                }
                            const uintptr_t source = input +
                                (uint32_t)(ih0 + (int64_t)rlo * d.stride_h) * d.input_strides[2] +
                                (uint32_t)(iw0 + (int64_t)clo * d.stride_w) * 2u;
                            MAPS_DW_COUNT(3, d.stride_w == 1u ? 1u : rhi - rlo);
                            if (d.stride_w == 1u) {
                                if (maps_idma_copy_2d_ticketed(1u,
                                    row + (rlo * width + clo) * 2u, source,
                                    (chi - clo) * 2u, width * 2u,
                                    d.stride_h * d.input_strides[2], rhi - rlo) != 0)
                                    return -4;
                            } else {
                                /* The GVSoC DMA implements only one repetition
                                 * dimension. Gather strided halfwords with one
                                 * 2-D descriptor per output row; do not rely on
                                 * a nested 3-D transfer being implemented. */
                                for (uint32_t r = rlo; r < rhi; ++r)
                                    if (maps_idma_copy_2d_ticketed(1u,
                                            row + (r * width + clo) * 2u,
                                            source + (r - rlo) * d.stride_h * d.input_strides[2],
                                            2u, 2u, d.stride_w * 2u, chi - clo) != 0)
                                        return -4;
                            }
                        }
                        if (pitch != count) *(uint16_t *)(row + count * 2u) = 0u;
                    }
                p->im2col = columns;
                p->im2col_columns = count;
                p->im2col_stride = pitch;
                p->output = columns + kernel * pitch * 2u;
                asm volatile("fence rw,rw" ::: "memory");
                MAPS_DW_TIME(0, prepare_start);
                MAPS_DW_COUNT(4, 1u);
                MAPS_DW_STAMP(spatz_start);
                spatz_run_task_with_params(runtime->depthwise_conv_fp16_task, (uint32_t)p);
                const int result = maps_wait_for_spatz(runtime);
                MAPS_DW_TIME(1, spatz_start);
                if (result != 0) return result;
                MAPS_DW_STAMP(copy_start);
                MAPS_DW_COUNT(5, 1u);
                const uintptr_t output = d.output + n * d.output_strides[0] +
                    c * d.output_strides[1] + y * d.output_strides[2];
                if (maps_idma_copy_2d_ticketed(1u,
                        output, p->output, width * 2u,
                        d.output_strides[2], width * 2u, rows) != 0)
                    return -4;
                MAPS_DW_TIME(2, copy_start);
            }
        }
    return 0;
}

#endif
