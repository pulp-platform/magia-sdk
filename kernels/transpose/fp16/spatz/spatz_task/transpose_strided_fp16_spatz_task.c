#include "tile.h"
#include "transpose_strided_fp16_spatz_params.h"

/* Raw integer interleave: preserves every FP16 bit, including NaN payloads. */
int transpose_strided_fp16_spatz_task(void)
{
    const volatile transpose_strided_fp16_spatz_params_t *shared =
        (const volatile transpose_strided_fp16_spatz_params_t *)mmio32(SPATZ_DATA);
    const uint32_t rank = shared->rank, paired = shared->packed_pairs;
    /* Scalar volatile metadata copies prevent compiler-generated e64/fld
     * struct copies on the RVF-only Spatz core. Vector data moves below are
     * explicit and use supported e16/e32 instructions. */
    volatile uint32_t shape[6], source_stride[6], output_stride[6], coord[6];
    for (uint32_t axis = 0; axis < rank; ++axis) {
        shape[axis] = shared->shape[axis];
        source_stride[axis] = shared->source_stride[axis];
        output_stride[axis] = shared->output_stride[axis];
        coord[axis] = 0u;
    }
    const uint32_t outer = rank - (paired ? 2u : 1u);
    const uint32_t columns = shape[outer];
    uint32_t rows = 1u;
    for (uint32_t axis = 0; axis < outer; ++axis) rows *= shape[axis];
    uintptr_t src = shared->input, dst = shared->output;
    for (uint32_t row = 0; row < rows; ++row) {
        if (!paired) {
            const uint32_t last = rank - 1u;
            if ((src | dst) & 3u) {
                /* The VLSU requires word-aligned starting addresses. Raw
                 * scalar halfword copies handle unaligned rows exactly. */
                for (uint32_t x = 0; x < shape[last]; ++x)
                    *(uint16_t *)(dst + x * output_stride[last]) =
                        *(const uint16_t *)(src + x * source_stride[last]);
            } else {
                for (uint32_t x = 0; x < shape[last];) {
                    if (((src + x * source_stride[last]) | (dst + x * output_stride[last])) & 3u) {
                        *(uint16_t *)(dst + x * output_stride[last]) =
                            *(const uint16_t *)(src + x * source_stride[last]);
                        ++x;
                        continue;
                    }
                    uint32_t vl;
                    asm volatile("vsetvli %0, %1, e16, m8, ta, ma"
                                 : "=r"(vl) : "r"(shape[last] - x));
                    asm volatile("vlse16.v v0, (%0), %1"
                                 :: "r"(src + x * source_stride[last]), "r"(source_stride[last]) : "memory");
                    asm volatile("vsse16.v v0, (%0), %1"
                                 :: "r"(dst + x * output_stride[last]), "r"(output_stride[last]) : "memory");
                    x += vl;
                }
            }
        } else {
            for (uint32_t column = 0; column < columns;) {
                if ((src + column * 2u) & 3u) {
                    const uint32_t lo = *(const uint16_t *)(src + column * 2u);
                    const uint32_t hi = *(const uint16_t *)(src + source_stride[rank - 1u] + column * 2u);
                    *(uint32_t *)(dst + column * 4u) = lo | (hi << 16);
                    ++column;
                    continue;
                }
                uint32_t vl;
                asm volatile("vsetvli %0, %1, e16, m1, ta, ma"
                             : "=r"(vl) : "r"(columns - column));
                asm volatile("vle16.v v8, (%0)" :: "r"(src + column * 2u) : "memory");
                asm volatile("vle16.v v12, (%0)"
                             :: "r"(src + source_stride[rank - 1u] + column * 2u) : "memory");
                /* Unsigned widening add with zero implements bit extension
                 * using instructions supported by the current GVSoC model. */
                asm volatile("vwaddu.vx v16, v8, zero\n\t"
                             "vwaddu.vx v20, v12, zero");
                asm volatile("vsetvli zero, %0, e32, m2, ta, ma" :: "r"(vl));
                asm volatile("vsll.vi v20, v20, 16\n\t"
                             "vor.vv v16, v16, v20");
                asm volatile("vse32.v v16, (%0)" :: "r"(dst + column * 4u) : "memory");
                column += vl;
            }
        }
        /* Increment the outer coordinates without per-row divisions. */
        for (uint32_t axis = outer; axis-- > 0u;) {
            if (++coord[axis] < shape[axis]) {
                src += source_stride[axis];
                dst += output_stride[axis];
                break;
            }
            coord[axis] = 0u;
            src -= (shape[axis] - 1u) * source_stride[axis];
            dst -= (shape[axis] - 1u) * output_stride[axis];
        }
    }
    asm volatile("fence rw,rw" ::: "memory");
    return 0;
}
