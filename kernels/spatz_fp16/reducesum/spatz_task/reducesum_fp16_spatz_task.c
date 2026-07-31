#include "tile.h"
#include "reducesum_fp16_spatz_params.h"

/* Sums 'reduce_dim' rows of 'inner_dim' elements each. The accumulator is FP16,
 * like every other operand of these kernels, so long reductions lose precision
 * and can reach Inf beyond 65504. */
static inline void reduce_sum_core(const _Float16 *src, _Float16 *dst, const size_t reduce_dim, const size_t inner_dim)
{
    size_t avl = inner_dim;
    size_t vl;

    for (; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile ("vmv.v.i v0, 0");

        for (size_t r = 0; r < reduce_dim; r++) {
            const _Float16 *p_src = src + (r * inner_dim) + (inner_dim - avl);

            asm volatile ("vle16.v v8, (%0)" :: "r"(p_src));
            asm volatile ("vfadd.vv v0, v0, v8");
        }

        _Float16 *p_dst = dst + (inner_dim - avl);
        asm volatile ("vse16.v v0, (%0)" :: "r"(p_dst) : "memory");
    }
}

/* Same accumulation order as the vector path, one output element at a time.
 * Used when the vector accesses would not be 4-byte aligned. */
static inline void reduce_sum_core_scalar(const _Float16 *src, _Float16 *dst, const size_t reduce_dim, const size_t inner_dim)
{
    for (size_t i = 0; i < inner_dim; i++) {
        _Float16 acc = 0.0f;

        for (size_t r = 0; r < reduce_dim; r++)
            acc = (_Float16)(acc + src[(r * inner_dim) + i]);

        dst[i] = acc;
    }
}

/* The Spatz VLSU corrupts vector accesses to non-aligned addresses. Within a row
 * the unit-stride accesses keep the alignment of its base (VLMAX is even), but the
 * loads step by inner_dim between rows, so that has to be even as well. */
static inline int reduce_sum_vector_safe(const _Float16 *src, const _Float16 *dst, const size_t inner_dim)
{
    if ((inner_dim % 2) != 0)
        return 0;

    return (((uintptr_t)src & 3u) == 0) && (((uintptr_t)dst & 3u) == 0);
}

int reducesum_fp16_spatz_task(void)
{
    volatile reducesum_fp16_spatz_params_t *params;
    uintptr_t params_addr;

    const _Float16 *src;
    _Float16 *dst;
    uint32_t reduce_dim;
    uint32_t inner_dim;
    uint32_t outer_len;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile reducesum_fp16_spatz_params_t *) params_addr;

    src = (_Float16 *) params->shard_X;
    dst = (_Float16 *) params->shard_Y;
    reduce_dim = params->reduce_dim;
    inner_dim = params->inner_dim;
    outer_len = params->outer_len;

    uint32_t in_stride  = reduce_dim * inner_dim;
    uint32_t out_stride = inner_dim;

    for (uint32_t o = 0; o < outer_len; o++) {
        const _Float16 *row_src = src + (o * in_stride);
        _Float16 *row_dst = dst + (o * out_stride);

        if (reduce_sum_vector_safe(row_src, row_dst, inner_dim))
            reduce_sum_core(row_src, row_dst, reduce_dim, inner_dim);
        else
            reduce_sum_core_scalar(row_src, row_dst, reduce_dim, inner_dim);
    }

    return 0;
}
