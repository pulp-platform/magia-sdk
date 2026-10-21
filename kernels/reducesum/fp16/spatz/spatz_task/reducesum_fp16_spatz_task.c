#include "tile.h"
#include "reducesum_fp16_spatz_params.h"

/* Sums accumulate in fp32: fp16 inputs are loaded at e16/m4 and widened into e32/m8 accumulators
   with vfwadd, and the result is narrowed back to fp16 only once, at the end.
   Register use: v0 (m8) accumulator, v8 (m4) loads, v16 (m8) misaligned-row accumulator,
   v24 (m8) temporary. */

/* Unit-stride vector accesses must be 4-byte aligned. */
#define MISALIGNED(p)  (((uintptr_t)(p)) & 0x2)

/* Narrow vl fp32 sums of v0 to fp16 and store them; peel the first one if p is misaligned.
   Clobbers v8, v12. */
static inline void narrow_store_v0(_Float16 *p, const size_t vl)
{
    _Float16 first;

    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(vl));
    asm volatile("vfncvt.f.f.w v8, v0");

    if (!MISALIGNED(p)) {
        asm volatile("vse16.v v8, (%0)" ::"r"(p) : "memory");
        return;
    }

    asm volatile("vfmv.f.s %0, v8" : "=f"(first));
    *p++ = first;
    if (vl > 1) {
        asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(vl - 1));
        asm volatile("vslidedown.vi v12, v8, 1");
        asm volatile("vse16.v v12, (%0)" ::"r"(p) : "memory");
    }
}

/* ReduceSum of a tensor of size [outer_dims..., reduce_dim, inner_dim] with
   inner_dim > 1: vectorize across inner_dim, element-wise fp32 sum over reduce_dim rows.
   In case of aligned rows (step==1):
                         inner_dim
              |<--------------------->|
          ^   +-----------------------+   r0   --(1)--> v0 = wide(r0)
          |   +-----------------------+   r1   --(2)--> v0 = v0 + wide(r1)
          |   +-----------------------+   r2   --(3)--> v0 = v0 + wide(r2)
  reduce  |   +-----------------------+   r3   --(4)--> v0 = v0 + wide(r3)
   _dim   |   :                       :    :              :
          |   +-----------------------+   rR-1 --(R)--> v0 = v0 + wide(rR-1)
          v                                               |
                                                       vfncvt
                                                          |
                                                          v
                                            +-----------------------+
                                            |          dst          |  1
                                            +-----------------------+
                                            |<----- inner_dim ----->|

    In case of misaligned rows (step==2):
                       inner_dim
               |<--------------------->|
          ^    +-----------------------+   r0  --(a1)--> v0  = wide(r0)
          |  x +-----------------------+   r1  --(b1)--> v16 = wide(r1')            (vl+1 lanes)
          |    +-----------------------+   r2  --(a2)--> v0  = v0  + wide(r2)
  reduce  |  x +-----------------------+   r3  --(b2)--> v16 = v16 + wide(r3')
   _dim   |    +-----------------------+   r4  --(a3)--> v0  = v0  + wide(r4)
          |  x +-----------------------+   r5  --(b3)--> v16 = v16 + wide(r5')
          |    :                       :    :                :
          v
                                                 v0             v16
                                                 |               |
                                                 |        vslidedown 1   (drop x lane)
                                                 |               |
                                                 +-----> + <-----+
                                                         |
                                                      vfncvt
                                                         |
                                                         v
                                          +-----------------------+
                                          |          dst          |  1
                                          +-----------------------+
                                          |<----- inner_dim ----->|

  order: a1 a2 a3 ... (all v0 rows), then b1 b2 b3 ... (all v16 rows), then merge

*/
static __attribute__((noinline)) void reduce_sum_noninnermost(const _Float16 *src,
                                           _Float16 *dst,
                                           const size_t reduce_dim,
                                           const size_t inner_dim)
{
    register _Float16 zero asm("fs0") = 0.0f;
    const size_t step = (inner_dim & 1) ? 2 : 1;    /* stride between same-alignment rows */
    size_t avl = inner_dim;
    size_t vlmax, vl_cap, vl;

    asm volatile("vsetvli %0, zero, e16, m4, ta, ma" : "=r"(vlmax));
    vl_cap = ((inner_dim & 1) || MISALIGNED(src)) ? vlmax - 1 : vlmax;

    for (; avl > 0; avl -= vl) {
        const size_t col    = inner_dim - avl;
        const _Float16 *p_0 = src + col;
        size_t r_al, r_mis;     /* first aligned / misaligned row, reduce_dim if none */

        if (MISALIGNED(p_0)) {
            r_mis = 0;
            r_al  = (step == 2) ? 1 : reduce_dim;
        } else {
            r_al  = 0;
            r_mis = (step == 2) ? 1 : reduce_dim;
        }

        asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(avl < vl_cap ? avl : vl_cap));

        /* Seed with the first row (widened by adding 0): saves a zero-fill and one add. */
        if (r_al < reduce_dim) {
            asm volatile("vle16.v v8, (%0)" ::"r"(p_0 + r_al * inner_dim));
            asm volatile("vfwadd.vf v0, v8, %0" ::"f"(zero));
            for (size_t r = r_al + step; r < reduce_dim; r += step) {
                asm volatile("vle16.v v8, (%0)" ::"r"(p_0 + r * inner_dim));
                asm volatile("vfwadd.wv v0, v0, v8");
            }
        }

        if (r_mis < reduce_dim) {
            asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(vl + 1));
            asm volatile("vle16.v v8, (%0)" ::"r"(p_0 + r_mis * inner_dim - 1));
            asm volatile("vfwadd.vf v16, v8, %0" ::"f"(zero));
            for (size_t r = r_mis + step; r < reduce_dim; r += step) {
                asm volatile("vle16.v v8, (%0)" ::"r"(p_0 + r * inner_dim - 1));
                asm volatile("vfwadd.wv v16, v16, v8");
            }

            asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(vl));
            if (r_al < reduce_dim) {
                asm volatile("vslidedown.vi v24, v16, 1");
                asm volatile("vfadd.vv v0, v0, v24");
            } else {
                asm volatile("vslidedown.vi v0, v16, 1");
            }
        }

        narrow_store_v0(dst + col, vl);
    }
}

/* ReduceSum of a tensor of size [outer_dims..., reduce_dim, inner_dim] with
   inner_dim == 1: element-wise fp32 sum over VL-wide chunks, then one vfredusum into v16[0].
   The reduce_dim row is contiguous; if src is misaligned, src[0] seeds v16[0] and is
   skipped so that the chunks c0..cN-1 start aligned, otherwise v16[0] = 0:

       src[0] --(0)--> v16[0] = src[0] or 0
                          vl
              |<--------------------->|
          ^   +-----------------------+   c0   --(1)--> v0 = wide(c0)
          |   +-----------------------+   c1   --(2)--> v0 = v0 + wide(c1)
  reduce  |   +-----------------------+   c2   --(3)--> v0 = v0 + wide(c2)
   _dim   |   :                       :    :              :
          |   +-----------+               cN-1 --(N)--> v0 = v0 + wide(cN-1)   (tu: keeps upper lanes)
          v                                               |
                                                     vfredusum.vs   (fp32 sum of v16[0] and all v0 lanes)
                                                          |
                                                          v
                                                       +-----+
                                                       | dst |  1
                                                       +-----+
*/
static inline void reduce_sum_innermost(const _Float16 *src, _Float16 *dst, const size_t reduce_dim)
{
    register _Float16 zero asm("fs0") = 0.0f;
    size_t avl = reduce_dim;
    size_t vl, vl_first;
    float sum;

    /* Unlike max, sum is not idempotent: src[0] goes to v16 only if it is skipped. */
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(1));
    if (MISALIGNED(src)) {
        asm volatile("vfmv.s.f v16, %0" ::"f"((float)src[0]));
        src++;
        avl--;
    } else {
        asm volatile("vfmv.s.f v16, %0" ::"f"(0.0f));
    }

    if (avl > 0) {
        asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl_first) : "r"(avl));
        asm volatile("vle16.v v8, (%0)" ::"r"(src));
        asm volatile("vfwadd.vf v0, v8, %0" ::"f"(zero));
        src += vl_first;
        avl -= vl_first;

        /* tu: a short last chunk keeps the upper lanes of v0. */
        for (; avl > 0; avl -= vl) {
            asm volatile("vsetvli %0, %1, e16, m4, tu, ma" : "=r"(vl) : "r"(avl));
            asm volatile("vle16.v v8, (%0)" ::"r"(src));
            asm volatile("vfwadd.wv v0, v0, v8");
            src += vl;
        }

        asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(vl_first));
        asm volatile("vfredusum.vs v16, v0, v16");
    }

    asm volatile("vfmv.f.s %0, v16" : "=f"(sum));
    *dst = (_Float16)sum;
}

/* ReduceSum of a tensor of size [outer_dims..., reduce_dim, inner_dim] with
   inner_dim == 1, batched: vectorize across rows with strided column loads.
   Each lane owns one row; each column of vl rows is one vlse16 with stride reduce_dim:

                        reduce_dim
              |<------------------------->|
          ^   +----+----+----+-----+------+
          |   |    |    |    |     |      |
     vl   |   | k0 | k1 | k2 | ... | kR-1 |
    rows  |   |    |    |    |     |      |
          v   +----+----+----+-----+------+
                |    |    |           |
               (1)  (2)  (3)         (R)
                v    v    v           v
               v0 = wide(k0)                             +-----+   ^
                    v0 = v0 + wide(k1)                   |     |   |
                         v0 = v0 + wide(k2)   --vfncvt-->| dst |   vl
                                 :                       |     |   |
                                v0 = v0 + wide(kR-1)     +-----+   v
                                                            1
*/
static inline void reduce_sum_innermost_strided(const _Float16 *src,
                                                _Float16 *dst,
                                                size_t rows,
                                                const size_t reduce_dim)
{
    register _Float16 zero asm("fs0") = 0.0f;
    const size_t row_stride = reduce_dim * sizeof(_Float16);
    size_t vl;

    for (; rows > 0; rows -= vl, src += vl * reduce_dim, dst += vl) {
        asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(rows));

        asm volatile("vlse16.v v8, (%0), %1" ::"r"(src), "r"(row_stride));
        asm volatile("vfwadd.vf v0, v8, %0" ::"f"(zero));
        for (size_t c = 1; c < reduce_dim; c++) {
            asm volatile("vlse16.v v8, (%0), %1" ::"r"(src + c), "r"(row_stride));
            asm volatile("vfwadd.wv v0, v0, v8");
        }

        narrow_store_v0(dst, vl);
    }
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
    params      = (volatile reducesum_fp16_spatz_params_t *)params_addr;

    src        = (_Float16 *)params->shard_X;
    dst        = (_Float16 *)params->shard_Y;
    reduce_dim = params->reduce_dim;
    inner_dim  = params->inner_dim;
    outer_len  = params->outer_len;

    uint32_t in_stride  = reduce_dim * inner_dim;
    uint32_t out_stride = inner_dim;

    /* Batch many short rows; long rows are faster one at a time. */
    if (inner_dim == 1 && outer_len >= 8 && reduce_dim <= 256) {
        reduce_sum_innermost_strided(src, dst, outer_len, reduce_dim);
        return 0;
    }

    for (uint32_t o = 0; o < outer_len; o++) {
        if (inner_dim == 1)
            reduce_sum_innermost(src + (o * in_stride), dst + o, reduce_dim);
        else
            reduce_sum_noninnermost(src + (o * in_stride), dst + (o * out_stride), reduce_dim, inner_dim);
    }

    return 0;
}
