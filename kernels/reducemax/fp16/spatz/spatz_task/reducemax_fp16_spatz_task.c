#include "tile.h"
#include "reducemax_fp16_spatz_params.h"

/* Unit-stride vector accesses must be 4-byte aligned. */
#define MISALIGNED(p)  (((uintptr_t)(p)) & 0x2)

/* Store vl elements of v0; peel the first one if p is misaligned. Clobbers v8. */
static inline void store_v0(_Float16 *p, const size_t vl)
{
    _Float16 first;

    if (!MISALIGNED(p)) {
        asm volatile("vse16.v v0, (%0)" ::"r"(p) : "memory");
        return;
    }

    asm volatile("vfmv.f.s %0, v0" : "=f"(first));
    *p++ = first;
    if (vl > 1) {
        asm volatile("vsetvli zero, %0, e16, m8, ta, ma" ::"r"(vl - 1));
        asm volatile("vslidedown.vi v8, v0, 1");
        asm volatile("vse16.v v8, (%0)" ::"r"(p) : "memory");
    }
}

/* ReduceMax of a tensor of size [outer_dims..., reduce_dim, inner_dim] with
   inner_dim > 1: vectorize across inner_dim, element-wise max over reduce_dim rows.
   In case of aligned rows (step==1):
                         inner_dim
              |<--------------------->|
          ^   +-----------------------+   r0   --(1)--> v0 = r0
          |   +-----------------------+   r1   --(2)--> v0 = max(v0, r1)
          |   +-----------------------+   r2   --(3)--> v0 = max(v0, r2)
  reduce  |   +-----------------------+   r3   --(4)--> v0 = max(v0, r3)
   _dim   |   :                       :    :              :
          |   +-----------------------+   rR-1 --(R)--> v0 = max(v0, rR-1)
          v                                               |
                                                          v
                                            +-----------------------+
                                            |          dst          |  1
                                            +-----------------------+
                                            |<----- inner_dim ----->|
    
    In case of misaligned rows (step==2):
                       inner_dim
               |<--------------------->|
          ^    +-----------------------+   r0  --(a1)--> v0  = r0
          |  x +-----------------------+   r1  --(b1)--> v16 = r1'            (vl+1 lanes)
          |    +-----------------------+   r2  --(a2)--> v0  = max(v0,  r2)
  reduce  |  x +-----------------------+   r3  --(b2)--> v16 = max(v16, r3')
   _dim   |    +-----------------------+   r4  --(a3)--> v0  = max(v0,  r4)
          |  x +-----------------------+   r5  --(b3)--> v16 = max(v16, r5')
          |    :                       :    :                :
          v
                                                 v0             v16
                                                 |               |
                                                 |        vslidedown 1   (drop x lane)
                                                 |               |
                                                 +---> max <-----+
                                                        |
                                                        v
                                          +-----------------------+
                                          |          dst          |  1
                                          +-----------------------+
                                          |<----- inner_dim ----->|

  order: a1 a2 a3 ... (all v0 rows), then b1 b2 b3 ... (all v16 rows), then merge

*/
static __attribute__((noinline)) void reduce_max_noninnermost(const _Float16 *src,
                                           _Float16 *dst,
                                           const size_t reduce_dim,
                                           const size_t inner_dim)
{
    const size_t step = (inner_dim & 1) ? 2 : 1;    /* stride between same-alignment rows */
    size_t avl = inner_dim;
    size_t vlmax, vl_cap, vl;

    asm volatile("vsetvli %0, zero, e16, m8, ta, ma" : "=r"(vlmax));
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

        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl < vl_cap ? avl : vl_cap));

        /* Seed with the first row, not 0, to handle all-negative inputs. */
        if (r_al < reduce_dim) {
            asm volatile("vle16.v v0, (%0)" ::"r"(p_0 + r_al * inner_dim));
            for (size_t r = r_al + step; r < reduce_dim; r += step) {
                asm volatile("vle16.v v8, (%0)" ::"r"(p_0 + r * inner_dim));
                asm volatile("vfmax.vv v0, v0, v8");
            }
        }

        if (r_mis < reduce_dim) {
            asm volatile("vsetvli zero, %0, e16, m8, ta, ma" ::"r"(vl + 1));
            asm volatile("vle16.v v16, (%0)" ::"r"(p_0 + r_mis * inner_dim - 1));
            for (size_t r = r_mis + step; r < reduce_dim; r += step) {
                asm volatile("vle16.v v8, (%0)" ::"r"(p_0 + r * inner_dim - 1));
                asm volatile("vfmax.vv v16, v16, v8");
            }

            asm volatile("vsetvli zero, %0, e16, m8, ta, ma" ::"r"(vl));
            if (r_al < reduce_dim) {
                asm volatile("vslidedown.vi v8, v16, 1");
                asm volatile("vfmax.vv v0, v0, v8");
            } else {
                asm volatile("vslidedown.vi v0, v16, 1");
            }
        }

        store_v0(dst + col, vl);
    }
}

/* ReduceMax of a tensor of size [outer_dims..., reduce_dim, inner_dim] with
   inner_dim == 1: element-wise max over VL-wide chunks, then one vfredmax into v16[0].
   The reduce_dim row is contiguous; src[0] seeds v16[0] and, if misaligned, is skipped
   so that the chunks c0..cN-1 start aligned:

       src[0] --(0)--> v16[0] = src[0]
                          vl
              |<--------------------->|
          ^   +-----------------------+   c0   --(1)--> v0 = c0
          |   +-----------------------+   c1   --(2)--> v0 = max(v0, c1)
  reduce  |   +-----------------------+   c2   --(3)--> v0 = max(v0, c2)
   _dim   |   :                       :    :              :
          |   +-----------+               cN-1 --(N)--> v0 = max(v0, cN-1)   (tu: keeps upper lanes)
          v                                               |
                                                     vfredmax.vs   (max of v16[0] and all v0 lanes)
                                                          |
                                                          v
                                                       +-----+
                                                       | dst |  1
                                                       +-----+
*/
static inline void reduce_max_innermost(const _Float16 *src, _Float16 *dst, const size_t reduce_dim)
{
    size_t avl = reduce_dim;
    size_t vl, vl_first;
    _Float16 max;

    asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vfmv.s.f v16, %0" ::"f"(src[0]));

    /* src[0] is already in v16: skip it to realign. */
    if (MISALIGNED(src)) {
        src++;
        avl--;
    }

    if (avl > 0) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl_first) : "r"(avl));
        asm volatile("vle16.v v0, (%0)" ::"r"(src));
        src += vl_first;
        avl -= vl_first;

        if (avl > 0) {
            /* tu: a short last chunk keeps the upper lanes of v0. */
            for (; avl > 0; avl -= vl) {
                asm volatile("vsetvli %0, %1, e16, m8, tu, ma" : "=r"(vl) : "r"(avl));
                asm volatile("vle16.v v8, (%0)" ::"r"(src));
                asm volatile("vfmax.vv v0, v0, v8");
                src += vl;
            }
            asm volatile("vsetvli zero, %0, e16, m8, ta, ma" ::"r"(vl_first));
        }

        asm volatile("vfredmax.vs v16, v0, v16");
    }

    asm volatile("vfmv.f.s %0, v16" : "=f"(max));
    *dst = max;
}

/* ReduceMax of a tensor of size [outer_dims..., reduce_dim, inner_dim] with
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
               v0 = k0                                   +-----+   ^
                    v0 = max(v0, k1)                     |     |   |
                         v0 = max(v0, k2)       ---->    | dst |   vl
                                 :                       |     |   |
                                v0 = max(v0, kR-1)       +-----+   v
                                                            1
*/
static inline void reduce_max_innermost_strided(const _Float16 *src,
                                                _Float16 *dst,
                                                size_t rows,
                                                const size_t reduce_dim)
{
    const size_t row_stride = reduce_dim * sizeof(_Float16);
    size_t vl;

    for (; rows > 0; rows -= vl, src += vl * reduce_dim, dst += vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(rows));

        asm volatile("vlse16.v v0, (%0), %1" ::"r"(src), "r"(row_stride));
        for (size_t c = 1; c < reduce_dim; c++) {
            asm volatile("vlse16.v v8, (%0), %1" ::"r"(src + c), "r"(row_stride));
            asm volatile("vfmax.vv v0, v0, v8");
        }

        store_v0(dst, vl);
    }
}

int reducemax_fp16_spatz_task(void)
{
    volatile reducemax_fp16_spatz_params_t *params;
    uintptr_t params_addr;

    const _Float16 *src;
    _Float16 *dst;
    uint32_t reduce_dim;
    uint32_t inner_dim;
    uint32_t outer_len;

    params_addr = mmio32(SPATZ_DATA);
    params      = (volatile reducemax_fp16_spatz_params_t *)params_addr;

    src        = (_Float16 *)params->shard_X;
    dst        = (_Float16 *)params->shard_Y;
    reduce_dim = params->reduce_dim;
    inner_dim  = params->inner_dim;
    outer_len  = params->outer_len;

    uint32_t in_stride  = reduce_dim * inner_dim;
    uint32_t out_stride = inner_dim;

    /* Batch many short rows; long rows are faster one at a time. */
    if (inner_dim == 1 && outer_len >= 8 && reduce_dim <= 256) {
        reduce_max_innermost_strided(src, dst, outer_len, reduce_dim);
        return 0;
    }

    for (uint32_t o = 0; o < outer_len; o++) {
        if (inner_dim == 1)
            reduce_max_innermost(src + (o * in_stride), dst + o, reduce_dim);
        else
            reduce_max_noninnermost(src + (o * in_stride), dst + (o * out_stride), reduce_dim, inner_dim);
    }

    return 0;
}
