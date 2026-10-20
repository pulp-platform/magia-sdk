/*
 * Batched matmul, fp16, one shard of A's rows per tile.
 *
 * Y[b] = A[b] * B[b], with A [M, K], B [K, O] and Y [M, O]. Either operand may be shared
 * by every batch (a_batched / b_batched == 0); the batch loop then reads the same matrix
 * each time.
 *
 * The accumulation is FP16 - vfmacc over ascending k, one rounding per term - so the
 * scalar fallback below has to use a fused multiply-add too, not `acc += a * b`, which
 * would round twice.
 */
#include "tile.h"
#include "matmul_rows_fp16_spatz_params.h"

/* One FP16 fused multiply-add, c + a * b with a single rounding, exactly like vfmacc. */
static inline _Float16 fma16(_Float16 a, _Float16 b, _Float16 c)
{
    _Float16 out;

    asm volatile("fmadd.h %0, %1, %2, %3" : "=f"(out) : "f"(a), "f"(b), "f"(c));

    return out;
}

/*
 * Same order and the same roundings as the vector path: accumulate over ascending k with
 * fused multiply-adds. Used when the unit-stride vector accesses would not be 4-byte
 * aligned.
 */
static void matmul_scalar(const _Float16 *A, const _Float16 *B, _Float16 *Y, const size_t dim_M, const size_t dim_K, const size_t dim_O)
{
    for (size_t m = 0; m < dim_M; m++) {
        for (size_t o = 0; o < dim_O; o++) {
            _Float16 acc = 0.0f;

            for (size_t k = 0; k < dim_K; k++)
                acc = fma16(A[(m * dim_K) + k], B[(k * dim_O) + o], acc);

            Y[(m * dim_O) + o] = acc;
        }
    }
}

static void matmul_transposed_b_scalar(
    const _Float16 *A, const _Float16 *B, _Float16 *Y,
    const size_t dim_M, const size_t dim_K, const size_t dim_O)
{
    for (size_t m = 0; m < dim_M; ++m) {
        for (size_t o = 0; o < dim_O; ++o) {
            _Float16 acc = 0.0f;
            for (size_t k = 0; k < dim_K; ++k)
                acc = fma16(A[m * dim_K + k], B[o * dim_K + k], acc);
            Y[m * dim_O + o] = acc;
        }
    }
}

#ifndef MATMUL_TRANSPOSED_B_VECTOR
#define MATMUL_TRANSPOSED_B_VECTOR 1
#endif

/* Vectorize over output columns while preserving ascending-K FP16 FMAs.
 * With even K, a word-aligned B row provides two adjacent weights per word.
 * Gather those words across classes, select each half through aligned vector
 * scratch, and reuse the word load for two successive accumulation steps. */
static void matmul_transposed_b_vector(
    const _Float16 *A, const _Float16 *B, _Float16 *Y,
    const size_t dim_M, const size_t dim_K, const size_t dim_O)
{
    volatile uint32_t packed[64] __attribute__((aligned(16)));
    volatile _Float16 output[64] __attribute__((aligned(16)));
    const uint32_t step=dim_K*sizeof(_Float16), half_step=4u;
    for (size_t m=0;m<dim_M;++m)
        for (size_t o=0;o<dim_O;) {
            size_t count=dim_O-o;
            if (count>64u) count=64u;
            size_t vl;
            asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(count));
            asm volatile("vfmv.v.f v8, %0" :: "f"((_Float16)0.0f));
            for (size_t k=0;k<dim_K;k+=2u) {
                const _Float16 *weights=B+o*dim_K+k;
                asm volatile("vsetvli zero, %0, e32, m8, ta, ma" :: "r"(vl));
                asm volatile("vlse32.v v16, (%0), %1" :: "r"(weights), "r"(step) : "memory");
                asm volatile("vse32.v v16, (%0)" :: "r"(packed) : "memory");
                asm volatile("vsetvli zero, %0, e16, m4, ta, ma" :: "r"(vl));
                asm volatile("vlse16.v v4, (%0), %1" :: "r"(packed), "r"(half_step) : "memory");
                const _Float16 a0=A[m*dim_K+k];
                asm volatile("vfmacc.vf v8, %0, v4" :: "f"(a0));
                asm volatile("vsetvli zero, %0, e32, m8, ta, ma" :: "r"(vl));
                asm volatile("vsrl.vi v16, v16, 16");
                asm volatile("vse32.v v16, (%0)" :: "r"(packed) : "memory");
                asm volatile("vsetvli zero, %0, e16, m4, ta, ma" :: "r"(vl));
                asm volatile("vlse16.v v4, (%0), %1" :: "r"(packed), "r"(half_step) : "memory");
                const _Float16 a1=A[m*dim_K+k+1u];
                asm volatile("vfmacc.vf v8, %0, v4" :: "f"(a1));
            }
            _Float16 *dest=Y+m*dim_O+o;
            if (((uintptr_t)dest&3u)==0u) {
                asm volatile("vse16.v v8, (%0)" :: "r"(dest) : "memory");
            } else {
                asm volatile("vse16.v v8, (%0)" :: "r"(output) : "memory");
                for (size_t lane=0;lane<vl;++lane) dest[lane]=output[lane];
            }
            o+=vl;
        }
}

/*
 * The Spatz VLSU corrupts vector accesses to non-aligned addresses. The B rows and the Y
 * rows are read and written unit-stride and step by dim_O, so every row is 4-byte aligned
 * only if dim_O is even and the bases are. A is only ever read with scalar loads, and so
 * is alignment-agnostic.
 */
static inline int matmul_vector_safe(const _Float16 *B, const _Float16 *Y, const size_t dim_O)
{
    if ((dim_O % 2) != 0)
        return 0;

    return ((((uintptr_t)B & 3u) == 0) && (((uintptr_t)Y & 3u) == 0));
}

static void matmul(const _Float16 *A, const _Float16 *B, _Float16 *Y, const size_t dim_M, const size_t dim_K, const size_t dim_O)
{
    register _Float16 ZERO asm ("fs0") = 0.0f;
    const _Float16 *A_col1_elem1;
    const _Float16 *A_col1_elem2;
    const _Float16 *A_col2_elem1;
    const _Float16 *A_col2_elem2;
    const _Float16 *B_row1;
    const _Float16 *B_row2;
    _Float16 *Y_row1;
    _Float16 *Y_row2;
    size_t avl;
    size_t vl;

    for (int o = 0; o < dim_O; o += vl) {
        avl = dim_O - o;
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        for (int m = 0; m < (dim_M - 1); m += 2) {
            Y_row1 = Y + (m * dim_O + o);
            Y_row2 = Y + ((m + 1) * dim_O + o);

            asm volatile ("vfmv.v.f v0, %0" :: "f"(ZERO));
            asm volatile ("vfmv.v.f v8, %0" :: "f"(ZERO));

            for (int k = 0; k < (dim_K - 1); k += 2) {
                B_row1 = B + (k * dim_O + o);
                B_row2 = B + ((k + 1) * dim_O + o);

                A_col1_elem1 = A + (m * dim_K + k);
                A_col2_elem1 = A + (m * dim_K + (k + 1));
                A_col1_elem2 = A + ((m + 1) * dim_K + k);
                A_col2_elem2 = A + ((m + 1) * dim_K + (k + 1));

                asm volatile ("vle16.v v16, (%0)" :: "r"(B_row1));
                asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(*A_col1_elem1));
                asm volatile ("vfmacc.vf v8, %0, v16" :: "f"(*A_col1_elem2));

                asm volatile ("vle16.v v24, (%0)" :: "r"(B_row2));
                asm volatile ("vfmacc.vf v0, %0, v24" :: "f"(*A_col2_elem1));
                asm volatile ("vfmacc.vf v8, %0, v24" :: "f"(*A_col2_elem2));
            }

            if (dim_K % 2) {
                B_row1 = B + ((dim_K - 1) * dim_O + o);
                A_col1_elem1 = A + (m * dim_K + (dim_K - 1));
                A_col1_elem2 = A + ((m + 1) * dim_K + (dim_K - 1));

                asm volatile ("vle16.v v16, (%0)" :: "r"(B_row1));
                asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(*A_col1_elem1));
                asm volatile ("vfmacc.vf v8, %0, v16" :: "f"(*A_col1_elem2));
            }

            asm volatile ("vse16.v v0, (%0)" :: "r"(Y_row1));
            asm volatile ("vse16.v v8, (%0)" :: "r"(Y_row2));
        }

        if (dim_M % 2) {
            Y_row1 = Y + ((dim_M - 1) * dim_O + o);

            asm volatile ("vfmv.v.f v0, %0" :: "f"(ZERO));

            for (int k = 0; k < (dim_K - 1); k += 2) {
                B_row1 = B + (k * dim_O + o);
                B_row2 = B + ((k + 1) * dim_O + o);
                A_col1_elem1 = A + ((dim_M - 1) * dim_K + k);
                A_col2_elem1 = A + ((dim_M - 1) * dim_K + (k + 1));

                asm volatile ("vle16.v v16, (%0)" :: "r"(B_row1));
                asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(*A_col1_elem1));

                asm volatile ("vle16.v v24, (%0)" :: "r"(B_row2));
                asm volatile ("vfmacc.vf v0, %0, v24" :: "f"(*A_col2_elem1));
            }

            if (dim_K % 2) {
                B_row1 = B + ((dim_K - 1) * dim_O + o);
                A_col1_elem1 = A + ((dim_M - 1) * dim_K + (dim_K - 1));

                asm volatile ("vle16.v v16, (%0)" :: "r"(B_row1));
                asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(*A_col1_elem1));
            }

            asm volatile ("vse16.v v0, (%0)" :: "r"(Y_row1));
        }
    }
}

#ifndef MATMUL_VECTOR_BIAS
#define MATMUL_VECTOR_BIAS 1
#endif

#if !MATMUL_VECTOR_BIAS
static void add_bias_scalar(const _Float16 *bias, _Float16 *Y, const size_t dim_M,
                     const size_t dim_O, uint32_t mode)
{
    if (!bias || mode == MATMUL_BIAS_NONE)
        return;
    for (size_t m = 0; m < dim_M; ++m)
        for (size_t o = 0; o < dim_O; ++o) {
            const size_t bias_index = mode == MATMUL_BIAS_SCALAR
                ? 0 : (mode == MATMUL_BIAS_ROW
                    ? m : (mode == MATMUL_BIAS_COLUMN ? o : m * dim_O + o));
            Y[m * dim_O + o] += bias[bias_index];
        }
}
#endif

static void add_bias(const _Float16 *bias, _Float16 *Y, const size_t dim_M,
                     const size_t dim_O, uint32_t mode)
{
#if MATMUL_VECTOR_BIAS
    if (!bias || mode==MATMUL_BIAS_NONE) return;
    const int scalar_bias=mode==MATMUL_BIAS_SCALAR || mode==MATMUL_BIAS_ROW;
    for (size_t m=0;m<dim_M;++m) {
        const _Float16 *b=mode==MATMUL_BIAS_COLUMN ? bias :
            (scalar_bias ? bias+(mode==MATMUL_BIAS_ROW ? m : 0u) : bias+m*dim_O);
        for (size_t o=0;o<dim_O;) {
            _Float16 *dest=Y+m*dim_O+o;
            const _Float16 *source=b+(scalar_bias ? 0u : o);
            if (((uintptr_t)dest&3u) || (!scalar_bias && ((uintptr_t)source&3u))) {
                *dest+=*source;
                ++o;
                continue;
            }
            size_t vl;
            asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(dim_O-o));
            asm volatile("vle16.v v0, (%0)" :: "r"(dest) : "memory");
            if (scalar_bias) {
                const _Float16 value=*source;
                asm volatile("vfadd.vf v0, v0, %0" :: "f"(value));
            } else {
                asm volatile("vle16.v v8, (%0)" :: "r"(source) : "memory");
                asm volatile("vfadd.vv v0, v0, v8");
            }
            asm volatile("vse16.v v0, (%0)" :: "r"(dest) : "memory");
            o+=vl;
        }
    }
#else
    add_bias_scalar(bias,Y,dim_M,dim_O,mode);
#endif
}

static inline uint32_t matmul_cycle(void)
{
    uint32_t value;
    asm volatile("csrr %0, cycle" : "=r"(value) :: "memory");
    return value;
}

int matmul_rows_fp16_spatz_task(void)
{
    volatile matmul_rows_fp16_spatz_params_t *params;
    uintptr_t params_addr;
    const _Float16 *shard_A;
    const _Float16 *shard_B;
    _Float16 *shard_Y;
    size_t M, K, O;
    uint32_t batch_len;
    uint32_t a_batched;
    uint32_t b_batched;
    uint32_t bias_mode;
    uint32_t bias_batched;
    uint32_t transpose_b;
    const _Float16 *bias;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile matmul_rows_fp16_spatz_params_t *) params_addr;

    shard_A   = (const _Float16 *) params->shard_A;
    shard_B   = (const _Float16 *) params->shard_B;
    shard_Y   = (_Float16 *) params->shard_Y;
    batch_len = params->batch_len;
    a_batched = params->a_batched;
    b_batched = params->b_batched;
    bias = (const _Float16 *)params->shard_bias;
    bias_mode = params->bias_mode;
    bias_batched = params->bias_batched;
    transpose_b = params->transpose_b;
    M = params->M;
    K = params->K;
    O = params->O;
#if defined(MAPS_MATMUL_PROFILE) && MAPS_MATMUL_PROFILE
    /* Diagnostic builds reserve two counters after the shared parameter block. */
    volatile uint32_t *profile = (volatile uint32_t *)
        ((params_addr + sizeof(*params) + 15u) & ~(uintptr_t)15u);
#else
    volatile uint32_t *profile = 0;
#endif
    uint32_t compute_cycles=0u, bias_cycles=0u;

    if (M == 0 || batch_len == 0)
        return 0;

    size_t size_A_2d = M * K;
    size_t size_B_2d = K * O;
    size_t size_Y_2d = M * O;

    for (uint32_t b = 0; b < batch_len; b++) {
        const _Float16 *current_A = shard_A + ((a_batched ? b : 0) * size_A_2d);
        const _Float16 *current_B = shard_B + ((b_batched ? b : 0) * size_B_2d);
        _Float16 *current_Y = shard_Y + (b * size_Y_2d);

        uint32_t stamp=profile ? matmul_cycle() : 0u;
        if (transpose_b) {
#if MATMUL_TRANSPOSED_B_VECTOR
            if ((K&1u)==0u && ((uintptr_t)current_B&3u)==0u)
                matmul_transposed_b_vector(current_A, current_B, current_Y, M, K, O);
            else
#endif
                matmul_transposed_b_scalar(current_A, current_B, current_Y, M, K, O);
        }
        else if (matmul_vector_safe(current_B, current_Y, O))
            matmul(current_A, current_B, current_Y, M, K, O);
        else
            matmul_scalar(current_A, current_B, current_Y, M, K, O);
        if (profile) compute_cycles+=matmul_cycle()-stamp;
        const _Float16 *current_bias = bias;
        if (bias_batched) {
            const size_t bias_batch_elements = bias_mode == MATMUL_BIAS_MATRIX
                ? size_Y_2d : (bias_mode == MATMUL_BIAS_ROW
                    ? M : (bias_mode == MATMUL_BIAS_COLUMN ? O : 1u));
            current_bias += b * bias_batch_elements;
        }
        if (profile) stamp=matmul_cycle();
        add_bias(current_bias, current_Y, M, O, bias_mode);
        if (profile) bias_cycles+=matmul_cycle()-stamp;
    }

    if (profile) {
        profile[0]=compute_cycles;
        profile[1]=bias_cycles;
        asm volatile("fence rw,rw" ::: "memory");
    }
    return 0;
}
