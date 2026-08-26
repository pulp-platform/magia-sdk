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
#include "matmul_fp16_spatz_params.h"

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

static void add_bias(const _Float16 *bias, _Float16 *Y, const size_t dim_M,
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

int matmul_fp16_spatz_task(void)
{
    volatile matmul_fp16_spatz_params_t *params;
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
    params = (volatile matmul_fp16_spatz_params_t *) params_addr;

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

    if (M == 0 || batch_len == 0)
        return 0;

    size_t size_A_2d = M * K;
    size_t size_B_2d = K * O;
    size_t size_Y_2d = M * O;

    for (uint32_t b = 0; b < batch_len; b++) {
        const _Float16 *current_A = shard_A + ((a_batched ? b : 0) * size_A_2d);
        const _Float16 *current_B = shard_B + ((b_batched ? b : 0) * size_B_2d);
        _Float16 *current_Y = shard_Y + (b * size_Y_2d);

        if (transpose_b)
            matmul_transposed_b_scalar(current_A, current_B, current_Y, M, K, O);
        else if (matmul_vector_safe(current_B, current_Y, O))
            matmul(current_A, current_B, current_Y, M, K, O);
        else
            matmul_scalar(current_A, current_B, current_Y, M, K, O);
        const _Float16 *current_bias = bias;
        if (bias_batched) {
            const size_t bias_batch_elements = bias_mode == MATMUL_BIAS_MATRIX
                ? size_Y_2d : (bias_mode == MATMUL_BIAS_ROW
                    ? M : (bias_mode == MATMUL_BIAS_COLUMN ? O : 1u));
            current_bias += b * bias_batch_elements;
        }
        add_bias(current_bias, current_Y, M, O, bias_mode);
    }

    return 0;
}
