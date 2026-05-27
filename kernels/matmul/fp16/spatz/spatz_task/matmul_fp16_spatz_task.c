#include "tile.h"
#include "matmul_fp16_spatz_params.h"

#if 0

static void matmul_fp32_acc(const _Float16 *A, const _Float16 *B, _Float16 *Y, const size_t dim_M, const size_t dim_K, const size_t dim_O)
{
    for (size_t m = 0; m + 1 < dim_M; m += 2) {
        for (size_t o = 0; o < dim_O; o++) {
            float acc0 = 0.0f;
            float acc1 = 0.0f;

            for (size_t k = 0; k < dim_K; k++) {
                const float a0 = A[m * dim_K + k];
                const float a1 = A[(m + 1) * dim_K + k];
                const float b = B[k * dim_O + o];

                acc0 += a0 * b;
                acc1 += a1 * b;
            }

            Y[m * dim_O + o] = acc0;
            Y[(m + 1) * dim_O + o] = acc1;
        }
    }

    if (dim_M % 2) {
        const size_t m = dim_M - 1;
        for (size_t o = 0; o < dim_O; o++) {
            float acc = 0.0f;
            for (size_t k = 0; k < dim_K; k++) {
                acc += A[m * dim_K + k] * B[k * dim_O + o];
            }
            Y[m * dim_O + o] = acc;
        }
    }
}

static void matmul_fp16_acc(const _Float16 *A, const _Float16 *B, _Float16 *Y, const size_t dim_M, const size_t dim_K, const size_t dim_O)
{
    for (size_t m = 0; m + 1 < dim_M; m += 2) {
        for (size_t o = 0; o < dim_O; o++) {
            _Float16 acc0 = 0.0f;
            _Float16 acc1 = 0.0f;

            for (size_t k = 0; k < dim_K; k++) {
                const _Float16 a0 = A[m * dim_K + k];
                const _Float16 a1 = A[(m + 1) * dim_K + k];
                const _Float16 b = B[k * dim_O + o];

                acc0 += a0 * b;
                acc1 += a1 * b;
            }

            Y[m * dim_O + o] = acc0;
            Y[(m + 1) * dim_O + o] = acc1;
        }
    }

    if (dim_M % 2) {
        const size_t m = dim_M - 1;
        for (size_t o = 0; o < dim_O; o++) {
            _Float16 acc = 0.0f;
            for (size_t k = 0; k < dim_K; k++) {
                acc += A[m * dim_K + k] * B[k * dim_O + o];
            }
            Y[m * dim_O + o] = acc;
        }
    }
}

#endif

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

int matmul_fp16_spatz_task(void)
{
    volatile matmul_fp16_spatz_params_t *params;
    uintptr_t params_addr;
    const _Float16 *shard_A;
    const _Float16 *shard_B;
    _Float16 *shard_Y;
    size_t M, K, O;
    uint32_t batch_len;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile matmul_fp16_spatz_params_t *) params_addr;

    shard_A = (const _Float16 *) params->shard_A;
    shard_B = (const _Float16 *) params->shard_B;
    shard_Y = (_Float16 *) params->shard_Y;
    batch_len = params->batch_len;
    M = params->M;
    K = params->K;
    O = params->O;

    size_t size_A_2d = M * K;
    size_t size_B_2d = K * O;
    size_t size_Y_2d = M * O;

    for (uint32_t b = 0; b < batch_len; b++) {
        const _Float16 *current_A = shard_A + (b * size_A_2d);
        const _Float16 *current_B = shard_B + (b * size_B_2d);
        _Float16 *current_Y = shard_Y + (b * size_Y_2d);

        matmul(current_A, current_B, current_Y, M, K, O);
    }

    return 0;
}
