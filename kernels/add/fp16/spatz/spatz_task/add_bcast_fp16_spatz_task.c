/*
 * Elementwise add with one broadcast operand, the counterpart of mul's broadcast task.
 *
 * Both shapes that occur in the networks here are "A is [rows, row_len], B is smaller":
 *
 *   ADD_BCAST_ROW     Y[r][i] = A[r][i] + B[i]   - a bias over the last axis, and the
 *                                                  attention bias shared by every window
 *   ADD_BCAST_SCALAR  Y[r][i] = A[r][i] + B[r]   - one scalar per row
 *
 * Per element this is the same single FP16 add the plain kernel does, so a golden is just
 * numpy's broadcasting.
 */
#include "tile.h"
#include "add_bcast_fp16_spatz_params.h"

/* Rows are row_len apart in L1, so an odd row_len alternates the alignment of the row
 * bases and the Spatz VLSU corrupts every other row. Rare enough to just walk scalar. */
static inline void add_row_scalar(const _Float16 *a, const _Float16 *b, _Float16 *y, size_t len, size_t b_step)
{
    for (size_t i = 0; i < len; i++)
        y[i] = (_Float16)(a[i] + b[i * b_step]);
}

int add_bcast_fp16_spatz_task(void)
{
    volatile add_bcast_fp16_spatz_params_t *params;
    uintptr_t params_addr;

    const _Float16 *A;
    const _Float16 *B;
    _Float16 *Y;

    uint32_t row_len;
    uint32_t rows;
    uint32_t mode;
    int vector_safe;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile add_bcast_fp16_spatz_params_t *) params_addr;

    A       = (const _Float16 *) params->shard_A;
    B       = (const _Float16 *) params->shard_B;
    Y       = (_Float16 *) params->shard_Y;
    rows    = params->rows;
    row_len = params->row_len;
    mode    = params->mode;

    vector_safe = ((row_len & 1u) == 0);

    for (uint32_t r = 0; r < rows; r++) {
        const _Float16 *p_a = A + (size_t)r * row_len;
        _Float16 *p_y       = Y + (size_t)r * row_len;
        size_t avl          = row_len;
        size_t vl;

        if (mode == ADD_BCAST_SCALAR) {
            _Float16 s = B[r];

            if (!vector_safe) {
                add_row_scalar(p_a, &s, p_y, row_len, 0);
                continue;
            }

            for (; avl > 0; avl -= vl) {
                asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

                asm volatile ("vle16.v v0, (%0)" :: "r"(p_a));
                asm volatile ("vfadd.vf v0, v0, %0" :: "f"(s));
                asm volatile ("vse16.v v0, (%0)" :: "r"(p_y) : "memory");

                p_a += vl;
                p_y += vl;
            }
        } else {
            const _Float16 *p_b = B;

            if (!vector_safe) {
                add_row_scalar(p_a, p_b, p_y, row_len, 1);
                continue;
            }

            for (; avl > 0; avl -= vl) {
                asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

                asm volatile ("vle16.v v0, (%0)" :: "r"(p_a));
                asm volatile ("vle16.v v8, (%0)" :: "r"(p_b));
                asm volatile ("vfadd.vv v0, v0, v8");
                asm volatile ("vse16.v v0, (%0)" :: "r"(p_y) : "memory");

                p_a += vl;
                p_b += vl;
                p_y += vl;
            }
        }
    }

    return 0;
}
