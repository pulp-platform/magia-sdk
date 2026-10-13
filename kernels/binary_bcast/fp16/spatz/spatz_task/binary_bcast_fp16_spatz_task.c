#include "tile.h"
#include "binary_bcast_fp16_spatz_params.h"

static inline void binary_scalar(const _Float16 *a, const _Float16 *b,
                                 _Float16 *y, size_t length, size_t step,
                                 uint32_t operation)
{
    for (size_t index = 0; index < length; ++index)
        y[index] = operation == BINARY_BCAST_SUB
            ? a[index] - b[index * step] : a[index] / b[index * step];
}

int binary_bcast_fp16_spatz_task(void)
{
    volatile binary_bcast_fp16_spatz_params_t *params =
        (volatile binary_bcast_fp16_spatz_params_t *)mmio32(SPATZ_DATA);
    const _Float16 *a = (const _Float16 *)params->shard_A;
    const _Float16 *b = (const _Float16 *)params->shard_B;
    _Float16 *y = (_Float16 *)params->shard_Y;

    for (uint32_t row = 0; row < params->rows; ++row) {
        const _Float16 *row_a = a + (size_t)row * params->row_len;
        const _Float16 *row_b = params->mode == BINARY_BCAST_SCALAR ? b + row : b;
        _Float16 *row_y = y + (size_t)row * params->row_len;
        if ((params->row_len & 1u) != 0u) {
            binary_scalar(row_a, row_b, row_y, params->row_len,
                          params->mode == BINARY_BCAST_SCALAR ? 0u : 1u,
                          params->operation);
            continue;
        }
        for (size_t avl = params->row_len; avl > 0u;) {
            size_t vl;
            asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
            asm volatile ("vle16.v v0, (%0)" :: "r"(row_a));
            if (params->mode == BINARY_BCAST_SCALAR) {
                const _Float16 scalar = *row_b;
                if (params->operation == BINARY_BCAST_SUB)
                    asm volatile ("vfsub.vf v0, v0, %0" :: "f"(scalar));
                else
                    asm volatile ("vfdiv.vf v0, v0, %0" :: "f"(scalar));
            } else {
                asm volatile ("vle16.v v8, (%0)" :: "r"(row_b));
                if (params->operation == BINARY_BCAST_SUB)
                    asm volatile ("vfsub.vv v0, v0, v8");
                else
                    asm volatile ("vfdiv.vv v0, v0, v8");
                row_b += vl;
            }
            asm volatile ("vse16.v v0, (%0)" :: "r"(row_y) : "memory");
            row_a += vl;
            row_y += vl;
            avl -= vl;
        }
    }
    return 0;
}
