#include "tile.h"
#include "reducemax_fp16_spatz_params.h"

int reducemax_fp16_spatz_task(void)
{
    volatile reducemax_fp16_spatz_params_t *params =
        (volatile reducemax_fp16_spatz_params_t *)mmio32(SPATZ_DATA);
    const _Float16 *src = (const _Float16 *)params->shard_X;
    _Float16 *dst = (_Float16 *)params->shard_Y;

    for (uint32_t element = 0u; element < params->inner_dim; ++element) {
        _Float16 result = src[element];
        for (uint32_t row = 1u; row < params->reduce_dim; ++row) {
            const _Float16 value = src[row * params->inner_dim + element];
            result = value > result ? value : result;
        }
        dst[element] = result;
    }
    return 0;
}
