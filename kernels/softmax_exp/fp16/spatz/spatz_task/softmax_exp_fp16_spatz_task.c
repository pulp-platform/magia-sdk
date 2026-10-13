#include "tile.h"
#include "softmax_exp_fp16_spatz_params.h"

static _Float16 softmax_fastexp(_Float16 value)
{
    _Float16 scaled = (_Float16)(value * (_Float16)1486.0f);
    scaled = (_Float16)(scaled + (_Float16)15360.0f);
    float converted = (float)scaled;
    uint16_t bits = converted <= 0.0f
                        ? 0u
                        : (converted >= 65535.0f ? 65535u
                                                : (uint16_t)converted);
    _Float16 result;
    __builtin_memcpy(&result, &bits, sizeof(result));
    return result;
}

int softmax_exp_fp16_spatz_task(void)
{
    volatile softmax_exp_fp16_spatz_params_t *params =
        (volatile softmax_exp_fp16_spatz_params_t *)mmio32(SPATZ_DATA);
    const _Float16 *input = (const _Float16 *)params->input;
    _Float16 *output = (_Float16 *)params->output;
    for (uint32_t index = 0; index < params->len; ++index)
        output[index] = softmax_fastexp(input[index]);
    return 0;
}
