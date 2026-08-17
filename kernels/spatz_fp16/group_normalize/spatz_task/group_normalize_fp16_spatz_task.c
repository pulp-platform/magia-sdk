#include "tile.h"
#include "group_normalize_fp16_spatz_params.h"

int group_normalize_fp16_spatz_task(void)
{
    volatile group_normalize_fp16_spatz_params_t *params =
        (volatile group_normalize_fp16_spatz_params_t *)mmio32(SPATZ_DATA);
    if (params->local_spatial_elements == 0u)
        return -1;
    const _Float16 *input = (const _Float16 *)params->input;
    const _Float16 *scale = (const _Float16 *)params->scale;
    const _Float16 *bias = (const _Float16 *)params->bias;
    _Float16 *output = (_Float16 *)params->output;
    const _Float16 *mean = (const _Float16 *)params->mean;
    const _Float16 *variance = (const _Float16 *)params->variance;
    for (uint32_t index = 0; index < params->local_elements; ++index) {
        const uint32_t local_channel =
            index / params->local_spatial_elements;
        const uint32_t affine_channel =
            params->channel_offset + local_channel - params->scale_channel_offset;
        const uint32_t group =
            (params->channel_offset + local_channel) /
            params->channels_per_group;
        float stddev;
        asm volatile("fsqrt.s %0, %1"
                     : "=f"(stddev)
                     : "f"((float)variance[group] + params->epsilon));
        const float inverse_stddev = 1.0f / stddev;
        output[index] = (_Float16)(
            ((float)input[index] - (float)mean[group]) * inverse_stddev *
                (float)scale[affine_channel] +
            (float)bias[affine_channel]);
    }
    return 0;
}
