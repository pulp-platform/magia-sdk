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
    const uint32_t local_channels =
        params->local_elements / params->local_spatial_elements;
    uint32_t active_group = params->num_groups;
    float group_mean = 0.0f;
    float inverse_stddev = 0.0f;
    for (uint32_t local_channel = 0; local_channel < local_channels;
         ++local_channel) {
        const uint32_t affine_channel =
            params->channel_offset + local_channel - params->scale_channel_offset;
        const uint32_t group =
            (params->channel_offset + local_channel) /
            params->channels_per_group;
        /* Mean and inverse standard deviation are shared by every channel in
         * the group. Keep the element arithmetic unchanged, but do the costly
         * scalar square root and division only when the group changes. */
        if (group != active_group) {
            float stddev;
            asm volatile("fsqrt.s %0, %1"
                         : "=f"(stddev)
                         : "f"((float)variance[group] + params->epsilon));
            active_group = group;
            group_mean = (float)mean[group];
            inverse_stddev = 1.0f / stddev;
        }
        const uint32_t channel_start =
            local_channel * params->local_spatial_elements;
        const uint32_t channel_end =
            channel_start + params->local_spatial_elements;
        for (uint32_t index = channel_start; index < channel_end; ++index)
            output[index] = (_Float16)(
                ((float)input[index] - group_mean) * inverse_stddev *
                    (float)scale[affine_channel] +
                (float)bias[affine_channel]);
    }
    return 0;
}
