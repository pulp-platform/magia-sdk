#include "tile.h"
#include "group_reduce_fp16_spatz_params.h"

int group_reduce_fp16_spatz_task(void)
{
    volatile group_reduce_fp16_spatz_params_t *params =
        (volatile group_reduce_fp16_spatz_params_t *)mmio32(SPATZ_DATA);
    if (params->elements_per_group == 0u)
        return -1;
    const _Float16 *input = (const _Float16 *)params->input;
    _Float16 *output = (_Float16 *)params->output;
    const _Float16 scale =
        (_Float16)(1.0f / (float)params->elements_per_group);
    for (uint32_t group = 0; group < params->num_groups; ++group)
        output[group] = 0.0f;
    for (uint32_t index = 0; index < params->local_elements; ++index) {
        const uint32_t local_channel =
            index / params->local_spatial_elements;
        const uint32_t group =
            (params->channel_offset + local_channel) /
            params->channels_per_group;
        output[group] = (_Float16)(
            output[group] + (_Float16)(input[index] * scale));
    }
    return 0;
}
