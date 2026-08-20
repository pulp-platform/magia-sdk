#include "tile.h"
#include "group_centered_reduce_fp16_spatz_params.h"

int group_centered_reduce_fp16_spatz_task(void)
{
    volatile group_centered_reduce_fp16_spatz_params_t *params =
        (volatile group_centered_reduce_fp16_spatz_params_t *)mmio32(SPATZ_DATA);
    if (params->elements_per_group == 0u)
        return -1;
    const _Float16 *input = (const _Float16 *)params->input;
    const _Float16 *mean = (const _Float16 *)params->mean;
    _Float16 *output = (_Float16 *)params->output;
    const _Float16 scale =
        (_Float16)(1.0f / (float)params->elements_per_group);
    for (uint32_t group = 0; group < params->num_groups; ++group) {
        _Float16 lanes[256];
        for (uint32_t lane = 0; lane < 256u; ++lane)
            lanes[lane] = 0.0f;
        uint32_t group_index = 0u;
        for (uint32_t index = 0; index < params->local_elements; ++index) {
            const uint32_t local_channel =
                index / params->local_spatial_elements;
            const uint32_t element_group =
                (params->channel_offset + local_channel) /
                params->channels_per_group;
            if (element_group != group)
                continue;
            const _Float16 centered = (_Float16)(input[index] - mean[group]);
            const uint32_t lane = group_index++ & 255u;
            lanes[lane] = (_Float16)(lanes[lane] +
                (_Float16)((_Float16)(centered * scale) * centered));
        }
        _Float16 sum = 0.0f;
        for (uint32_t lane = 0; lane < 256u; ++lane)
            sum = (_Float16)(sum + lanes[lane]);
        output[group] = sum;
    }
    return 0;
}
