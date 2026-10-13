#include "tile.h"
#include "group_centered_reduce_fp16_spatz_params.h"

static inline _Float16 reduce_centered_group(const _Float16 *input,
                                             uint32_t elements,
                                             _Float16 mean, _Float16 scale)
{
    const _Float16 zero = 0.0f;
    size_t vl;

    asm volatile("vsetvli %0, %1, e16, m8, tu, mu"
                 : "=r"(vl) : "r"(128u));
    asm volatile("vfmv.v.f v8, %0" :: "f"(zero));
    asm volatile("vfmv.v.f v16, %0" :: "f"(zero));

    while (elements != 0u) {
        const uint32_t first = elements < 128u ? elements : 128u;
        asm volatile("vsetvli %0, %1, e16, m8, tu, mu"
                     : "=r"(vl) : "r"(first));
        asm volatile("vle16.v v0, (%0)" :: "r"(input));
        asm volatile("vfsub.vf v0, v0, %0" :: "f"(mean));
        asm volatile("vfmul.vf v24, v0, %0" :: "f"(scale));
        asm volatile("vfmacc.vv v8, v24, v0");
        input += first;
        elements -= first;

        const uint32_t second = elements < 128u ? elements : 128u;
        if (second != 0u) {
            asm volatile("vsetvli %0, %1, e16, m8, tu, mu"
                         : "=r"(vl) : "r"(second));
            asm volatile("vle16.v v0, (%0)" :: "r"(input));
            asm volatile("vfsub.vf v0, v0, %0" :: "f"(mean));
            asm volatile("vfmul.vf v24, v0, %0" :: "f"(scale));
            asm volatile("vfmacc.vv v16, v24, v0");
            input += second;
            elements -= second;
        }
    }

    _Float16 lanes[256];
    asm volatile("vsetvli %0, %1, e16, m8, tu, mu"
                 : "=r"(vl) : "r"(128u));
    asm volatile("vse16.v v8, (%0)" :: "r"(lanes) : "memory");
    asm volatile("vse16.v v16, (%0)" :: "r"(lanes + 128) : "memory");
    _Float16 sum = 0.0f;
    for (uint32_t lane = 0; lane < 256u; ++lane)
        sum = (_Float16)(sum + lanes[lane]);
    return sum;
}

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
    const uint32_t local_channels =
        params->local_elements / params->local_spatial_elements;
    const uint32_t channel_end = params->channel_offset + local_channels;
    for (uint32_t group = 0; group < params->num_groups; ++group) {
        const uint32_t group_channel_start = group * params->channels_per_group;
        const uint32_t group_channel_end =
            group_channel_start + params->channels_per_group;
        const uint32_t overlap_start = params->channel_offset > group_channel_start
            ? params->channel_offset : group_channel_start;
        const uint32_t overlap_end = channel_end < group_channel_end
            ? channel_end : group_channel_end;
        const uint32_t group_channels = overlap_end > overlap_start
            ? overlap_end - overlap_start : 0u;
        const uint32_t local_start = group_channels != 0u
            ? (overlap_start - params->channel_offset) *
                params->local_spatial_elements
            : 0u;
        output[group] = reduce_centered_group(
            input + local_start,
            group_channels * params->local_spatial_elements,
            mean[group], scale);
    }
    return 0;
}
