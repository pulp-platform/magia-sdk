#include "tile.h"
#include "group_normalize_fp16_spatz_params.h"

static inline float sqrtf_sp(float value)
{
    float result;
    asm volatile("fsqrt.s %0, %1" : "=f"(result) : "f"(value));
    return result;
}

static inline void normalize_channel(const _Float16 *input, _Float16 *output,
                                     const _Float16 mean,
                                     const _Float16 inverse_stddev,
                                     const uint32_t elements)
{
    uint32_t remaining = elements;
    size_t vl;
    while (remaining != 0u) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma"
                     : "=r"(vl) : "r"(remaining));
        asm volatile("vle16.v v0, (%0)" :: "r"(input));
        asm volatile("vfsub.vf v0, v0, %0" :: "f"(mean));
        asm volatile("vfmul.vf v0, v0, %0" :: "f"(inverse_stddev));
        asm volatile("vse16.v v0, (%0)" :: "r"(output));
        input += vl;
        output += vl;
        remaining -= vl;
    }
}

static inline void affine_channel(_Float16 *output, const _Float16 scale,
                                  const _Float16 bias,
                                  const uint32_t elements)
{
    uint32_t remaining = elements;
    size_t vl;
    while (remaining != 0u) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma"
                     : "=r"(vl) : "r"(remaining));
        asm volatile("vle16.v v0, (%0)" :: "r"(output));
        asm volatile("vfmul.vf v0, v0, %0" :: "f"(scale));
        asm volatile("vfadd.vf v0, v0, %0" :: "f"(bias));
        asm volatile("vse16.v v0, (%0)" :: "r"(output));
        output += vl;
        remaining -= vl;
    }
}

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
    _Float16 group_mean = 0.0f;
    _Float16 inverse_stddev = 0.0f;
    for (uint32_t local_channel = 0; local_channel < local_channels;
         ++local_channel) {
        const uint32_t affine_index =
            params->channel_offset + local_channel - params->scale_channel_offset;
        const uint32_t group =
            (params->channel_offset + local_channel) /
            params->channels_per_group;
        /* Match the full-mesh GroupNorm arithmetic: normalize in FP16, store,
         * then apply the FP16 affine transform in a separate vector pass. */
        if (group != active_group) {
            active_group = group;
            group_mean = mean[group];
            inverse_stddev = (_Float16)(1.0f / sqrtf_sp(
                (float)variance[group] + params->epsilon));
        }
        const uint32_t channel_start =
            local_channel * params->local_spatial_elements;
        normalize_channel(input + channel_start, output + channel_start,
                          group_mean, inverse_stddev,
                          params->local_spatial_elements);
        affine_channel(output + channel_start, scale[affine_index],
                       bias[affine_index], params->local_spatial_elements);
    }
    return 0;
}
