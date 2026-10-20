#ifndef MAPS_DMA_TASKS_H
#define MAPS_DMA_TASKS_H

#ifndef MAPS_TRANSPOSE_PACKED_PAIRS
#define MAPS_TRANSPOSE_PACKED_PAIRS 1u
#endif
#if MAPS_HAS_TRANSPOSE_SPATZ_TASK
#include "transpose_strided_fp16_spatz_params.h"
#endif

/* Interleave two contiguous input streams into adjacent FP16 output pairs.
 * No arithmetic or conversion is performed. Explicit zero extension also
 * avoids relying on upper register bits in the control-core halfword loads. */
static inline uint32_t maps_transpose_half(uintptr_t address)
{
    uint32_t value = *(const uint16_t *)address;
    asm volatile("slli %0, %0, 16\n\tsrli %0, %0, 16" : "+r"(value));
    return value;
}

static inline int maps_transpose_packed_pairs(
    uint32_t input, uint32_t output, const subslice_desc_t *out,
    const uint32_t *source_stride)
{
    const uint32_t outer_rank = out->rank - 2u;
    uint32_t rows = 1u;
    for (uint32_t axis = 0; axis < outer_rank; ++axis) {
        if (rows > UINT32_MAX / out->shape[axis]) return -4;
        rows *= out->shape[axis];
    }
    for (uint32_t row = 0; row < rows; ++row) {
        uint32_t remaining = row, src = input, dst = output;
        for (uint32_t axis = outer_rank; axis-- > 0u;) {
            const uint32_t coordinate = remaining % out->shape[axis];
            remaining /= out->shape[axis];
            src += coordinate * source_stride[axis];
            dst += coordinate * out->strides_bytes[axis];
        }
        const uint32_t other = src + source_stride[out->rank - 1u];
        for (uint32_t column = 0; column < out->shape[outer_rank]; ++column) {
            const uint32_t lo = maps_transpose_half(src + column * 2u);
            const uint32_t hi = maps_transpose_half(other + column * 2u);
            *(uint32_t *)(dst + column * 4u) = lo | (hi << 16);
        }
    }
    asm volatile("fence rw,rw" ::: "memory");
    return 0;
}

static inline uint32_t maps_dma_contiguous(const subslice_desc_t *sub)
{
    uint64_t stride = sub->elem_bytes;
    if (!stride || sub->rank > MAPS_MAX_RANK) return 0u;
    for (uint32_t axis = sub->rank; axis-- > 0u;) {
        if (!sub->shape[axis] || (sub->shape[axis] > 1u && sub->strides_bytes[axis] != stride)) return 0u;
        stride *= sub->shape[axis];
        if (stride > UINT32_MAX) return 0u;
    }
    return 1u;
}

static inline int maps_execute_reshape_dma(const tile_plan_t *plan,
    const op_desc_t *op, uint32_t slot, maps_operation_runtime_t *runtime)
{
    if (!runtime ||
        op->num_inputs != 1u || op->num_outputs != 1u ||
        op->inputs[0].elem_type != op->outputs[0].elem_type ||
        op->inputs[0].elem_bytes != op->outputs[0].elem_bytes ||
        !maps_dma_contiguous(&op->inputs[0]) || !maps_dma_contiguous(&op->outputs[0])) return -1;
    uint32_t input_bytes = maps_operation_elems(&op->inputs[0]) * op->inputs[0].elem_bytes;
    uint32_t output_bytes = maps_operation_elems(&op->outputs[0]) * op->outputs[0].elem_bytes;
    if (input_bytes != output_bytes) return -2;
    uint32_t input = local_subslice_addr(plan, &op->inputs[0], slot);
    uint32_t output = local_subslice_addr(plan, &op->outputs[0], slot);
    if (input == output) return 0;
    if ((uint64_t)input < (uint64_t)output + output_bytes && (uint64_t)output < (uint64_t)input + input_bytes) return -3;
    return maps_idma_copy_1d_ticketed(1u, output, input, output_bytes);
}

/* Translate the permutation into paired source/destination strides. Up to
 * three axes are handled by each DMA descriptor; outer axes choose descriptors.
 * Shard coordinates are rebased before submission, matching the scalar helper. */
static inline int maps_execute_transpose_dma(const tile_plan_t *plan,
    const op_desc_t *op, uint32_t slot, maps_operation_runtime_t *runtime)
{
    if (!runtime ||
        op->num_inputs != 1u || op->num_outputs != 1u ||
        op->inputs[0].rank != op->outputs[0].rank ||
        op->inputs[0].elem_type != op->outputs[0].elem_type ||
        op->inputs[0].elem_bytes != op->outputs[0].elem_bytes ||
        !op->outputs[0].elem_bytes || op->outputs[0].rank > MAPS_MAX_RANK) return -1;
    const subslice_desc_t *in = &op->inputs[0], *out = &op->outputs[0];
    uint32_t rank = out->rank, source_stride[MAPS_MAX_RANK], seen = 0u;
    uint32_t input = local_subslice_addr(plan, in, slot), output = local_subslice_addr(plan, out, slot);
    if (!rank) return maps_idma_copy_1d_ticketed(1u, output, input, out->elem_bytes);
    uint64_t source_offset = 0u, input_span = in->elem_bytes, output_span = out->elem_bytes;
    for (uint32_t axis = 0; axis < rank; ++axis) {
        uint32_t perm = op->params[axis];
        if (perm >= rank || (seen & (1u << perm)) || !out->shape[axis] || !in->shape[perm]) return -2;
        seen |= 1u << perm;
        uint32_t in_origin = maps_operation_global_offset(plan, in, perm);
        uint32_t out_origin = maps_operation_global_offset(plan, out, axis);
        if (out_origin < in_origin || (uint64_t)(out_origin - in_origin) + out->shape[axis] > in->shape[perm]) return -3;
        source_stride[axis] = in->strides_bytes[perm];
        source_offset += (uint64_t)(out_origin - in_origin) * source_stride[axis];
        input_span += (uint64_t)(in->shape[perm] - 1u) * source_stride[axis];
        output_span += (uint64_t)(out->shape[axis] - 1u) * out->strides_bytes[axis];
    }
    if (source_offset > UINT32_MAX || input_span > UINT32_MAX || output_span > UINT32_MAX) return -4;
    if ((uint64_t)input < (uint64_t)output + output_span && (uint64_t)output < (uint64_t)input + input_span) return -5;
    input += (uint32_t)source_offset;
#if MAPS_HAS_TRANSPOSE_SPATZ_TASK
    if (out->elem_type == ELEM_F16 && out->elem_bytes == 2u && rank > 0u &&
        runtime->spatz_initialized && runtime->transpose_fp16_task &&
        runtime->spatz_params && !((uintptr_t)runtime->spatz_params & 3u) &&
        runtime->spatz_params_bytes >= sizeof(transpose_strided_fp16_spatz_params_t)) {
        uint32_t valid = !((input | output) & 1u);
        uint32_t pairs = rank >= 2u && out->shape[rank - 1u] == 2u &&
            source_stride[rank - 1u] > 2u && source_stride[rank - 2u] == 2u &&
            out->strides_bytes[rank - 1u] == 2u && out->strides_bytes[rank - 2u] == 4u &&
            !(input & 3u) && !(output & 3u) && !(source_stride[rank - 1u] & 3u);
        uint32_t rows = 1u;
        for (uint32_t axis = 0; axis < rank; ++axis) {
            if ((source_stride[axis] | out->strides_bytes[axis]) & 1u) valid = 0u;
            if (rank >= 2u && axis < rank - 2u && out->shape[axis] > 1u &&
                ((source_stride[axis] | out->strides_bytes[axis]) & 3u)) pairs = 0u;
        }
        /* Decide outer dimensions after all alignment checks. */
        const uint32_t actual_outer = rank - (pairs ? 2u : 1u);
        for (uint32_t axis = 0; axis < actual_outer; ++axis) {
            if (rows > UINT32_MAX / out->shape[axis]) return -4;
            rows *= out->shape[axis];
        }
        if (valid) {
            volatile transpose_strided_fp16_spatz_params_t *p = runtime->spatz_params;
            p->input = input; p->output = output; p->rank = rank; p->packed_pairs = pairs;
            for (uint32_t axis = 0; axis < rank; ++axis) {
                p->shape[axis] = out->shape[axis];
                p->source_stride[axis] = source_stride[axis];
                p->output_stride[axis] = out->strides_bytes[axis];
            }
            asm volatile("fence rw,rw" ::: "memory");
            spatz_run_task_with_params(runtime->transpose_fp16_task, (uint32_t)p);
            return maps_wait_for_spatz(runtime);
        }
    }
#endif
#if MAPS_TRANSPOSE_PACKED_PAIRS
    if (rank >= 2u && out->elem_type == ELEM_F16 && out->elem_bytes == 2u &&
        out->shape[rank - 1u] == 2u && source_stride[rank - 1u] > 2u &&
        source_stride[rank - 2u] == 2u && out->strides_bytes[rank - 1u] == 2u &&
        out->strides_bytes[rank - 2u] == 4u && !(input & 1u) && !(output & 3u)) {
        uint32_t aligned = !(source_stride[rank - 1u] & 1u);
        for (uint32_t axis = 0; axis < rank - 2u; ++axis)
            if ((source_stride[axis] & 1u) || (out->strides_bytes[axis] & 3u)) aligned = 0u;
        if (aligned) {
            return maps_transpose_packed_pairs(input, output, out, source_stride);
        }
    }
#endif
    uint32_t bytes = out->elem_bytes, first_outer = rank;
    uint32_t reps2 = 1u, reps3 = 1u, src2 = 0u, dst2 = 0u, src3 = 0u, dst3 = 0u;
    if (source_stride[rank - 1u] == bytes && out->strides_bytes[rank - 1u] == bytes) {
        bytes *= out->shape[rank - 1u];
        --first_outer;
    }
    if (first_outer) {
        --first_outer; reps2 = out->shape[first_outer];
        src2 = source_stride[first_outer]; dst2 = out->strides_bytes[first_outer];
    }
    if (first_outer) {
        --first_outer; reps3 = out->shape[first_outer];
        src3 = source_stride[first_outer]; dst3 = out->strides_bytes[first_outer];
        /* The DMA model uses unsigned page increments. A transposed pair
         * can require a backwards page step; keep that axis in the descriptor
         * loop rather than wrapping the address outside local memory. */
        if ((uint64_t)src3 < (uint64_t)(reps2 - 1u) * src2 ||
            (uint64_t)dst3 < (uint64_t)(reps2 - 1u) * dst2) {
            ++first_outer;
            reps3 = 1u;
        }
    }
    if (reps3 == 1u) {
        src3 = (reps2 - 1u) * src2; dst3 = (reps2 - 1u) * dst2;
    }
    uint64_t descriptor_count = 1u;
    for (uint32_t axis = 0; axis < first_outer; ++axis) descriptor_count *= out->shape[axis];
    if (descriptor_count > UINT32_MAX) return -4;
    for (uint32_t index = 0; index < (uint32_t)descriptor_count; ++index) {
        uint32_t remaining = index, src = input, dst = output;
        for (uint32_t axis = first_outer; axis-- > 0u;) {
            uint32_t coordinate = remaining % out->shape[axis];
            remaining /= out->shape[axis];
            src += coordinate * source_stride[axis]; dst += coordinate * out->strides_bytes[axis];
        }
        if (maps_idma_copy_3d_ticketed(1u, dst, src, bytes,
                dst2, src2, reps2, dst3, src3, reps3) != 0) return -6;
    }
    return 0;
}
#endif
