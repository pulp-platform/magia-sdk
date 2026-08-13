#ifndef MAPS_IDMA_H
#define MAPS_IDMA_H

#include <stdint.h>

#include "eventunit.h"
#include "idma.h"

#define MAPS_IDMA_MAX_RANK 6u
#define IDMA_ND_MAX_RANK MAPS_IDMA_MAX_RANK

typedef struct {
    uint32_t start;
    uint32_t length;
    uint32_t stride;
} TensorRange;

typedef struct {
    uint32_t rank;
    uint32_t num_elems;
    TensorRange dims[MAPS_IDMA_MAX_RANK];
} tensor_sub_slice_t;

typedef struct {
    uint32_t rank;
    uint32_t length[MAPS_IDMA_MAX_RANK];
    uint32_t stride[MAPS_IDMA_MAX_RANK];
    uint32_t base_offset;
} maps_idma_normalized_t;

static inline void maps_idma_normalize(
    const tensor_sub_slice_t *slice, maps_idma_normalized_t *normalized)
{
    normalized->rank = 0u;
    normalized->base_offset = 0u;
    for (uint32_t dimension = 0u; dimension < slice->rank; ++dimension) {
        const TensorRange *range = &slice->dims[dimension];
        normalized->base_offset += range->start * range->stride;
        if (range->length == 1u)
            continue;
        if (normalized->rank != 0u &&
            normalized->stride[normalized->rank - 1u] ==
                range->length * range->stride) {
            normalized->length[normalized->rank - 1u] *= range->length;
            normalized->stride[normalized->rank - 1u] = range->stride;
            continue;
        }
        normalized->length[normalized->rank] = range->length;
        normalized->stride[normalized->rank] = range->stride;
        ++normalized->rank;
    }
}

static inline uint32_t maps_idma_address(
    uint32_t base, const maps_idma_normalized_t *normalized, uint32_t index)
{
    uint32_t address = base;
    for (uint32_t dimension = normalized->rank; dimension-- > 0u;) {
        address += (index % normalized->length[dimension]) *
                   normalized->stride[dimension];
        index /= normalized->length[dimension];
    }
    return address;
}

static inline uint32_t maps_idma_gcd(uint32_t lhs, uint32_t rhs)
{
    while (rhs != 0u) {
        const uint32_t remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    return lhs;
}

static inline void maps_idma_wait(
    eu_controller_t *event_unit, uint8_t direction)
{
    if (event_unit == 0)
        return;
    if (direction == 0u)
        eu_idma_wait_a2o(event_unit, WFE);
    else
        eu_idma_wait_o2a(event_unit, WFE);
}

static inline int idma_memcpy_md_to_nd(
    idma_controller_t *controller,
    uint8_t direction,
    uint32_t destination_address,
    uint32_t source_address,
    const tensor_sub_slice_t *source,
    const tensor_sub_slice_t *destination,
    uint32_t element_bytes,
    eu_controller_t *event_unit)
{
    if (source->rank > MAPS_IDMA_MAX_RANK ||
        destination->rank > MAPS_IDMA_MAX_RANK ||
        source->num_elems != destination->num_elems || element_bytes == 0u)
        return -1;
    if (source->num_elems == 0u)
        return 0;

    maps_idma_normalized_t normalized_source;
    maps_idma_normalized_t normalized_destination;
    maps_idma_normalize(source, &normalized_source);
    maps_idma_normalize(destination, &normalized_destination);
    source_address += normalized_source.base_offset;
    destination_address += normalized_destination.base_offset;

    uint32_t source_stride = 0u;
    uint32_t destination_stride = 0u;
    uint32_t row_bytes = 0u;
    uint32_t repetitions = 0u;
    if (normalized_source.rank == 2u &&
        normalized_destination.rank == 2u &&
        normalized_source.length[0] == normalized_destination.length[0] &&
        normalized_source.length[1] == normalized_destination.length[1] &&
        normalized_source.stride[1] == element_bytes &&
        normalized_destination.stride[1] == element_bytes) {
        source_stride = normalized_source.stride[0];
        destination_stride = normalized_destination.stride[0];
        row_bytes = normalized_source.length[1] * element_bytes;
        repetitions = normalized_source.length[0];
    }
    if (repetitions != 0u) {
        const uint32_t axi_address = direction == 0u
            ? source_address : destination_address;
        const uint32_t obi_address = direction == 0u
            ? destination_address : source_address;
        const uint32_t axi_stride = direction == 0u
            ? source_stride : destination_stride;
        const uint32_t obi_stride = direction == 0u
            ? destination_stride : source_stride;
        const int result = idma_memcpy_2d_ex(
            controller, direction, axi_address, obi_address, row_bytes,
            axi_stride, obi_stride, repetitions);
        maps_idma_wait(event_unit, direction);
        return result;
    }

    const uint32_t source_run =
        normalized_source.rank != 0u &&
        normalized_source.stride[normalized_source.rank - 1u] == element_bytes
            ? normalized_source.length[normalized_source.rank - 1u] : 1u;
    const uint32_t destination_run =
        normalized_destination.rank != 0u &&
        normalized_destination.stride[normalized_destination.rank - 1u] == element_bytes
            ? normalized_destination.length[normalized_destination.rank - 1u] : 1u;
    const uint32_t block_elements = maps_idma_gcd(source_run, destination_run);
    const uint32_t block_bytes = block_elements * element_bytes;
    for (uint32_t index = 0u; index < source->num_elems;
         index += block_elements) {
        const uint32_t source_block = maps_idma_address(
            source_address, &normalized_source, index);
        const uint32_t destination_block = maps_idma_address(
            destination_address, &normalized_destination, index);
        const uint32_t axi_address = direction == 0u
            ? source_block : destination_block;
        const uint32_t obi_address = direction == 0u
            ? destination_block : source_block;
        const int result = idma_memcpy_1d(
            controller, direction, axi_address, obi_address, block_bytes);
        if (result != 0)
            return result;
        maps_idma_wait(event_unit, direction);
    }
    return 0;
}

#endif
