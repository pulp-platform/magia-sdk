#include <stdint.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"
#include "utils/maps_idma.h"
#include "utils/printf.h"

static const uint8_t source[128]
    __attribute__((section(".l2_bulk.maps_transfer"))) = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    };

static tensor_sub_slice_t slice(
    uint32_t rank, uint32_t elements, const TensorRange *ranges)
{
    tensor_sub_slice_t result = {.rank = rank, .num_elems = elements};
    for (uint32_t index = 0u; index < rank; ++index)
        result.dims[index] = ranges[index];
    return result;
}

static uint32_t check(const uint8_t *actual, const uint8_t *expected, uint32_t count)
{
    uint32_t errors = 0u;
    for (uint32_t index = 0u; index < count; ++index)
        errors += actual[index] != expected[index];
    return errors;
}

int main(void)
{
    const uint32_t hartid = get_hartid();
    if (hartid != 0u)
        return 0;

    idma_config_t idma_config = {.hartid = hartid};
    idma_controller_t idma = {.base = 0, .cfg = &idma_config, .api = &idma_api};
    eu_config_t event_config = {.hartid = hartid};
    eu_controller_t event_unit = {
        .base = 0, .cfg = &event_config, .api = &eu_api};
    idma_init(&idma);
    eu_init(&event_unit);
    eu_clear_events(0xffffffffu);
    eu_idma_init(&event_unit, 0u);

    uint8_t *l1 = (uint8_t *)get_l1_base(hartid);
    uint32_t errors = 0u;

    const TensorRange contiguous_range[] = {{0u, 8u, 1u}};
    tensor_sub_slice_t contiguous = slice(1u, 8u, contiguous_range);
    idma_memcpy_md_to_nd(
        &idma, 0u, (uint32_t)l1, (uint32_t)source,
        &contiguous, &contiguous, 1u, &event_unit);
    errors += check(l1, source, 8u);

    const TensorRange source_2d_ranges[] = {
        {0u, 2u, 8u}, {0u, 3u, 1u}};
    const TensorRange destination_2d_ranges[] = {
        {0u, 2u, 10u}, {0u, 3u, 1u}};
    tensor_sub_slice_t source_2d = slice(2u, 6u, source_2d_ranges);
    tensor_sub_slice_t destination_2d = slice(2u, 6u, destination_2d_ranges);
    idma_memcpy_md_to_nd(
        &idma, 0u, (uint32_t)(l1 + 32u), (uint32_t)source,
        &source_2d, &destination_2d, 1u, &event_unit);
    const uint8_t expected_2d[] = {0u, 1u, 2u, 8u, 9u, 10u};
    uint8_t actual_2d[] = {
        l1[32], l1[33], l1[34], l1[42], l1[43], l1[44]};
    errors += check(actual_2d, expected_2d, 6u);

    const TensorRange sparse_ranges[] = {
        {0u, 2u, 8u}, {0u, 3u, 2u}};
    tensor_sub_slice_t sparse = slice(2u, 6u, sparse_ranges);
    idma_memcpy_md_to_nd(
        &idma, 0u, (uint32_t)(l1 + 64u), (uint32_t)source,
        &sparse, &sparse, 1u, &event_unit);
    const uint8_t expected_sparse[] = {0u, 2u, 4u, 8u, 10u, 12u};
    uint8_t actual_sparse[] = {
        l1[64], l1[66], l1[68], l1[72], l1[74], l1[76]};
    errors += check(actual_sparse, expected_sparse, 6u);

    const TensorRange source_3d_ranges[] = {
        {0u, 2u, 16u}, {0u, 2u, 4u}, {0u, 2u, 1u}};
    const TensorRange destination_3d_ranges[] = {
        {0u, 2u, 4u}, {0u, 2u, 2u}, {0u, 2u, 1u}};
    tensor_sub_slice_t source_3d = slice(3u, 8u, source_3d_ranges);
    tensor_sub_slice_t destination_3d = slice(3u, 8u, destination_3d_ranges);
    idma_memcpy_md_to_nd(
        &idma, 0u, (uint32_t)(l1 + 96u), (uint32_t)source,
        &source_3d, &destination_3d, 1u, &event_unit);
    const uint8_t expected_3d[] = {0u, 1u, 4u, 5u, 16u, 17u, 20u, 21u};
    errors += check(l1 + 96u, expected_3d, 8u);

    printf("MAPS transfer errors: %u\n", errors);
    return (int)errors;
}
