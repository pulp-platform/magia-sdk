#include <stdint.h>

#include "tile.h"
#include "utils/maps_operations.h"
#include "utils/printf.h"

#define ELEMENTS 512u

static uint32_t read_cycle(void)
{
    uint32_t cycle;
    __asm__ volatile("rdcycle %0" : "=r"(cycle));
    return cycle;
}

int main(void)
{
    const uint32_t hartid = get_hartid();
    if (hartid != 0u)
        return 0;

    uint16_t *input = (uint16_t *)(get_l1_base(hartid) + 4096u);
    uint16_t *output = (uint16_t *)(get_l1_base(hartid) + 5120u);
    for (uint32_t index = 0u; index < ELEMENTS; ++index) {
        input[index] = (uint16_t)index;
        output[index] = 0xffffu;
    }

    const slice_desc_t slices[] = {
        {
            .slice_id = 0u,
            .global_kind = GLOBAL_INTERMEDIATE,
            .owner_hartid = hartid,
            .rank = 1u,
            .shape = {ELEMENTS},
            .elem_type = ELEM_F16,
            .elem_bytes = sizeof(uint16_t),
            .l1_offset_bytes = 4096u,
            .slot_bytes = ELEMENTS * sizeof(uint16_t),
            .num_slots = 1u,
            .strides_bytes = {sizeof(uint16_t)},
        },
        {
            .slice_id = 1u,
            .global_kind = GLOBAL_INTERMEDIATE,
            .owner_hartid = hartid,
            .rank = 1u,
            .shape = {ELEMENTS},
            .elem_type = ELEM_F16,
            .elem_bytes = sizeof(uint16_t),
            .l1_offset_bytes = 5120u,
            .slot_bytes = ELEMENTS * sizeof(uint16_t),
            .num_slots = 1u,
            .strides_bytes = {sizeof(uint16_t)},
        },
    };
    const tile_plan_t plan = {
        .hartid = hartid,
        .l1_data_base = get_l1_base(hartid),
        .num_token_slots = 1u,
        .num_slices = 2u,
        .slices = slices,
    };
    const op_desc_t op = {
        .kind = OP_ALL_REDUCE_SUM,
        .num_inputs = 1u,
        .inputs = {{
            .slice_id = 0u,
            .rank = 1u,
            .shape = {ELEMENTS},
            .elem_type = ELEM_F16,
            .elem_bytes = sizeof(uint16_t),
            .strides_bytes = {sizeof(uint16_t)},
        }},
        .num_outputs = 1u,
        .outputs = {{
            .slice_id = 1u,
            .rank = 1u,
            .shape = {ELEMENTS},
            .elem_type = ELEM_F16,
            .elem_bytes = sizeof(uint16_t),
            .strides_bytes = {sizeof(uint16_t)},
        }},
        .collective = {
            .num_token_slots = 1u,
            .num_participants = 1u,
            .participants = {{
                .hartid = 0u,
                .input_l1_offset_bytes = 4096u,
                .input_slot_bytes = ELEMENTS * sizeof(uint16_t),
                .output_l1_offset_bytes = 5120u,
                .output_slot_bytes = ELEMENTS * sizeof(uint16_t),
            }},
        },
    };

    const uint32_t start = read_cycle();
    uint32_t errors = maps_execute_all_reduce(&plan, &op, 0u) != 0;
    const uint32_t cycles = read_cycle() - start;
    for (uint32_t index = 0u; index < ELEMENTS; ++index)
        errors += output[index] != input[index];
    errors += cycles >= 20000u;

    printf("MAPS singleton all-reduce cycles: %u\n", cycles);
    printf("MAPS singleton all-reduce errors: %u\n", errors);
    return (int)errors;
}
