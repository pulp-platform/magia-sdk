#include <stdint.h>

#include "fsync.h"
#include "test_maps_collective_task_bin.h"
#include "tile.h"
#include "utils/maps_operations.h"
#include "utils/printf.h"

#define INPUT_OFFSET 4096u
#define OUTPUT_OFFSET 4112u
#define READY_FLAGS_OFFSET 1024u
#define MAX_PARTICIPANTS 64u
#define SCALING_CASES 7u
#define MAX_64_WAY_CYCLES 20000u

static volatile uint32_t measured_cycles[MAX_PARTICIPANTS]
    __attribute__((section(".l2_arena")));
static volatile uint32_t measured_errors[MAX_PARTICIPANTS]
    __attribute__((section(".l2_arena")));

static uint32_t read_cycle(void)
{
    uint32_t cycle;
    __asm__ volatile("rdcycle %0" : "=r"(cycle));
    return cycle;
}

static void global_barrier(fsync_controller_t *fsync, eu_controller_t *event_unit)
{
    fsync_sync_global(fsync);
    eu_fsync_wait(event_unit, WFE);
}

int main(void)
{
    static const uint32_t participant_counts[SCALING_CASES] = {
        1u, 2u, 4u, 8u, 16u, 32u, 64u,
    };
    const uint32_t hartid = get_hartid();

    idma_config_t idma_config = {.hartid = hartid};
    idma_controller_t idma = {
        .base = 0u, .cfg = &idma_config, .api = &idma_api,
    };
    fsync_config_t fsync_config = {.hartid = hartid};
    fsync_controller_t fsync = {
        .base = 0u, .cfg = &fsync_config, .api = &fsync_api,
    };
    eu_config_t eu_config = {.hartid = hartid};
    eu_controller_t event_unit = {
        .base = 0u, .cfg = &eu_config, .api = &eu_api,
    };
    idma_init(&idma);
    fsync_init(&fsync);
    eu_init(&event_unit);
    eu_clear_events(0xffffffffu);
    eu_fsync_init(&event_unit, 0u);
    eu_idma_init(&event_unit, 0u);

    maps_operation_runtime_t runtime = {
        .idma_ctrl = &idma,
        .redmule_ctrl = 0,
        .eu_ctrl = &event_unit,
        .kernel_abi_version = MAPS_KERNEL_ABI_VERSION,
        .task_bundle_abi_version = MAPS_TASK_BUNDLE_ABI_VERSION,
        .spatz_binary_start = SPATZ_BINARY_START,
        .add_fp16_task = 0u,
        .mul_bcast_fp16_task = 0u,
        .matmul_fp16_task = 0u,
        .relu_fp16_task = 0u,
        .softmax_exp_fp16_task = 0u,
        .group_reduce_fp16_task = 0u,
        .group_centered_reduce_fp16_task = 0u,
        .group_normalize_fp16_task = 0u,
        .reducesum_fp16_task = REDUCESUM_FP16_SPATZ_TASK,
        .reducemax_fp16_task = REDUCEMAX_FP16_SPATZ_TASK,
        .binary_bcast_fp16_task = 0u,
        .spatz_params =
            (void *)(get_l1_base(hartid) + MAPS_OPERATION_TASK_SCRATCH_OFFSET),
        .spatz_params_bytes = MAPS_OPERATION_TASK_SCRATCH_BYTES,
        .spatz_initialized = 0u,
    };
    if (maps_operation_runtime_init(&runtime) != 0)
        return 1;

    uint16_t *input = (uint16_t *)(get_l1_base(hartid) + INPUT_OFFSET);
    uint16_t *output = (uint16_t *)(get_l1_base(hartid) + OUTPUT_OFFSET);
    volatile uint32_t *ready_flags =
        (volatile uint32_t *)(get_l1_base(hartid) + READY_FLAGS_OFFSET);
    for (uint32_t index = 0u; index < 2u * MAX_PARTICIPANTS; ++index)
        ready_flags[index] = 0u;

    const slice_desc_t slices[] = {
        {
            .slice_id = 0u,
            .global_kind = GLOBAL_INTERMEDIATE,
            .owner_hartid = hartid,
            .rank = 1u,
            .shape = {1u},
            .elem_type = ELEM_F16,
            .elem_bytes = sizeof(uint16_t),
            .l1_offset_bytes = INPUT_OFFSET,
            .slot_bytes = sizeof(uint16_t),
            .num_slots = 1u,
            .strides_bytes = {sizeof(uint16_t)},
        },
        {
            .slice_id = 1u,
            .global_kind = GLOBAL_INTERMEDIATE,
            .owner_hartid = hartid,
            .rank = 1u,
            .shape = {1u},
            .elem_type = ELEM_F16,
            .elem_bytes = sizeof(uint16_t),
            .l1_offset_bytes = OUTPUT_OFFSET,
            .slot_bytes = sizeof(uint16_t),
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
        .ready_flags_base = READY_FLAGS_OFFSET,
        .ready_flags_count = 2u * MAX_PARTICIPANTS,
    };
    static op_desc_t op = {
        .kind = OP_ALL_REDUCE_SUM,
        .num_inputs = 1u,
        .inputs = {{
            .slice_id = 0u,
            .rank = 1u,
            .shape = {1u},
            .elem_type = ELEM_F16,
            .elem_bytes = sizeof(uint16_t),
            .strides_bytes = {sizeof(uint16_t)},
        }},
        .num_outputs = 1u,
        .outputs = {{
            .slice_id = 1u,
            .rank = 1u,
            .shape = {1u},
            .elem_type = ELEM_F16,
            .elem_bytes = sizeof(uint16_t),
            .strides_bytes = {sizeof(uint16_t)},
        }},
        .collective = {
            .num_token_slots = 1u,
        },
    };
    for (uint32_t participant = 0u; participant < MAX_PARTICIPANTS;
         ++participant) {
        collective_participant_desc_t *descriptor =
            &op.collective.participants[participant];
        descriptor->hartid = participant;
        descriptor->l1_data_base_offset = 0u;
        descriptor->input_l1_offset_bytes = INPUT_OFFSET;
        descriptor->input_slot_bytes = sizeof(uint16_t);
        descriptor->output_l1_offset_bytes = OUTPUT_OFFSET;
        descriptor->output_slot_bytes = sizeof(uint16_t);
    }

    uint32_t errors = 0u;
    for (uint32_t test_case = 0u; test_case < SCALING_CASES; ++test_case) {
        const uint32_t participants = participant_counts[test_case];
        if (participants > NUM_HARTS)
            continue;
        *input = maps_operation_f32_to_f16(1.0f);
        *output = 0xffffu;
        measured_cycles[hartid] = 0u;
        measured_errors[hartid] = 0u;
        op.collective.num_participants = participants;

        global_barrier(&fsync, &event_unit);
        if (hartid < participants) {
            const uint32_t start = read_cycle();
            const int result =
                maps_execute_all_reduce(&plan, &op, 0u, &runtime);
            measured_cycles[hartid] = read_cycle() - start;
            measured_errors[hartid] =
                result != 0 ||
                *output != maps_operation_f32_to_f16((float)participants);
        }
        global_barrier(&fsync, &event_unit);

        if (hartid == 0u) {
            uint32_t maximum = 0u;
            uint32_t case_errors = 0u;
            for (uint32_t participant = 0u; participant < participants;
                 ++participant) {
                if (measured_cycles[participant] > maximum)
                    maximum = measured_cycles[participant];
                case_errors += measured_errors[participant];
            }
            printf("MAPS all-reduce participants=%u cycles=%u errors=%u\n",
                   participants, maximum, case_errors);
            errors += case_errors;
            if (participants == MAX_PARTICIPANTS)
                errors += maximum >= MAX_64_WAY_CYCLES;
        }
        global_barrier(&fsync, &event_unit);
    }

    if (hartid == 0u)
        printf("MAPS all-reduce scaling errors: %u\n", errors);
    return hartid == 0u ? (int)errors : 0;
}
