#ifndef MAPS_UTILS_H
#define MAPS_UTILS_H

#include <stdint.h>
#include <stddef.h>

#include "addr_map/tile_addr_map.h"
#include "idma.h"
#include "utils/maps_idma.h"
#include "eventunit.h"
#ifndef MAPS_ENABLE_GENERIC_OP_DISPATCH
#define MAPS_ENABLE_GENERIC_OP_DISPATCH 0u
#endif
#if MAPS_ENABLE_GENERIC_OP_DISPATCH
#include "gelu.h"
#include "layernorm.h"
#include "patchify.h"
#include "softmax.h"
#include "transpose.h"
#endif
/*
#ifndef MAPS_ENABLE_TRACE
#define MAPS_ENABLE_TRACE 0u
#endif
*/
#if MAPS_ENABLE_TRACE
#include "utils/printf.h"
#endif

#ifndef MAPS_ENABLE_EVENT_TRACE
#define MAPS_ENABLE_EVENT_TRACE MAPS_ENABLE_TRACE
#endif

#ifndef MAPS_WAIT_MODE
#define MAPS_WAIT_MODE WFE
#endif

#define MAPS_MAX_RANK       6u
#define MAPS_MAX_OP_INPUTS  5u
#define MAPS_MAX_OP_OUTPUTS 3u
#define MAPS_MAX_COLLECTIVE_PARTICIPANTS 64u
#define MAPS_OP_ABI_VERSION 2u

/* Stable op_desc_t.params indices for ABI version 2. */
#define MAPS_PARAM_ROWS       0u
#define MAPS_PARAM_COLS       1u
#define MAPS_PARAM_INNER      2u
#define MAPS_PARAM_FP32_BITS  2u
#define MAPS_PARAM_MODE       1u
#define MAPS_PARAM_ELEM_TYPE  6u
#define MAPS_PARAM_CONV_STRIDE_H   0u
#define MAPS_PARAM_CONV_STRIDE_W   1u
#define MAPS_PARAM_CONV_DILATION_H 2u
#define MAPS_PARAM_CONV_DILATION_W 3u
#define MAPS_PARAM_CONV_PAD_TOP    4u
#define MAPS_PARAM_CONV_PAD_LEFT   5u
#define MAPS_PARAM_CONV_PAD_BOTTOM 6u
#define MAPS_PARAM_CONV_PAD_RIGHT  7u

#define MAX_RANK            MAPS_MAX_RANK
#define MAX_OP_INPUTS       MAPS_MAX_OP_INPUTS
#define MAX_OP_OUTPUTS      MAPS_MAX_OP_OUTPUTS

typedef enum {
    ELEM_F32 = 0,
    ELEM_F16 = 1,
    ELEM_I32 = 2,
    ELEM_U8  = 3,
} elem_type_t;

typedef enum {
    OP_INVALID = 0,
    OP_MATMUL,
    OP_ADD,
    OP_LOG,
    OP_EXP,
    OP_SOFTMAX_EXP,
    OP_REDUCE_SUM,
    OP_REDUCE_MAX,
    OP_COPY,
    OP_MATMUL_BIAS,
    OP_LAYERNORM,
    OP_GELU,
    OP_SOFTMAX_ROW,
    OP_TRANSPOSE_2D,
    OP_PATCHIFY,
    OP_CONV2D,
    OP_DEPTHWISE_CONV,
    OP_ABS,
    OP_DIV,
    OP_GROUP_NORMALIZE,
    OP_GROUP_REDUCE,
    OP_GROUP_CENTERED_REDUCE,
    OP_IM2COL,
    OP_OUTPUT_REFORMAT,
    OP_SPLIT,
    OP_MUL,
    OP_NEG,
    OP_POW,
    OP_RELU,
    OP_RESHAPE,
    OP_SIGMOID,
    OP_SLICE,
    OP_SQRT,
    OP_SUB,
    OP_TRANSPOSE,
    OP_ALL_REDUCE_MAX,
    OP_ALL_REDUCE_SUM,
} op_kind_t;

typedef enum {
    GLOBAL_INPUT = 0,
    GLOBAL_OUTPUT,
    GLOBAL_INITIALIZER,
    GLOBAL_INTERMEDIATE,
} global_kind_t;

/*
A slice is the tensor fragment that is used by the tile
in a local operation.
*/
typedef struct {

    uint32_t slice_id;
    uint32_t global_id;
    global_kind_t global_kind;
    uint32_t owner_hartid;
    uint32_t full_offset[MAPS_MAX_RANK];
    uint32_t rank;
    uint32_t shape[MAPS_MAX_RANK];
    elem_type_t elem_type;
    uint32_t elem_bytes;
    uint32_t l1_offset_bytes;
    uint32_t slot_bytes;
    uint32_t num_slots;
    uint32_t strides_bytes[MAPS_MAX_RANK];

} slice_desc_t;

/*
A subslice is the tensor fragment that is sent during
a transition. A subslice can degenerate into a slice
if all the slice gets sent
*/
typedef struct {

    uint32_t slice_id;
    uint32_t offset[MAPS_MAX_RANK];
    uint32_t rank;
    uint32_t shape[MAPS_MAX_RANK];
    elem_type_t elem_type;
    uint32_t elem_bytes;
    uint32_t strides_bytes[MAPS_MAX_RANK];

} subslice_desc_t;

// descriptor for l2 read operation
typedef struct {

    uint32_t global_id;
    global_kind_t global_kind;
    uint32_t src_l2_addr;
    uint32_t src_l2_token_stride_bytes;
    subslice_desc_t dst;
    tensor_sub_slice_t copy_src;
    tensor_sub_slice_t copy_dst;

} l2_read_desc_t;

// descriptor for receive (maybe not needed, need to talk about this)
typedef struct {

    uint32_t transition_id;
    uint32_t src_hartid;
    uint32_t dst_hartid;
    subslice_desc_t dst;
    uint32_t ready_flag_id;

} recv_desc_t;

// descriptor for send operations
typedef struct {

    uint32_t transition_id;
    uint32_t src_hartid;
    uint32_t dst_hartid;
    subslice_desc_t src;
    subslice_desc_t dst;
    uint32_t dst_l1_data_base;
    uint32_t dst_slice_l1_offset_bytes;
    uint32_t dst_slice_slot_bytes;
    tensor_sub_slice_t copy_src;
    tensor_sub_slice_t copy_dst;
    uint32_t ready_flag_id;

} send_desc_t;

typedef struct {
    uint32_t hartid;
    uint32_t l1_data_base_offset;
    uint32_t input_l1_offset_bytes;
    uint32_t input_slot_bytes;
    uint32_t output_l1_offset_bytes;
    uint32_t output_slot_bytes;
} collective_participant_desc_t;

typedef struct {
    uint32_t stage_id;
    uint32_t layer_index;
    uint32_t group_id;
    uint32_t flag_base;
    uint32_t num_token_slots;
    uint32_t num_participants;
    collective_participant_desc_t participants[MAPS_MAX_COLLECTIVE_PARTICIPANTS];
} collective_desc_t;

// descriptor for compute operations
typedef struct {

    op_kind_t kind;
    uint32_t num_inputs;
    subslice_desc_t inputs[MAPS_MAX_OP_INPUTS];
    uint32_t num_outputs;
    subslice_desc_t outputs[MAPS_MAX_OP_OUTPUTS];
    collective_desc_t collective;
    uint32_t params[8];

} op_desc_t;

// descriptor for l2 write operations
typedef struct {

    uint32_t global_id;
    global_kind_t global_kind;
    subslice_desc_t src;
    uint32_t dst_l2_addr;
    uint32_t dst_l2_token_stride_bytes;
    tensor_sub_slice_t copy_src;
    tensor_sub_slice_t copy_dst;

} l2_write_desc_t;

struct tile_plan;
/* template for custom kernel executors that can be generated
example:

static int my_kernel_executor(const tile_plan_t *plan,
                              const op_desc_t *op,
                              uint32_t slot,
                              void *user)
{
    if (op->kind == OP_MATMUL) {
        float *a = (float *)local_subslice_addr(plan, &op->inputs[0], slot);
        float *b = (float *)local_subslice_addr(plan, &op->inputs[1], slot);
        float *c = (float *)local_subslice_addr(plan, &op->outputs[0], slot);

        my_fast_matmul(a, b, c, op->params);

        return 0;
    }

    return maps_execute_builtin_op(plan, op, slot);
}

*/
typedef int (*maps_op_executor_t)(const struct tile_plan *plan,
                                  const op_desc_t *op,
                                  uint32_t slot,
                                  void *user);

typedef struct tile_plan {
    uint32_t hartid;

    uint32_t l1_data_base;      // base adress of the tile
    uint32_t ready_flags_base;  // base adress if flag-reserved space
    uint32_t ready_flags_count; // number of flags
    uint32_t num_token_slots;   // for double (maybe triple?) buffering

    // accounts for all slices (input slices, output slices, init slices)
    uint32_t num_slices;
    const slice_desc_t *slices;

    // initializer readings
    uint32_t num_init_l2_reads;
    const l2_read_desc_t *init_l2_reads;

    // l2 readings after initialization (inputs)
    uint32_t num_l2_reads;
    const l2_read_desc_t *l2_reads;

    // l1-l1 recieves after initialization (inputs/activations)
    uint32_t num_recvs;
    const recv_desc_t *recvs;

    // operations executed by tile
    uint32_t num_ops;
    const op_desc_t *ops;

    // l1-l1 sends (outputs/activations)
    uint32_t num_sends;
    const send_desc_t *sends;

    // l1-l2 writes (outputs)
    uint32_t num_l2_writes;
    const l2_write_desc_t *l2_writes;

    maps_op_executor_t execute_op; // pointer to custom kernel implementation
    void *execute_op_user;         // context for custom kernels

} tile_plan_t;

// error handling
static inline void maps_trap(void)
{
    for (;;) {
        __asm__ volatile("wfi" ::: "memory");
    }
}

static inline void trap_unsupported_rank(uint32_t rank)
{
    (void)rank;
    maps_trap();
}

static inline void trap_unsupported_op(op_kind_t kind)
{
    (void)kind;
    maps_trap();
}

static inline const slice_desc_t *get_slice(const tile_plan_t *plan, uint32_t slice_id)
{
    for (uint32_t i = 0; i < plan->num_slices; ++i) {
        if (plan->slices[i].slice_id == slice_id) {
            return &plan->slices[i];
        }
    }

    return NULL;
}

static inline uint32_t maps_elem_type_bytes(elem_type_t elem_type)
{
    switch (elem_type) {
    case ELEM_F32:
    case ELEM_I32:
        return 4u;
    case ELEM_F16:
        return 2u;
    case ELEM_U8:
        return 1u;
    default:
        return 0u;
    }
}

static inline uint32_t maps_shape_elems(uint32_t rank, const uint32_t shape[MAPS_MAX_RANK])
{
    uint32_t elems = 1u;

    for (uint32_t d = 0; d < rank; ++d) {
        elems *= shape[d];
    }

    return elems;
}

static inline uint32_t subslice_offset_bytes(const subslice_desc_t *sub)
{
    uint32_t off = 0u;

    for (uint32_t d = 0; d < sub->rank; ++d) {
        off += sub->offset[d] * sub->strides_bytes[d];
    }

    return off;
}

static inline uint32_t
local_slice_slot_addr(const tile_plan_t *plan, const slice_desc_t *slice, uint32_t slot)
{
    uint32_t physical_slot = slot;

    if (slice == NULL) {
        maps_trap();
    }

    if (slice->global_kind == GLOBAL_INITIALIZER) {
        physical_slot = 0u;
    }

    if (physical_slot >= slice->num_slots) {
        maps_trap();
    }

    return plan->l1_data_base + slice->l1_offset_bytes + physical_slot * slice->slot_bytes;
}

static inline uint32_t local_slice_addr(const tile_plan_t *plan, uint32_t slice_id, uint32_t slot)
{
    return local_slice_slot_addr(plan, get_slice(plan, slice_id), slot);
}

static inline uint32_t
local_subslice_addr(const tile_plan_t *plan, const subslice_desc_t *sub, uint32_t slot)
{
    return local_slice_addr(plan, sub->slice_id, slot) + subslice_offset_bytes(sub);
}

static inline uint32_t remote_tile_l1_base(uint32_t hartid)
{
    return L1_BASE + hartid * L1_TILE_OFFSET;
}

static inline uint32_t remote_subslice_addr(const send_desc_t *send, uint32_t slot)
{
    return remote_tile_l1_base(send->dst_hartid) + send->dst_l1_data_base +
           send->dst_slice_l1_offset_bytes + slot * send->dst_slice_slot_bytes +
           subslice_offset_bytes(&send->dst);
}

static inline uint32_t
maps_ready_flag_index(uint32_t transition_id, uint32_t ready_flag_id, uint32_t slot)
{
    return (transition_id * 16u) + (ready_flag_id * 4u) + slot;
}

static inline volatile uint32_t *maps_ready_flag_addr(uint32_t hartid,
                                                      uint32_t ready_flags_base,
                                                      uint32_t transition_id,
                                                      uint32_t ready_flag_id,
                                                      uint32_t slot)
{
    uint32_t index = maps_ready_flag_index(transition_id, ready_flag_id, slot);
    uint32_t addr  = remote_tile_l1_base(hartid) + ready_flags_base + index * sizeof(uint32_t);

    return (volatile uint32_t *)addr;
}

/* GVSoC's CV32 model implements cycle but not cycleh. */
static inline uint32_t maps_read_cycle(void)
{
#if MAPS_ENABLE_TRACE
    uint32_t cycle;
    __asm__ volatile("rdcycle %0" : "=r"(cycle));
    return cycle;
#else
    return 0u;
#endif
}

static inline void maps_trace_event(
    const tile_plan_t *plan, uint32_t token, uint32_t slot, const char *phase, uint32_t index)
{
#if MAPS_ENABLE_EVENT_TRACE
    printf("maps t%d tok %d slot %d %s %d\n", plan->hartid, token, slot, phase, index);
#else
    (void)plan;
    (void)token;
    (void)slot;
    (void)phase;
    (void)index;
#endif
}

static inline void maps_trace_duration(const tile_plan_t *plan,
                                       uint32_t token,
                                       uint32_t slot,
                                       const char *phase,
                                       uint32_t index,
                                       uint32_t cycles)
{
#if MAPS_ENABLE_TRACE
    printf("maps t%d tok %d slot %d %s %d cycles %u\n", plan->hartid, token, slot, phase,
           index, cycles);
#else
    (void)plan;
    (void)token;
    (void)slot;
    (void)phase;
    (void)index;
    (void)cycles;
#endif
}

static inline void maps_main_trace(uint32_t hartid, const char *phase)
{
#if MAPS_ENABLE_EVENT_TRACE
    if (hartid == 0u) {
        printf("\nmaps main %s\n\n", phase);
    }
#else
    (void)hartid;
    (void)phase;
#endif
}

static inline uint32_t
is_ready(const tile_plan_t *plan, uint32_t transition_id, uint32_t ready_flag_id, uint32_t slot)
{
    volatile uint32_t *flag = maps_ready_flag_addr(
        plan->hartid, plan->ready_flags_base, transition_id, ready_flag_id, slot);

    return *flag == 1u;
}


static inline void clear_ready(const tile_plan_t *plan,
                               uint32_t transition_id,
                               uint32_t ready_flag_id,
                               uint32_t slot)
{
    volatile uint32_t *flag = maps_ready_flag_addr(
        plan->hartid, plan->ready_flags_base, transition_id, ready_flag_id, slot);

    *flag = 0u;
    __asm__ volatile("fence rw, rw" ::: "memory");
}
__attribute__((noinline))
static void publish_ready(const tile_plan_t *plan,
                                 uint32_t dst_hartid,
                                 uint32_t transition_id,
                                 uint32_t ready_flag_id,
                                 uint32_t slot)
{
    volatile uint32_t *flag = maps_ready_flag_addr(
        dst_hartid, plan->ready_flags_base, transition_id, ready_flag_id, slot);

    __asm__ volatile("fence rw, w" ::: "memory");
    *flag = 1u;
    __asm__ volatile("fence w, w" ::: "memory");
}

__attribute__((noinline))
static void issue_l2_read_token(const tile_plan_t *plan,
                                       const l2_read_desc_t *read,
                                       uint32_t token,
                                       uint32_t slot,
                                       idma_controller_t *idma_ctrl,
                                       eu_controller_t *eu_ctrl)
{
    uint32_t dst_addr = local_subslice_addr(plan, &read->dst, slot);
    uint32_t src_addr = read->src_l2_addr + token * read->src_l2_token_stride_bytes;

    idma_memcpy_md_to_nd(idma_ctrl, 0u, dst_addr, src_addr, &read->copy_src, &read->copy_dst,
                         read->dst.elem_bytes, eu_ctrl);
}

__attribute__((noinline))
static void issue_l2_write_token(const tile_plan_t *plan,
                                        const l2_write_desc_t *write,
                                        uint32_t token,
                                        uint32_t slot,
                                        idma_controller_t *idma_ctrl,
                                        eu_controller_t *eu_ctrl)
{
    uint32_t src_addr = local_subslice_addr(plan, &write->src, slot);
    uint32_t dst_addr = write->dst_l2_addr + token * write->dst_l2_token_stride_bytes;

    idma_memcpy_md_to_nd(idma_ctrl, 1u, dst_addr, src_addr, &write->copy_src, &write->copy_dst,
                         write->src.elem_bytes, eu_ctrl);
}

__attribute__((noinline))
static void issue_send(const tile_plan_t *plan,
                              const send_desc_t *send,
                              uint32_t slot,
                              idma_controller_t *idma_ctrl,
                              eu_controller_t *eu_ctrl)
{
    uint32_t src_addr = local_subslice_addr(plan, &send->src, slot);
    uint32_t dst_addr = remote_subslice_addr(send, slot);

    idma_memcpy_md_to_nd(idma_ctrl, 1u, dst_addr, src_addr, &send->copy_src, &send->copy_dst,
                         send->src.elem_bytes, eu_ctrl);
    publish_ready(plan, send->dst_hartid, send->transition_id, send->ready_flag_id, slot);
}

__attribute__((noinline))
static void wait_recv_token(const tile_plan_t *plan,
                            const recv_desc_t *recv,
                            uint32_t token,
                            uint32_t slot)
{
    while (!is_ready(plan, recv->transition_id, recv->ready_flag_id, slot)) {
        __asm__ volatile("" ::: "memory");
    }

    __asm__ volatile("fence r, rw" ::: "memory");
    clear_ready(plan, recv->transition_id, recv->ready_flag_id, slot);
}

////////////////////////////////////////////////////////////////////////////////////
// builtin operations

static inline void maps_memcpy_bytes(uint32_t dst_addr, uint32_t src_addr, uint32_t size_bytes)
{
    volatile uint8_t *dst       = (volatile uint8_t *)dst_addr;
    volatile const uint8_t *src = (volatile const uint8_t *)src_addr;

    for (uint32_t i = 0; i < size_bytes; ++i) {
        dst[i] = src[i];
    }
}

static inline int maps_execute_copy_op(const tile_plan_t *plan, const op_desc_t *op, uint32_t slot)
{
    if (op->num_inputs == 0u || op->num_outputs == 0u) {
        return -1;
    }

    uint32_t src_addr   = local_subslice_addr(plan, &op->inputs[0], slot);
    uint32_t dst_addr   = local_subslice_addr(plan, &op->outputs[0], slot);
    uint32_t elem_bytes = op->outputs[0].elem_bytes;
    uint32_t size_bytes = maps_shape_elems(op->outputs[0].rank, op->outputs[0].shape) * elem_bytes;

    maps_memcpy_bytes(dst_addr, src_addr, size_bytes);
    return 0;
}

static inline int
maps_execute_add_f32_op(const tile_plan_t *plan, const op_desc_t *op, uint32_t slot)
{
    if (op->num_inputs < 2u || op->num_outputs == 0u) {
        return -1;
    }

    uint32_t elems = maps_shape_elems(op->outputs[0].rank, op->outputs[0].shape);
    volatile const float *a =
        (volatile const float *)local_subslice_addr(plan, &op->inputs[0], slot);
    volatile const float *b =
        (volatile const float *)local_subslice_addr(plan, &op->inputs[1], slot);
    volatile float *out = (volatile float *)local_subslice_addr(plan, &op->outputs[0], slot);

    if (op->inputs[0].elem_bytes != 4u || op->inputs[1].elem_bytes != 4u ||
        op->outputs[0].elem_bytes != 4u) {
        return -1;
    }

    for (uint32_t i = 0; i < elems; ++i) {
        out[i] = a[i] + b[i];
    }

    return 0;
}

#if MAPS_ENABLE_GENERIC_OP_DISPATCH
static inline float maps_param_as_f32(uint32_t bits)
{
    union {
        uint32_t bits;
        float value;
    } conversion = {.bits = bits};
    return conversion.value;
}

static inline int maps_execute_layernorm_op(const tile_plan_t *plan,
                                            const op_desc_t *op, uint32_t slot)
{
    if (op->num_inputs < 3u || op->num_outputs == 0u ||
        op->outputs[0].elem_type != ELEM_F16)
        return -1;
    maps_layernorm_f16(
        (float16 *)local_subslice_addr(plan, &op->outputs[0], slot),
        (const float16 *)local_subslice_addr(plan, &op->inputs[0], slot),
        (const float16 *)local_subslice_addr(plan, &op->inputs[1], slot),
        (const float16 *)local_subslice_addr(plan, &op->inputs[2], slot),
        op->params[0], op->params[1], maps_param_as_f32(op->params[2]));
    return 0;
}

static inline int maps_execute_gelu_op(const tile_plan_t *plan,
                                       const op_desc_t *op, uint32_t slot)
{
    if (op->num_inputs == 0u || op->num_outputs == 0u ||
        op->params[1] > MAPS_GELU_TANH)
        return -1;
    maps_gelu_f16(
        (float16 *)local_subslice_addr(plan, &op->outputs[0], slot),
        (const float16 *)local_subslice_addr(plan, &op->inputs[0], slot),
        op->params[0], (maps_gelu_mode_t)op->params[1]);
    return 0;
}

static inline int maps_execute_softmax_op(const tile_plan_t *plan,
                                          const op_desc_t *op, uint32_t slot)
{
    if (op->num_inputs == 0u || op->num_outputs == 0u)
        return -1;
    maps_softmax_row_f16(
        (float16 *)local_subslice_addr(plan, &op->outputs[0], slot),
        (const float16 *)local_subslice_addr(plan, &op->inputs[0], slot),
        op->params[0], op->params[1], maps_param_as_f32(op->params[2]));
    return 0;
}

static inline int maps_execute_transpose_op(const tile_plan_t *plan,
                                            const op_desc_t *op, uint32_t slot)
{
    if (op->num_inputs == 0u || op->num_outputs == 0u)
        return -1;
    maps_transpose_2d_f16(
        (float16 *)local_subslice_addr(plan, &op->outputs[0], slot),
        (const float16 *)local_subslice_addr(plan, &op->inputs[0], slot),
        op->params[0], op->params[1], op->params[2], op->params[3]);
    return 0;
}

static inline int maps_execute_patchify_op(const tile_plan_t *plan,
                                           const op_desc_t *op, uint32_t slot)
{
    if (op->num_inputs == 0u || op->num_outputs == 0u)
        return -1;
    maps_patchify_nchw_f16(
        (float16 *)local_subslice_addr(plan, &op->outputs[0], slot),
        (const float16 *)local_subslice_addr(plan, &op->inputs[0], slot),
        op->params[0], op->params[1], op->params[2], op->params[3],
        op->params[4]);
    return 0;
}
#endif
//////////////////////////////////////////////////////////////////////////////////////

// builtin operations entry point
static inline int
maps_execute_builtin_op(const tile_plan_t *plan, const op_desc_t *op, uint32_t slot)
{
    switch (op->kind) {
    case OP_INVALID:
        return 0;
    case OP_COPY:
        return maps_execute_copy_op(plan, op, slot);
    case OP_ADD:
        return maps_execute_add_f32_op(plan, op, slot);
#if MAPS_ENABLE_GENERIC_OP_DISPATCH
    case OP_LAYERNORM:
        return maps_execute_layernorm_op(plan, op, slot);
    case OP_GELU:
        return maps_execute_gelu_op(plan, op, slot);
    case OP_SOFTMAX_ROW:
        return maps_execute_softmax_op(plan, op, slot);
    case OP_TRANSPOSE_2D:
        return maps_execute_transpose_op(plan, op, slot);
    case OP_PATCHIFY:
        return maps_execute_patchify_op(plan, op, slot);
#endif
    case OP_MATMUL_BIAS:
        /* Requires accelerator controllers and is handled by generated dispatch. */
        return -1;
    default:
        return -1;
    }
}

// high level tile operation executor
static inline void execute_op(const tile_plan_t *plan,
                              const op_desc_t *op,
                              uint32_t slot,
                              idma_controller_t *idma_ctrl,
                              eu_controller_t *eu_ctrl)
{
    int rc;

    (void)idma_ctrl;
    (void)eu_ctrl;

    // if there is a custom kernel, run it
    if (plan->execute_op != NULL) {
        rc = plan->execute_op(plan, op, slot, plan->execute_op_user);
    }
    // else check if a builtin kernel exists
    else {
        rc = maps_execute_builtin_op(plan, op, slot);
    }

    // error handling for no custom kernel and no builtin kernel
    if (rc != 0) {
        trap_unsupported_op(op->kind);
    }
}

// helper for double buffering
static inline uint32_t maps_token_slot(const tile_plan_t *plan, uint32_t token)
{
    uint32_t slots = plan->num_token_slots;

    if (slots == 0u) {
        slots = 1u;
    }

    return token % slots;
}

// main tile initialization function, basically just read l2 initializers
static inline void
init_tile(const tile_plan_t *plan, idma_controller_t *idma_ctrl, eu_controller_t *eu_ctrl)
{
    maps_trace_event(plan, 0u, 0u, "init-begin", plan->num_init_l2_reads);

    for (uint32_t i = 0; i < plan->num_init_l2_reads; ++i) {
        maps_trace_event(plan, 0u, 0u, "init-read", i);
        issue_l2_read_token(plan, &plan->init_l2_reads[i], 0u, 0u, idma_ctrl, eu_ctrl);
        maps_trace_event(plan, 0u, 0u, "init-read-done", i);
    }

    maps_trace_event(plan, 0u, 0u, "init-done", 0u);
}

// main tokenized tile loop
static inline void run_tile_token(const tile_plan_t *plan,
                                  uint32_t token,
                                  idma_controller_t *idma_ctrl,
                                  eu_controller_t *eu_ctrl)
{
    // Set double buffering slot
    uint32_t slot = maps_token_slot(plan, token);
    uint32_t token_start = maps_read_cycle();

    // Do L2 reads
    for (uint32_t i = 0; i < plan->num_l2_reads; ++i) {
        uint32_t step_start = maps_read_cycle();
        issue_l2_read_token(plan, &plan->l2_reads[i], token, slot, idma_ctrl, eu_ctrl);
        maps_trace_duration(plan, token, slot, "l2-read", i,
                            maps_read_cycle() - step_start);
    }

    // Do L1-L1 receives
    for (uint32_t i = 0; i < plan->num_recvs; ++i) {
        uint32_t step_start = maps_read_cycle();
        wait_recv_token(plan, &plan->recvs[i], token, slot);
        maps_trace_duration(plan, token, slot, "recv", plan->recvs[i].transition_id,
                            maps_read_cycle() - step_start);
    }

    // Perform actual operations
    for (uint32_t i = 0; i < plan->num_ops; ++i) {
        uint32_t step_start = maps_read_cycle();
        execute_op(plan, &plan->ops[i], slot, idma_ctrl, eu_ctrl);
        maps_trace_duration(plan, token, slot, "op", i, maps_read_cycle() - step_start);
    }

    // Do L1-L1 sends
    for (uint32_t i = 0; i < plan->num_sends; ++i) {
        uint32_t step_start = maps_read_cycle();
        issue_send(plan, &plan->sends[i], slot, idma_ctrl, eu_ctrl);
        maps_trace_duration(plan, token, slot, "send", plan->sends[i].transition_id,
                            maps_read_cycle() - step_start);
    }

    // Do L2 writes
    for (uint32_t i = 0; i < plan->num_l2_writes; ++i) {
        uint32_t step_start = maps_read_cycle();
        issue_l2_write_token(plan, &plan->l2_writes[i], token, slot, idma_ctrl, eu_ctrl);
        maps_trace_duration(plan, token, slot, "l2-write", i,
                            maps_read_cycle() - step_start);
    }

    maps_trace_duration(plan, token, slot, "token", token, maps_read_cycle() - token_start);
}

// top level wrapper for multi-token execution
static inline void run_tile_tokens(const tile_plan_t *plan,
                                   uint32_t num_tokens,
                                   idma_controller_t *idma_ctrl,
                                   eu_controller_t *eu_ctrl)
{
    uint32_t run_start = maps_read_cycle();

    for (uint32_t token = 0; token < num_tokens; ++token) {
        run_tile_token(plan, token, idma_ctrl, eu_ctrl);
    }

    maps_trace_duration(plan, 0u, 0u, "run", num_tokens, maps_read_cycle() - run_start);
}

#endif /* MAPS_UTILS_H */
