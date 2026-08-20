#ifndef MAPS_UTILS_V2_H
#define MAPS_UTILS_V2_H

/*
 * FIFO-backed MAPS transport.
 *
 * This intentionally leaves maps_utils.h unchanged.  The compute, slice and
 * L2 descriptor types are reused verbatim; only the point-to-point transport
 * descriptors and tile plan are versioned here.  Keeping the common fields in
 * the same order as their v1 counterparts keeps generator changes mechanical.
 */

#include "utils/maps_utils.h"
#include "utils/maps_operations.h"
#include "utils/l1_fifo.h"

typedef struct {
    uint32_t transition_id;
    uint32_t src_hartid;
    uint32_t dst_hartid;
    subslice_desc_t dst;
    uint32_t producer_idx; /* Compact, destination-local FIFO sub-ring index. */
} fifo_recv_desc_t;

typedef struct {
    uint32_t transition_id;
    uint32_t src_hartid;
    uint32_t dst_hartid;
    subslice_desc_t src;
    subslice_desc_t dst;
    uint32_t dst_l1_data_base;          /* Retained for generator compatibility. */
    uint32_t dst_slice_l1_offset_bytes; /* Retained for generator compatibility. */
    uint32_t dst_slice_slot_bytes;      /* Retained for generator compatibility. */
    tensor_sub_slice_t copy_src;
    tensor_sub_slice_t copy_dst;
    uint32_t producer_idx;
} fifo_send_desc_t;

typedef struct {
    uint32_t l1_offset_bytes;
    uint32_t num_producers;
    uint32_t num_slots;
    const uint32_t *slot_data_sizes;
    uint32_t slot_data_size;
} fifo_desc_t;

typedef struct fifo_tile_plan {
    uint32_t hartid;

    uint32_t l1_data_base;
    uint32_t ready_flags_base;  /* Retained for compatible generated layouts. */
    uint32_t ready_flags_count; /* Unused by FIFO transport. */
    uint32_t num_token_slots;

    uint32_t num_slices;
    const slice_desc_t *slices;
    uint32_t num_init_l2_reads;
    const l2_read_desc_t *init_l2_reads;
    uint32_t num_l2_reads;
    const l2_read_desc_t *l2_reads;

    uint32_t num_recvs;
    const fifo_recv_desc_t *recvs;
    uint32_t num_ops;
    const op_desc_t *ops;
    uint32_t num_sends;
    const fifo_send_desc_t *sends;
    uint32_t num_l2_writes;
    const l2_write_desc_t *l2_writes;

    maps_operation_runtime_t *operation_runtime;
    fifo_desc_t fifo;
} fifo_tile_plan_t;

static inline uint32_t maps_fifo_tag(uint32_t transition_id, uint32_t slot)
{
    /* Preserve the old ready-flag namespace: transition_id * 16 + slot. */
    return transition_id * 16u + slot;
}

static inline void maps_fifo_init(const fifo_tile_plan_t *plan)
{
    uint32_t data_offset = plan->l1_data_base - get_l1_base(plan->hartid);
    uint32_t fifo_bytes = FIFO_HEADER_SIZE +
        plan->fifo.num_producers * FIFO_RING_STATE_SIZE;
    if (plan->fifo.slot_data_sizes) {
        fifo_bytes += (plan->fifo.num_producers + 1u) * FIFO_SLOT_OFFSET_SIZE;
        for (uint32_t producer = 0u; producer < plan->fifo.num_producers; ++producer)
            fifo_bytes += plan->fifo.num_slots * (FIFO_SLOT_META_SIZE +
                ((plan->fifo.slot_data_sizes[producer] + 3u) & ~3u));
    } else {
        fifo_bytes += plan->fifo.num_producers * plan->fifo.num_slots *
            (FIFO_SLOT_META_SIZE + ((plan->fifo.slot_data_size + 3u) & ~3u));
    }

    if (plan->fifo.num_producers == 0u)
        return;
    /* l1_fifo.h currently places its header at offset zero. */
    if (plan->fifo.l1_offset_bytes != 0u || fifo_bytes > data_offset)
        maps_trap();

    if (plan->fifo.slot_data_sizes)
        fifo_init_variable(plan->hartid, plan->fifo.num_producers,
                           plan->fifo.num_slots, plan->fifo.slot_data_sizes);
    else
        fifo_init(plan->hartid, plan->fifo.num_producers,
                  plan->fifo.num_slots, plan->fifo.slot_data_size);
}

/* Copy a packed FIFO payload to a potentially strided MAPS destination. */
static inline void maps_fifo_unpack(const fifo_msg_t *msg, const subslice_desc_t *dst,
                                    uint32_t dst_addr)
{
    const uint8_t *src = (const uint8_t *)msg->data_ptr;
    volatile uint8_t *out = (volatile uint8_t *)dst_addr;

    if (msg->desc.rank != dst->rank || msg->elem_bytes != dst->elem_bytes ||
        msg->desc.num_elems != maps_shape_elems(dst->rank, dst->shape)) {
        maps_trap();
    }

    /* MAPS matrix fragments are packed in FIFO slots.  The common rank-2
     * destination has contiguous elements within each row but a stride between
     * rows.  Copying it row-wise avoids the generic per-element div/mod address
     * reconstruction below. */
    if (dst->rank == 2u && dst->strides_bytes[1] == msg->elem_bytes) {
        uint32_t row_bytes = dst->shape[1] * msg->elem_bytes;

        if (msg->elem_bytes == sizeof(uint16_t) &&
            (((uint32_t)src | dst_addr | dst->strides_bytes[0]) & 1u) == 0u) {
            const uint16_t *src16 = (const uint16_t *)src;
            volatile uint16_t *out16 = (volatile uint16_t *)out;
            uint32_t row_elems = dst->shape[1];

            for (uint32_t row = 0u; row < dst->shape[0]; ++row) {
                uint32_t src_offset = row * row_elems;
                uint32_t dst_offset = row * (dst->strides_bytes[0] / sizeof(uint16_t));

                for (uint32_t elem = 0u; elem < row_elems; ++elem)
                    out16[dst_offset + elem] = src16[src_offset + elem];
            }
            return;
        }

        for (uint32_t row = 0u; row < dst->shape[0]; ++row) {
            uint32_t src_offset = row * row_bytes;
            uint32_t dst_offset = row * dst->strides_bytes[0];

            for (uint32_t byte = 0u; byte < row_bytes; ++byte)
                out[dst_offset + byte] = src[src_offset + byte];
        }
        return;
    }

    for (uint32_t elem = 0; elem < msg->desc.num_elems; ++elem) {
        uint32_t remaining = elem;
        uint32_t dst_offset = 0u;

        for (uint32_t d = dst->rank; d-- > 0u;) {
            uint32_t coordinate = remaining % dst->shape[d];
            remaining /= dst->shape[d];
            dst_offset += coordinate * dst->strides_bytes[d];
        }

        for (uint32_t byte = 0; byte < msg->elem_bytes; ++byte)
            out[dst_offset + byte] = src[elem * msg->elem_bytes + byte];
    }
}

static inline void maps_fifo_issue_send(const fifo_tile_plan_t *plan,
                                        const fifo_send_desc_t *send,
                                        uint32_t token,
                                        uint32_t slot,
                                        idma_controller_t *idma_ctrl,
                                        eu_controller_t *eu_ctrl)
{
    maps_trace_event((const tile_plan_t *)plan, token, slot, "fifo-send", send->transition_id);
    fifo_push_req_t req = {
        .target_hartid = send->dst_hartid,
        .producer_idx  = send->producer_idx,
        .src_base_addr = local_subslice_addr((const tile_plan_t *)plan, &send->src, slot),
        .src           = &send->copy_src,
        .desc          = &send->copy_dst,
        .tag           = maps_fifo_tag(send->transition_id, slot),
        .elem_bytes    = send->src.elem_bytes,
    };

    if (fifo_push(idma_ctrl, eu_ctrl, &req) != 0)
        maps_trap();
    maps_trace_event((const tile_plan_t *)plan, token, slot, "fifo-sent", send->transition_id);
}

/* MAPS receives are dependency-addressed, so inspect the requested producer's
 * sub-ring directly instead of accepting an unrelated ready message from the
 * FIFO's fair round-robin scan. */
static inline uint32_t maps_fifo_peek_from(const fifo_tile_plan_t *plan,
                                           uint32_t producer_idx,
                                           fifo_msg_t *out)
{
    fifo_header_t *hdr = fifo_get_header(plan->hartid);
    fifo_ring_state_t *rs = fifo_ring_state(hdr, producer_idx);

    if (rs->tail == rs->head)
        return 0u;

    asm volatile("fence r, r" ::: "memory");

    fifo_slot_t *slot = fifo_slot_at(hdr, producer_idx, rs->head);
    out->data_ptr     = (uint32_t)fifo_slot_data(slot);
    out->src          = producer_idx;
    out->tag          = slot->tag;
    out->elem_bytes   = slot->elem_bytes;
    out->data_size    = slot->data_size;
    out->desc         = slot->desc;
    return 1u;
}

static inline void maps_fifo_wait_recv(const fifo_tile_plan_t *plan,
                                       const fifo_recv_desc_t *recv,
                                       uint32_t token,
                                       uint32_t slot)
{
    fifo_msg_t msg;
    uint32_t expected_tag = maps_fifo_tag(recv->transition_id, slot);

    maps_trace_event((const tile_plan_t *)plan, token, slot, "fifo-wait",
                     recv->transition_id);

    for (;;) {
        if (!maps_fifo_peek_from(plan, recv->producer_idx, &msg)) {
            __asm__ volatile("" ::: "memory");
            continue;
        }

        if (msg.tag != expected_tag) {
            maps_trace_event((const tile_plan_t *)plan, token, slot, "fifo-bad-tag",
                             msg.tag);
            maps_trap();
        }

        maps_fifo_unpack(&msg, &recv->dst,
                         local_subslice_addr((const tile_plan_t *)plan, &recv->dst, slot));
        fifo_release(plan->hartid, msg.src);
        maps_trace_event((const tile_plan_t *)plan, token, slot, "fifo-recv",
                         recv->transition_id);
        return;
    }
}

static inline void maps_fifo_init_tile(const fifo_tile_plan_t *plan,
                                       idma_controller_t *idma_ctrl,
                                       eu_controller_t *eu_ctrl)
{
    for (uint32_t i = 0; i < plan->num_init_l2_reads; ++i)
        issue_l2_read_token((const tile_plan_t *)plan, &plan->init_l2_reads[i], 0u, 0u,
                            idma_ctrl, eu_ctrl);
}

static inline void maps_fifo_run_tile_token(const fifo_tile_plan_t *plan, uint32_t token,
                                            idma_controller_t *idma_ctrl,
                                            eu_controller_t *eu_ctrl)
{
    uint32_t slot = maps_token_slot((const tile_plan_t *)plan, token);
    uint32_t token_start = maps_read_cycle();

    for (uint32_t i = 0; i < plan->num_l2_reads; ++i) {
        uint32_t step_start = maps_read_cycle();
        issue_l2_read_token((const tile_plan_t *)plan, &plan->l2_reads[i], token, slot,
                            idma_ctrl, eu_ctrl);
        maps_trace_duration((const tile_plan_t *)plan, token, slot, "l2-read", i,
                            maps_read_cycle() - step_start);
    }
    for (uint32_t i = 0; i < plan->num_recvs; ++i) {
        uint32_t step_start = maps_read_cycle();
        maps_fifo_wait_recv(plan, &plan->recvs[i], token, slot);
        maps_trace_duration((const tile_plan_t *)plan, token, slot, "recv",
                            plan->recvs[i].transition_id,
                            maps_read_cycle() - step_start);
    }
    for (uint32_t i = 0; i < plan->num_ops; ++i) {
        uint32_t step_start = maps_read_cycle();
        if (maps_execute_operation((const tile_plan_t *)plan, &plan->ops[i], slot,
                                   plan->operation_runtime) != 0)
            maps_trap();
        maps_trace_duration((const tile_plan_t *)plan, token, slot, "op", i,
                            maps_read_cycle() - step_start);
    }
    for (uint32_t i = 0; i < plan->num_sends; ++i) {
        uint32_t step_start = maps_read_cycle();
        maps_fifo_issue_send(plan, &plan->sends[i], token, slot, idma_ctrl, eu_ctrl);
        maps_trace_duration((const tile_plan_t *)plan, token, slot, "send",
                            plan->sends[i].transition_id,
                            maps_read_cycle() - step_start);
    }
    for (uint32_t i = 0; i < plan->num_l2_writes; ++i) {
        uint32_t step_start = maps_read_cycle();
        issue_l2_write_token((const tile_plan_t *)plan, &plan->l2_writes[i], token, slot,
                             idma_ctrl, eu_ctrl);
        maps_trace_duration((const tile_plan_t *)plan, token, slot, "l2-write", i,
                            maps_read_cycle() - step_start);
    }

    maps_trace_duration((const tile_plan_t *)plan, token, slot, "token", token,
                        maps_read_cycle() - token_start);
}

static inline void maps_fifo_run_tile_tokens(const fifo_tile_plan_t *plan, uint32_t num_tokens,
                                             idma_controller_t *idma_ctrl,
                                             eu_controller_t *eu_ctrl)
{
    uint32_t run_start = maps_read_cycle();

    for (uint32_t token = 0; token < num_tokens; ++token)
        maps_fifo_run_tile_token(plan, token, idma_ctrl, eu_ctrl);

    maps_trace_duration((const tile_plan_t *)plan, 0u, 0u, "run", num_tokens,
                        maps_read_cycle() - run_start);
}

#endif /* MAPS_UTILS_V2_H */
