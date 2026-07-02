#ifndef MAPS_HANDWRITTEN_TEST_H
#define MAPS_HANDWRITTEN_TEST_H

#include <stdint.h>

#ifndef MAPS_ENABLE_TRACE
#define MAPS_ENABLE_TRACE 1u
#endif

#include "eventunit.h"
#include "fsync.h"
#include "idma.h"
#include "redmule.h"
#include "tile.h"
#include "utils/maps_utils.h"

#define MAPS_TILE_MM0           0u
#define MAPS_TILE_MM1           1u
#define MAPS_TILE_MM2           8u
#define MAPS_TILE_LOG0          12u
#define MAPS_TILE_LOG1          13u
#define MAPS_TILE_ADD           35u

#define MAPS_NUM_TOKENS         2u
#define MAPS_NUM_TOKEN_SLOTS    2u
#define MAPS_MATRIX_RANK        2u
#define MAPS_N                  32u
#define MAPS_K                  32u
#define MAPS_MM0_ROWS           11u
#define MAPS_MM1_ROWS           11u
#define MAPS_MM2_ROWS           10u
#define MAPS_MM1_LOG0_ROWS      5u
#define MAPS_MM1_LOG1_ROWS      6u
#define MAPS_LOG_ROWS           16u
#define MAPS_ELEM_BYTES         2u

#define MAPS_MM0_ELEMS          (MAPS_MM0_ROWS * MAPS_N)
#define MAPS_MM1_ELEMS          (MAPS_MM1_ROWS * MAPS_N)
#define MAPS_MM2_ELEMS          (MAPS_MM2_ROWS * MAPS_N)
#define MAPS_B_ELEMS            (MAPS_K * MAPS_N)
#define MAPS_LOG_ELEMS          (MAPS_LOG_ROWS * MAPS_N)

#define MAPS_MM0_BYTES          (MAPS_MM0_ELEMS * MAPS_ELEM_BYTES)
#define MAPS_MM1_BYTES          (MAPS_MM1_ELEMS * MAPS_ELEM_BYTES)
#define MAPS_MM2_BYTES          (MAPS_MM2_ELEMS * MAPS_ELEM_BYTES)
#define MAPS_B_BYTES            (MAPS_B_ELEMS * MAPS_ELEM_BYTES)
#define MAPS_LOG_BYTES          (MAPS_LOG_ELEMS * MAPS_ELEM_BYTES)

#define MAPS_L1_DATA_OFFSET     0x0000u
#define MAPS_READY_FLAGS_OFFSET 0x7000u
#define MAPS_READY_FLAGS_WORDS  256u

#define MAPS_L1_A_OFFSET        0x0000u
#define MAPS_L1_B_OFFSET        0x0800u
#define MAPS_L1_MATMUL_OFFSET   0x1000u
#define MAPS_L1_LOG_IN_OFFSET   0x0000u
#define MAPS_L1_LOG_OUT_OFFSET  0x0800u
#define MAPS_L1_ADD_IN0_OFFSET  0x0000u
#define MAPS_L1_ADD_IN1_OFFSET  0x0800u
#define MAPS_L1_FINAL_OFFSET    0x1000u

#ifndef MAPS_HANDWRITTEN_USE_REDMULE
#define MAPS_HANDWRITTEN_USE_REDMULE 1u
#endif

enum {
    MAPS_GLOBAL_A0 = 0,
    MAPS_GLOBAL_B0,
    MAPS_GLOBAL_A1,
    MAPS_GLOBAL_B1,
    MAPS_GLOBAL_A2,
    MAPS_GLOBAL_B2,
    MAPS_GLOBAL_ZERO_11,
    MAPS_GLOBAL_ZERO_10,
    MAPS_GLOBAL_MM0,
    MAPS_GLOBAL_MM1,
    MAPS_GLOBAL_MM2,
    MAPS_GLOBAL_LOG0,
    MAPS_GLOBAL_LOG1,
    MAPS_GLOBAL_FINAL,
};

enum {
    MAPS_SLICE_A = 0,
    MAPS_SLICE_B,
    MAPS_SLICE_MATMUL_OUT,
    MAPS_SLICE_LOG_IN,
    MAPS_SLICE_LOG_OUT,
    MAPS_SLICE_ADD_IN0,
    MAPS_SLICE_ADD_IN1,
    MAPS_SLICE_FINAL,
};

enum {
    MAPS_TRANS_MM0_LOG0 = 0,
    MAPS_TRANS_MM1_LOG0,
    MAPS_TRANS_MM1_LOG1,
    MAPS_TRANS_MM2_LOG1,
    MAPS_TRANS_LOG0_ADD,
    MAPS_TRANS_LOG1_ADD,
};

typedef struct {
    redmule_controller_t *redmule_ctrl;
    eu_controller_t *eu_ctrl;
} maps_generated_runtime_t;

static float16 maps_l2_a0[MAPS_NUM_TOKENS * MAPS_MM0_ELEMS];
static float16 maps_l2_a1[MAPS_NUM_TOKENS * MAPS_MM1_ELEMS];
static float16 maps_l2_a2[MAPS_NUM_TOKENS * MAPS_MM2_ELEMS];
static float16 maps_l2_b0[MAPS_B_ELEMS];
static float16 maps_l2_b1[MAPS_B_ELEMS];
static float16 maps_l2_b2[MAPS_B_ELEMS];
static float16 maps_l2_zero_11[MAPS_MM1_ELEMS];
static float16 maps_l2_zero_10[MAPS_MM2_ELEMS];
static float16 maps_l2_final[MAPS_NUM_TOKENS * MAPS_LOG_ELEMS];
static float16 maps_l2_golden[MAPS_NUM_TOKENS * MAPS_LOG_ELEMS];

static const uint16_t maps_generated_value_lut[8] = {
    0x3c00u,
    0x4000u,
    0x4200u,
    0x4400u,
    0x4500u,
    0x4600u,
    0x4700u,
    0x4800u,
};

static const uint16_t maps_generated_log_lut[8] = {
    0x0000u,
    0x398cu,
    0x3c65u,
    0x3d8cu,
    0x3e70u,
    0x3f2bu,
    0x3fc9u,
    0x4029u,
};

static const uint16_t maps_generated_log_add_lut[8][8] = {
    {0x0000u, 0x398cu, 0x3c65u, 0x3d8cu, 0x3e70u, 0x3f2bu, 0x3fc9u, 0x4029u},
    {0x398cu, 0x3d8cu, 0x3f2bu, 0x4029u, 0x409bu, 0x40f8u, 0x4148u, 0x418cu},
    {0x3c65u, 0x3f2bu, 0x4065u, 0x40f8u, 0x416au, 0x41c8u, 0x4217u, 0x425cu},
    {0x3d8cu, 0x4029u, 0x40f8u, 0x418cu, 0x41feu, 0x425cu, 0x42aau, 0x42efu},
    {0x3e70u, 0x409bu, 0x416au, 0x41feu, 0x4270u, 0x42ceu, 0x431cu, 0x4361u},
    {0x3f2bu, 0x40f8u, 0x41c8u, 0x425cu, 0x42ceu, 0x432bu, 0x437au, 0x43beu},
    {0x3fc9u, 0x4148u, 0x4217u, 0x42aau, 0x431cu, 0x437au, 0x43c9u, 0x4407u},
    {0x4029u, 0x418cu, 0x425cu, 0x42efu, 0x4361u, 0x43beu, 0x4407u, 0x4429u},
};

static inline uint32_t maps_generated_lcg(uint32_t *state)
{
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static inline uint16_t maps_generated_rand_half(uint32_t *state)
{
    uint32_t v = (maps_generated_lcg(state) >> 24) & 0x7u;

    return maps_generated_value_lut[v];
}

static inline void maps_generated_fill_random(float16 *dst, uint32_t elems, uint32_t seed)
{
    uint32_t state     = seed;
    uint16_t *dst_half = (uint16_t *)dst;

    for (uint32_t i = 0; i < elems; ++i) {
        dst_half[i] = maps_generated_rand_half(&state);
    }
}

static inline void maps_generated_fill_identity(float16 *dst)
{
    uint16_t *dst_half = (uint16_t *)dst;

    for (uint32_t r = 0; r < MAPS_K; ++r) {
        for (uint32_t c = 0; c < MAPS_N; ++c) {
            dst_half[r * MAPS_N + c] = (r == c) ? 0x3c00u : 0x0000u;
        }
    }
}

static inline void maps_generated_zero(float16 *dst, uint32_t elems)
{
    uint16_t *dst_half = (uint16_t *)dst;

    for (uint32_t i = 0; i < elems; ++i) {
        dst_half[i] = 0x0000u;
    }
}

static inline uint32_t maps_generated_value_index(uint16_t value)
{
    switch (value) {
    case 0x3c00u:
        return 0u;
    case 0x4000u:
        return 1u;
    case 0x4200u:
        return 2u;
    case 0x4400u:
        return 3u;
    case 0x4500u:
        return 4u;
    case 0x4600u:
        return 5u;
    case 0x4700u:
        return 6u;
    case 0x4800u:
        return 7u;
    default:
        return 0u;
    }
}

static inline uint32_t maps_generated_log_index(uint16_t value)
{
    switch (value) {
    case 0x398cu:
        return 1u;
    case 0x3c65u:
        return 2u;
    case 0x3d8cu:
        return 3u;
    case 0x3e70u:
        return 4u;
    case 0x3f2bu:
        return 5u;
    case 0x3fc9u:
        return 6u;
    case 0x4029u:
        return 7u;
    case 0x0000u:
    default:
        return 0u;
    }
}

static inline void
maps_generated_cpu_matmul(const float16 *a, const float16 *b, float16 *out, uint32_t rows)
{
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < MAPS_N; ++c) {
            float16 acc = (float16)0.0f;

            for (uint32_t k = 0; k < MAPS_K; ++k) {
                acc = acc + a[r * MAPS_K + k] * b[k * MAPS_N + c];
            }

            out[r * MAPS_N + c] = acc;
        }
    }
}

static inline void
maps_generated_copy_identity_matmul(uint32_t dst_addr, uint32_t src_addr, uint32_t elems)
{
    volatile uint16_t *dst       = (volatile uint16_t *)dst_addr;
    volatile const uint16_t *src = (volatile const uint16_t *)src_addr;

    for (uint32_t i = 0; i < elems; ++i) {
        dst[i] = src[i];
    }
}

static inline void maps_generated_compute_golden_token(uint32_t token)
{
    const uint16_t *a0 = (const uint16_t *)&maps_l2_a0[token * MAPS_MM0_ELEMS];
    const uint16_t *a1 = (const uint16_t *)&maps_l2_a1[token * MAPS_MM1_ELEMS];
    const uint16_t *a2 = (const uint16_t *)&maps_l2_a2[token * MAPS_MM2_ELEMS];
    uint16_t *golden   = (uint16_t *)&maps_l2_golden[token * MAPS_LOG_ELEMS];

    for (uint32_t r = 0; r < MAPS_LOG_ROWS; ++r) {
        for (uint32_t c = 0; c < MAPS_N; ++c) {
            uint16_t lhs_value;
            uint16_t rhs_value;

            if (r < MAPS_MM0_ROWS) {
                lhs_value = a0[r * MAPS_N + c];
            } else {
                lhs_value = a1[(r - MAPS_MM0_ROWS) * MAPS_N + c];
            }

            if (r < MAPS_MM1_LOG1_ROWS) {
                rhs_value = a1[(MAPS_MM1_LOG0_ROWS + r) * MAPS_N + c];
            } else {
                rhs_value = a2[(r - MAPS_MM1_LOG1_ROWS) * MAPS_N + c];
            }

            golden[r * MAPS_N + c] =
                maps_generated_log_add_lut[maps_generated_value_index(lhs_value)]
                                          [maps_generated_value_index(rhs_value)];
        }
    }
}

static inline void maps_generated_init_tensors(uint32_t hartid)
{
    if (hartid != 0u) {
        return;
    }

#if MAPS_ENABLE_TRACE
    printf("maps init fill-a\n");
#endif
    maps_generated_fill_random(maps_l2_a0, MAPS_NUM_TOKENS * MAPS_MM0_ELEMS, 0x1234u);
    maps_generated_fill_random(maps_l2_a1, MAPS_NUM_TOKENS * MAPS_MM1_ELEMS, 0x5678u);
    maps_generated_fill_random(maps_l2_a2, MAPS_NUM_TOKENS * MAPS_MM2_ELEMS, 0x9abcu);
#if MAPS_ENABLE_TRACE
    printf("maps init fill-b\n");
#endif
    maps_generated_fill_identity(maps_l2_b0);
    maps_generated_fill_identity(maps_l2_b1);
    maps_generated_fill_identity(maps_l2_b2);
#if MAPS_ENABLE_TRACE
    printf("maps init zero\n");
#endif
    maps_generated_zero(maps_l2_zero_11, MAPS_MM1_ELEMS);
    maps_generated_zero(maps_l2_zero_10, MAPS_MM2_ELEMS);
    maps_generated_zero(maps_l2_final, MAPS_NUM_TOKENS * MAPS_LOG_ELEMS);
    maps_generated_zero(maps_l2_golden, MAPS_NUM_TOKENS * MAPS_LOG_ELEMS);

#if MAPS_ENABLE_TRACE
    printf("maps init golden\n");
#endif
    for (uint32_t token = 0; token < MAPS_NUM_TOKENS; ++token) {
        maps_generated_compute_golden_token(token);
    }
#if MAPS_ENABLE_TRACE
    printf("maps init done\n");
#endif
}

/* One side (src or dst) of a maps copy: a packed rows x cols matrix sub-slice. */
#define MAPS_SUBSLICE_INIT(rows, cols)                                                             \
    {                                                                                              \
        .rank      = MAPS_MATRIX_RANK,                                                             \
        .num_elems = (rows) * (cols),                                                              \
        .dims      = {{0u, (rows), (cols) * MAPS_ELEM_BYTES}, {0u, (cols), MAPS_ELEM_BYTES}},      \
    }

/* Fills both trailing (copy_src, copy_dst) fields; maps src/dst shapes match. */
#define MAPS_COPY_INIT(rows, cols) MAPS_SUBSLICE_INIT(rows, cols), MAPS_SUBSLICE_INIT(rows, cols)

#define MAPS_SUB_INIT(slice, row_off, rows, cols)                                                  \
    {                                                                                              \
        .slice_id      = (slice),                                                                  \
        .offset        = {(row_off), 0u, 0u, 0u},                                                  \
        .rank          = MAPS_MATRIX_RANK,                                                         \
        .shape         = {(rows), (cols), 0u, 0u},                                                 \
        .elem_type     = ELEM_F16,                                                                 \
        .elem_bytes    = MAPS_ELEM_BYTES,                                                          \
        .strides_bytes = {MAPS_N * MAPS_ELEM_BYTES, MAPS_ELEM_BYTES, 0u, 0u},                      \
    }

#define MAPS_SLICE_INIT(slice, global, kind, rows, cols, offset, bytes, slots)                     \
    {                                                                                              \
        .slice_id        = (slice),                                                                \
        .global_id       = (global),                                                               \
        .global_kind     = (kind),                                                                 \
        .owner_hartid    = 0u,                                                                     \
        .full_offset     = {0u, 0u, 0u, 0u},                                                       \
        .rank            = MAPS_MATRIX_RANK,                                                       \
        .shape           = {(rows), (cols), 0u, 0u},                                               \
        .elem_type       = ELEM_F16,                                                               \
        .elem_bytes      = MAPS_ELEM_BYTES,                                                        \
        .l1_offset_bytes = (offset),                                                               \
        .slot_bytes      = (bytes),                                                                \
        .num_slots       = (slots),                                                                \
        .strides_bytes   = {(cols) * MAPS_ELEM_BYTES, MAPS_ELEM_BYTES, 0u, 0u},                    \
    }

#define MAPS_SUB_A(rows)           MAPS_SUB_INIT(MAPS_SLICE_A, 0u, rows, MAPS_K)
#define MAPS_SUB_B                 MAPS_SUB_INIT(MAPS_SLICE_B, 0u, MAPS_K, MAPS_N)
#define MAPS_SUB_MM(row, rows)     MAPS_SUB_INIT(MAPS_SLICE_MATMUL_OUT, row, rows, MAPS_N)
#define MAPS_SUB_LOG_IN(row, rows) MAPS_SUB_INIT(MAPS_SLICE_LOG_IN, row, rows, MAPS_N)
#define MAPS_SUB_LOG_OUT           MAPS_SUB_INIT(MAPS_SLICE_LOG_OUT, 0u, MAPS_LOG_ROWS, MAPS_N)
#define MAPS_SUB_ADD_IN0           MAPS_SUB_INIT(MAPS_SLICE_ADD_IN0, 0u, MAPS_LOG_ROWS, MAPS_N)
#define MAPS_SUB_ADD_IN1           MAPS_SUB_INIT(MAPS_SLICE_ADD_IN1, 0u, MAPS_LOG_ROWS, MAPS_N)
#define MAPS_SUB_FINAL             MAPS_SUB_INIT(MAPS_SLICE_FINAL, 0u, MAPS_LOG_ROWS, MAPS_N)

static const slice_desc_t maps_matmul_slices_11[3] = {
    MAPS_SLICE_INIT(MAPS_SLICE_A,
                    MAPS_GLOBAL_A0,
                    GLOBAL_INPUT,
                    MAPS_MM1_ROWS,
                    MAPS_K,
                    MAPS_L1_A_OFFSET,
                    MAPS_MM1_BYTES,
                    MAPS_NUM_TOKEN_SLOTS),
    MAPS_SLICE_INIT(MAPS_SLICE_B,
                    MAPS_GLOBAL_B0,
                    GLOBAL_INITIALIZER,
                    MAPS_K,
                    MAPS_N,
                    MAPS_L1_B_OFFSET,
                    MAPS_B_BYTES,
                    1u),
    MAPS_SLICE_INIT(MAPS_SLICE_MATMUL_OUT,
                    MAPS_GLOBAL_MM0,
                    GLOBAL_INTERMEDIATE,
                    MAPS_MM1_ROWS,
                    MAPS_N,
                    MAPS_L1_MATMUL_OFFSET,
                    MAPS_MM1_BYTES,
                    MAPS_NUM_TOKEN_SLOTS),
};

static const slice_desc_t maps_matmul_slices_10[3] = {
    MAPS_SLICE_INIT(MAPS_SLICE_A,
                    MAPS_GLOBAL_A2,
                    GLOBAL_INPUT,
                    MAPS_MM2_ROWS,
                    MAPS_K,
                    MAPS_L1_A_OFFSET,
                    MAPS_MM2_BYTES,
                    MAPS_NUM_TOKEN_SLOTS),
    MAPS_SLICE_INIT(MAPS_SLICE_B,
                    MAPS_GLOBAL_B2,
                    GLOBAL_INITIALIZER,
                    MAPS_K,
                    MAPS_N,
                    MAPS_L1_B_OFFSET,
                    MAPS_B_BYTES,
                    1u),
    MAPS_SLICE_INIT(MAPS_SLICE_MATMUL_OUT,
                    MAPS_GLOBAL_MM2,
                    GLOBAL_INTERMEDIATE,
                    MAPS_MM2_ROWS,
                    MAPS_N,
                    MAPS_L1_MATMUL_OFFSET,
                    MAPS_MM2_BYTES,
                    MAPS_NUM_TOKEN_SLOTS),
};

static const slice_desc_t maps_log_slices[2] = {
    MAPS_SLICE_INIT(MAPS_SLICE_LOG_IN,
                    MAPS_GLOBAL_MM0,
                    GLOBAL_INTERMEDIATE,
                    MAPS_LOG_ROWS,
                    MAPS_N,
                    MAPS_L1_LOG_IN_OFFSET,
                    MAPS_LOG_BYTES,
                    MAPS_NUM_TOKEN_SLOTS),
    MAPS_SLICE_INIT(MAPS_SLICE_LOG_OUT,
                    MAPS_GLOBAL_LOG0,
                    GLOBAL_INTERMEDIATE,
                    MAPS_LOG_ROWS,
                    MAPS_N,
                    MAPS_L1_LOG_OUT_OFFSET,
                    MAPS_LOG_BYTES,
                    MAPS_NUM_TOKEN_SLOTS),
};

static const slice_desc_t maps_add_slices[3] = {
    MAPS_SLICE_INIT(MAPS_SLICE_ADD_IN0,
                    MAPS_GLOBAL_LOG0,
                    GLOBAL_INTERMEDIATE,
                    MAPS_LOG_ROWS,
                    MAPS_N,
                    MAPS_L1_ADD_IN0_OFFSET,
                    MAPS_LOG_BYTES,
                    MAPS_NUM_TOKEN_SLOTS),
    MAPS_SLICE_INIT(MAPS_SLICE_ADD_IN1,
                    MAPS_GLOBAL_LOG1,
                    GLOBAL_INTERMEDIATE,
                    MAPS_LOG_ROWS,
                    MAPS_N,
                    MAPS_L1_ADD_IN1_OFFSET,
                    MAPS_LOG_BYTES,
                    MAPS_NUM_TOKEN_SLOTS),
    MAPS_SLICE_INIT(MAPS_SLICE_FINAL,
                    MAPS_GLOBAL_FINAL,
                    GLOBAL_OUTPUT,
                    MAPS_LOG_ROWS,
                    MAPS_N,
                    MAPS_L1_FINAL_OFFSET,
                    MAPS_LOG_BYTES,
                    MAPS_NUM_TOKEN_SLOTS),
};

static const l2_read_desc_t maps_init_l2_reads_tile0[1] = {
    {MAPS_GLOBAL_B0,
     GLOBAL_INITIALIZER,
     (uint32_t)maps_l2_b0,
     0u,
     MAPS_SUB_B,
     MAPS_COPY_INIT(MAPS_K, MAPS_N)},
};

static const l2_read_desc_t maps_init_l2_reads_tile1[1] = {
    {MAPS_GLOBAL_B1,
     GLOBAL_INITIALIZER,
     (uint32_t)maps_l2_b1,
     0u,
     MAPS_SUB_B,
     MAPS_COPY_INIT(MAPS_K, MAPS_N)},
};

static const l2_read_desc_t maps_init_l2_reads_tile8[1] = {
    {MAPS_GLOBAL_B2,
     GLOBAL_INITIALIZER,
     (uint32_t)maps_l2_b2,
     0u,
     MAPS_SUB_B,
     MAPS_COPY_INIT(MAPS_K, MAPS_N)},
};

static const l2_read_desc_t maps_l2_reads_tile0[2] = {
    {MAPS_GLOBAL_A0,
     GLOBAL_INPUT,
     (uint32_t)maps_l2_a0,
     MAPS_MM0_BYTES,
     MAPS_SUB_A(MAPS_MM0_ROWS),
     MAPS_COPY_INIT(MAPS_MM0_ROWS, MAPS_K)},
    {MAPS_GLOBAL_ZERO_11,
     GLOBAL_INTERMEDIATE,
     (uint32_t)maps_l2_zero_11,
     0u,
     MAPS_SUB_MM(0u, MAPS_MM0_ROWS),
     MAPS_COPY_INIT(MAPS_MM0_ROWS, MAPS_N)},
};

static const l2_read_desc_t maps_l2_reads_tile1[2] = {
    {MAPS_GLOBAL_A1,
     GLOBAL_INPUT,
     (uint32_t)maps_l2_a1,
     MAPS_MM1_BYTES,
     MAPS_SUB_A(MAPS_MM1_ROWS),
     MAPS_COPY_INIT(MAPS_MM1_ROWS, MAPS_K)},
    {MAPS_GLOBAL_ZERO_11,
     GLOBAL_INTERMEDIATE,
     (uint32_t)maps_l2_zero_11,
     0u,
     MAPS_SUB_MM(0u, MAPS_MM1_ROWS),
     MAPS_COPY_INIT(MAPS_MM1_ROWS, MAPS_N)},
};

static const l2_read_desc_t maps_l2_reads_tile8[2] = {
    {MAPS_GLOBAL_A2,
     GLOBAL_INPUT,
     (uint32_t)maps_l2_a2,
     MAPS_MM2_BYTES,
     MAPS_SUB_A(MAPS_MM2_ROWS),
     MAPS_COPY_INIT(MAPS_MM2_ROWS, MAPS_K)},
    {MAPS_GLOBAL_ZERO_10,
     GLOBAL_INTERMEDIATE,
     (uint32_t)maps_l2_zero_10,
     0u,
     MAPS_SUB_MM(0u, MAPS_MM2_ROWS),
     MAPS_COPY_INIT(MAPS_MM2_ROWS, MAPS_N)},
};

#define MAPS_SEND_MM_TO_LOG(trans, src_hart, dst_hart, src_row, dst_row, rows)                     \
    {                                                                                              \
        .transition_id             = (trans),                                                      \
        .src_hartid                = (src_hart),                                                   \
        .dst_hartid                = (dst_hart),                                                   \
        .src                       = MAPS_SUB_MM(src_row, rows),                                   \
        .dst                       = MAPS_SUB_LOG_IN(dst_row, rows),                               \
        .dst_l1_data_base          = MAPS_L1_DATA_OFFSET,                                          \
        .dst_slice_l1_offset_bytes = MAPS_L1_LOG_IN_OFFSET,                                        \
        .dst_slice_slot_bytes      = MAPS_LOG_BYTES,                                               \
        .copy_src                  = MAPS_SUBSLICE_INIT(rows, MAPS_N),                             \
        .copy_dst                  = MAPS_SUBSLICE_INIT(rows, MAPS_N),                             \
        .ready_flag_id             = 0u,                                                           \
    }

static const send_desc_t maps_sends_tile0[1] = {
    MAPS_SEND_MM_TO_LOG(MAPS_TRANS_MM0_LOG0, MAPS_TILE_MM0, MAPS_TILE_LOG0, 0u, 0u, MAPS_MM0_ROWS),
};

static const send_desc_t maps_sends_tile1[2] = {
    MAPS_SEND_MM_TO_LOG(
        MAPS_TRANS_MM1_LOG0, MAPS_TILE_MM1, MAPS_TILE_LOG0, 0u, MAPS_MM0_ROWS, MAPS_MM1_LOG0_ROWS),
    MAPS_SEND_MM_TO_LOG(MAPS_TRANS_MM1_LOG1,
                        MAPS_TILE_MM1,
                        MAPS_TILE_LOG1,
                        MAPS_MM1_LOG0_ROWS,
                        0u,
                        MAPS_MM1_LOG1_ROWS),
};

static const send_desc_t maps_sends_tile8[1] = {
    MAPS_SEND_MM_TO_LOG(
        MAPS_TRANS_MM2_LOG1, MAPS_TILE_MM2, MAPS_TILE_LOG1, 0u, MAPS_MM1_LOG1_ROWS, MAPS_MM2_ROWS),
};

static const recv_desc_t maps_recvs_tile12[2] = {
    {MAPS_TRANS_MM0_LOG0, MAPS_TILE_MM0, MAPS_TILE_LOG0, MAPS_SUB_LOG_IN(0u, MAPS_MM0_ROWS), 0u},
    {MAPS_TRANS_MM1_LOG0,
     MAPS_TILE_MM1,
     MAPS_TILE_LOG0,
     MAPS_SUB_LOG_IN(MAPS_MM0_ROWS, MAPS_MM1_LOG0_ROWS),
     0u},
};

static const recv_desc_t maps_recvs_tile13[2] = {
    {MAPS_TRANS_MM1_LOG1,
     MAPS_TILE_MM1,
     MAPS_TILE_LOG1,
     MAPS_SUB_LOG_IN(0u, MAPS_MM1_LOG1_ROWS),
     0u},
    {MAPS_TRANS_MM2_LOG1,
     MAPS_TILE_MM2,
     MAPS_TILE_LOG1,
     MAPS_SUB_LOG_IN(MAPS_MM1_LOG1_ROWS, MAPS_MM2_ROWS),
     0u},
};

static const recv_desc_t maps_recvs_tile35[2] = {
    {MAPS_TRANS_LOG0_ADD, MAPS_TILE_LOG0, MAPS_TILE_ADD, MAPS_SUB_ADD_IN0, 0u},
    {MAPS_TRANS_LOG1_ADD, MAPS_TILE_LOG1, MAPS_TILE_ADD, MAPS_SUB_ADD_IN1, 0u},
};

#define MAPS_SEND_LOG_TO_ADD(trans, src_hart, src_slice, dst_slice, dst_offset)                    \
    {                                                                                              \
        .transition_id             = (trans),                                                      \
        .src_hartid                = (src_hart),                                                   \
        .dst_hartid                = MAPS_TILE_ADD,                                                \
        .src                       = src_slice,                                                    \
        .dst                       = dst_slice,                                                    \
        .dst_l1_data_base          = MAPS_L1_DATA_OFFSET,                                          \
        .dst_slice_l1_offset_bytes = (dst_offset),                                                 \
        .dst_slice_slot_bytes      = MAPS_LOG_BYTES,                                               \
        .copy_src                  = MAPS_SUBSLICE_INIT(MAPS_LOG_ROWS, MAPS_N),                    \
        .copy_dst                  = MAPS_SUBSLICE_INIT(MAPS_LOG_ROWS, MAPS_N),                    \
        .ready_flag_id             = 0u,                                                           \
    }

static const send_desc_t maps_sends_tile12[1] = {
    MAPS_SEND_LOG_TO_ADD(MAPS_TRANS_LOG0_ADD,
                         MAPS_TILE_LOG0,
                         MAPS_SUB_LOG_OUT,
                         MAPS_SUB_ADD_IN0,
                         MAPS_L1_ADD_IN0_OFFSET),
};

static const send_desc_t maps_sends_tile13[1] = {
    MAPS_SEND_LOG_TO_ADD(MAPS_TRANS_LOG1_ADD,
                         MAPS_TILE_LOG1,
                         MAPS_SUB_LOG_OUT,
                         MAPS_SUB_ADD_IN1,
                         MAPS_L1_ADD_IN1_OFFSET),
};

static const op_desc_t maps_ops_matmul_11[1] = {
    {
        .kind        = OP_MATMUL,
        .num_inputs  = 2u,
        .inputs      = {MAPS_SUB_A(MAPS_MM1_ROWS), MAPS_SUB_B},
        .num_outputs = 1u,
        .outputs     = {MAPS_SUB_MM(0u, MAPS_MM1_ROWS)},
        .params      = {MAPS_MM1_ROWS, MAPS_K, MAPS_N, 0u, 0u, 0u, 0u, 0u},
    },
};

static const op_desc_t maps_ops_matmul_10[1] = {
    {
        .kind        = OP_MATMUL,
        .num_inputs  = 2u,
        .inputs      = {MAPS_SUB_A(MAPS_MM2_ROWS), MAPS_SUB_B},
        .num_outputs = 1u,
        .outputs     = {MAPS_SUB_MM(0u, MAPS_MM2_ROWS)},
        .params      = {MAPS_MM2_ROWS, MAPS_K, MAPS_N, 0u, 0u, 0u, 0u, 0u},
    },
};

static const op_desc_t maps_ops_log[1] = {
    {
        .kind        = OP_LOG,
        .num_inputs  = 1u,
        .inputs      = {MAPS_SUB_LOG_IN(0u, MAPS_LOG_ROWS)},
        .num_outputs = 1u,
        .outputs     = {MAPS_SUB_LOG_OUT},
        .params      = {MAPS_LOG_ROWS, MAPS_N, 0u, 0u, 0u, 0u, 0u, 0u},
    },
};

static const op_desc_t maps_ops_add[1] = {
    {
        .kind        = OP_ADD,
        .num_inputs  = 2u,
        .inputs      = {MAPS_SUB_ADD_IN0, MAPS_SUB_ADD_IN1},
        .num_outputs = 1u,
        .outputs     = {MAPS_SUB_FINAL},
        .params      = {MAPS_LOG_ROWS, MAPS_N, 0u, 0u, 0u, 0u, 0u, 0u},
    },
};

static const l2_write_desc_t maps_l2_writes_tile35[1] = {
    {MAPS_GLOBAL_FINAL,
     GLOBAL_OUTPUT,
     MAPS_SUB_FINAL,
     (uint32_t)maps_l2_final,
     MAPS_LOG_BYTES,
     MAPS_COPY_INIT(MAPS_LOG_ROWS, MAPS_N)},
};

static inline void maps_generated_clear_mailbox(uint32_t hartid)
{
    volatile uint32_t *flags = (volatile uint32_t *)(get_l1_base(hartid) + MAPS_READY_FLAGS_OFFSET);

    for (uint32_t i = 0; i < MAPS_READY_FLAGS_WORDS; ++i) {
        flags[i] = 0u;
    }
}

static inline void maps_generated_log(uint32_t dst_addr, uint32_t src_addr, uint32_t elems)
{
    volatile uint16_t *dst       = (volatile uint16_t *)dst_addr;
    volatile const uint16_t *src = (volatile const uint16_t *)src_addr;

    for (uint32_t i = 0; i < elems; ++i) {
        dst[i] = maps_generated_log_lut[maps_generated_value_index(src[i])];
    }
}

static inline void
maps_generated_add(uint32_t dst_addr, uint32_t lhs_addr, uint32_t rhs_addr, uint32_t elems)
{
    volatile uint16_t *dst       = (volatile uint16_t *)dst_addr;
    volatile const uint16_t *lhs = (volatile const uint16_t *)lhs_addr;
    volatile const uint16_t *rhs = (volatile const uint16_t *)rhs_addr;

    for (uint32_t i = 0; i < elems; ++i) {
        dst[i] = maps_generated_log_add_lut[maps_generated_log_index(lhs[i])]
                                           [maps_generated_log_index(rhs[i])];
    }
}

static inline int
maps_generated_execute_op(const tile_plan_t *plan, const op_desc_t *op, uint32_t slot, void *user)
{
    maps_generated_runtime_t *runtime = (maps_generated_runtime_t *)user;
    (void)runtime;

    if (op->kind == OP_MATMUL) {
        uint32_t a   = local_subslice_addr(plan, &op->inputs[0], slot);
        uint32_t b   = local_subslice_addr(plan, &op->inputs[1], slot);
        uint32_t out = local_subslice_addr(plan, &op->outputs[0], slot);

#if MAPS_HANDWRITTEN_USE_REDMULE
        maps_trace_event(plan, slot, slot, "redmule-start", op->params[0]);
        redmule_gemm(runtime->redmule_ctrl,
                     a,
                     b,
                     out,
                     (uint16_t)op->params[0],
                     (uint16_t)op->params[1],
                     (uint16_t)op->params[2]);
        eu_redmule_wait(runtime->eu_ctrl, MAPS_WAIT_MODE);
        maps_trace_event(plan, slot, slot, "redmule-done", op->params[0]);
#else
        (void)b;
        maps_generated_copy_identity_matmul(out, a, op->params[0] * op->params[2]);
#endif
        return 0;
    }

    if (op->kind == OP_LOG) {
        uint32_t in  = local_subslice_addr(plan, &op->inputs[0], slot);
        uint32_t out = local_subslice_addr(plan, &op->outputs[0], slot);

        maps_generated_log(out, in, op->params[0] * op->params[1]);
        return 0;
    }

    if (op->kind == OP_ADD) {
        uint32_t lhs = local_subslice_addr(plan, &op->inputs[0], slot);
        uint32_t rhs = local_subslice_addr(plan, &op->inputs[1], slot);
        uint32_t out = local_subslice_addr(plan, &op->outputs[0], slot);

        maps_generated_add(out, lhs, rhs, op->params[0] * op->params[1]);
        return 0;
    }

    return -1;
}

static inline uint32_t
maps_generated_fill_plan(tile_plan_t *plan, uint32_t hartid, maps_generated_runtime_t *runtime)
{
    *plan = (tile_plan_t){
        .hartid            = hartid,
        .l1_data_base      = get_l1_base(hartid) + MAPS_L1_DATA_OFFSET,
        .ready_flags_base  = MAPS_READY_FLAGS_OFFSET,
        .ready_flags_count = MAPS_READY_FLAGS_WORDS,
        .num_token_slots   = MAPS_NUM_TOKEN_SLOTS,
        .execute_op        = maps_generated_execute_op,
        .execute_op_user   = runtime,
    };

    switch (hartid) {
    case MAPS_TILE_MM0:
        plan->num_slices        = 3u;
        plan->slices            = maps_matmul_slices_11;
        plan->num_init_l2_reads = 1u;
        plan->init_l2_reads     = maps_init_l2_reads_tile0;
        plan->num_l2_reads      = 2u;
        plan->l2_reads          = maps_l2_reads_tile0;
        plan->num_ops           = 1u;
        plan->ops               = maps_ops_matmul_11;
        plan->num_sends         = 1u;
        plan->sends             = maps_sends_tile0;
        return 1u;
    case MAPS_TILE_MM1:
        plan->num_slices        = 3u;
        plan->slices            = maps_matmul_slices_11;
        plan->num_init_l2_reads = 1u;
        plan->init_l2_reads     = maps_init_l2_reads_tile1;
        plan->num_l2_reads      = 2u;
        plan->l2_reads          = maps_l2_reads_tile1;
        plan->num_ops           = 1u;
        plan->ops               = maps_ops_matmul_11;
        plan->num_sends         = 2u;
        plan->sends             = maps_sends_tile1;
        return 1u;
    case MAPS_TILE_MM2:
        plan->num_slices        = 3u;
        plan->slices            = maps_matmul_slices_10;
        plan->num_init_l2_reads = 1u;
        plan->init_l2_reads     = maps_init_l2_reads_tile8;
        plan->num_l2_reads      = 2u;
        plan->l2_reads          = maps_l2_reads_tile8;
        plan->num_ops           = 1u;
        plan->ops               = maps_ops_matmul_10;
        plan->num_sends         = 1u;
        plan->sends             = maps_sends_tile8;
        return 1u;
    case MAPS_TILE_LOG0:
        plan->num_slices = 2u;
        plan->slices     = maps_log_slices;
        plan->num_recvs  = 2u;
        plan->recvs      = maps_recvs_tile12;
        plan->num_ops    = 1u;
        plan->ops        = maps_ops_log;
        plan->num_sends  = 1u;
        plan->sends      = maps_sends_tile12;
        return 1u;
    case MAPS_TILE_LOG1:
        plan->num_slices = 2u;
        plan->slices     = maps_log_slices;
        plan->num_recvs  = 2u;
        plan->recvs      = maps_recvs_tile13;
        plan->num_ops    = 1u;
        plan->ops        = maps_ops_log;
        plan->num_sends  = 1u;
        plan->sends      = maps_sends_tile13;
        return 1u;
    case MAPS_TILE_ADD:
        plan->num_slices    = 3u;
        plan->slices        = maps_add_slices;
        plan->num_recvs     = 2u;
        plan->recvs         = maps_recvs_tile35;
        plan->num_ops       = 1u;
        plan->ops           = maps_ops_add;
        plan->num_l2_writes = 1u;
        plan->l2_writes     = maps_l2_writes_tile35;
        return 1u;
    default:
        return 0u;
    }
}

static inline uint32_t maps_generated_check_output(void)
{
    uint32_t errors        = 0u;
    const uint16_t *final  = (const uint16_t *)maps_l2_final;
    const uint16_t *golden = (const uint16_t *)maps_l2_golden;

    for (uint32_t i = 0; i < MAPS_NUM_TOKENS * MAPS_LOG_ELEMS; ++i) {
        if (final[i] != golden[i]) {
            errors++;
        }
    }

    return errors;
}

#endif /* MAPS_HANDWRITTEN_TEST_H */
