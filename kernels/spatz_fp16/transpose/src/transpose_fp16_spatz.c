/*
 * Transpose, fp16, one shard of one axis per tile.
 *
 * Which axis gets sharded is the whole story here. The obvious choice - the input's axis
 * 0 - only works when the perm leaves that axis in place: the tile's slice of the *output*
 * is then a contiguous run too, so both transfers are plain memcpys. As soon as perm[0]
 * != 0 that stops being true, and a transpose whose leading axis the perm moves (a plain
 * 2-D transpose, an NCHW <-> NHWC swap, a head split) would have to run entirely on tile
 * 0 - and would try to stage the whole tensor in one tile's L1 while doing it.
 *
 * So the tile shards whichever of the input's axis 0 and the output's axis 0 is longer,
 * and the side that is then non-contiguous is moved with a 2-D transfer instead of a 1-D
 * one:
 *
 *   input axis 0  ->  contiguous in,  rectangle out (the reduced axis is q, perm[q] == 0)
 *   output axis 0 ->  rectangle in,   contiguous out (the reduced axis is p0 = perm[0])
 *
 * Either way L1 holds a sub-tensor with exactly one axis shortened to iter_len, so the
 * Spatz task needs no change at all: it is already rank- and perm-general, and handing it
 * the shortened shapes (and strides recomputed from them) is enough.
 */
#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "kernels_dma_utils.h"

#include "transpose_fp16_spatz.h"
#include "transpose_fp16_spatz_params.h"
#include "transpose_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "transpose_fp16_spatz"
#define TRANSPOSE_MAX_RANK (8)

/* One side's transfer, as reps rows of len elements with a stride on each side. */
typedef struct {
    uint32_t off;        /* element offset of the tile's rectangle in L2 */
    uint32_t len;        /* elements per row                             */
    uint32_t l2_stride;  /* elements between rows in L2                  */
    uint32_t l1_stride;  /* elements between rows in L1                  */
    uint32_t reps;       /* rows (1 when the run is contiguous)          */
} transpose_move_t;

typedef struct {
    transpose_move_t in;
    transpose_move_t out;
    uint32_t in_shape_r[TRANSPOSE_MAX_RANK];   /* shapes with the sharded axis shortened */
    uint32_t out_shape_r[TRANSPOSE_MAX_RANK];
    uint32_t shard_elems;                      /* elements per L1 buffer                 */
    uint32_t iter_len;                         /* extent of the sharded axis on this tile */
} transpose_geom_t;

static inline uint32_t prod_range(const uint32_t *shape, uint32_t from, uint32_t to)
{
    uint32_t p = 1;

    for (uint32_t i = from; i < to; i++)
        p *= shape[i];

    return p;
}

/*
 * Pick the axis to shard, split it, and work out both transfers. Returns 0 on success,
 * EINVAL if the rank is out of range.
 */
static int plan_shards(transpose_geom_t *g, const uint32_t *perm, const uint32_t *in_shape, const uint32_t *out_shape, uint32_t rank)
{
    uint32_t axis_in;      /* input axis being sharded  */
    uint32_t axis_out;     /* output axis being sharded */
    uint32_t extent;
    uint32_t iter_start;
    uint32_t iter_len;
    uint32_t shard;
    uint32_t left;
    uint32_t inner;
    uint32_t total;

    if (rank == 0 || rank > TRANSPOSE_MAX_RANK)
        return EINVAL;

    /*
     * Shard the input's axis 0 when it is at least as long as the output's, otherwise the
     * output's. perm[0] == 0 makes the two the same axis, and the input branch below then
     * degenerates to two contiguous transfers.
     */
    if (in_shape[0] >= out_shape[0]) {
        axis_in = 0;
        for (axis_out = 0; axis_out < rank && perm[axis_out] != 0; axis_out++)
            ;
        if (axis_out == rank)
            return EINVAL;
    } else {
        axis_out = 0;
        axis_in = perm[0];
        if (axis_in >= rank)
            return EINVAL;
    }

    extent = in_shape[axis_in];

    shard = extent / NUM_HARTS;
    left  = extent % NUM_HARTS;

    iter_start = HID * shard + (HID < left ? HID : left);
    iter_len   = shard + (HID < left ? 1 : 0);

    for (uint32_t i = 0; i < rank; i++) {
        g->in_shape_r[i]  = in_shape[i];
        g->out_shape_r[i] = out_shape[i];
    }
    g->in_shape_r[axis_in]   = iter_len;
    g->out_shape_r[axis_out] = iter_len;

    total = prod_range(in_shape, 0, rank);
    g->shard_elems = (extent != 0) ? (total / extent) * iter_len : 0;
    g->iter_len    = iter_len;

    /* Input side: rows of the rectangle that keeps axis_in inside [iter_start, +iter_len). */
    inner = prod_range(in_shape, axis_in + 1, rank);
    g->in.off       = iter_start * inner;
    g->in.len       = iter_len * inner;
    g->in.l2_stride = in_shape[axis_in] * inner;
    g->in.l1_stride = iter_len * inner;
    g->in.reps      = prod_range(in_shape, 0, axis_in);

    /* Output side: the mirror image about axis_out. */
    inner = prod_range(out_shape, axis_out + 1, rank);
    g->out.off       = iter_start * inner;
    g->out.len       = iter_len * inner;
    g->out.l2_stride = out_shape[axis_out] * inner;
    g->out.l1_stride = iter_len * inner;
    g->out.reps      = prod_range(out_shape, 0, axis_out);

    return 0;
}

static int alloc_l1(void **params, const transpose_geom_t *g, uint32_t rank)
{
    volatile transpose_fp16_spatz_params_t *trans_params;

    uintptr_t shard_input;
    uintptr_t shard_output;
    uintptr_t out_shape_l1;
    uintptr_t in_strides_l1;
    uintptr_t perm_l1;
    uintptr_t coord_l1;
    uint32_t stride;

    l1_alloc_init();

    trans_params = l1_alloc(sizeof(transpose_fp16_spatz_params_t));
    if (!trans_params)
        return ENOMEM;

    shard_input = (uintptr_t) l1_alloc(g->shard_elems * sizeof(float16));
    if (!shard_input)
        return ENOMEM;

    shard_output = (uintptr_t) l1_alloc(g->shard_elems * sizeof(float16));
    if (!shard_output)
        return ENOMEM;

    out_shape_l1 = (uintptr_t) l1_alloc(rank * sizeof(uint32_t));
    if (!out_shape_l1)
        return ENOMEM;

    in_strides_l1 = (uintptr_t) l1_alloc(rank * sizeof(uint32_t));
    if (!in_strides_l1)
        return ENOMEM;

    perm_l1 = (uintptr_t) l1_alloc(rank * sizeof(uint32_t));
    if (!perm_l1)
        return ENOMEM;

    coord_l1 = (uintptr_t) l1_alloc(rank * sizeof(uint32_t));
    if (!coord_l1)
        return ENOMEM;

    trans_params->shard_input     = shard_input;
    trans_params->shard_output    = shard_output;
    trans_params->out_shape       = out_shape_l1;
    trans_params->in_strides      = in_strides_l1;
    trans_params->perm            = perm_l1;
    trans_params->coord           = coord_l1;
    trans_params->rank            = rank;
    trans_params->iteration_start = 0;
    /* The task walks the *staged* output, whose leading axis may be the full one. */
    trans_params->iteration_len   = g->out_shape_r[0];

    for (uint32_t i = 0; i < rank; i++)
        mmio32(out_shape_l1 + i * sizeof(uint32_t)) = g->out_shape_r[i];

    /* Strides of the sub-tensor actually staged, not of the whole input. */
    stride = 1;
    for (int i = (int) rank - 1; i >= 0; i--) {
        mmio32(in_strides_l1 + i * sizeof(uint32_t)) = stride;
        stride *= g->in_shape_r[i];
    }

    *params = (void *) trans_params;

    return 0;
}

static int init_input_params(void *params, kdma_t *d, const transpose_geom_t *g, const float16 *input, const uint32_t *perm, uint32_t rank)
{
    volatile transpose_fp16_spatz_params_t *trans_params;
    uintptr_t perm_base;

    trans_params = (volatile transpose_fp16_spatz_params_t *) params;
    perm_base    = trans_params->perm;

    /* reps == 1 is the contiguous case; kdma_in_2d covers it, but say so plainly. */
    if (g->in.reps == 1)
        kdma_in(d,
                (uintptr_t)(input + g->in.off),
                trans_params->shard_input,
                g->in.len * sizeof(float16));
    else
        kdma_in_2d(d,
                   (uintptr_t)(input + g->in.off),
                   trans_params->shard_input,
                   g->in.len * sizeof(float16),
                   g->in.l2_stride * sizeof(float16),
                   g->in.l1_stride * sizeof(float16),
                   g->in.reps);

    for (uint32_t i = 0; i < rank; i++)
        mmio32(perm_base + i * sizeof(uint32_t)) = perm[i];

    return 0;
}

static int offload_spatz_task(kdma_t *d, void *params)
{
    int ret;

    spatz_run_task_with_params(TRANSPOSE_FP16_SPATZ_TASK, (uint32_t)params);

    ret = eu_spatz_wait(&d->eu, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n", HID, KERNEL_NAME, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

/* Mirror of the staging, about the output's sharded axis. */
static int store_result(void *params, kdma_t *d, const transpose_geom_t *g, float16 *output)
{
    volatile transpose_fp16_spatz_params_t *trans_params;

    trans_params = (volatile transpose_fp16_spatz_params_t *) params;

    if (g->out.reps == 1)
        kdma_out(d,
                 (uintptr_t)(output + g->out.off),
                 trans_params->shard_output,
                 g->out.len * sizeof(float16));
    else
        kdma_out_2d(d,
                    (uintptr_t)(output + g->out.off),
                    trans_params->shard_output,
                    g->out.len * sizeof(float16),
                    g->out.l2_stride * sizeof(float16),
                    g->out.l1_stride * sizeof(float16),
                    g->out.reps);

    return 0;
}

/*
 * `iterations` is kept for source compatibility with the callers that pass in_shape[0];
 * the shard is worked out from the shapes and the perm now, since the axis that is worth
 * sharding is not always axis 0.
 */
void MAGIA_transpose_fp16_spatz(const float16 *input, float16 *output, uint32_t *perm, uint32_t *in_shape, uint32_t *out_shape, uint32_t rank, uint32_t iterations)
{
    volatile transpose_fp16_spatz_params_t *params;
    transpose_geom_t g;
    kdma_t d;
    int ret;

    (void) iterations;

    ret = plan_shards(&g, perm, in_shape, out_shape, rank);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Unsupported rank %d\n", HID, KERNEL_NAME, (int)rank);
        return;
    }

    /* More tiles than the sharded axis is long: nothing to stage, run or write back. */
    if (g.iter_len == 0 || g.shard_elems == 0)
        return;

    ret = alloc_l1((void **)&params, &g, rank);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    kdma_open(&d);

    ret = init_input_params((void *)params, &d, &g, input, perm, rank);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = offload_spatz_task(&d, (void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *)params, &d, &g, output);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
