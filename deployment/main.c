#include <stdint.h>

#include "eventunit.h"
#include "fsync.h"
#include "idma.h"
#include "tile.h"

#include "kernels_compare_utils.h"
#include "kernels_profiling_utils.h"

#include "network.h"
#include "data.h"

#ifdef ENABLE_NODE_PROFILING

/* Phase counters in the per-tile .tile_bss (L1) */
uint32_t prof_cyc_alloc __attribute__((section(".tile_bss")));
uint32_t prof_cyc_data_in __attribute__((section(".tile_bss")));
uint32_t prof_cyc_compute __attribute__((section(".tile_bss")));
uint32_t prof_cyc_data_out __attribute__((section(".tile_bss")));
uint32_t prof_cyc_sync __attribute__((section(".tile_bss")));

/* The snapshots stay in shared L2, a single tile reads every tile's entries
 * to print the report. One extra slot closes the last node. */
uint32_t prof_node[(PROF_N_NODES + 1) * PROF_N_TILES * PROF_NODE_FIELDS];

/* Called by a single tile after the barrier: every tile has written its
 * snapshots and the tables live in shared L2, so one tile can print the whole
 * report and the lines come out ordered instead of interleaved.
 *
 * Walks the network layer by layer: the nodes of a layer, then that layer's
 * per-tile recap, and a final recap at the end. The layer and node names are
 * emitted by generate_with_spatz.py into the generated network.c. */
static void prof_report(void)
{
    static uint32_t total[PROF_N_TILES][PROF_NODE_FIELDS];
    static uint32_t layer[PROF_N_TILES][PROF_NODE_FIELDS];

    for (uint32_t h = 0; h < PROF_N_TILES; h++)
        for (uint32_t f = 0; f < PROF_NODE_FIELDS; f++)
            total[h][f] = 0;

    for (uint32_t l = 0; l < PROF_N_LAYERS; l++) {
        for (uint32_t h = 0; h < PROF_N_TILES; h++)
            for (uint32_t f = 0; f < PROF_NODE_FIELDS; f++)
                layer[h][f] = 0;

        printf("[PROF] ### Profiling for Layer %d (%s) ###\n", l, prof_layer_name[l]);

        for (uint32_t n = 0; n < PROF_N_NODES; n++) {
            if (prof_node_layer[n] != l)
                continue;

            for (uint32_t h = 0; h < PROF_N_TILES; h++) {
                uint32_t *cur  = PROF_NODE_AT(n, h);
                uint32_t *next = PROF_NODE_AT(n + 1, h);
                uint32_t d[PROF_NODE_FIELDS];

                /* A node's cost is the delta between consecutive snapshots. */
                for (uint32_t f = 0; f < PROF_NODE_FIELDS; f++) {
                    d[f] = next[f] - cur[f];
                    layer[h][f] += d[f];
                    total[h][f] += d[f];
                }

                printf("[PROF] hid=%d layer_idx=%d node_idx=%d node=%s op=%s "
                       "node_cycles=%d alloc=%d in=%d cmp=%d out=%d snc=%d\n",
                       h,
                       l,
                       n,
                       prof_node_name[n],
                       prof_node_op[n],
                       d[0],
                       d[1],
                       d[2],
                       d[3],
                       d[4],
                       d[5]);
            }
        }

        printf("[PROF] Layer %d recap:\n", l);
        for (uint32_t h = 0; h < PROF_N_TILES; h++)
            printf("[PROF] hid=%d layer_idx=%d layer_cycles=%d "
                   "alloc=%d in=%d cmp=%d out=%d snc=%d\n",
                   h,
                   l,
                   layer[h][0],
                   layer[h][1],
                   layer[h][2],
                   layer[h][3],
                   layer[h][4],
                   layer[h][5]);
    }

    printf("[PROF] ### Final recap ###\n");
    for (uint32_t h = 0; h < PROF_N_TILES; h++)
        printf("[PROF] hid=%d total_cycles=%d "
               "alloc=%d in=%d cmp=%d out=%d snc=%d\n",
               h,
               total[h][0],
               total[h][1],
               total[h][2],
               total[h][3],
               total[h][4],
               total[h][5]);
}
#endif

int init_fsync(fsync_controller_t *fsync_ctrl)
{
    fsync_config_t fsync_cfg;

    fsync_init();

    return 0;
}

int init_idma()
{
    idma_config_t idma_cfg;

    idma_init();

    return 0;
}

int init_event_unit()
{
    eu_config_t eu_cfg;

    eu_init();
    eu_spatz_init(0);
    eu_fsync_init(0);
    eu_idma_init(0);

    return 0;
}

int init_spatz()
{
    spatz_init(SPATZ_BINARY_START);

    return 0;
}

int deinit_spatz()
{
    spatz_clk_dis();

    return 0;
}

void sync()
{
    fsync_sync_global();
    eu_fsync_wait(WFE);
}

void input_copy()
{
    for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
        memcpy(DeeployNetwork_inputs[buf], inputs[buf], DeeployNetwork_inputs_bytes[buf]);
    }
}

int check_result()
{
    int n_mismatch = 0;

    for (uint32_t i = 0; i < OUTPUTS_NUM; i++)
        n_mismatch += compare_fp16_bitwise((const float16 *)DeeployNetwork_outputs[i],
                                           (const float16 *)outputs[i],
                                           outputs_size[i]);

    return n_mismatch;
}

int main(void)
{
    uint32_t cycle_start;
    uint32_t cycle_stop;
    int hid;
    int ret;

    hid = get_hartid();

    ret = init_fsync();
    if (ret) {
        printf("[CV32 (%d)] Fsync initialization failed with errno: %d\n", HID, ret);
        return ret;
    }

    ret = init_event_unit();
    if (ret) {
        printf("[CV32 (%d)] Event Unit initialization failed with errno: %d\n", HID, ret);
        return ret;
    }

    ret = init_idma();
    if (ret) {
        printf("[CV32 (%d)] iDMA initialization failed with errno: %d\n", HID, ret);
        return ret;
    }

    ret = init_spatz();
    if (ret) {
        printf("[CV32 (%d)] Spatz initialization failed with errno: %d\n", HID, ret);
        return ret;
    }

    sync();

    if (hid == 0)
        InitNetwork();

    sync();

    if (hid == 0)
        input_copy();

    sync();

    perf_enable_cycles();

    cycle_start = perf_get_cycles();
    RunNetwork();
    cycle_stop = perf_get_cycles();

#ifdef ENABLE_NODE_PROFILING
    /* Closes the last node */
    PROF_NODE(PROF_N_NODES);
#endif

    sync(&fsync_ctrl, &eu_ctrl);

#ifdef ENABLE_NODE_PROFILING
    if (hid == 0)
        prof_report();

    /* Hold the others until the report is out */
    sync(&fsync_ctrl, &eu_ctrl);
#endif

    printf("[CV32 (%d)] Run completed in %d cycles\n", hid, cycle_stop - cycle_start);

    sync();

    ret = deinit_spatz();
    if (ret) {
        printf("[CV32 (%d)] Spatz deinitialization failed with errno: %d\n", HID, ret);
        return ret;
    }

    if (hid == 0) {
        ret = check_result();
        printf("[CV32] Test completed with %d mismatches\n", ret);
    }

    sync();

    return ret;
}
