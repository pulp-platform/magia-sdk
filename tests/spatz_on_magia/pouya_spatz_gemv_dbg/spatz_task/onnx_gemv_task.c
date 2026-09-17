#include "tile.h"
#include "onnx_gemv_params.h"
#include "data.h" /* debug-only: golden A/x/C/G + DEBUG_PARTIAL/DEBUG_ROWSUM.
                    * Available because CMakeLists.txt's add_spatz_task()
                    * already puts test_data on this target's INCLUDE_DIRS.
                    * Remove this include once debugging is done. */

/*
 * GEMV: Y = alpha * A * x + beta * C
 * (see original onnx_gemv_task.c for the vectorization rationale)
 *
 * ============================== HOW TO READ THE OUTPUT ==============================
 * Every fp16 value is printed as an integer scaled by 1000 (this platform's
 * printf only safely supports %d). Example: a printed value of 2500 means
 * the real fp16 value is 2.500; -1500 means -1.500.
 *
 * Every single "asm volatile(...)" instruction below prints its own block:
 *   [INSTR] <the actual instruction>
 *     purpose: <what it's supposed to do, in plain words>
 *     input:   <scalar operands going into it>
 *     output:  <what it produced, per output lane if it's a vector op>
 *     expected:<what it should have produced, per lane>
 *     verdict: MATCH or *** MISMATCH ***
 * =======================================================================================
 */

#define DBG_BUF_LEN 64

/* Scratch buffer that vector register dumps are vse16.v'd into so the
 * scalar core can read them lane-by-lane.
 */
static _Float16 dbg_buf[DBG_BUF_LEN];

/* Only %d is safe to printf on this platform, so every fp16 value is
 * converted to an integer scaled by 1000 before printing.
 */
static inline int32_t fp16_to_milli(_Float16 v)
{
    return (int32_t)((float)v * 1000.0f);
}

/* Prints dbg_buf[0..vl) against expected[0..vl) (expected given as plain
 * integers, converted to milli inside). row_base is the first output row
 * this chunk covers, so dbg_buf lane 'l' == output row (row_base + l).
 */
static void dump_vector_lanes(unsigned int vl, unsigned int row_base,
                               const int32_t *expected)
{
    for (unsigned int lane = 0; lane < vl; lane++) {
        int32_t actual_milli   = fp16_to_milli(dbg_buf[lane]);
        int32_t expected_milli = expected[lane] * 1000;
        int32_t diff           = actual_milli - expected_milli;

        printf("    row=%d  actual=%d  expected=%d  diff=%d\n",
               (int)(row_base + lane), actual_milli, expected_milli, diff);

        if (diff != 0) {
            printf("    *** MISMATCH at row=%d ***\n", (int)(row_base + lane));
        }
    }
}

static void gemv_debug(const _Float16 *A_dev,
                        const _Float16 *x_dev,
                        const _Float16 *C_dev,
                        _Float16 *Y_dev,
                        _Float16 alpha,
                        _Float16 beta,
                        const size_t dim_M,
                        const size_t dim_K)
{
    register _Float16 ZERO asm("fs0") = 0.0f;
    const _Float16 *A_col;
    const _Float16 *C_seg;
    _Float16 *Y_seg;
    size_t stride;
    size_t avl;
    size_t vl;
    size_t VLMAX_MEASURED;
    size_t probe_avl;
    int32_t expected_lane[DBG_BUF_LEN];

    printf("\n================ GEMV INSTRUCTION-LEVEL TRACE ================\n");
    printf("All fp16 values below are printed as integers scaled by 1000.\n");
    printf("Example: printed 2500 means the real fp16 value is 2.500\n");
    printf("================================================================\n");

    if (dim_M > DBG_BUF_LEN) {
        printf("!! dim_M=%d exceeds DBG_BUF_LEN=%d -- aborting trace, grow "
               "DBG_BUF_LEN before using a dataset this large\n",
               (int)dim_M, (int)DBG_BUF_LEN);
        return;
    }

    stride = dim_K * sizeof(_Float16);

    /* ---- ONE-TIME PROBE: measure hardware VLMAX for e16,m8 ---- */
    probe_avl = (size_t)-1; /* request an unbounded AVL to force VLMAX */
    asm volatile("vsetvli %0, %1, e16, m8, ta, ma"
                 : "=r"(VLMAX_MEASURED)
                 : "r"(probe_avl));
    printf("\n[INSTR] vsetvli VLMAX_MEASURED, -1, e16, m8, ta, ma   (one-time probe)\n");
    printf("    purpose: measure the hardware VLMAX for e16 elements at LMUL=m8\n");
    printf("    input:   avl=(all-ones, i.e. an unbounded request)\n");
    printf("    output:  VLMAX_MEASURED=%d\n", (int)VLMAX_MEASURED);
    printf("    (this value is used below as the 'expected vl' whenever avl >= VLMAX)\n");

    for (unsigned int m = 0; m < dim_M; m += vl) {
        avl = dim_M - m;

        printf("\n---------------- row block starting m=%d ----------------\n", (int)m);

        /* ================= vsetvli ================= */
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
        {
            size_t expected_vl = (avl < VLMAX_MEASURED) ? avl : VLMAX_MEASURED;
            printf("\n[INSTR] vsetvli vl, avl, e16, m8, ta, ma\n");
            printf("    purpose: request vl active lanes for avl=%d remaining rows\n", (int)avl);
            printf("    input:   avl=%d\n", (int)avl);
            printf("    output:  vl=%d\n", (int)vl);
            printf("    expected: vl=%d  (= min(avl, VLMAX=%d))\n",
                   (int)expected_vl, (int)VLMAX_MEASURED);
            if (vl == expected_vl)
                printf("    verdict: MATCH\n");
            else
                printf("    verdict: *** MISMATCH ***\n");
        }

        C_seg = C_dev + m;
        Y_seg = Y_dev + m;

        /* ================= vfmv.v.f (init accumulator to 0) ================= */
        asm volatile("vfmv.v.f v0, %0" ::"f"(ZERO));
        asm volatile("vse16.v v0, (%0)" ::"r"(dbg_buf) : "memory");
        printf("\n[INSTR] vfmv.v.f v0, fs0        (fs0 = 0.0)\n");
        printf("    purpose: initialize accumulator v0 to zero for this row block\n");
        printf("    input:   fs0=0\n");
        printf("    output:  v0[lane] for lane=0..%d (per-row below)\n", (int)vl - 1);
        for (unsigned int lane = 0; lane < vl; lane++)
            expected_lane[lane] = 0;
        printf("    expected: 0 for every row\n");
        dump_vector_lanes(vl, m, expected_lane);

        if (alpha != 0.0f) {
            for (unsigned int k = 0; k < dim_K; k++) {
                /* ================= vlse16.v (strided load of A column k) ================= */
                A_col = A_dev + (m * dim_K + k);
                asm volatile("vlse16.v v16, (%0), %1" ::"r"(A_col), "r"(stride));
                asm volatile("vse16.v v16, (%0)" ::"r"(dbg_buf) : "memory");
                printf("\n[INSTR] vlse16.v v16, (A_col), stride    (k=%d)\n", (int)k);
                printf("    purpose: strided load of column k=%d of A, one element per row\n", (int)k);
                printf("    input:   k=%d, stride_bytes=%d\n", (int)k, (int)stride);
                printf("    output:  v16[lane] for lane=0..%d (per-row below)\n", (int)vl - 1);
                for (unsigned int lane = 0; lane < vl; lane++)
                    expected_lane[lane] = (int32_t)A[(m + lane) * dim_K + k];
                printf("    expected: A[row][%d] for each row\n", (int)k);
                dump_vector_lanes(vl, m, expected_lane);

                /* ================= vfmacc.vf (accumulate x[k] * A column) ================= */
                asm volatile("vfmacc.vf v0, %0, v16" ::"f"(*(x_dev + k)));
                asm volatile("vse16.v v0, (%0)" ::"r"(dbg_buf) : "memory");
                printf("\n[INSTR] vfmacc.vf v0, x[k], v16          (k=%d)\n", (int)k);
                printf("    purpose: v0 += x[%d] * v16   (running dot product so far)\n", (int)k);
                printf("    input:   x[%d]=%d\n", (int)k, fp16_to_milli(*(x_dev + k)));
                printf("    output:  v0[lane] for lane=0..%d (per-row below)\n", (int)vl - 1);
                for (unsigned int lane = 0; lane < vl; lane++)
                    expected_lane[lane] = DEBUG_PARTIAL[m + lane][k];
                printf("    expected: sum of A[row][0..%d] * x[0..%d] for each row\n", (int)k, (int)k);
                dump_vector_lanes(vl, m, expected_lane);
            }

            /* ================= vfmul.vf (scale by alpha) ================= */
            asm volatile("vfmul.vf v0, v0, %0" ::"f"(alpha));
            asm volatile("vse16.v v0, (%0)" ::"r"(dbg_buf) : "memory");
            printf("\n[INSTR] vfmul.vf v0, v0, alpha\n");
            printf("    purpose: scale the completed dot product by alpha\n");
            printf("    input:   alpha=%d\n", fp16_to_milli(alpha));
            printf("    output:  v0[lane] for lane=0..%d (per-row below)\n", (int)vl - 1);
            for (unsigned int lane = 0; lane < vl; lane++)
                expected_lane[lane] = DEBUG_ROWSUM[m + lane];
            printf("    expected: alpha * (A@x)[row] for each row\n");
            dump_vector_lanes(vl, m, expected_lane);
        }

        if (beta != 0.0f) {
            /* ================= vle16.v (contiguous load of C) ================= */
            asm volatile("vle16.v v16, (%0)" ::"r"(C_seg));
            asm volatile("vse16.v v16, (%0)" ::"r"(dbg_buf) : "memory");
            printf("\n[INSTR] vle16.v v16, (C_seg)\n");
            printf("    purpose: contiguous load of C[m..m+vl-1]\n");
            printf("    input:   (no scalar operand)\n");
            printf("    output:  v16[lane] for lane=0..%d (per-row below)\n", (int)vl - 1);
            for (unsigned int lane = 0; lane < vl; lane++)
                expected_lane[lane] = (int32_t)C[m + lane];
            printf("    expected: C[row] for each row\n");
            dump_vector_lanes(vl, m, expected_lane);

            /* ================= vfmacc.vf (accumulate beta * C) ================= */
            asm volatile("vfmacc.vf v0, %0, v16" ::"f"(beta));
            asm volatile("vse16.v v0, (%0)" ::"r"(dbg_buf) : "memory");
            printf("\n[INSTR] vfmacc.vf v0, beta, v16\n");
            printf("    purpose: v0 += beta * C   (this produces the final result)\n");
            printf("    input:   beta=%d\n", fp16_to_milli(beta));
            printf("    output:  v0[lane] for lane=0..%d (per-row below)\n", (int)vl - 1);
            for (unsigned int lane = 0; lane < vl; lane++)
                expected_lane[lane] = (int32_t)G[m + lane];
            printf("    expected: Y[row] = alpha*(A@x)[row] + beta*C[row] for each row\n");
            dump_vector_lanes(vl, m, expected_lane);
        }

        /* ================= vse16.v (store final result to memory) ================= */
        asm volatile("vse16.v v0, (%0)" ::"r"(Y_seg) : "memory");
        printf("\n[INSTR] vse16.v v0, (Y_seg)\n");
        printf("    purpose: store the final v0 accumulator out to Y in memory\n");
        printf("    input:   (no scalar operand)\n");
        printf("    output:  Y[row] read back from memory, for lane=0..%d (per-row below)\n", (int)vl - 1);
        for (unsigned int lane = 0; lane < vl; lane++)
            dbg_buf[lane] = Y_seg[lane]; /* read back what actually landed in memory */
        for (unsigned int lane = 0; lane < vl; lane++)
            expected_lane[lane] = (int32_t)G[m + lane];
        printf("    expected: G[row] (golden) for each row\n");
        dump_vector_lanes(vl, m, expected_lane);
    }

    printf("\n================ END GEMV INSTRUCTION-LEVEL TRACE ================\n");
}

/* Unmodified transposed-A path, kept only so onnx_gemv_task() still
 * compiles/works if TRANS_A is ever flipped true. Not instrumented.
 */
static void gemv_Atrans(const _Float16 *A,
                        const _Float16 *x,
                        const _Float16 *C,
                        _Float16 *Y,
                        _Float16 alpha,
                        _Float16 beta,
                        const size_t dim_M,
                        const size_t dim_K)
{
    register _Float16 ZERO asm("fs0") = 0.0f;
    const _Float16 *A_row;
    const _Float16 *C_seg;
    _Float16 *Y_seg;
    size_t avl;
    size_t vl;

    for (unsigned int m = 0; m < dim_M; m += vl) {
        avl = dim_M - m;
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        C_seg = C + m;
        Y_seg = Y + m;

        asm volatile("vfmv.v.f v0, %0" ::"f"(ZERO));

        if (alpha != 0.0f) {
            for (unsigned int k = 0; k < dim_K; k++) {
                A_row = A + (k * dim_M + m);
                asm volatile("vle16.v v16, (%0)" ::"r"(A_row));
                asm volatile("vfmacc.vf v0, %0, v16" ::"f"(*(x + k)));
            }
            asm volatile("vfmul.vf v0, v0, %0" ::"f"(alpha));
        }

        if (beta != 0.0f) {
            asm volatile("vle16.v v16, (%0)" ::"r"(C_seg));
            asm volatile("vfmacc.vf v0, %0, v16" ::"f"(beta));
        }

        asm volatile("vse16.v v0, (%0)" ::"r"(Y_seg));
    }
}

int onnx_gemv_task(void)
{
    volatile onnx_gemv_params_t *params;
    uintptr_t params_addr;
    _Float16 alpha;
    _Float16 beta;
    _Float16 *A;
    _Float16 *x;
    _Float16 *C;
    _Float16 *Y;
    bool transA;
    size_t M;
    size_t K;

    params_addr = mmio32(SPATZ_DATA);
    params      = (volatile onnx_gemv_params_t *)params_addr;

    alpha  = *(_Float16 *)params->addr_alpha;
    beta   = *(_Float16 *)params->addr_beta;
    A      = (_Float16 *)params->addr_A;
    x      = (_Float16 *)params->addr_x;
    C      = (_Float16 *)params->addr_C;
    Y      = (_Float16 *)params->addr_Y;
    transA = params->transA;
    M      = params->M;
    K      = params->K;

    if (transA)
        gemv_Atrans(A, x, C, Y, alpha, beta, M, K);
    else
        gemv_debug(A, x, C, Y, alpha, beta, M, K); /* debug build */

    return 0;
}