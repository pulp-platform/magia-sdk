#include "tile.h"
#include "im2col_fp16_spatz_params.h"

/* dst[0 .. n - 1] = 0 */
static inline void zero_fp16(_Float16 *dst, size_t n)
{
    size_t vl;

    for (; n > 0; n -= vl, dst += vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vmv.v.i v0, 0");
        asm volatile("vse16.v v0, (%0)" SPATZ_VST_FENCE ::"r"(dst) : "memory");
    }
}

/* dst[0 .. n - 1] = src[0 .. n - 1] */
static inline void copy_fp16(_Float16 *dst, const _Float16 *src, size_t n)
{
    size_t vl;

    for (; n > 0; n -= vl, dst += vl, src += vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vle16.v v8, (%0)" ::"r"(src));
        asm volatile("vse16.v v8, (%0)" SPATZ_VST_FENCE ::"r"(dst) : "memory");
    }
}

/*
 * im2col for a strip of `cp` consecutive output pixels in one output row:
 *   (row, x0), (row, x0 + 1), ..., (row, x0 + cp - 1)
 *
 * Convolution parameters: ksize x ksize kernel, stride 1, `pad` zeros on each side.
 *
 * Output layout:
 *   Row p of dst is the patch of output pixel (row, x0 + p), flattened as (ky, kx, ci):
 *     dst[p][(ky * ksize + kx) * cin + ci] = in[row - pad + ky][x0 + p - pad + kx][ci]
 *   Positions outside the image are 0.
 *
 * Input buffer (src, HWC layout):
 *   - Starts at input row lo and holds columns src_x .. src_x + src_w - 1.
 *   - Must contain every in-image row and column that any of the cp patches reads.
 *
 * Copy pattern:
 *   In HWC, one patch row (fixed ky, all kx and ci) is ksize * cin contiguous values in the
 *   input row. So for each (p, ky) there is one contiguous copy of the in-image columns, plus
 *   zero fill for the columns outside the image.
 */
int im2col_fp16_spatz_task(void)
{
    const im2col_fp16_spatz_params_t *params;
    const _Float16 *src;
    _Float16 *dst;
    int32_t height, width, cin, ksize, pad, row, lo, x0, cp, src_w, src_x;
    int32_t seg, p, ky;

    params = (const im2col_fp16_spatz_params_t *)mmio32(SPATZ_DATA);
    src    = (const _Float16 *)params->src;
    dst    = (_Float16 *)params->dst;
    height = params->height;
    width  = params->width;
    cin    = params->cin;
    ksize  = params->ksize;
    pad    = params->pad;
    row    = params->row;
    lo     = params->lo;
    x0     = params->x0;
    cp     = params->cp;
    src_w  = params->src_w;
    src_x  = params->src_x;
    seg    = ksize * cin; /* one patch row: ksize columns x cin channels */

    for (p = 0; p < cp; p++) {
        int32_t x      = x0 + p;
        int32_t kx_lo  = pad - x > 0 ? pad - x : 0;                       /* first column inside the image */
        int32_t kx_hi  = width + pad - x < ksize ? width + pad - x : ksize; /* past the last one            */
        _Float16 *d    = dst + p * ksize * seg;

        for (ky = 0; ky < ksize; ky++, d += seg) {
            int32_t iy = row - pad + ky;

            if (iy < 0 || iy >= height) {
                zero_fp16(d, seg);
                continue;
            }
            if (kx_lo > 0)
                zero_fp16(d, kx_lo * cin);
            copy_fp16(d + kx_lo * cin, src + ((iy - lo) * src_w + x - pad + kx_lo - src_x) * cin, (kx_hi - kx_lo) * cin);
            if (kx_hi < ksize)
                zero_fp16(d + kx_hi * cin, (ksize - kx_hi) * cin);
        }
    }

    return 0;
}
