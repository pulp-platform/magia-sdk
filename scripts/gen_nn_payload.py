#!/usr/bin/env python3
# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Build the raw payload the VFIO host DMAs into MAGIA's L2 for nn_is_bench.
#
# The bit patterns must match what gen_nn_is_data.py embedded in the ELF, so
# that the golden samples carried by the ELF stay valid and the device can
# self-check its own result with NN_FLAG_VERIFY. Both scripts derive the data
# from the same seed with the same RNG calls, in the same order -- keep the two
# generators in step.
#
# Output: one flat blob, X followed by W, to be DMAed as a single transfer to
# L2 offset NN_X_ADDR - 0xC0000000. The 64-byte nn_hdr_t is NOT part of it: the
# host rewrites the header itself every iteration because iter_id changes.
#
# Usage:
#   python3 scripts/gen_nn_payload.py --size 512 -o /tmp/nn_payload_512.bin

import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=None,
                    help="shorthand for --m/--n/--k all equal")
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--seed", type=lambda v: int(v, 0), default=0xC0FFEE,
                    help="must match NN_DATA_SEED in the generated nn_is_size.h")
    ap.add_argument("--bias", action="store_true",
                    help="also emit the Y bias plane as <out>.bias")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    if args.size is not None:
        args.m = args.n = args.k = args.size
    m, n, k = args.m, args.n, args.k

    rng = np.random.default_rng(args.seed)

    # Identical call sequence to gen_nn_is_data.py: magnitude first, then sign,
    # X before W. Any divergence here silently desynchronises the two datasets.
    def rand_block(shape):
        mag = rng.uniform(0.25, 2.0, size=shape)
        sign = rng.integers(0, 2, size=shape) * 2 - 1
        return (mag * sign).astype(np.float16)

    x = rand_block((m, n))
    w = rand_block((n, k))
    bias_row = (rng.uniform(-0.5, 0.5, size=k)).astype(np.float16)

    blob = np.concatenate([x.reshape(-1).view(np.uint16),
                           w.reshape(-1).view(np.uint16)])
    with open(args.out, "wb") as fh:
        fh.write(blob.tobytes())

    print("wrote %s (%d bytes = X %d KiB + W %d KiB)"
          % (args.out, blob.nbytes, x.nbytes // 1024, w.nbytes // 1024))
    print("  DMA it to L2 offset (NN_X_ADDR - 0xC0000000) = 0x%08x" % 0x08001000)

    if args.bias:
        # Y is an in/out buffer: the leftmost mesh column reads it as the GEMM
        # bias and the rightmost writes the result back over it. It therefore
        # has to be re-staged every iteration, like X and W.
        yb = np.broadcast_to(bias_row, (m, k)).astype(np.float16).copy()
        path = args.out + ".bias"
        with open(path, "wb") as fh:
            fh.write(yb.reshape(-1).view(np.uint16).tobytes())
        print("wrote %s (%d bytes)" % (path, yb.nbytes))
        print("  DMA it to L2 offset (NN_Y_ADDR - 0xC0000000) = 0x%08x" % 0x08101000)

    return 0


if __name__ == "__main__":
    sys.exit(main())
