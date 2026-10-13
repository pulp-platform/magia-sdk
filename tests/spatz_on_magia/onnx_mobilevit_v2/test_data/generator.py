#!/usr/bin/env python3
# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""
Builds the data and the layer table for the onnx_mobilevit_v2 whole-network test.

Regenerate with: python3 generator.py

Reads ../mobilevitv2.onnx (batchnorm folded, GroupNormalization scale/bias fused,
opset 21: 61 Conv, 43 Mul, 25 Sigmoid, 21 GroupNormalization, 19 Add, 12 Reshape,
10 Softmax, 9 Split, 9 ReduceSum, 9 Relu, 6 Transpose, 1 GlobalAveragePool, 1 Flatten,
1 Gemm) at its native 256x256 input, and emits

  mobilevit_data.bin       - one blob: every weight and bias in FP16, the input image,
                             and the two goldens. Pulled into the .l2_bulk section by
                             mobilevit_data.S with .incbin.
  ../include/mobilevit_graph.h
                           - the layer table, the offsets into the blob, and the
                             activation arena layout.
  layers/<nnn>_<op>.npy    - every intermediate tensor, for bringing the test up one
                             layer at a time. Not compiled in.

THREE NODE TYPES EMIT NO LAYER

Reshape, Split and Flatten are all pure views here, so they are aliased in place rather
than run, which is what takes 227 nodes down to 205 layers and removes the need for a
Split kernel:

  Reshape  every one of the 12 is a contiguous NCHW view - [1,64,32,32] -> [1,64,16,2,16,2]
           splits H into 16x2 and W into 16x2, and so on. Same bytes, same order.
  Split    every one is axis 1 with contiguous channel runs, so each output is a byte
           range of its input.
  Flatten  [1,256,1,1] -> [1,256], as in ResNet18.

TWO CALL-SHAPE DECISIONS

  Transpose  the kernel shards axis 0, which is the batch - 1 here, so everything would
             land on tile 0. All six perms fix axes 0 and 1, so the two are merged into
             one: rank 6 [1,64,16,2,16,2] perm [0,1,3,5,2,4] is passed as rank 5
             [64,16,2,16,2] perm [0,2,4,1,3], iterations 64.
  Gemm       called transposed (A = the [1000,256] fc weight, B = the embedding as a
             column, Y = [1000,1], the same bytes as [1,1000]) because the kernel shards
             the GEMM's M: the ONNX orientation gives M = 1 and leaves the whole layer on
             tile 0. Same as ResNet18.

GOLDEN MODEL

Each op is replayed exactly as the corresponding kernel computes it, so the network
output is reproducible bit for bit rather than merely close:

  Conv     conv2dgemm, one group at a time: the group's own [K_g, N] im2col against its
           [C_out/group, K_g] filters, the GEMM accumulated over ascending k with one
           fused multiply-add per k, then the bias folded in with one more. Blocking, the
           output-channel sharding and the grouping are exact partitions of that same
           per-element computation.
  Sigmoid  NOT a true sigmoid: a Schraudolph fast exp (COEF 1477) on -x, clamped at
           x = -11 first, then 1/(1+e). See sigmoid_fp16_spatz/test_data/generator.py -
           without that clamp this network's sigmoid inputs, which reach -898, come out
           NaN.
  Softmax  NOT a true softmax either: fast exp with COEF 1486 on (x - max), summed
           through the lane accumulator and folded with vfredosum.
  GroupNorm
           mean and variance taken in units of sh = the smallest power of two >= sqrt(n)
           (exact, sh is a power of two) and rescaled in FP32, because with 65536 elements
           per group the plain sum, the sum of squares and (_Float16)n all leave FP16
           range. See groupnorm_fp16_spatz/test_data/generator.py.
  Mul      one FP16 multiply per element, including the two broadcast forms.
  Add      vfadd. Relu: vfmax against zero.
  ReduceSum
           inner_dim is 1 here, so the kernel takes its scalar path: ascending FP16 adds.
  Transpose
           pure data movement.
  GlobalAveragePool
           vfredosum: ascending sum with FP16 rounding at every step, then divide by H*W.
  Gemm     same accumulation as Conv's GEMM.
"""

import os

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(HERE)
MODEL = os.path.join(TEST_DIR, "mobilevitv2.onnx")
BLOB = os.path.join(HERE, "mobilevit_data.bin")
HEADER = os.path.join(TEST_DIR, "include", "mobilevit_graph.h")
LAYER_DIR = os.path.join(HERE, "layers")

SEED = 20260729

# Schraudolph fast exp constants, from the two task files
SIGMOID_COEF = np.float16(1477.0)
SOFTMAX_COEF = np.float16(1486.0)
FASTEXP_BIAS = np.float16(15360.0)
SIGMOID_MIN = np.float16(-11.0)

# VLMAX at e16/m8: the magia_v3 GVSoC model gives Spatz VLEN = 512, so 512/16 * 8.
VLMAX = 256

# Broadcast modes, from mul_bcast_fp16_spatz_params.h
MUL_BCAST_ROW = 0
MUL_BCAST_SCALAR = 1

# Ops the driver knows about; must match mvit_op_t in the generated header.
OPS = {
    "Conv": 0,
    "Relu": 1,
    "Add": 2,
    "Mul": 3,
    "MulBcast": 4,
    "Sigmoid": 5,
    "GroupNormalization": 6,
    "Softmax": 7,
    "ReduceSum": 8,
    "Transpose": 9,
    "GlobalAveragePool": 10,
    "Gemm": 11,
}

# Merged transpose rank, see the note above. 6 - 1 = 5, rounded up for headroom.
MAX_RANK = 6


# --- exact FP16 replays -------------------------------------------------------


def lane_fold(vals):
    """Whole VLMAX-wide chunks added element-wise into a lane accumulator (vfadd.vv,
    ascending), then the lanes folded ascending from zero (vfredosum.vs)."""
    lanes = np.zeros(min(VLMAX, vals.size), dtype=np.float16)

    for off in range(0, vals.size, VLMAX):
        chunk = vals[off : off + VLMAX]
        lanes[: chunk.size] = (
            lanes[: chunk.size].astype(np.float64) + chunk.astype(np.float64)
        ).astype(np.float16)

    acc = np.float16(0.0)
    for v in lanes:
        acc = (acc + v).astype(np.float16)

    return acc


def im2col(X_b, kh, kw, sh, sw, ph, pw, h_out, w_out, ic0, cin_g):
    """B[k, n], k = (ic*K_h + ki)*K_w + kj over the group's cin_g channels starting at
    ic0, n = oh*W_out + ow, zero outside the input."""
    h_in, w_in = X_b.shape[1], X_b.shape[2]
    Xp = np.zeros((cin_g, h_in + 2 * ph, w_in + 2 * pw), dtype=np.float16)
    Xp[:, ph : ph + h_in, pw : pw + w_in] = X_b[ic0 : ic0 + cin_g]

    oh = np.arange(h_out) * sh
    ow = np.arange(w_out) * sw
    out = np.empty((cin_g * kh * kw, h_out * w_out), dtype=np.float16)
    ic = np.arange(cin_g)

    for ki in range(kh):
        for kj in range(kw):
            patch = Xp[:, ki + oh[:, None], kj + ow[None, :]]
            out[(ic * kh + ki) * kw + kj, :] = patch.reshape(cin_g, -1)

    return out


def gemm_fp16_exact(A, B, bias):
    """
    alpha = beta = 1, so: acc = 0; for ascending k, acc = fp16(acc + A[m,k]*B[k,n]);
    then acc = fp16(bias[m] + acc). float64 keeps each individual fused multiply-add
    exact before it is rounded back to FP16, which is what fmadd.h does.
    """
    K = A.shape[1]
    N = B.shape[1]

    acc = np.zeros((A.shape[0], N), dtype=np.float64)
    A64 = A.astype(np.float64)
    B64 = B.astype(np.float64)

    for k in range(K):
        acc += A64[:, k : k + 1] * B64[k : k + 1, :]
        acc = acc.astype(np.float16).astype(np.float64)

    if bias is not None:
        acc = (bias.astype(np.float64)[:, None] + acc).astype(np.float16)
    else:
        acc = acc.astype(np.float16)

    return acc


def conv_fp16_exact(X, W, B, kh, kw, sh, sw, ph, pw, h_out, w_out, group):
    n, c_in = X.shape[0], X.shape[1]
    c_out = W.shape[0]
    cin_g = c_in // group
    cout_g = c_out // group
    K_g = cin_g * kh * kw
    Y = np.empty((n, c_out, h_out, w_out), dtype=np.float16)

    for b in range(n):
        for g in range(group):
            Bmat = im2col(X[b], kh, kw, sh, sw, ph, pw, h_out, w_out, g * cin_g, cin_g)
            A = W[g * cout_g : (g + 1) * cout_g].reshape(cout_g, K_g)
            bias = B[g * cout_g : (g + 1) * cout_g] if B is not None else None
            Y[b, g * cout_g : (g + 1) * cout_g] = gemm_fp16_exact(A, Bmat, bias).reshape(
                cout_g, h_out, w_out
            )

    return Y


def sigmoid_fp16_exact(X):
    """vfmax.vf against MIN, vfsgnjn, then the fast exp and 1/(1+e)."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x = np.maximum(X.astype(np.float16), SIGMOID_MIN).astype(np.float16)
        t = ((-x).astype(np.float16) * SIGMOID_COEF).astype(np.float16)
        t = (t + FASTEXP_BIAS).astype(np.float16)
        u = np.clip(np.trunc(t.astype(np.float64)), 0, 65535).astype(np.uint16)
        e = u.view(np.float16)
        return (np.float16(1.0) / (np.float16(1.0) + e).astype(np.float16)).astype(np.float16)


def softmax_fp16_exact(rows):
    """Row max, fast exp of (x - max), the lane-folded sum, then a per-element divide."""
    G = np.zeros_like(rows)

    for r in range(rows.shape[0]):
        row = rows[r]
        v = (row - row.max()).astype(np.float16)
        t = (v * SOFTMAX_COEF).astype(np.float16)
        t = (t + FASTEXP_BIAS).astype(np.float16)
        u = np.clip(np.trunc(t.astype(np.float64)), 0, 65535).astype(np.uint16)
        e = u.view(np.float16)

        G[r] = (e / lane_fold(e)).astype(np.float16)

    return G


def stat_scale(n):
    """Smallest power of two >= sqrt(n)."""
    sh = 1
    while sh < 32768 and sh * sh < n:
        sh <<= 1
    return sh


def groupnorm_fp16_exact(X, scale, bias, num_groups, eps):
    n, c, h, w = X.shape
    c_per_g = c // num_groups
    npg = c_per_g * h * w

    sh = stat_scale(npg)
    inv_sh = np.float16(1.0 / sh)
    sh_over_n = np.float32(sh) / np.float32(npg)
    sh2_over_n = (np.float32(sh) * np.float32(sh)) / np.float32(npg)
    eps16 = np.float16(eps)

    G = np.zeros_like(X)

    for b in range(n):
        for g in range(num_groups):
            block = X[b, g * c_per_g : (g + 1) * c_per_g].reshape(-1)

            s = lane_fold((block * inv_sh).astype(np.float16))
            mean = np.float16(np.float32(s) * sh_over_n)

            d = (block - mean).astype(np.float16)
            ss = lane_fold((((d * inv_sh).astype(np.float16)) ** 2).astype(np.float16))
            var = np.float32(ss) * sh2_over_n

            denom = np.float16(np.float32(1.0) / np.sqrt(var + np.float32(eps16)))

            norm = (d * denom).astype(np.float16).reshape(c_per_g, h, w)
            for ci in range(c_per_g):
                ch = g * c_per_g + ci
                G[b, ch] = ((norm[ci] * scale[ch]).astype(np.float16) + bias[ch]).astype(
                    np.float16
                )

    return G


def reducesum_last_fp16_exact(X):
    """inner_dim == 1, so the kernel's scalar path: ascending FP16 adds per row."""
    rows = X.reshape(-1, X.shape[-1])
    out = np.zeros(rows.shape[0], dtype=np.float16)

    for r in range(rows.shape[0]):
        acc = np.float16(0.0)
        for v in rows[r]:
            acc = (acc + v).astype(np.float16)
        out[r] = acc

    return out.reshape(X.shape[:-1] + (1,))


def gap_fp16_exact(X):
    """vfredosum: ascending sum, FP16 rounding at every step, then divide by H*W."""
    n, c, h, w = X.shape
    flat = X.reshape(n * c, h * w)
    acc = np.zeros(n * c, dtype=np.float64)

    for i in range(flat.shape[1]):
        acc += flat[:, i].astype(np.float64)
        acc = acc.astype(np.float16).astype(np.float64)

    return (acc.astype(np.float16) / np.float16(h * w)).astype(np.float16).reshape(n, c, 1, 1)


# --- blob and header emission -------------------------------------------------


class Blob:
    """Append-only FP16 blob; hands back 4-byte-aligned byte offsets."""

    def __init__(self):
        self.parts = []
        self.size = 0

    def add(self, arr):
        arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float16))
        if self.size % 4:
            pad = 4 - (self.size % 4)
            self.parts.append(np.zeros(pad, dtype=np.uint8))
            self.size += pad
        off = self.size
        self.parts.append(arr.view(np.uint8).reshape(-1))
        self.size += arr.nbytes
        return off

    def write(self, path):
        with open(path, "wb") as f:
            for p in self.parts:
                f.write(p.tobytes())
        return self.size


class Arena:
    """Byte offsets for the activation buffers, all live at once."""

    def __init__(self):
        self.size = 0

    def alloc(self, nelem):
        if self.size % 4:
            self.size += 4 - (self.size % 4)
        off = self.size
        self.size += nelem * 2
        return off


def attrs(node):
    out = {}
    for a in node.attribute:
        if a.type == onnx.AttributeProto.INTS:
            out[a.name] = list(a.ints)
        elif a.type == onnx.AttributeProto.INT:
            out[a.name] = a.i
        elif a.type == onnx.AttributeProto.FLOAT:
            out[a.name] = a.f
    return out


def merge_transpose(shape, perm):
    """Fold the leading axes the perm leaves in place into one, so the kernel - which
    shards axis 0 - has something bigger than the batch to shard."""
    k = 0
    while k < len(perm) and perm[k] == k:
        k += 1
    k = max(k - 1, 0)  # keep at least one axis to shard

    lead = int(np.prod(shape[: k + 1])) if k > 0 else shape[0]
    in_shape = [lead] + list(shape[k + 1 :])
    perm_m = [0] + [p - k for p in perm[k + 1 :]]
    out_shape = [in_shape[p] for p in perm_m]

    return in_shape, perm_m, out_shape


def mul_broadcast(a_shape, b_shape):
    """(rows, row_len, mode) for the two broadcast forms, or None if elementwise."""
    if list(a_shape) == list(b_shape):
        return None

    n, c, h, w = a_shape
    if list(b_shape) == [n, 1, h, w]:
        return c, h * w, MUL_BCAST_ROW
    if list(b_shape) == [n, c, h, 1]:
        return c * h, w, MUL_BCAST_SCALAR

    raise RuntimeError(f"unhandled Mul broadcast {a_shape} * {b_shape}")


def main():
    os.makedirs(LAYER_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(HEADER), exist_ok=True)

    model = shape_inference.infer_shapes(onnx.load(MODEL))
    graph = model.graph
    init = {t.name: numpy_helper.to_array(t) for t in graph.initializer}

    rng = np.random.default_rng(SEED)
    in_name = graph.input[0].name
    in_shape = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]

    blob = Blob()
    arena = Arena()

    # A fixed pseudo-random image. The network is fully convolutional up to the pool, so
    # any input exercises every layer; a real photo would only change the class index.
    X0 = rng.standard_normal(in_shape).astype(np.float16)
    input_off = blob.add(X0)

    tensors = {in_name: X0}
    place = {in_name: ("blob", input_off)}
    layers = []
    aliased = 0
    gn_eps = None

    print(f"input {in_shape}, {len(graph.node)} nodes")

    for idx, node in enumerate(graph.node):
        a = attrs(node)
        op = node.op_type
        x = tensors[node.input[0]]
        geom = {}

        # --- nodes that are pure views: alias, emit no layer ---------------------
        if op in ("Reshape", "Flatten"):
            if op == "Flatten":
                y = x.reshape(x.shape[0], -1)
            else:
                y = x.reshape([int(v) for v in init[node.input[1]]])
            tensors[node.output[0]] = y
            place[node.output[0]] = place[node.input[0]]
            aliased += 1
            continue

        if op == "Split":
            assert a["axis"] == 1, "only an axis-1 Split is a contiguous view"
            space, base = place[node.input[0]]
            sizes = [int(v) for v in init[node.input[1]]]
            inner = int(np.prod(x.shape[2:]))
            c0 = 0
            for out, size in zip(node.output, sizes):
                tensors[out] = x[:, c0 : c0 + size]
                place[out] = (space, base + c0 * inner * 2)
                c0 += size
            aliased += 1
            continue

        # --- real layers ---------------------------------------------------------
        if op == "Conv":
            W = init[node.input[1]].astype(np.float16)
            B = init[node.input[2]].astype(np.float16) if len(node.input) > 2 else None
            kh, kw = a["kernel_shape"]
            sh, sw = a["strides"]
            ph, pw = a["pads"][0], a["pads"][1]
            group = a.get("group", 1)
            h_out = (x.shape[2] + 2 * ph - kh) // sh + 1
            w_out = (x.shape[3] + 2 * pw - kw) // sw + 1
            y = conv_fp16_exact(x, W, B, kh, kw, sh, sw, ph, pw, h_out, w_out, group)
            geom = dict(
                k_h=kh, k_w=kw, s_h=sh, s_w=sw, p_h=ph, p_w=pw, group=group,
                has_bias=1 if B is not None else 0,
                w_off=blob.add(W), b_off=blob.add(B) if B is not None else 0,
            )

        elif op == "Relu":
            y = np.maximum(x, np.float16(0.0)).astype(np.float16)

        elif op == "Sigmoid":
            y = sigmoid_fp16_exact(x)

        elif op == "Add":
            b = tensors[node.input[1]]
            assert x.shape == b.shape, f"broadcast Add {x.shape} {b.shape}"
            y = (x.astype(np.float64) + b.astype(np.float64)).astype(np.float16)

        elif op == "Mul":
            b = tensors[node.input[1]]
            bc = mul_broadcast(x.shape, b.shape)
            y = (x.astype(np.float64) * b.astype(np.float64)).astype(np.float16)
            if bc is not None:
                rows, row_len, mode = bc
                op = "MulBcast"
                geom = dict(rows=rows, row_len=row_len, bcast_mode=mode)

        elif op == "GroupNormalization":
            scale = init[node.input[1]].astype(np.float16)
            bias = init[node.input[2]].astype(np.float16)
            num_groups = a["num_groups"]
            eps = a["epsilon"]
            assert gn_eps in (None, eps), "one MVIT_GN_EPS for the whole graph"
            gn_eps = eps
            y = groupnorm_fp16_exact(x, scale, bias, num_groups, eps)
            geom = dict(num_groups=num_groups, w_off=blob.add(scale), b_off=blob.add(bias))

        elif op == "Softmax":
            assert a.get("axis", -1) in (-1, len(x.shape) - 1), "softmax must be over the last axis"
            rows = int(np.prod(x.shape[:-1]))
            row_len = x.shape[-1]
            y = softmax_fp16_exact(x.reshape(rows, row_len)).reshape(x.shape)
            geom = dict(sm_rows=rows, sm_len=row_len)

        elif op == "ReduceSum":
            axes = [int(v) for v in init[node.input[1]]]
            assert axes == [-1] or axes == [len(x.shape) - 1], f"ReduceSum over {axes}"
            assert a.get("keepdims", 1) == 1
            y = reducesum_last_fp16_exact(x)
            geom = dict(outer=int(np.prod(x.shape[:-1])), reduce=x.shape[-1], inner=1)

        elif op == "Transpose":
            perm = a["perm"]
            y = np.ascontiguousarray(x.transpose(perm))
            t_in, perm_m, t_out = merge_transpose(list(x.shape), perm)
            geom = dict(rank=len(t_in), iterations=t_in[0], perm=perm_m,
                        t_in_shape=t_in, t_out_shape=t_out)

        elif op == "GlobalAveragePool":
            y = gap_fp16_exact(x)

        elif op == "Gemm":
            W = init[node.input[1]].astype(np.float16)  # [1000, 256], ONNX transB = 1
            B = init[node.input[2]].astype(np.float16)
            y = gemm_fp16_exact(W, x.reshape(-1, 1), B).reshape(1, -1)
            geom = dict(has_bias=1, w_off=blob.add(W), b_off=blob.add(B))

        else:
            raise RuntimeError(f"unhandled op {node.op_type}")

        tensors[node.output[0]] = y
        dst_off = arena.alloc(y.size)
        place[node.output[0]] = ("arena", dst_off)

        src = [place[n] for n in node.input if n in place]
        # Only the 4D ops read in_shape/out_shape; a rank-6 Transpose carries its own.
        ish = (list(x.shape) + [1, 1, 1, 1])[:4]
        osh = (list(y.shape) + [1, 1, 1, 1])[:4]

        layers.append(dict(
            idx=idx, op=op, name=f"{idx:03d}_{node.op_type}",
            src0=src[0], src1=src[1] if len(src) > 1 else ("arena", 0),
            dst=dst_off, in_shape=ish, out_shape=osh, numel=int(y.size), **geom,
        ))

        np.save(os.path.join(LAYER_DIR, f"{idx:03d}_{node.op_type}.npy"), y)
        print(f"{idx:3d} {op:20s} {list(x.shape)} -> {list(y.shape)}  |max| = "
              f"{float(np.abs(y.astype(np.float64)).max()):.5g}")

    # --- goldens -------------------------------------------------------------
    logits = tensors[graph.node[-2].output[0]]  # Gemm output
    output = tensors[graph.output[0].name]      # Softmax output
    logits_place = place[graph.node[-2].output[0]]
    output_place = place[graph.output[0].name]

    golden_logits_off = blob.add(logits)
    golden_output_off = blob.add(output)

    top1 = int(np.argmax(output.reshape(-1).astype(np.float64)))
    top1_logits = int(np.argmax(logits.reshape(-1).astype(np.float64)))
    assert top1 == top1_logits, "softmax is monotone, top-1 must agree with the logits"

    l64 = logits.reshape(-1).astype(np.float64)
    true_sm = np.exp(l64 - l64.max())
    true_sm /= true_sm.sum()
    dev = float(np.max(np.abs(output.reshape(-1).astype(np.float64) - true_sm)))

    blob_size = blob.write(BLOB)

    # --- header --------------------------------------------------------------
    def clist(vals, n):
        vals = list(vals) + [0] * (n - len(vals))
        return "{" + ", ".join(str(int(v)) for v in vals) + "}"

    with open(HEADER, "w") as f:
        f.write("// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.\n")
        f.write("// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n")
        f.write("// SPDX-License-Identifier: Apache-2.0\n\n")
        f.write("/* Auto-generated by test_data/generator.py -- do not edit by hand.\n")
        f.write(" * Regenerate with: cd test_data && python3 generator.py\n */\n\n")
        f.write("#ifndef MOBILEVIT_GRAPH_H_\n#define MOBILEVIT_GRAPH_H_\n\n")
        f.write("#include <stdint.h>\n\n")

        f.write("/* Operations the driver dispatches on. */\n")
        f.write("typedef enum {\n")
        for name, code in OPS.items():
            f.write(f"    MVIT_{name.upper()} = {code},\n")
        f.write("} mvit_op_t;\n\n")

        f.write("/* Where an operand lives. Only the first layer reads from the blob. */\n")
        f.write("#define MVIT_SPACE_ARENA (0)\n")
        f.write("#define MVIT_SPACE_BLOB  (1)\n\n")
        f.write(f"#define MVIT_MAX_RANK    ({MAX_RANK})\n\n")

        f.write("typedef struct {\n")
        f.write("    uint8_t op;           /* mvit_op_t                                  */\n")
        f.write("    uint8_t has_bias;\n")
        f.write("    uint8_t src0_space;   /* MVIT_SPACE_*                               */\n")
        f.write("    uint8_t src1_space;\n")
        f.write("    uint32_t src0_off;    /* byte offset in its space                   */\n")
        f.write("    uint32_t src1_off;\n")
        f.write("    uint32_t dst_off;     /* byte offset in the arena                   */\n")
        f.write("    uint32_t w_off;       /* blob: conv/gemm weights, groupnorm scale   */\n")
        f.write("    uint32_t b_off;       /* blob: bias, groupnorm bias                 */\n")
        f.write("    uint32_t in_shape[4];\n")
        f.write("    uint32_t out_shape[4];\n")
        f.write("    uint32_t kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, group;\n")
        f.write("    uint32_t numel;       /* output elements, for the flat ops          */\n")
        f.write("    uint32_t num_groups;  /* GroupNormalization                         */\n")
        f.write("    uint32_t sm_rows, sm_len;             /* Softmax                    */\n")
        f.write("    uint32_t outer, reduce, inner;        /* ReduceSum                  */\n")
        f.write("    uint32_t rows, row_len, bcast_mode;   /* MulBcast                   */\n")
        f.write("    uint32_t rank, iterations;            /* Transpose, axes merged     */\n")
        f.write("    uint32_t perm[MVIT_MAX_RANK];\n")
        f.write("    uint32_t t_in_shape[MVIT_MAX_RANK];\n")
        f.write("    uint32_t t_out_shape[MVIT_MAX_RANK];\n")
        f.write("    const char *name;\n")
        f.write("} mvit_layer_t;\n\n")

        f.write(f"#define MVIT_NUM_LAYERS       ({len(layers)})\n")
        f.write(f"#define MVIT_BLOB_BYTES       ({blob_size})\n")
        f.write(f"#define MVIT_ARENA_BYTES      ({arena.size})\n")
        f.write(f"#define MVIT_INPUT_OFF        ({input_off})\n")
        f.write(f"#define MVIT_GN_EPS           ({float(np.float16(gn_eps)):.9g}f)\n\n")

        f.write("/* Pass/fail is the logits: bit-exact, and produced by the 204 layers before\n")
        f.write(" * the softmax. The softmax output is reported but not failed on. */\n")
        f.write(f"#define MVIT_LOGITS_OFF       ({logits_place[1]})\n")
        f.write(f"#define MVIT_LOGITS_LEN       ({logits.size})\n")
        f.write(f"#define MVIT_LOGITS_ULP_TOLL  (0)\n")
        f.write(f"#define MVIT_GOLDEN_LOGITS_OFF ({golden_logits_off})\n\n")
        f.write(f"#define MVIT_OUTPUT_OFF       ({output_place[1]})\n")
        f.write(f"#define MVIT_OUTPUT_LEN       ({output.size})\n")
        f.write(f"#define MVIT_GOLDEN_OUTPUT_OFF ({golden_output_off})\n")
        f.write(f"#define MVIT_TOP1             ({top1})\n\n")

        f.write("/* Stop after this many layers, for bringing the test up one at a time. */\n")
        f.write("#ifndef MVIT_MAX_LAYERS\n")
        f.write("#define MVIT_MAX_LAYERS       MVIT_NUM_LAYERS\n")
        f.write("#endif\n\n")

        f.write("/* clang-format off */\n")
        f.write(f"static const mvit_layer_t mobilevit_layers[{len(layers)}] = {{\n")
        for L in layers:
            f.write("    { ")
            f.write(f".op = MVIT_{L['op'].upper()}, ")
            f.write(f".has_bias = {L.get('has_bias', 0)}, ")
            f.write(f".src0_space = MVIT_SPACE_{L['src0'][0].upper()}, ")
            f.write(f".src1_space = MVIT_SPACE_{L['src1'][0].upper()},\n      ")
            f.write(f".src0_off = {L['src0'][1]}, .src1_off = {L['src1'][1]}, ")
            f.write(f".dst_off = {L['dst']},\n      ")
            f.write(f".w_off = {L.get('w_off', 0)}, .b_off = {L.get('b_off', 0)},\n      ")
            f.write(f".in_shape = {clist(L['in_shape'], 4)}, ")
            f.write(f".out_shape = {clist(L['out_shape'], 4)},\n      ")
            f.write(f".kernel_h = {L.get('k_h', 0)}, .kernel_w = {L.get('k_w', 0)}, ")
            f.write(f".stride_h = {L.get('s_h', 0)}, .stride_w = {L.get('s_w', 0)}, ")
            f.write(f".pad_h = {L.get('p_h', 0)}, .pad_w = {L.get('p_w', 0)}, ")
            f.write(f".group = {L.get('group', 1)},\n      ")
            f.write(f".numel = {L['numel']}, .num_groups = {L.get('num_groups', 0)},\n      ")
            f.write(f".sm_rows = {L.get('sm_rows', 0)}, .sm_len = {L.get('sm_len', 0)}, ")
            f.write(f".outer = {L.get('outer', 0)}, .reduce = {L.get('reduce', 0)}, ")
            f.write(f".inner = {L.get('inner', 0)},\n      ")
            f.write(f".rows = {L.get('rows', 0)}, .row_len = {L.get('row_len', 0)}, ")
            f.write(f".bcast_mode = {L.get('bcast_mode', 0)},\n      ")
            f.write(f".rank = {L.get('rank', 0)}, .iterations = {L.get('iterations', 0)},\n      ")
            f.write(f".perm = {clist(L.get('perm', []), MAX_RANK)}, ")
            f.write(f".t_in_shape = {clist(L.get('t_in_shape', []), MAX_RANK)},\n      ")
            f.write(f".t_out_shape = {clist(L.get('t_out_shape', []), MAX_RANK)},\n      ")
            f.write(f".name = \"{L['name']}\" }},\n")
        f.write("};\n")
        f.write("/* clang-format on */\n\n")
        f.write("#endif /* MOBILEVIT_GRAPH_H_ */\n")

    print()
    print(f"{len(layers)} layers emitted, {aliased} nodes aliased away")
    print(f"blob    {BLOB} : {blob_size/1024/1024:.2f} MB")
    print(f"arena                                : {arena.size/1024/1024:.2f} MB")
    print(f"header  {HEADER}")
    print(f"layers  {LAYER_DIR}/ : {len(layers)} tensors")
    print(f"top-1 class {top1}, logit {float(l64[top1]):.4f}, "
          f"p = {float(output.reshape(-1)[top1]):.4f}")
    print(f"fast-exp softmax deviates from a true softmax by up to {dev:.4f}")


if __name__ == "__main__":
    main()
