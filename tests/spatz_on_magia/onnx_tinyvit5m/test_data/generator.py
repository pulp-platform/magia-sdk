#!/usr/bin/env python3
# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""
Builds the data and the layer table for the onnx_tinyvit5m whole-network test.

Regenerate with: python3 generator.py

Reads ../tiny_vit_5m.onnx (batchnorm folded, GELU fused, LayerNorm kept as one op,
opset 20: 72 Add, 72 Transpose, 64 Reshape, 60 MatMul, 27 Conv, 23 Gelu,
21 LayerNormalization, 11 Softmax, 10 Split, 1 GlobalAveragePool, 1 Flatten, 1 Gemm)
at its native 224x224 input, and emits

  tinyvit_data.bin         - one blob: every weight and bias in FP16, the input image,
                             and the two goldens. Pulled into the .l2_bulk section by
                             tinyvit_data.S with .incbin.
  ../include/tinyvit_graph.h
                           - the layer table, the offsets into the blob, and the
                             activation arena layout.
  layers/<nnn>_<op>.npy    - every intermediate tensor, for bringing the test up one
                             layer at a time. Not compiled in.

WHICH NODES EMIT NO LAYER

Reshape and Flatten are pure views here and are aliased in place rather than run. Every
Reshape in this graph reads a buffer some other op has just materialised in row-major
order, and ONNX Reshape never permutes that order - it only re-factors the multi-index -
so all 64 are the same bytes in the same order. Two Transposes (357 and 359) permute only
axes of extent 1 and are aliased on the same grounds.

SPLIT IS NOT A VIEW HERE, UNLIKE IN MOBILEVIT

All ten Splits are axis 3 of [..., H, 96] into three runs of 32, i.e. the *innermost*
axis. Each of q, k and v is therefore a strided view - runs of 32 elements 96 apart -
not a byte range, and cannot be aliased the way mobilevit's axis-1 Splits were. Each one
becomes a single de-interleaving Transpose instead: the input read as [R, 3, 32] with
R = prod(leading dims), permuted to [3, R, 32], after which q, k and v *are* contiguous
byte ranges of that output and are aliased. R is 3136 / 980 / 490, so the layer shards
well.

CALL-SHAPE DECISIONS

  Transpose  the kernel shards whichever of the input's and the output's leading axis is
             longer, so the leading axes the perm leaves fixed are merged into one - and
             dropped entirely when their product is 1, which is the case for every
             batch-1 transpose here. [1,28,28,128] perm [0,3,1,2] goes in as [28,28,128]
             perm [2,0,1], which shards 128 ways instead of 1.
  MatMul     against a weight (40 of the 60) it is a plain 2-D GEMM once the leading axes
             are folded into M, and goes to the gemm kernel with beta = 0 - which shards
             M and has an alignment guard. C is passed as the destination buffer: gemm
             stages C unconditionally but never reads it when beta is 0. The 20 batched
             attention MatMuls go to the matmul kernel, which shards M within each batch.
  Add        46 of the 72 broadcast. 36 are a MatMul bias over the last axis and 2 are an
             attention bias shared by 16 windows, both of which are the add_bcast kernel's
             row form; the other 8 have a broadcast operand with the same element count as
             their input, so they are plain elementwise adds.
  LayerNorm  20 of the 21 have no bias. The kernel takes both, so a shared zero vector per
             row length goes in the blob.
  Gemm       called transposed (A = the [1000,320] fc weight, B = the embedding as a
             column, Y = [1000,1], the same bytes as [1,1000]) because the kernel shards
             the GEMM's M: the ONNX orientation gives M = 1 and leaves the whole layer on
             tile 0. Same as ResNet18 and MobileViT.

GOLDEN MODEL

Each op is replayed exactly as the corresponding kernel computes it, so the network
output is reproducible bit for bit rather than merely close:

  Conv     conv2dgemm, one group at a time: the group's own [K_g, N] im2col against its
           [C_out/group, K_g] filters, the GEMM accumulated over ascending k with one
           fused multiply-add per k, then the bias folded in with one more.
  MatMul   the same ascending-k fused multiply-add, with no bias and no alpha/beta. The
           matmul kernel's vector path (vfmacc) and its scalar fallback (fmadd.h) agree,
           so one model covers both - which matters, because the four 7x7 QK^T MatMuls
           have an odd O = 49 and take the fallback.
  Gelu     NOT an exact-erf GELU: the tanh form, with a Schraudolph fast exp (COEF 1486)
           standing in for exp(2t) and its argument clamped to +/-5 first. See
           gelu_fp16_spatz/test_data/generator.py.
  Softmax  NOT a true softmax either: fast exp with COEF 1486 on (x - max), summed
           through the lane accumulator and folded with vfredosum. The kernel's scalar
           fallback, which the four 49-long softmaxes take, sums ascending - identical to
           the lane fold for any row that fits one chunk.
  LayerNorm
           mean and variance taken in units of sh = the smallest power of two >= sqrt(n)
           (exact, sh is a power of two) and rescaled in FP32. See
           layernorm_fp16_spatz/test_data/generator.py.
  Add      vfadd, including the broadcast form.
  Transpose
           pure data movement.
  GlobalAveragePool
           vfredosum: ascending sum with FP16 rounding at every step, then divide by H*W.
"""

import os

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(HERE)
MODEL = os.path.join(TEST_DIR, "tiny_vit_5m.onnx")
BLOB = os.path.join(HERE, "tinyvit_data.bin")
HEADER = os.path.join(TEST_DIR, "include", "tinyvit_graph.h")
LAYER_DIR = os.path.join(HERE, "layers")

SEED = 20260729

# Schraudolph fast exp constants, from the two task files
SOFTMAX_COEF = np.float16(1486.0)
GELU_COEF = np.float16(1486.0)
FASTEXP_BIAS = np.float16(15360.0)

# gelu's polynomial and clamp, from gelu_fp16_spatz_task.c
GELU_C0 = np.float16(0.044715)
GELU_C1 = np.float16(0.797884561)  # sqrt(2/pi)
GELU_TMIN = np.float16(-5.0)
GELU_TMAX = np.float16(5.0)

# VLMAX at e16/m8: the magia_v3 GVSoC model gives Spatz VLEN = 512, so 512/16 * 8.
VLMAX = 256

# Broadcast modes, from add_bcast_fp16_spatz_params.h
ADD_BCAST_ROW = 0
ADD_BCAST_SCALAR = 1

# Ops the driver dispatches on; must match tvit_op_t in the generated header.
OPS = {
    "Conv": 0,
    "Gelu": 1,
    "LayerNormalization": 2,
    "MatMul2D": 3,
    "MatMul": 4,
    "Add": 5,
    "AddBcast": 6,
    "Softmax": 7,
    "Transpose": 8,
    "GlobalAveragePool": 9,
    "Gemm": 10,
}

# Every perm here merges down to rank 5 or less.
MAX_RANK = 5


# --- exact FP16 replays -------------------------------------------------------


def lane_fold_rows(vals):
    """Whole VLMAX-wide chunks added element-wise into a lane accumulator (vfadd.vv,
    ascending), then the lanes folded ascending from zero (vfredosum.vs). Vectorised
    over the rows: `vals` is [R, L] and the result is [R].

    For a row that fits one chunk this is a plain ascending sum, which is also what the
    scalar fallbacks in softmax and mul/add_bcast do - so one model covers both paths.
    """
    R, L = vals.shape
    n_lanes = min(VLMAX, L)

    lanes = np.zeros((R, n_lanes), dtype=np.float16)
    for off in range(0, L, VLMAX):
        chunk = vals[:, off : off + VLMAX]
        w = chunk.shape[1]
        lanes[:, :w] = (
            lanes[:, :w].astype(np.float64) + chunk.astype(np.float64)
        ).astype(np.float16)

    acc = np.zeros(R, dtype=np.float16)
    for i in range(n_lanes):
        acc = (acc.astype(np.float64) + lanes[:, i].astype(np.float64)).astype(np.float16)

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
    acc = 0; for ascending k, acc = fp16(acc + A[m,k]*B[k,n]); then, if there is one,
    acc = fp16(bias[m] + acc). float64 keeps each individual fused multiply-add exact
    before it is rounded back to FP16, which is what vfmacc and fmadd.h do.
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


def gelu_fp16_exact(X):
    """x^3 / the polynomial / the clamp / the fast tanh / 0.5 * x * (1 + tanh).

    x^3 overflows to +/-Inf above |x| ~ 40.3 - which the hardware does too - and the
    clamp two steps later turns the Inf back into +/-5.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        x = X.astype(np.float16)

        t = (x * x).astype(np.float16)
        t = (t * x).astype(np.float16)
        t = (t * GELU_C0).astype(np.float16)
        t = (t + x).astype(np.float16)
        t = (t * GELU_C1).astype(np.float16)

        t = np.minimum(t, GELU_TMAX).astype(np.float16)
        t = np.maximum(t, GELU_TMIN).astype(np.float16)

        u = (t * np.float16(2.0)).astype(np.float16)
        u = (u * GELU_COEF).astype(np.float16)
        u = (u + FASTEXP_BIAS).astype(np.float16)

        bits = np.clip(np.trunc(u.astype(np.float64)), 0, 65535).astype(np.uint16)
        e = bits.view(np.float16)

        den = (e + np.float16(1.0)).astype(np.float16)
        num = (e - np.float16(1.0)).astype(np.float16)
        th = (num / den).astype(np.float16)

        r = (th + np.float16(1.0)).astype(np.float16)
        r = (r * x).astype(np.float16)
        return (r * np.float16(0.5)).astype(np.float16)


def softmax_fp16_exact(rows):
    """Row max, fast exp of (x - max), the lane-folded sum, then a per-element divide."""
    m = rows.max(axis=1, keepdims=True)
    v = (rows - m).astype(np.float16)

    t = (v * SOFTMAX_COEF).astype(np.float16)
    t = (t + FASTEXP_BIAS).astype(np.float16)
    bits = np.clip(np.trunc(t.astype(np.float64)), 0, 65535).astype(np.uint16)
    e = bits.view(np.float16)

    s = lane_fold_rows(e)

    return (e / s[:, None]).astype(np.float16)


def stat_scale(n):
    """Smallest power of two >= sqrt(n)."""
    sh = 1
    while sh < 32768 and sh * sh < n:
        sh <<= 1
    return sh


def layernorm_fp16_exact(X, scale, bias, eps):
    """X is [rows, w_len]; both reductions are taken in units of sh and rescaled in FP32,
    where the variance may exceed the FP16 range because only 1/sqrt(var + eps) is ever
    rounded back."""
    w_len = X.shape[1]

    sh = stat_scale(w_len)
    inv_sh = np.float16(1.0 / sh)  # exact: sh is a power of two
    sh_over_n = np.float32(sh) / np.float32(w_len)
    sh2_over_n = (np.float32(sh) * np.float32(sh)) / np.float32(w_len)
    eps16 = np.float16(eps)

    s = lane_fold_rows((X * inv_sh).astype(np.float16))
    mean = (s.astype(np.float32) * sh_over_n).astype(np.float16)

    d = (X - mean[:, None]).astype(np.float16)
    ds = (d * inv_sh).astype(np.float16)
    ss = lane_fold_rows((ds * ds).astype(np.float16))
    var = ss.astype(np.float32) * sh2_over_n

    denom = (np.float32(1.0) / np.sqrt(var + np.float32(eps16))).astype(np.float16)

    norm = (d * denom[:, None]).astype(np.float16)
    return ((norm * scale).astype(np.float16) + bias).astype(np.float16)


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
    """Merge the leading axes the perm leaves in place into one, and drop them outright
    when their product is 1.

    The kernel shards whichever of the input's and the output's leading axis is longer,
    so a fixed leading axis of extent 1 - which every batch-1 transpose here has - is
    worth removing rather than keeping as the shard axis.
    """
    f = 0
    while f < len(perm) and perm[f] == f:
        f += 1

    if f == 0:
        return list(shape), list(perm), [shape[p] for p in perm]

    lead = int(np.prod(shape[:f]))

    if lead == 1 and f < len(perm):
        in_shape = list(shape[f:])
        perm_m = [p - f for p in perm[f:]]
    else:
        in_shape = [lead] + list(shape[f:])
        perm_m = [0] + [p - f + 1 for p in perm[f:]]

    return in_shape, perm_m, [in_shape[p] for p in perm_m]


def transpose_is_view(shape, perm):
    """True when the perm only moves axes of extent 1, i.e. the output is the input's
    bytes in the same order and the node can be aliased."""
    kept = [perm[i] for i in range(len(perm)) if shape[perm[i]] > 1]
    return all(a < b for a, b in zip(kept, kept[1:]))


def add_broadcast(a_shape, b_shape):
    """(rows, row_len, mode) for the broadcast form, or None when the two operands have
    the same element count and a plain elementwise add covers it."""
    na = int(np.prod(a_shape))
    nb = int(np.prod(b_shape))

    if na == nb:
        return None

    # B has to be a suffix of A's shape, so that A is [rows, row_len] with row_len = |B|.
    assert list(b_shape) == list(a_shape[len(a_shape) - len(b_shape) :]), (
        f"unhandled Add broadcast {a_shape} + {b_shape}"
    )

    return na // nb, nb, ADD_BCAST_ROW


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
    ln_eps = None

    blob_consts = {}   # initializer name -> byte offset, so a shared one is stored once
    zero_bias = {}     # row length -> byte offset of a zero vector

    def const_place(name):
        """Byte offset of an initializer in the blob, adding it the first time."""
        if name not in blob_consts:
            blob_consts[name] = blob.add(init[name].astype(np.float16))
        return ("blob", blob_consts[name])

    def operand_place(name):
        return place[name] if name in place else const_place(name)

    def zeros_place(n):
        if n not in zero_bias:
            zero_bias[n] = blob.add(np.zeros(n, dtype=np.float16))
        return zero_bias[n]

    print(f"input {in_shape}, {len(graph.node)} nodes")

    for idx, node in enumerate(graph.node):
        a = attrs(node)
        op = node.op_type
        x = tensors[node.input[0]] if node.input[0] in tensors else init[node.input[0]]
        geom = {}
        src_override = None

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

        if op == "Transpose" and transpose_is_view(list(x.shape), a["perm"]):
            tensors[node.output[0]] = np.ascontiguousarray(x.transpose(a["perm"]))
            place[node.output[0]] = place[node.input[0]]
            aliased += 1
            continue

        # --- Split: one de-interleaving Transpose, then three aliases ------------
        if op == "Split":
            axis = a["axis"]
            sizes = [int(v) for v in init[node.input[1]]]
            assert axis == len(x.shape) - 1, "Split must be over the last axis"
            assert len(set(sizes)) == 1, "Split must be into equal runs"

            u = sizes[0]
            n_out = len(sizes)
            r = int(np.prod(x.shape[:-1]))

            # [R, n_out, U] -> [n_out, R, U]: after this each output is contiguous.
            y = np.ascontiguousarray(x.reshape(r, n_out, u).transpose(1, 0, 2))

            dst_off = arena.alloc(y.size)
            t_in, perm_m, t_out = merge_transpose([r, n_out, u], [1, 0, 2])

            layers.append(dict(
                idx=idx, op="Transpose", name=f"{idx:03d}_Split",
                src0=place[node.input[0]], src1=("arena", 0), dst=dst_off,
                in_shape=[1, 1, 1, 1], out_shape=[1, 1, 1, 1], numel=int(y.size),
                rank=len(t_in), iterations=t_in[0], perm=perm_m,
                t_in_shape=t_in, t_out_shape=t_out,
            ))
            np.save(os.path.join(LAYER_DIR, f"{idx:03d}_Split.npy"), y)

            for j, out in enumerate(node.output):
                tensors[out] = np.ascontiguousarray(
                    x[..., j * u : (j + 1) * u]
                )
                place[out] = ("arena", dst_off + j * r * u * 2)

            print(f"{idx:3d} {'Split':20s} {list(x.shape)} -> {n_out} x "
                  f"{list(tensors[node.output[0]].shape)} (de-interleave)")
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

        elif op == "Gelu":
            y = gelu_fp16_exact(x)

        elif op == "LayerNormalization":
            axis = a.get("axis", -1)
            assert axis in (-1, len(x.shape) - 1), f"LayerNorm over axis {axis}"
            eps = a["epsilon"]
            assert ln_eps in (None, eps), "one TVIT_LN_EPS for the whole graph"
            ln_eps = eps

            scale = init[node.input[1]].astype(np.float16)
            w_len = x.shape[-1]
            rows = int(np.prod(x.shape[:-1]))

            if len(node.input) > 2:
                bias = init[node.input[2]].astype(np.float16)
                b_off = blob.add(bias)
            else:
                bias = np.zeros(w_len, dtype=np.float16)
                b_off = zeros_place(w_len)

            y = layernorm_fp16_exact(x.reshape(rows, w_len), scale, bias, eps).reshape(x.shape)
            geom = dict(ln_rows=rows, ln_len=w_len, w_off=blob.add(scale), b_off=b_off)

        elif op == "MatMul":
            b_name = node.input[1]

            if b_name in init:
                # 2-D weight: fold the leading axes of A into M and let gemm shard it.
                W = init[b_name].astype(np.float16)
                assert W.ndim == 2, f"MatMul weight of rank {W.ndim}"
                m = int(np.prod(x.shape[:-1]))
                k, o = W.shape
                y = gemm_fp16_exact(x.reshape(m, k), W, None).reshape(x.shape[:-1] + (o,))
                op = "MatMul2D"
                geom = dict(mm_m=m, mm_k=k, mm_o=o, mm_batches=1, w_off=blob.add(W))
            else:
                bt = tensors[b_name]
                assert x.ndim >= 3 and x.shape[:-2] == bt.shape[:-2], (
                    f"MatMul {x.shape} @ {bt.shape}"
                )
                batches = int(np.prod(x.shape[:-2]))
                m, k = x.shape[-2], x.shape[-1]
                o = bt.shape[-1]
                xr = x.reshape(batches, m, k)
                br = bt.reshape(batches, k, o)
                y = np.stack(
                    [gemm_fp16_exact(xr[i], br[i], None) for i in range(batches)]
                ).reshape(x.shape[:-1] + (o,))
                geom = dict(mm_m=m, mm_k=k, mm_o=o, mm_batches=batches)

        elif op == "Add":
            # The constant is input[0] for every bias add and input[1] for every
            # attention bias, so pick the bigger operand as A either way.
            n0 = int(np.prod(np.asarray(x).shape))
            other = node.input[1]
            xo = tensors[other] if other in tensors else init[other]
            n1 = int(np.prod(np.asarray(xo).shape))

            if n1 > n0:
                a_name, b_name = other, node.input[0]
                av, bv = xo, x
            else:
                a_name, b_name = node.input[0], other
                av, bv = x, xo

            av = av.astype(np.float16)
            bv = bv.astype(np.float16)
            bc = add_broadcast(av.shape, bv.shape)

            y = (av.astype(np.float64) + bv.astype(np.float64)).astype(np.float16)
            src_override = (operand_place(a_name), operand_place(b_name))

            if bc is not None:
                rows, row_len, mode = bc
                op = "AddBcast"
                geom = dict(rows=rows, row_len=row_len, bcast_mode=mode)

        elif op == "Softmax":
            assert a.get("axis", -1) in (-1, len(x.shape) - 1), "softmax over the last axis"
            rows = int(np.prod(x.shape[:-1]))
            row_len = x.shape[-1]
            y = softmax_fp16_exact(x.reshape(rows, row_len)).reshape(x.shape)
            geom = dict(sm_rows=rows, sm_len=row_len)

        elif op == "Transpose":
            perm = a["perm"]
            y = np.ascontiguousarray(x.transpose(perm))
            t_in, perm_m, t_out = merge_transpose(list(x.shape), perm)
            geom = dict(rank=len(t_in), iterations=t_in[0], perm=perm_m,
                        t_in_shape=t_in, t_out_shape=t_out)

        elif op == "GlobalAveragePool":
            y = gap_fp16_exact(x)

        elif op == "Gemm":
            assert a.get("transB", 0) == 1 and a.get("transA", 0) == 0
            assert float(a.get("alpha", 1.0)) == 1.0 and float(a.get("beta", 1.0)) == 1.0
            W = init[node.input[1]].astype(np.float16)  # [1000, 320], ONNX transB = 1
            B = init[node.input[2]].astype(np.float16)
            y = gemm_fp16_exact(W, x.reshape(-1, 1), B).reshape(1, -1)
            geom = dict(has_bias=1, w_off=blob.add(W), b_off=blob.add(B))

        else:
            raise RuntimeError(f"unhandled op {node.op_type}")

        tensors[node.output[0]] = y
        dst_off = arena.alloc(y.size)
        place[node.output[0]] = ("arena", dst_off)

        if src_override is not None:
            src = list(src_override)
        else:
            src = [place[n] for n in node.input if n in place]

        # Only the 4-D ops read in_shape/out_shape; the others carry their own geometry.
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

    # An FP32 reference on the same input, reported and not asserted. The kernels' GELU
    # is a Schraudolph fast tanh applied 23 times and their softmax is another fast exp,
    # so the class this network predicts in FP16 need not be the one ONNX predicts - a
    # disagreement would be a finding about the gelu kernel, not a broken test.
    ref = None
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
        ref = sess.run(None, {in_name: X0.astype(np.float32)})[0].reshape(-1)
    except Exception as exc:  # onnxruntime missing, or the model rejects the input
        print(f"(no FP32 cross-check: {exc})")

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
        f.write("#ifndef TINYVIT_GRAPH_H_\n#define TINYVIT_GRAPH_H_\n\n")
        f.write("#include <stdint.h>\n\n")

        f.write("/* Operations the driver dispatches on. */\n")
        f.write("typedef enum {\n")
        for name, code in OPS.items():
            f.write(f"    TVIT_{name.upper()} = {code},\n")
        f.write("} tvit_op_t;\n\n")

        f.write("/* Where an operand lives: the arena, or the weight blob. */\n")
        f.write("#define TVIT_SPACE_ARENA (0)\n")
        f.write("#define TVIT_SPACE_BLOB  (1)\n\n")
        f.write(f"#define TVIT_MAX_RANK    ({MAX_RANK})\n\n")

        f.write("typedef struct {\n")
        f.write("    uint8_t op;           /* tvit_op_t                                  */\n")
        f.write("    uint8_t has_bias;\n")
        f.write("    uint8_t src0_space;   /* TVIT_SPACE_*                               */\n")
        f.write("    uint8_t src1_space;\n")
        f.write("    uint32_t src0_off;    /* byte offset in its space                   */\n")
        f.write("    uint32_t src1_off;\n")
        f.write("    uint32_t dst_off;     /* byte offset in the arena                   */\n")
        f.write("    uint32_t w_off;       /* blob: conv/gemm/matmul weights, ln scale   */\n")
        f.write("    uint32_t b_off;       /* blob: bias, ln bias                        */\n")
        f.write("    uint32_t in_shape[4];\n")
        f.write("    uint32_t out_shape[4];\n")
        f.write("    uint32_t kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, group;\n")
        f.write("    uint32_t numel;       /* output elements, for the flat ops          */\n")
        f.write("    uint32_t ln_rows, ln_len;             /* LayerNormalization         */\n")
        f.write("    uint32_t sm_rows, sm_len;             /* Softmax                    */\n")
        f.write("    uint32_t rows, row_len, bcast_mode;   /* AddBcast                   */\n")
        f.write("    uint32_t mm_m, mm_k, mm_o, mm_batches;/* MatMul, MatMul2D           */\n")
        f.write("    uint32_t rank, iterations;            /* Transpose, axes merged     */\n")
        f.write("    uint32_t perm[TVIT_MAX_RANK];\n")
        f.write("    uint32_t t_in_shape[TVIT_MAX_RANK];\n")
        f.write("    uint32_t t_out_shape[TVIT_MAX_RANK];\n")
        f.write("    const char *name;\n")
        f.write("} tvit_layer_t;\n\n")

        f.write(f"#define TVIT_NUM_LAYERS       ({len(layers)})\n")
        f.write(f"#define TVIT_BLOB_BYTES       ({blob_size})\n")
        f.write(f"#define TVIT_ARENA_BYTES      ({arena.size})\n")
        f.write(f"#define TVIT_INPUT_OFF        ({input_off})\n")
        f.write(f"#define TVIT_LN_EPS           ({float(np.float16(ln_eps)):.9g}f)\n\n")

        f.write("/* Pass/fail is the logits: bit-exact, and the product of every layer but\n")
        f.write(" * the softmax. The softmax output is reported but not failed on. */\n")
        f.write(f"#define TVIT_LOGITS_OFF       ({logits_place[1]})\n")
        f.write(f"#define TVIT_LOGITS_LEN       ({logits.size})\n")
        f.write("#define TVIT_LOGITS_ULP_TOLL  (0)\n")
        f.write(f"#define TVIT_GOLDEN_LOGITS_OFF ({golden_logits_off})\n\n")
        f.write(f"#define TVIT_OUTPUT_OFF       ({output_place[1]})\n")
        f.write(f"#define TVIT_OUTPUT_LEN       ({output.size})\n")
        f.write(f"#define TVIT_GOLDEN_OUTPUT_OFF ({golden_output_off})\n")
        f.write(f"#define TVIT_TOP1             ({top1})\n\n")

        f.write("/* Stop after this many layers, for bringing the test up one at a time. */\n")
        f.write("#ifndef TVIT_MAX_LAYERS\n")
        f.write("#define TVIT_MAX_LAYERS       TVIT_NUM_LAYERS\n")
        f.write("#endif\n\n")

        f.write("/* clang-format off */\n")
        f.write(f"static const tvit_layer_t tinyvit_layers[{len(layers)}] = {{\n")
        for L in layers:
            f.write("    { ")
            f.write(f".op = TVIT_{L['op'].upper()}, ")
            f.write(f".has_bias = {L.get('has_bias', 0)}, ")
            f.write(f".src0_space = TVIT_SPACE_{L['src0'][0].upper()}, ")
            f.write(f".src1_space = TVIT_SPACE_{L['src1'][0].upper()},\n      ")
            f.write(f".src0_off = {L['src0'][1]}, .src1_off = {L['src1'][1]}, ")
            f.write(f".dst_off = {L['dst']},\n      ")
            f.write(f".w_off = {L.get('w_off', 0)}, .b_off = {L.get('b_off', 0)},\n      ")
            f.write(f".in_shape = {clist(L['in_shape'], 4)}, ")
            f.write(f".out_shape = {clist(L['out_shape'], 4)},\n      ")
            f.write(f".kernel_h = {L.get('k_h', 0)}, .kernel_w = {L.get('k_w', 0)}, ")
            f.write(f".stride_h = {L.get('s_h', 0)}, .stride_w = {L.get('s_w', 0)}, ")
            f.write(f".pad_h = {L.get('p_h', 0)}, .pad_w = {L.get('p_w', 0)}, ")
            f.write(f".group = {L.get('group', 1)},\n      ")
            f.write(f".numel = {L['numel']},\n      ")
            f.write(f".ln_rows = {L.get('ln_rows', 0)}, .ln_len = {L.get('ln_len', 0)}, ")
            f.write(f".sm_rows = {L.get('sm_rows', 0)}, .sm_len = {L.get('sm_len', 0)},\n      ")
            f.write(f".rows = {L.get('rows', 0)}, .row_len = {L.get('row_len', 0)}, ")
            f.write(f".bcast_mode = {L.get('bcast_mode', 0)},\n      ")
            f.write(f".mm_m = {L.get('mm_m', 0)}, .mm_k = {L.get('mm_k', 0)}, ")
            f.write(f".mm_o = {L.get('mm_o', 0)}, .mm_batches = {L.get('mm_batches', 0)},\n      ")
            f.write(f".rank = {L.get('rank', 0)}, .iterations = {L.get('iterations', 0)},\n      ")
            f.write(f".perm = {clist(L.get('perm', []), MAX_RANK)}, ")
            f.write(f".t_in_shape = {clist(L.get('t_in_shape', []), MAX_RANK)},\n      ")
            f.write(f".t_out_shape = {clist(L.get('t_out_shape', []), MAX_RANK)},\n      ")
            f.write(f".name = \"{L['name']}\" }},\n")
        f.write("};\n")
        f.write("/* clang-format on */\n\n")
        f.write("#endif /* TINYVIT_GRAPH_H_ */\n")

    print()
    print(f"{len(layers)} layers emitted, {aliased} nodes aliased away")
    print(f"blob    {BLOB} : {blob_size/1024/1024:.2f} MB")
    print(f"arena                                : {arena.size/1024/1024:.2f} MB")
    print(f"header  {HEADER}")
    print(f"layers  {LAYER_DIR}/ : {len(layers)} tensors")
    print(f"top-1 class {top1}, logit {float(l64[top1]):.4f}, "
          f"p = {float(output.reshape(-1)[top1]):.4f}")
    print(f"fast-exp softmax deviates from a true softmax by up to {dev:.4f}")

    if ref is not None:
        ref64 = ref.astype(np.float64)
        top1_ref = int(np.argmax(ref64))
        rank = int(np.where(np.argsort(-ref64) == top1)[0][0])
        print(f"onnxruntime FP32 top-1 is class {top1_ref} "
              f"({'agrees' if top1_ref == top1 else 'DIFFERS'}); "
              f"the FP16 winner ranks {rank + 1} there")


if __name__ == "__main__":
    main()
