#!/usr/bin/env python3
# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""
Builds the data and the layer table for the onnx_resnet18 whole-network test.

Regenerate with: python3 generator.py

Reads ../resnet18.onnx (batchnorm already folded into the convolutions: 20 Conv,
17 Relu, 8 Add, 1 MaxPool, 1 GlobalAveragePool, 1 Flatten, 1 Gemm, 1 Softmax) and emits

  resnet18_data.bin        - one blob: every weight and bias in FP16, the input image,
                             and the two goldens. Pulled into the .l2_bulk section by
                             resnet18_data.S with .incbin, because the 11.7 M weights
                             would be a ~180 MB C source as an array.
  ../include/resnet18_graph.h
                           - the layer table, the offsets into the blob, and the
                             activation arena layout.
  layers/<nn>_<op>.npy     - every intermediate tensor, for bringing the test up one
                             layer at a time. Not compiled in.

GOLDEN MODEL

Each op is replayed exactly as the corresponding kernel computes it, so the network
output is reproducible bit for bit rather than merely close:

  Conv   conv2dgemm: im2col, then the GEMM accumulated over ascending k with one fused
         multiply-add per k (exact product, single rounding to FP16 each step), scaled
         by alpha, then the bias folded in with one more fused multiply-add. Blocking
         and the output-channel sharding are exact partitions of that same per-element
         computation, so neither changes the result.
  Relu   vfmax against zero.
  Add    vfadd.
  MaxPool
         max over the window clamped to the input - which is what ONNX's -inf padding
         amounts to.
  GlobalAveragePool
         vfredosum: ascending sum with FP16 rounding at every step, then divide by H*W.
  Gemm   same accumulation as Conv's GEMM. Called transposed (A = the [1000, 512] fc
         weight, B = the 512-long embedding, Y = [1000, 1]) because the kernel shards
         the GEMM's M: calling it the ONNX way round would make M = 1 and leave the
         whole layer on tile 0.
  Softmax
         NOT a true softmax. The kernel uses a Schraudolph fast exp - COEF * x + BIAS
         in FP16, truncated to an unsigned integer and reinterpreted as FP16 bits -
         which deviates from exp by ~0.01. The deviation is reported below.

         Its sum accumulates whole vector chunks element-wise and then folds the lanes
         with vfredosum, which RVV does pin down, so the golden replays both parts and
         the softmax is reproducible too. It is still reported rather than failed on:
         the pass criterion is the logits, which are bit-exact and cover the 49 layers
         before it, plus the predicted class.
"""

import os

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(HERE)
MODEL = os.path.join(TEST_DIR, "resnet18.onnx")
BLOB = os.path.join(HERE, "resnet18_data.bin")
HEADER = os.path.join(TEST_DIR, "include", "resnet18_graph.h")
LAYER_DIR = os.path.join(HERE, "layers")

SEED = 20260728

# Schraudolph fast exp constants, from softmax_fp16_spatz_task.c
COEF = np.float16(1486.0)
BIAS = np.float16(15360.0)

# VLMAX at e16/m8, i.e. how many elements share a lane accumulator. The magia_v3 GVSoC
# model gives its Spatz VLEN = 512 (gvsoc_work/gvsoc_config.json), so 512/16 * 8 = 256 -
# NOT the 128 that cmake/spatz_config.cmake's (unused) SPATZ_VLEN = 256 would suggest.
VLMAX = 256

# Ops the driver knows about; must match rn18_op_t in the generated header.
OPS = {
    "Conv": 0,
    "Relu": 1,
    "Add": 2,
    "MaxPool": 3,
    "GlobalAveragePool": 4,
    "Gemm": 5,
    "Softmax": 6,
}


# --- exact FP16 replays -------------------------------------------------------


def f16(x):
    return np.asarray(x).astype(np.float16)


def im2col(X_b, kh, kw, sh, sw, ph, pw, h_out, w_out):
    """B[k, n], k = (ic*K_h + ki)*K_w + kj, n = oh*W_out + ow, zero outside the input."""
    c_in, h_in, w_in = X_b.shape
    Xp = np.zeros((c_in, h_in + 2 * ph, w_in + 2 * pw), dtype=np.float16)
    Xp[:, ph : ph + h_in, pw : pw + w_in] = X_b

    oh = np.arange(h_out) * sh
    ow = np.arange(w_out) * sw
    out = np.empty((c_in * kh * kw, h_out * w_out), dtype=np.float16)
    ic = np.arange(c_in)

    for ki in range(kh):
        for kj in range(kw):
            patch = Xp[:, ki + oh[:, None], kj + ow[None, :]]
            out[(ic * kh + ki) * kw + kj, :] = patch.reshape(c_in, -1)

    return out


def gemm_fp16_exact(A, B, bias):
    """
    alpha = beta = 1, so: acc = 0; for ascending k, acc = fp16(acc + A[m,k]*B[k,n]);
    then acc = fp16(bias[m] + acc). float64 keeps each individual fused multiply-add
    exact before it is rounded back to FP16, which is what fmadd.h does.
    """
    M, K = A.shape
    N = B.shape[1]

    acc = np.zeros((M, N), dtype=np.float64)
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


def conv_fp16_exact(X, W, B, kh, kw, sh, sw, ph, pw, h_out, w_out):
    n, c_in = X.shape[0], X.shape[1]
    c_out = W.shape[0]
    A = W.reshape(c_out, -1)
    Y = np.empty((n, c_out, h_out, w_out), dtype=np.float16)

    for b in range(n):
        Bmat = im2col(X[b], kh, kw, sh, sw, ph, pw, h_out, w_out)
        Y[b] = gemm_fp16_exact(A, Bmat, B).reshape(c_out, h_out, w_out)

    return Y


def maxpool_fp16_exact(X, kh, kw, sh, sw, ph, pw, h_out, w_out):
    n, c, h_in, w_in = X.shape
    Y = np.empty((n, c, h_out, w_out), dtype=np.float16)

    for oh in range(h_out):
        h0 = max(oh * sh - ph, 0)
        h1 = min(oh * sh - ph + kh, h_in)
        for ow in range(w_out):
            w0 = max(ow * sw - pw, 0)
            w1 = min(ow * sw - pw + kw, w_in)
            Y[:, :, oh, ow] = X[:, :, h0:h1, w0:w1].max(axis=(2, 3))

    return Y


def gap_fp16_exact(X):
    """vfredosum: ascending sum, FP16 rounding at every step, then divide by H*W."""
    n, c, h, w = X.shape
    flat = X.reshape(n * c, h * w)
    acc = np.zeros(n * c, dtype=np.float64)

    for i in range(flat.shape[1]):
        acc += flat[:, i].astype(np.float64)
        acc = acc.astype(np.float16).astype(np.float64)

    return (acc.astype(np.float16) / np.float16(h * w)).astype(np.float16).reshape(n, c, 1, 1)


def fastexp_fp16(v):
    """vfmul.vf / vfadd.vf / vfcvt.rtz.xu.f.v, the result reinterpreted as FP16 bits."""
    t = (v.astype(np.float16) * COEF).astype(np.float16)
    t = (t + BIAS).astype(np.float16)
    u = np.clip(np.trunc(t.astype(np.float64)), 0, 65535).astype(np.uint16)
    return u.view(np.float16)


def softmax_fp16_exact(rows):
    """
    Row max, fast exp of (x - max), then the sum: whole VLMAX-wide chunks are added
    element-wise into an accumulator vector (ascending chunk order, the tail lanes of a
    short final chunk left as they were), then the lanes are folded ascending from zero
    with vfredosum.
    """
    G = np.zeros_like(rows)

    for r in range(rows.shape[0]):
        row = rows[r]
        mx = row.max()
        e = fastexp_fp16((row - mx).astype(np.float16))

        lanes = np.zeros(min(VLMAX, e.size), dtype=np.float16)
        for off in range(0, e.size, VLMAX):
            chunk = e[off : off + VLMAX]
            lanes[: chunk.size] = (lanes[: chunk.size].astype(np.float64)
                                   + chunk.astype(np.float64)).astype(np.float16)

        acc = np.float16(0.0)
        for v in lanes:
            acc = (acc + v).astype(np.float16)

        G[r] = (e / acc).astype(np.float16)

    return G


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
        out[a.name] = list(a.ints) if a.ints else a.i
    return out


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

    # A fixed pseudo-random image. ResNet18 is fully convolutional up to the pool, so
    # any input exercises every layer; a real photo would only change the class index.
    X0 = rng.standard_normal(in_shape).astype(np.float16)
    input_off = blob.add(X0)

    tensors = {in_name: X0}           # name -> FP16 value
    place = {in_name: ("blob", input_off)}  # name -> (space, byte offset)
    layers = []

    print(f"input {in_shape}, {len(graph.node)} nodes")

    for idx, node in enumerate(graph.node):
        a = attrs(node)
        op = node.op_type
        x = tensors[node.input[0]]

        if op == "Flatten":
            # NCHW [1, 512, 1, 1] and [1, 512] are the same bytes: alias, no layer.
            tensors[node.output[0]] = x.reshape(x.shape[0], -1)
            place[node.output[0]] = place[node.input[0]]
            print(f"{idx:2d} Flatten (alias, no layer emitted)")
            continue

        if op == "Conv":
            W = init[node.input[1]].astype(np.float16)
            B = init[node.input[2]].astype(np.float16) if len(node.input) > 2 else None
            kh, kw = a["kernel_shape"]
            sh, sw = a["strides"]
            ph, pw = a["pads"][0], a["pads"][1]
            h_out = (x.shape[2] + 2 * ph - kh) // sh + 1
            w_out = (x.shape[3] + 2 * pw - kw) // sw + 1
            y = conv_fp16_exact(x, W, B, kh, kw, sh, sw, ph, pw, h_out, w_out)
            w_off = blob.add(W)
            b_off = blob.add(B) if B is not None else 0
            geom = dict(k_h=kh, k_w=kw, s_h=sh, s_w=sw, p_h=ph, p_w=pw,
                        has_bias=1 if B is not None else 0, w_off=w_off, b_off=b_off)
            assert a.get("group", 1) == 1, "grouped convs not expected in resnet18"

        elif op == "Relu":
            y = np.maximum(x, np.float16(0.0)).astype(np.float16)
            geom = {}

        elif op == "Add":
            y = (x.astype(np.float64) + tensors[node.input[1]].astype(np.float64)).astype(
                np.float16
            )
            geom = {}

        elif op == "MaxPool":
            kh, kw = a["kernel_shape"]
            sh, sw = a["strides"]
            ph, pw = a["pads"][0], a["pads"][1]
            h_out = (x.shape[2] + 2 * ph - kh) // sh + 1
            w_out = (x.shape[3] + 2 * pw - kw) // sw + 1
            y = maxpool_fp16_exact(x, kh, kw, sh, sw, ph, pw, h_out, w_out)
            geom = dict(k_h=kh, k_w=kw, s_h=sh, s_w=sw, p_h=ph, p_w=pw)

        elif op == "GlobalAveragePool":
            y = gap_fp16_exact(x)
            geom = {}

        elif op == "Gemm":
            W = init[node.input[1]].astype(np.float16)   # [1000, 512], ONNX transB = 1
            B = init[node.input[2]].astype(np.float16)
            # Transposed call: A = W, B = the embedding as a column, Y = [1000, 1].
            y = gemm_fp16_exact(W, x.reshape(-1, 1), B).reshape(1, -1)
            geom = dict(has_bias=1, w_off=blob.add(W), b_off=blob.add(B))

        elif op == "Softmax":
            y = softmax_fp16_exact(x.reshape(1, -1)).reshape(x.shape)
            geom = {}

        else:
            raise RuntimeError(f"unhandled op {op}")

        tensors[node.output[0]] = y
        dst_off = arena.alloc(y.size)
        place[node.output[0]] = ("arena", dst_off)

        src = [place[n] for n in node.input if n in place]
        ish = list(x.shape) + [1] * (4 - len(x.shape))
        osh = list(y.shape) + [1] * (4 - len(y.shape))

        layers.append(dict(
            idx=idx, op=op, name=f"{idx:02d}_{op}",
            src0=src[0], src1=src[1] if len(src) > 1 else ("arena", 0),
            nsrc=len(src), dst=dst_off,
            in_shape=ish, out_shape=osh, numel=int(y.size), **geom,
        ))

        np.save(os.path.join(LAYER_DIR, f"{idx:02d}_{op}.npy"), y)
        print(f"{idx:2d} {op:18s} {ish} -> {osh}  |max| = "
              f"{float(np.abs(y.astype(np.float64)).max()):.5g}")

    # --- goldens -------------------------------------------------------------
    logits = tensors[graph.node[-2].output[0]]     # Gemm output
    output = tensors[graph.output[0].name]         # Softmax output
    logits_place = place[graph.node[-2].output[0]]
    output_place = place[graph.output[0].name]

    golden_logits_off = blob.add(logits)
    golden_output_off = blob.add(output)

    top1 = int(np.argmax(output.reshape(-1).astype(np.float64)))
    top1_logits = int(np.argmax(logits.reshape(-1).astype(np.float64)))
    assert top1 == top1_logits, "softmax is monotone, top-1 must agree with the logits"

    # How far the fast exp puts the output from a real softmax, for the record.
    l64 = logits.reshape(-1).astype(np.float64)
    true_sm = np.exp(l64 - l64.max())
    true_sm /= true_sm.sum()
    dev = float(np.max(np.abs(output.reshape(-1).astype(np.float64) - true_sm)))

    blob_size = blob.write(BLOB)

    # --- header --------------------------------------------------------------
    def cshape(s):
        return "{" + ", ".join(str(int(v)) for v in s) + "}"

    with open(HEADER, "w") as f:
        f.write("// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.\n")
        f.write("// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n")
        f.write("// SPDX-License-Identifier: Apache-2.0\n\n")
        f.write("/* Auto-generated by test_data/generator.py -- do not edit by hand.\n")
        f.write(" * Regenerate with: cd test_data && python3 generator.py\n */\n\n")
        f.write("#ifndef RESNET18_GRAPH_H_\n#define RESNET18_GRAPH_H_\n\n")
        f.write("#include <stdint.h>\n\n")

        f.write("/* Operations the driver dispatches on. */\n")
        f.write("typedef enum {\n")
        for name, code in OPS.items():
            f.write(f"    RN18_{name.upper()} = {code},\n")
        f.write("} rn18_op_t;\n\n")

        f.write("/* Where an operand lives. Only the first layer reads from the blob. */\n")
        f.write("#define RN18_SPACE_ARENA (0)\n")
        f.write("#define RN18_SPACE_BLOB  (1)\n\n")

        f.write("typedef struct {\n")
        f.write("    uint8_t op;           /* rn18_op_t                                  */\n")
        f.write("    uint8_t has_bias;\n")
        f.write("    uint8_t src0_space;   /* RN18_SPACE_*                               */\n")
        f.write("    uint8_t src1_space;\n")
        f.write("    uint32_t src0_off;    /* byte offset in its space                   */\n")
        f.write("    uint32_t src1_off;\n")
        f.write("    uint32_t dst_off;     /* byte offset in the arena                   */\n")
        f.write("    uint32_t w_off;       /* byte offset in the blob                    */\n")
        f.write("    uint32_t b_off;\n")
        f.write("    uint32_t in_shape[4];\n")
        f.write("    uint32_t out_shape[4];\n")
        f.write("    uint32_t kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w;\n")
        f.write("    uint32_t numel;       /* output elements, for the flat ops           */\n")
        f.write("    const char *name;\n")
        f.write("} rn18_layer_t;\n\n")

        f.write(f"#define RN18_NUM_LAYERS       ({len(layers)})\n")
        f.write(f"#define RN18_BLOB_BYTES       ({blob_size})\n")
        f.write(f"#define RN18_ARENA_BYTES      ({arena.size})\n")
        f.write(f"#define RN18_INPUT_OFF        ({input_off})\n\n")

        f.write("/* Pass/fail is the logits: bit-exact, and produced by the 49 layers before\n")
        f.write(" * the softmax. The softmax output is reported but not failed on, though it is\n")
        f.write(" * reproducible now that its lane fold is vfredosum. */\n")
        f.write(f"#define RN18_LOGITS_OFF       ({logits_place[1]})\n")
        f.write(f"#define RN18_LOGITS_LEN       ({logits.size})\n")
        f.write(f"#define RN18_LOGITS_ULP_TOLL  (0)\n")
        f.write(f"#define RN18_GOLDEN_LOGITS_OFF ({golden_logits_off})\n\n")
        f.write(f"#define RN18_OUTPUT_OFF       ({output_place[1]})\n")
        f.write(f"#define RN18_OUTPUT_LEN       ({output.size})\n")
        f.write(f"#define RN18_GOLDEN_OUTPUT_OFF ({golden_output_off})\n")
        f.write(f"#define RN18_TOP1             ({top1})\n\n")

        f.write("/* Stop after this many layers, for bringing the test up one at a time. */\n")
        f.write("#ifndef RN18_MAX_LAYERS\n")
        f.write("#define RN18_MAX_LAYERS       RN18_NUM_LAYERS\n")
        f.write("#endif\n\n")

        f.write("/* clang-format off */\n")
        f.write(f"static const rn18_layer_t resnet18_layers[{len(layers)}] = {{\n")
        for L in layers:
            f.write("    { ")
            f.write(f".op = RN18_{L['op'].upper()}, ")
            f.write(f".has_bias = {L.get('has_bias', 0)}, ")
            f.write(f".src0_space = RN18_SPACE_{L['src0'][0].upper()}, ")
            f.write(f".src1_space = RN18_SPACE_{L['src1'][0].upper()},\n      ")
            f.write(f".src0_off = {L['src0'][1]}, .src1_off = {L['src1'][1]}, ")
            f.write(f".dst_off = {L['dst']},\n      ")
            f.write(f".w_off = {L.get('w_off', 0)}, .b_off = {L.get('b_off', 0)},\n      ")
            f.write(f".in_shape = {cshape(L['in_shape'])}, ")
            f.write(f".out_shape = {cshape(L['out_shape'])},\n      ")
            f.write(f".kernel_h = {L.get('k_h', 0)}, .kernel_w = {L.get('k_w', 0)}, ")
            f.write(f".stride_h = {L.get('s_h', 0)}, .stride_w = {L.get('s_w', 0)}, ")
            f.write(f".pad_h = {L.get('p_h', 0)}, .pad_w = {L.get('p_w', 0)},\n      ")
            f.write(f".numel = {L['numel']}, .name = \"{L['name']}\" }},\n")
        f.write("};\n")
        f.write("/* clang-format on */\n\n")
        f.write("#endif /* RESNET18_GRAPH_H_ */\n")

    print()
    print(f"blob    {BLOB} : {blob_size/1024/1024:.2f} MB")
    print(f"arena                                : {arena.size/1024/1024:.2f} MB")
    print(f"header  {HEADER}")
    print(f"layers  {LAYER_DIR}/ : {len(layers)} tensors")
    print(f"top-1 class {top1}, logit {float(l64[top1]):.4f}, "
          f"p = {float(output.reshape(-1)[top1]):.4f}")
    print(f"fast-exp softmax deviates from a true softmax by up to {dev:.4f}")


if __name__ == "__main__":
    main()
