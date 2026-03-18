#!/usr/bin/env python3
"""
Generate test.h for the FlatAttention test from scratch.
Computes: O = softmax(Q @ K^T) @ V  using float16 arithmetic.
Also generates intermediate golden arrays for step-by-step debugging.

Usage:
  python3 tests/magia/mesh/flatatt/gen_golden.py
  python3 tests/magia/mesh/flatatt/gen_golden.py --s-size 16 --d-size 8 --mesh 2 --seed 42
  python3 tests/magia/mesh/flatatt/gen_golden.py -o custom/path/test.h

Intermediate arrays:
  s_golden[S_SIZE * S_SIZE]  - scores after Q @ K^T
  m_golden[S_SIZE]           - row-wise max of scores (init=0, matching HW)
  e_golden[S_SIZE * S_SIZE]  - exp(scores - rowmax) using soft_expf
  l_golden[S_SIZE]           - row-wise sum of exp scores (fp16 accumulation)
"""

import argparse
import os
import sys

import numpy as np

DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "include", "test.h"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate test.h for the FlatAttention test."
    )
    parser.add_argument("--s-size", type=int, default=16, help="Sequence length (default: 16)")
    parser.add_argument("--d-size", type=int, default=8, help="Head dimension (default: 8)")
    parser.add_argument("--mesh", type=int, default=2, help="Mesh size, NxN tiles (default: 2)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument("-o", "--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output path (default: {DEFAULT_OUTPUT})")
    return parser.parse_args()


def validate_params(s, d, mesh):
    errors = []

    if s % mesh != 0:
        errors.append(
            f"S_SIZE ({s}) must be divisible by mesh size ({mesh}).\n"
            f"  Hint: try --s-size {mesh * ((s // mesh) + 1)} or --s-size {mesh * (s // mesh) if s // mesh > 0 else mesh}"
        )

    if d % 4 != 0:
        errors.append(
            f"D_SIZE ({d}) must be divisible by 4 (RedMule PE-array alignment).\n"
            f"  Hint: try --d-size {4 * ((d // 4) + 1)} or --d-size {4 * (d // 4) if d // 4 > 0 else 4}"
        )

    if s % mesh == 0:
        tile_dim = s // mesh
        if tile_dim % 4 != 0:
            errors.append(
                f"S_SIZE/mesh = {tile_dim} must be divisible by 4 (tile alignment).\n"
                f"  Hint: tile_dim={tile_dim} is not aligned. Try --s-size {mesh * 4 * ((tile_dim // 4) + 1)}"
            )

    if errors:
        print("ERROR: Invalid parameters:\n", file=sys.stderr)
        for e in errors:
            print(f"  - {e}\n", file=sys.stderr)
        sys.exit(1)


def generate_inputs(s, d, seed):
    rng = np.random.default_rng(seed)
    q_inp = rng.uniform(-2, 2, size=(s, d)).astype(np.float16)
    # K^T layout (D x S), scaled for numerical stability
    k_inp = (rng.uniform(-1, 1, size=(d, s)) * (1.0 / (d * 3))).astype(np.float16)
    v_inp = rng.uniform(-2, 2, size=(s, d)).astype(np.float16)
    return q_inp, k_inp, v_inp


def soft_expf(x):
    """Match the C soft_expf implementation exactly."""
    x = float(x)
    if x > 11.0:
        x = 11.0
    if x < -17.0:
        return 0.0

    # Range reduction: x = n*ln2 + r
    n = int(x * 1.4426950408889634 + (0.5 if x >= 0 else -0.5))
    r = x - n * 0.6931471805599453

    # Taylor expansion
    e = 1.0 + r * (1.0 + r * (0.5 + r * (0.16666667 + r * (0.04166667 + r * 0.00833333))))

    # Multiply by 2^n
    if n > 0:
        for _ in range(n):
            e *= 2.0
    else:
        for _ in range(-n):
            e *= 0.5

    return e


def compute_golden(Q, K_T, V, S, D):
    # Step 1: S = Q @ K^T (RedMule: fp32 accumulation, fp16 output)
    scores = (Q.astype(np.float32) @ K_T.astype(np.float32)).astype(np.float16)
    s_golden = scores.copy()
    print(f"Scores range: [{float(scores.min()):.4f}, {float(scores.max()):.4f}]")

    # Step 2: Row-wise max, initialized to 0 (matching HW rowmax())
    m_golden = np.zeros(S, dtype=np.float16)
    for i in range(S):
        row_max = np.float16(0.0)
        for j in range(S):
            if scores[i, j] > row_max:
                row_max = scores[i, j]
        m_golden[i] = row_max
    print(f"Row max range: [{float(m_golden.min()):.4f}, {float(m_golden.max()):.4f}]")

    # Step 3: Subtract max (fp16 rowdiff)
    for i in range(S):
        for j in range(S):
            scores[i, j] = np.float16(float(scores[i, j]) - float(m_golden[i]))

    # Step 4: soft_expf (fp16 -> f32 Taylor -> fp16)
    for i in range(S):
        for j in range(S):
            scores[i, j] = np.float16(soft_expf(float(scores[i, j])))
    e_golden = scores.copy()
    print(f"Exp scores range: [{float(e_golden.min()):.6f}, {float(e_golden.max()):.4f}]")

    # Step 5: Row sum (fp16 accumulation, matching HW rowsum())
    l_golden = np.zeros(S, dtype=np.float16)
    for i in range(S):
        row_sum = np.float16(0.0)
        for j in range(S):
            row_sum = np.float16(float(row_sum) + float(scores[i, j]))
        l_golden[i] = row_sum
    print(f"Row sum range: [{float(l_golden.min()):.4f}, {float(l_golden.max()):.4f}]")

    # Step 6: S @ V (RedMule: fp32 accumulation, fp16 output)
    O = (scores.astype(np.float32) @ V.astype(np.float32)).astype(np.float16)

    # Step 7: Divide by row sum (fp16 rowdiv)
    for i in range(S):
        for j in range(D):
            O[i, j] = np.float16(float(O[i, j]) / float(l_golden[i]))
    o_golden = O

    print(f"Output shape: {o_golden.shape}")
    print(f"Output sample [0,:]: {o_golden[0,:]}")
    print(f"Output sample [-1,:]: {o_golden[-1,:]}")

    return o_golden, s_golden, m_golden, e_golden, l_golden


def format_array(arr, name, size_expr=None):
    """Format a numpy float16 array as a C array declaration."""
    if size_expr:
        decl = f'extern float16alt {name:<12s}[{size_expr}] = {{'
    else:
        decl = f'extern float16alt {name:<12s}[] = {{'

    vals = []
    for v in arr.flat:
        f32 = float(v)
        vals.append(f'{f32}f')

    lines = [decl]
    for i in range(0, len(vals), 8):
        chunk = ', '.join(vals[i:i+8])
        if i + 8 < len(vals):
            chunk += ','
        lines.append(chunk)
    lines.append('};')
    return '\n'.join(lines)


def write_test_h(output_path, s, d, seed, mesh,
                 q_inp, k_inp, v_inp,
                 o_golden, s_golden, m_golden, e_golden, l_golden):
    parts = []

    # Generation parameters comment
    parts.append(f"""\
// Generated by gen_golden.py (S={s}, D={d}, mesh={mesh}x{mesh}, seed={seed})""")

    # Include guard and defines
    parts.append("""\

#ifndef _TEST_FLATATT_INCLUDE_GUARD_
#define _TEST_FLATATT_INCLUDE_GUARD_""")

    parts.append(f"""
#define S_SIZE ({s})
#define D_SIZE ({d})
""")

    # Input arrays
    parts.append(format_array(q_inp.flatten(), 'q_inp'))
    parts.append("")
    parts.append(format_array(k_inp.flatten(), 'k_inp'))
    parts.append("")
    parts.append(format_array(v_inp.flatten(), 'v_inp'))
    parts.append("")

    # Output buffer
    parts.append(f'extern float16alt {"o_out":<12s}[S_SIZE * D_SIZE] = {{}};')

    # Golden output
    parts.append(format_array(o_golden.flatten(), 'o_golden'))
    parts.append("")

    # Intermediate golden arrays
    parts.append("// Intermediate golden arrays for step-by-step checking")
    parts.append(format_array(s_golden.flatten(), 's_golden', 'S_SIZE * S_SIZE'))
    parts.append("")
    parts.append(format_array(m_golden.flatten(), 'm_golden', 'S_SIZE'))
    parts.append("")
    parts.append(format_array(e_golden.flatten(), 'e_golden', 'S_SIZE * S_SIZE'))
    parts.append("")
    parts.append(format_array(l_golden.flatten(), 'l_golden', 'S_SIZE'))

    # Close include guard
    parts.append("#endif //_TEST_FLATATT_INCLUDE_GUARD_")
    parts.append("")  # trailing newline

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(parts))


def main():
    args = parse_args()
    s, d, mesh = args.s_size, args.d_size, args.mesh

    validate_params(s, d, mesh)

    print(f"Generating test.h: S_SIZE={s}, D_SIZE={d}, mesh={mesh}x{mesh}, seed={args.seed}")

    q_inp, k_inp, v_inp = generate_inputs(s, d, args.seed)
    print(f"Q shape: {q_inp.shape}, K^T shape: {k_inp.shape}, V shape: {v_inp.shape}")

    o_golden, s_golden, m_golden, e_golden, l_golden = compute_golden(
        q_inp, k_inp, v_inp, s, d
    )

    write_test_h(
        args.output, s, d, args.seed, mesh,
        q_inp, k_inp, v_inp,
        o_golden, s_golden, m_golden, e_golden, l_golden,
    )

    print(f"\nWrote {args.output}:")
    print(f"  q_inp:     {s*d} values ({s}x{d})")
    print(f"  k_inp:     {d*s} values ({d}x{s}, K^T layout)")
    print(f"  v_inp:     {s*d} values ({s}x{d})")
    print(f"  o_golden:  {s*d} values")
    print(f"  s_golden:  {s*s} values")
    print(f"  m_golden:  {s} values")
    print(f"  e_golden:  {s*s} values")
    print(f"  l_golden:  {s} values")


if __name__ == '__main__':
    main()
