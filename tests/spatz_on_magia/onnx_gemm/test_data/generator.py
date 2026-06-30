import argparse
import onnx
import os
import numpy as np
import onnxruntime as ort


def positive_int(value):
    try:
        val = int(value)

    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer number")

    if val <= 0:
        raise argparse.ArgumentTypeError(f"'{value}' must be positive ({value}).")

    return val


def float_value(value):
    try:
        val = float(value)

    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid float number.")

    return val


def parse_args():
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for ONNX GEMM test.")

    parser.add_argument("M", type=positive_int, help="Rows of A anc C")
    parser.add_argument("N", type=positive_int, help="Columns of B and C")
    parser.add_argument("K", type=positive_int, help="Columns of A - Rows of B")

    parser.add_argument("--alpha", type=float_value, default=1.0, help="Scalar multiplier for the product of input tensors A * B.")
    parser.add_argument("--beta", type=float_value, default=1.0, help="Scalar multiplier for input tensor C.")
    parser.add_argument("--transA", type=int, choices=[0, 1], default=0, help="Whether A should be transposed (default: 0)")
    parser.add_argument("--transB", type=int, choices=[0, 1], default=0, help="Whether B should be transposed (default: 0)")

    args = parser.parse_args()
    return args


def generate_input_data(args):
    M = args.M
    N = args.N
    K = args.K

    # A
    if args.transA:
        A = np.random.randn(K, M).astype(np.float16)
    else:
        A = np.random.randn(M, K).astype(np.float16)

    # B
    if args.transB:
        B = np.random.randn(N, K).astype(np.float16)
    else:
        B = np.random.randn(K, N).astype(np.float16)

    C = np.random.randn(M, N).astype(np.float16)

    return A, B, C

def run_onnx_gemm(A, B, C, args):
    transA = args.transA
    transB = args.transB
    alpha = args.alpha
    beta = args.beta

    A_info = onnx.helper.make_tensor_value_info('A', onnx.TensorProto.FLOAT16, A.shape)
    B_info = onnx.helper.make_tensor_value_info('B', onnx.TensorProto.FLOAT16, B.shape)
    C_info = onnx.helper.make_tensor_value_info('C', onnx.TensorProto.FLOAT16, C.shape)
    Y_info = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT16,(args.M, args.N))

    opset = onnx.helper.make_operatorsetid("", 13)
    node_def = onnx.helper.make_node('Gemm', ['A', 'B', 'C'], ['Y'], alpha=alpha, beta=beta, transA=transA, transB=transB)
    graph_def = onnx.helper.make_graph([node_def], 'onnx-gemm-test', [A_info, B_info, C_info], [Y_info])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    ses = ort.InferenceSession(model_def.SerializeToString())
    res = ses.run(None, {'A': A, 'B': B, 'C': C})

    return res[0]


def format_array(array):
    flat = array.flatten()
    return "{ " + ", ".join(f"{x:f}f" for x in flat) + " }"


def format_float(value):
    return f"{value:f}f"

def generate_header_file(A, B, C, Y, args, filename="data.h"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    with open(filepath, "w") as f:
        f.write("/* Automatically generated header file for Spatz ONNX GEMM */\n")
        f.write("#ifndef DATA_H_\n")
        f.write("#define DATA_H_\n\n")

        f.write(f"#define DIM_M {args.M}\n")
        f.write(f"#define DIM_N {args.N}\n")
        f.write(f"#define DIM_K {args.K}\n\n")

        f.write(f"#define TRANS_A {args.transA}\n")
        f.write(f"#define TRANS_B {args.transB}\n\n")

        f.write(f"static const float16 ALPHA = {format_float(args.alpha)};\n")
        f.write(f"static const float16 BETA  = {format_float(args.beta)};\n\n")

        f.write(f"static const float16 A[] = {format_array(A)};\n\n")
        f.write(f"static const float16 B[] = {format_array(B)};\n\n")
        f.write(f"static const float16 C[] = {format_array(C)};\n\n")

        f.write(f"static const float16 G[] = {format_array(Y)};\n\n")

        f.write("#endif   /* DATA_H_ */\n")



def main():
    args = parse_args()

    A, B, C = generate_input_data(args)

    Y = run_onnx_gemm(A, B, C, args)

    generate_header_file(A, B, C, Y, args)

    print(f"File 'data.h' successfully generated (M={args.M} N={args.N} K={args.K} alpha={args.alpha} beta={args.beta} transA={args.transA} transB={args.transB})")


if __name__ == "__main__":
    main()
