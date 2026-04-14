import argparse
import onnx
import os
import numpy as np
import onnxruntime as ort


def positive_int(value):
    try:
        ival = int(value)

    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid Integer number.")

    if ival <= 0:
        raise argparse.ArgumentTypeError(f"Value must be positive ({value}).")

    return ival


def positive_float(value):
    try:
        fvalue = float(value)

    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid Real number.")

    if fvalue <= 0:
        raise argparse.ArgumentTypeError(f"Epsilon must be positive ({value}).")

    return fvalue


def parse_args():
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for ONNX Hardsigmoid test")

    parser.add_argument("N", type=positive_int, help="Batch size")
    parser.add_argument("C", type=positive_int, help="Number of input channels")
    parser.add_argument("H", type=positive_int, help="Spatial height dimension")
    parser.add_argument("W", type=positive_int, help="Spatial width dimension")

    parser.add_argument("--alpha", type=positive_float, default=0.2, help="Value of alpha (default 0.2)")
    parser.add_argument("--beta", type=positive_float, default=0.5, help="Value of beta (default 0.5)")

    args = parser.parse_args()
    return args


def generate_input_data(args):
    shape = (args.N, args.C, args.H, args.W)
    X = np.random.randn(*shape).astype(np.float16)
    return X


def run_onnx_hardsigmoid(X, args):
    X_info = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT16, X.shape)
    Y_info = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT16, X.shape)

    opset = onnx.helper.make_operatorsetid("", 22)
    node_def = onnx.helper.make_node('HardSigmoid', ['X'], ['Y'], alpha=args.alpha, beta=args.beta)
    graph_def = onnx.helper.make_graph([node_def], 'onnx-hardsigmoid-test', [X_info], [Y_info])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    ses = ort.InferenceSession(model_def.SerializePartialToString())
    res = ses.run(None, {'X': X})

    return res[0]


def format_array(array):
    return "{ " + ", ".join(f"{x:f}f" for x in array.flatten()) + " }"


def format_float(value):
    return f"{value:f}f"


def generate_header_file(args, X, G, filename="data.h"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    with open(filepath, "w") as f:
        f.write(f"/* Automatically generated header file for Spatz ONNX testing */\n")
        f.write(f"#ifndef DATA_H_\n")
        f.write(f"#define DATA_H_\n\n")

        f.write(f"#define BATCH {args.N}\n")
        f.write(f"#define CHANNELS {args.C}\n")
        f.write(f"#define HEIGHT {args.H}\n")
        f.write(f"#define WIDTH {args.W}\n\n")

        f.write(f"static const float16 alpha = {format_float(args.alpha)};\n")
        f.write(f"static const float16 beta = {format_float(args.beta)};\n\n")

        f.write(f"static const float16 X[] = {format_array(X)};\n")
        f.write(f"static const float16 G[] = {format_array(G)};\n\n")

        f.write(f"#endif   /* DATA_H_ */\n")


def main():
    args = parse_args()

    X = generate_input_data(args)

    G = run_onnx_hardsigmoid(X, args)

    generate_header_file(args, X, G)

    print(f"File 'data.h' successfully generated with [N:{args.N}, C:{args.C}, H:{args.H}, W:{args.W}]")


if __name__ == "__main__":
    main()
