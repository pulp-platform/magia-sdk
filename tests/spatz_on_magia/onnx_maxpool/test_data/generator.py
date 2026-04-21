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


def non_negative_int(value):
    try:
        ival = int(value)

    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid Integer number.")

    if ival < 0:
        raise argparse.ArgumentTypeError(f"Value must be non-negative ({value}).")

    return ival


def parse_args():
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for ONNX MaxPool test.")

    parser.add_argument("length", type=non_negative_int, help="Input vector length")
    parser.add_argument("--dilation", type=positive_int, default=1, help="Dilations (default: 1)")
    parser.add_argument("--kernel_shape", type=positive_int, default=16, help="Shape size (default)")
    parser.add_argument("--pad", type=non_negative_int, default=0, help="Padding (left=right=pad)")
    parser.add_argument("--strides", type=positive_int, default=16, help="Strides (default: 16)")

    args = parser.parse_args()
    return args


def generate_input_data(length):
    input = (np.random.randn(length)).astype(np.float16)
    return input


def run_onnx_maxpool(input, args):
    X = input.reshape(1, 1, -1)

    input_info = onnx.helper.make_tensor_value_info('I', onnx.TensorProto.FLOAT16, X.shape)
    output_info = onnx.helper.make_tensor_value_info('O', onnx.TensorProto.FLOAT16, None)

    opset = onnx.helper.make_operatorsetid("", 22)
    node_def = onnx.helper.make_node('MaxPool', ['I'], ['O'], dilations=[args.dilation], kernel_shape=[args.kernel_shape], pads=[args.pad, args.pad], strides=[args.strides])
    graph_def = onnx.helper.make_graph([node_def], 'onnx-maxpool-test', [input_info], [output_info])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    ses = ort.InferenceSession(model_def.SerializeToString())
    res = ses.run(None, {'I':X})

    return res[0].reshape(-1)


def format_array(array):
    return "{ " + ", ".join(f"{x:f}f" for x in array) + " }"


def format_float(value):
    return f"{value:f}f"


def generate_header_file(args, input, expected, filename="data.h"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    len_input = args.length
    len_output = len(expected)

    with open(filepath, "w") as f:
        f.write(f"/* Automatically generated header file for Spatz ONNX testing */\n")
        f.write(f"#ifndef DATA_H_\n")
        f.write(f"#define DATA_H_\n\n")

        f.write(f"#define LEN_INPUT  {len_input}\n")
        f.write(f"#define LEN_OUTPUT {len_output}\n\n")

        f.write(f"#define DILATION {args.dilation}\n")
        f.write(f"#define SHAPE {args.kernel_shape}\n")
        f.write(f"#define STRIDE {args.strides}\n")
        f.write(f"#define PAD {args.pad}\n\n")

        f.write(f"static const float16 input_vec[] = {format_array(input)};\n\n")

        f.write(f"static const float16 expected_vec[] = {format_array(expected)};\n\n")

        f.write(f"#endif   /* DATA_H_ */\n")


def main():
    args = parse_args()

    input = generate_input_data(args.length)

    expected = run_onnx_maxpool(input, args)

    generate_header_file(args, input, expected)

    print(f"File 'data.h' successfully generated")


if __name__ == "__main__":
    main()
