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


def parse_args():
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for ONNX Sub test")
    parser.add_argument("length", type=positive_int, help="Input vector length")
    args = parser.parse_args()

    return args


def generate_input_data(length):
    input_a = (np.random.randn(length)).astype(np.float16)
    input_b = (np.random.randn(length)).astype(np.float16)
    return input_a, input_b


def run_onnx_sub(input_a, input_b):
    a_info = onnx.helper.make_tensor_value_info('A', onnx.TensorProto.FLOAT16, input_a.shape)
    b_info = onnx.helper.make_tensor_value_info('B', onnx.TensorProto.FLOAT16, input_b.shape)
    output_info = onnx.helper.make_tensor_value_info('Diff', onnx.TensorProto.FLOAT16, input_a.shape)

    opset = onnx.helper.make_operatorsetid("", 14)
    node_def = onnx.helper.make_node('Sub', ['A', 'B'], ['Diff'])
    graph_def = onnx.helper.make_graph([node_def], 'onnx-sub-test', [a_info, b_info], [output_info])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    ses = ort.InferenceSession(model_def.SerializePartialToString())
    res = ses.run(None, {'A': input_a, 'B': input_b})

    return res[0]


def format_array(array):
    return "{ " + ", ".join(f"{x:f}f" for x in array) + " }"


def generate_header_file(args, input_a, input_b, expected, filename="data.h"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    with open(filepath, "w") as f:
        f.write(f"/* Automatically generated header file for Spatz ONNX testing */\n")
        f.write(f"#ifndef DATA_H_\n")
        f.write(f"#define DATA_H_\n\n")

        f.write(f"#define LEN {args.length}\n\n")

        f.write(f"static const float16 vec_a[] = {format_array(input_a)};\n")
        f.write(f"static const float16 vec_b[] = {format_array(input_b)};\n")
        f.write(f"static const float16 expected_vec[] = {format_array(expected)};\n\n")

        f.write(f"#endif   /* DATA_H_ */\n")


def main():
    args = parse_args()

    input_a, input_b = generate_input_data(args.length)

    expected = run_onnx_sub(input_a, input_b)

    generate_header_file(args, input_a, input_b, expected)

    print(f"File 'data.h' successfully generated")


if __name__ == "__main__":
    main()
