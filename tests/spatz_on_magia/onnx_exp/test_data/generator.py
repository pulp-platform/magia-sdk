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
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for ONNX Exp test")
    parser.add_argument("length", type=positive_int, help="Input vector length")
    args = parser.parse_args()

    return args


def generate_input_data(length):
    input =  np.random.uniform(-10, 10, length).astype(np.float16)
    return input

def run_onnx_exp(input):
    input_info = onnx.helper.make_tensor_value_info('I', onnx.TensorProto.FLOAT16, input.shape)
    output_info = onnx.helper.make_tensor_value_info('O', onnx.TensorProto.FLOAT16, input.shape)

    opset = onnx.helper.make_operatorsetid("", 13)
    node_def = onnx.helper.make_node('Exp', ['I'], ['O'])
    graph_def = onnx.helper.make_graph([node_def], 'onnx-exp-test', [input_info], [output_info])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    ses = ort.InferenceSession(model_def.SerializePartialToString())
    res = ses.run(None, {'I':input})

    return res[0]


def format_array(array):
    return "{ " + ", ".join(f"{x:f}f" for x in array) + " }"


def generate_header_file(args, input, expected, filename="data.h"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    with open(filepath, "w") as f:
        f.write(f"/* Automatically generated header file for Spatz ONNX testing */\n")
        f.write(f"#ifndef DATA_H_\n")
        f.write(f"#define DATA_H_\n\n")

        f.write(f"#define LEN {args.length}\n\n")

        f.write(f"static const float16 input_vec[] = {format_array(input)};\n")
        f.write(f"static const float16 expected_vec[] = {format_array(expected)};\n\n")

        f.write(f"#endif   /* DATA_H_ */\n")


def main():
    args = parse_args()

    input = generate_input_data(args.length)

    expected = run_onnx_exp(input)

    generate_header_file(args, input, expected)

    print(f"File 'data.h' successfully generated")


if __name__ == "__main__":
    main()
