import argparse
import onnx
import os
import numpy as np
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for ONNX Hardswish test")
    parser.add_argument("length", type=int, help="Input vector length")
    args = parser.parse_args()

    return args


def generate_input_data(length):
    input = (np.random.randn(length)).astype(np.float16)
    return input


def run_onnx_hardswish(input):
    input_info = onnx.helper.make_tensor_value_info('I', onnx.TensorProto.FLOAT16, input.shape)
    output_info = onnx.helper.make_tensor_value_info('O', onnx.TensorProto.FLOAT16, input.shape)

    opset = onnx.helper.make_operatorsetid("", 22)
    node_def = onnx.helper.make_node('HardSwish', ['I'], ['O'])
    graph_def = onnx.helper.make_graph([node_def], 'onnx-hardswish-test', [input_info], [output_info])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    ses = ort.InferenceSession(model_def.SerializePartialToString())
    res = ses.run(None, {'I':input})

    return res[0]


def format_array(array):
    return "{ " + ", ".join(f"{x:f}f" for x in array) + " }"


def generate_header_file(len, input, expected, filename="data.h"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    with open(filepath, "w") as f:
        f.write(f"/* Automatically generated header file for Spatz ONNX testing */\n")
        f.write(f"#ifndef DATA_H_\n")
        f.write(f"#define DATA_H_\n\n")

        f.write(f"#define LEN {len}\n\n")

        f.write(f"static const float16 input_vec[] = {format_array(input)};\n")
        f.write(f"static const float16 expected_vec[] = {format_array(expected)};\n\n")

        f.write(f"#endif   /* DATA_H_ */\n")


def main():
    args = parse_args()

    input = generate_input_data(args.length)

    expected = run_onnx_hardswish(input)

    generate_header_file(args.length, input, expected)

    print(f"File 'data.h' successfully generated")


if __name__ == "__main__":
    main()
