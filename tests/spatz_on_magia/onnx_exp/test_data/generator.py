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

    parser.add_argument("N", type=positive_int, help="Batch size")
    parser.add_argument("C", type=positive_int, help="Number of input channels")
    parser.add_argument("H", type=positive_int, help="Spatial height dimension")
    parser.add_argument("W", type=positive_int, help="Spatial width dimension")

    args = parser.parse_args()
    return args


def generate_input_data(args):
    shape = (args.N, args.C, args.H, args.W)
    input = np.random.randn(*shape).astype(np.float16)
    return input


def run_onnx_exp(input):
    input_info = onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT16, input.shape)
    output_info = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT16, input.shape)

    opset = onnx.helper.make_operatorsetid("", 13)
    node_def = onnx.helper.make_node('Exp', ['input'], ['output'])
    graph_def = onnx.helper.make_graph([node_def], 'onnx-exp-test', [input_info], [output_info])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    ses = ort.InferenceSession(model_def.SerializePartialToString())
    res = ses.run(None, {'input':input})

    return res[0]


def format_array(array):
    return "{ " + ", ".join(f"{x:f}f" for x in array.flatten()) + " }"


def generate_header_file(args, input, output, filename="data.h"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    with open(filepath, "w") as f:
        f.write("/* Automatically generated header file for Spatz ONNX testing */\n")
        f.write("#ifndef DATA_H_\n")
        f.write("#define DATA_H_\n\n")

        f.write(f"#define BATCH {args.N}\n")
        f.write(f"#define CHANNELS {args.C}\n")
        f.write(f"#define HEIGHT {args.H}\n")
        f.write(f"#define WIDTH {args.W}\n\n")

        f.write(f"static const float16 input[] = {format_array(input)};\n")
        f.write(f"static const float16 golden[] = {format_array(output)};\n\n")

        f.write("#endif  /* DATA_H_ */\n")


def main():
    args = parse_args()

    input = generate_input_data(args)

    output = run_onnx_exp(input)

    generate_header_file(args, input, output)

    print(f"File 'data.h' successfully generated with [N:{args.N}, C:{args.C}, H:{args.H}, W:{args.W}]")


if __name__ == "__main__":
    main()
