import argparse
import onnx
import os
import numpy as np
import onnxruntime as ort

def positive_float(value):
    try:
        fvalue = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid real number.")

    if fvalue <= 0:
        raise argparse.ArgumentTypeError(f"Epsilon must be positive ({value}).")

    return fvalue


def parse_args():
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for ONNX Groupnorm test")

    parser.add_argument("length", type=int, help="Input vector length")
    parser.add_argument("num_groups", type=int, help="The number of groups. It should be a divisor of len")
    parser.add_argument("--epsilon", type=positive_float, default=1e-05, help="Epsilon value for Groupnorm (default: 1e-05)")

    args = parser.parse_args()
    return args


def generate_input_data(length):
    input = (np.random.randn(length)).astype(np.float16)

    # gamma = (np.random.randn(length)).astype(np.float16)
    # beta  = (np.random.randn(length)).astype(np.float16)

    # gamma = np.ones(length, dtype=np.float16)
    # beta  = np.zeros(length, dtype=np.float16)

    gamma = (0.5 + 0.5 * np.random.randn(length)).astype(np.float16)
    beta  = (0.1 * np.random.randn(length)).astype(np.float16)

    return input, gamma, beta


def run_onnx_groupnorm(input, gamma, beta, args):
    if (args.length % args.num_groups) != 0:
        raise ValueError("The number of groups should be a divisor of len")

    X = input.reshape(1, args.length)
    input_info  = onnx.helper.make_tensor_value_info('I', onnx.TensorProto.FLOAT16, X.shape)
    scale_info  = onnx.helper.make_tensor_value_info('Scale', onnx.TensorProto.FLOAT16, gamma.shape)
    bias_info   = onnx.helper.make_tensor_value_info('B', onnx.TensorProto.FLOAT16, beta.shape)
    output_info = onnx.helper.make_tensor_value_info('O', onnx.TensorProto.FLOAT16, X.shape)

    opset = onnx.helper.make_operatorsetid("", 21)
    node_def = onnx.helper.make_node('GroupNormalization', ['I', 'Scale', 'B'], ['O'], num_groups=args.num_groups, epsilon=args.epsilon)
    graph_def = onnx.helper.make_graph([node_def], 'onnx-groupnorm-test', [input_info, scale_info, bias_info], [output_info])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    ses = ort.InferenceSession(model_def.SerializeToString())
    res = ses.run( None, {'I': X, 'Scale': gamma, 'B': beta})

    return res[0].reshape(args.length)


def format_array(array):
    return "{ " + ", ".join(f"{x:f}f" for x in array) + " }"


def format_float(value):
    return f"{value:f}f"


def generate_header_file(args, input, gamma, beta, expected, filename="data.h"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    with open(filepath, "w") as f:
        f.write(f"/* Automatically generated header file for Spatz ONNX testing */\n")
        f.write(f"#ifndef DATA_H_\n")
        f.write(f"#define DATA_H_\n\n")

        f.write(f"#define LEN {args.length}\n")
        f.write(f"#define NUM_GROUPS {args.num_groups}\n\n")

        f.write(f"static const float16 epsilon = {format_float(args.epsilon)};\n\n")

        f.write(f"static const float16 input_vec[] = {format_array(input)};\n")
        f.write(f"static const float16 gamma_vec[] = {format_array(gamma)};\n")
        f.write(f"static const float16 beta_vec[]  = {format_array(beta)};\n\n")

        f.write(f"static const float16 expected_vec[] = {format_array(expected)};\n\n")

        f.write(f"#endif   /* DATA_H_ */\n")


def main():
    args = parse_args()

    input, gamma, beta = generate_input_data(args.length)

    expected = run_onnx_groupnorm(input, gamma, beta, args)

    generate_header_file(args, input, gamma, beta, expected)

    print(f"File 'data.h' successfully generated with {args.length} elements.")


if __name__ == "__main__":
    main()
