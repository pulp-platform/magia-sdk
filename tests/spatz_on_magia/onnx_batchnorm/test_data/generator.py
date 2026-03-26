import argparse
import onnx
import os


import numpy as np
import onnxruntime as ort


def positive_float(value):
    try:
        fvalue = float(value)

    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid Real number.")

    if fvalue <= 0:
        raise argparse.ArgumentTypeError(f"Epsilon must be positive ({value}).")

    return fvalue


def parse_args():
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for ONNX BatchNorm test")

    parser.add_argument("length", type=int, help="Input vector length")
    parser.add_argument("--epsilon", type=positive_float, default=1e-05, help="Epsilon value for BatchNorm (default: 1e-05)")

    args = parser.parse_args()

    return args.length, args.epsilon


def generate_input_data(length):
    input = (np.random.randn(length)).astype(np.float16)

    gamma = (1.0 + 0.1 * np.random.randn(length)).astype(np.float16)
    beta = (0.1 * np.random.randn(length)).astype(np.float16)
    mean = (input + 0.1 * np.random.randn(length)).astype(np.float16)
    var = (0.5 + 0.25 * np.abs(np.random.randn(length))).astype(np.float16)

    return input, gamma, beta, mean, var


def run_onnx_batchnorm(input, gamma, beta, mean, var, epsilon):
    X = input.reshape(1, -1)

    input_info = onnx.helper.make_tensor_value_info('I', onnx.TensorProto.FLOAT16, X.shape)
    scale_info = onnx.helper.make_tensor_value_info('Scale', onnx.TensorProto.FLOAT16, gamma.shape)
    bias_info = onnx.helper.make_tensor_value_info('Bias', onnx.TensorProto.FLOAT16, beta.shape)
    mean_info = onnx.helper.make_tensor_value_info('Mean', onnx.TensorProto.FLOAT16, mean.shape)
    var_info = onnx.helper.make_tensor_value_info('Var', onnx.TensorProto.FLOAT16, var.shape)
    output_info = onnx.helper.make_tensor_value_info('O', onnx.TensorProto.FLOAT16, X.shape)

    opset = onnx.helper.make_operatorsetid("", 15)
    node_def = onnx.helper.make_node('BatchNormalization', ['I', 'Scale', 'Bias', 'Mean', 'Var'], ['O'], epsilon=epsilon)
    graph_def = onnx.helper.make_graph([node_def], 'onnx-batchnorm-test', [input_info, scale_info, bias_info, mean_info, var_info], [output_info])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    ses = ort.InferenceSession(model_def.SerializeToString())
    res = ses.run(None, {'I': X, 'Scale': gamma, 'Bias': beta, 'Mean': mean, 'Var': var})

    return res[0].reshape(-1)


def format_array(array):
    return "{ " + ", ".join(f"{x:f}f" for x in array) + " }"


def format_float(value):
    return f"{value:f}f"


def generate_header_file(length, epsilon, input, gamma, beta, mean, var, expected, filename="data.h"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    with open(filepath, "w") as f:
        f.write("/* Automatically generated header file for Spatz ONNX testing */\n")
        f.write("#ifndef DATA_H_\n")
        f.write("#define DATA_H_\n\n")

        f.write(f"#define LEN {length}\n\n")

        f.write(f"static const float16 epsilon = {format_float(epsilon)};\n\n")

        f.write(f"static const float16 input_vec[] = {format_array(input)};\n")
        f.write(f"static const float16 gamma_vec[] = {format_array(gamma)};\n")
        f.write(f"static const float16 beta_vec[]  = {format_array(beta)};\n")
        f.write(f"static const float16 mean_vec[]  = {format_array(mean)};\n")
        f.write(f"static const float16 var_vec[]   = {format_array(var)};\n\n")

        f.write(f"static const float16 expected_vec[] = {format_array(expected)};\n\n")

        f.write("#endif   /* DATA_H_ */\n")


def main():
    length, epsilon = parse_args()

    input, gamma, beta, mean, var = generate_input_data(length)

    expected = run_onnx_batchnorm(input, gamma, beta, mean, var, epsilon)

    generate_header_file(length, epsilon, input, gamma, beta, mean, var, expected)

    print(f"File 'data.h' successfully generated with {length} elements.")


if __name__ == "__main__":
    main()
