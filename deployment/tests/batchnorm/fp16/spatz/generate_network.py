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
        raise argparse.ArgumentTypeError((f"Value must be positive ({value})."))

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
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for BatchNormalization test")

    parser.add_argument("N", type=positive_int, help="Batch size")
    parser.add_argument("C", type=positive_int, help="Number of input channels")
    parser.add_argument("H", type=positive_int, help="Spatial height dimension")
    parser.add_argument("W", type=positive_int, help="Spatial width dimension")

    parser.add_argument("--epsilon", type=positive_float, default=1e-05, help="Epsilon value for BatchNorm (default: 1e-05)")

    args = parser.parse_args()
    return args


def generate_input_data(args):
    input_shape = (args.N, args.C, args.H, args.W)
    param_shape = (args.C, )

    X = np.random.randn(*input_shape).astype(np.float16)

    input_mean = np.mean(X, axis=(0, 2, 3)).astype(np.float16)

    input_var = np.var(X, axis=(0, 2, 3)).astype(np.float16)
    input_var = np.maximum(input_var, np.float16(0.1)).astype(np.float16)

    scale = (1.0 + 0.1 * np.random.randn(*param_shape)).astype(np.float16)

    B = (0.1 * np.random.randn(*param_shape)).astype(np.float16)

    return X, scale, B, input_mean, input_var


def run_onnx_batchnorm(X, scale, B, input_mean, input_var, epsilon):
    X_info = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT16, X.shape)

    scale_info = onnx.helper.make_tensor_value_info('scale', onnx.TensorProto.FLOAT16, scale.shape)
    B_info = onnx.helper.make_tensor_value_info('B', onnx.TensorProto.FLOAT16, B.shape)
    input_mean_info = onnx.helper.make_tensor_value_info('input_mean', onnx.TensorProto.FLOAT16, input_mean.shape)
    input_var_info = onnx.helper.make_tensor_value_info('input_var', onnx.TensorProto.FLOAT16, input_var.shape)
    Y_info = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT16, X.shape)

    opset = onnx.helper.make_operatorsetid("", 15)
    node_def = onnx.helper.make_node('BatchNormalization',['X', 'scale', 'B', 'input_mean', 'input_var'], ['Y'], epsilon=epsilon)
    graph_def = onnx.helper.make_graph([node_def], 'onnx-batchnorm-test', [X_info, scale_info, B_info, input_mean_info, input_var_info], [Y_info])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    sess = ort.InferenceSession(model_def.SerializeToString())
    res = sess.run(None, {'X': X, 'scale': scale, 'B': B, 'input_mean': input_mean, 'input_var': input_var})

    return res[0], model_def


def save_deployment_files(X, scale, B, input_mean, input_var, G, model_def):
    deployment_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(deployment_dir, exist_ok=True)

    onnx.save(model_def, os.path.join(deployment_dir, "network.onnx"))
    np.savez(os.path.join(deployment_dir, "inputs.npz"), X=X, scale=scale, B=B, input_mean=input_mean, input_var=input_var)

    np.savez(os.path.join(deployment_dir, "outputs.npz"), Y=G)


def main():
    args = parse_args()

    X, scale, B, input_mean, input_var = generate_input_data(args)

    G, model_def = run_onnx_batchnorm(X, scale, B, input_mean, input_var, args.epsilon)

    save_deployment_files(X, scale, B, input_mean, input_var, G, model_def)

    print(f"Deployment files generated with [N:{args.N}, C:{args.C}, H:{args.H}, W:{args.W}]")


if __name__ == "__main__":
    main()
