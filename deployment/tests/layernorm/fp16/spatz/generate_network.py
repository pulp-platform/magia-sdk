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
    param_shape = (args.W,)

    X = np.random.randn(*input_shape).astype(np.float16)
    scale = (0.5 + 0.5 * np.random.randn(*param_shape).astype(np.float16))
    B = (0.1 + np.random.randn(*param_shape)).astype(np.float16)

    return X, scale, B


def run_onnx_layernorm(X, scale, B, epsilon):
    X_info  = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT16, X.shape)
    scale_info  = onnx.helper.make_tensor_value_info('Scale', onnx.TensorProto.FLOAT16, scale.shape)
    B_info   = onnx.helper.make_tensor_value_info('B', onnx.TensorProto.FLOAT16, B.shape)
    Y_info = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT16, X.shape)

    opset = onnx.helper.make_operatorsetid("", 17)
    node_def = onnx.helper.make_node('LayerNormalization', ['X', 'Scale', 'B'], ['Y'], axis=-1, epsilon=epsilon)
    graph_def = onnx.helper.make_graph([node_def], 'onnx-layernorm-test', [X_info, scale_info, B_info], [Y_info])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    ses = ort.InferenceSession(model_def.SerializeToString())
    res = ses.run( None, {'X': X, 'Scale': scale, 'B': B})

    return res[0], model_def


def save_deployment_files(X, scale, B, Y, model_def):
    deployment_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(deployment_dir, exist_ok=True)

    onnx.save(model_def, os.path.join(deployment_dir, "network.onnx"))
    np.savez(os.path.join(deployment_dir, "inputs.npz"), X=X, scale=scale, B=B)
    np.savez(os.path.join(deployment_dir, "outputs.npz"), Y=Y)


def main():
    args = parse_args()

    X, scale, B = generate_input_data(args)

    G, model_def = run_onnx_layernorm(X, scale, B, args.epsilon)

    save_deployment_files(X, scale, B, G, model_def)

    print(f"Deployment files generated with [N:{args.N}, C:{args.C}, H:{args.H}, W:{args.W}]")


if __name__ == "__main__":
    main()
