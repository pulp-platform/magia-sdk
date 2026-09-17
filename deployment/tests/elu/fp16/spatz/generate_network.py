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


def nonnegative_flaot(value):
    try:
        fvalue = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid real number.")

    if fvalue < 0:
        raise argparse.ArgumentTypeError(f"Alpha must be positive ({value}).")

    return fvalue


def parse_args():
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for Elu test")

    parser.add_argument("N", type=positive_int, help="Batch size")
    parser.add_argument("C", type=positive_int, help="Number of input channels")
    parser.add_argument("H", type=positive_int, help="Spatial height dimension")
    parser.add_argument("W", type=positive_int, help="Spatial width dimension")

    parser.add_argument("--alpha", type=nonnegative_flaot, default=1.0, help="Coefficient of ELU (default: 1.0)")

    args = parser.parse_args()
    return args


def generate_input_data(args):
    X_shape = (args.N, args.C, args.H, args.W)
    X = np.random.uniform(-10, 5, size=X_shape).astype(np.float16)
    return X

def run_onnx_elu(X, alpha):
    X_info = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT16, X.shape)
    Y_info = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT16, X.shape)

    opset = onnx.helper.make_operatorsetid("", 22)
    node_def = onnx.helper.make_node('Elu', ['X'], ['Y'], alpha=alpha)
    graph_def = onnx.helper.make_graph([node_def], 'onnx-elu-test', [X_info], [Y_info])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    ses = ort.InferenceSession(model_def.SerializeToString())
    res = ses.run(None, {'X': X})

    return res[0], model_def


def save_deployment_files(X, Y, model_def):
    deployment_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(deployment_dir, exist_ok=True)

    onnx.save(model_def, os.path.join(deployment_dir, "network.onnx"))
    np.savez(os.path.join(deployment_dir, "inputs.npz"), X=X)
    np.savez(os.path.join(deployment_dir, "outputs.npz"), Y=Y)


def main():
    args = parse_args()

    X = generate_input_data(args)
    Y, model_def = run_onnx_elu(X, args.alpha)

    save_deployment_files(X, Y, model_def)

    print(f"Deployment files generated with [N:{args.N}, C:{args.C}, H:{args.H}, W:{args.W}]")


if __name__ == "__main__":
    main()
