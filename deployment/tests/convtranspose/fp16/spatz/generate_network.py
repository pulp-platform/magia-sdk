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


def non_negative_int(value):
    try:
        ival = int(value)

    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid Integer number.")

    if ival < 0:
        raise argparse.ArgumentTypeError(f"Value must be non-negative ({value}).")

    return ival


def parse_args():
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for CovTranspose test")

    parser.add_argument("N", type=positive_int, help="Batch size")
    parser.add_argument("C", type=positive_int, help="Number of input channels")
    parser.add_argument("H", type=positive_int, help="Spatial height dimension")
    parser.add_argument("W", type=positive_int, help="Spatial width dimension")

    parser.add_argument("--strides", type=positive_int, default=4, help="Strides (default: 4)")
    parser.add_argument("--pads", type=non_negative_int, default=0, help="Padding (left=right=pad)")
    parser.add_argument("--kernel_shape", type=positive_int, default=4, help="Shape size (default 4)")
    parser.add_argument("--group", type=positive_int, default=4, help="number of groups input channels and output channels are divided into (default 1)")

    args = parser.parse_args()
    return args


def generate_input_data(args):
    X_shape = (args.N, args.C, args.H, args.W)
    W_shape = (args.C, args.C // args.group, args.kernel_shape, args.kernel_shape)

    X = np.random.randn(*X_shape).astype(np.float16)
    W = np.random.randn(*W_shape).astype(np.float16)

    return X, W


def get_output_shape(X_shape, args):
    N, C, H_in, W_in = X_shape
    C_out = C

    H_out = args.strides * (H_in - 1) + args.kernel_shape - 2 * args.pads
    W_out = args.strides * (W_in - 1) + args.kernel_shape - 2 * args.pads

    return (N, C_out, H_out, W_out)


def run_onnx_convtranspose(X, W, args):
    Y_shape = get_output_shape(X.shape, args)

    X_info = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT16, X.shape)
    W_info = onnx.helper.make_tensor_value_info('W', onnx.TensorProto.FLOAT16, W.shape)
    Y_info = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT16, Y_shape)

    opset = onnx.helper.make_operatorsetid("", 22)
    node_def = onnx.helper.make_node('ConvTranspose', inputs=['X', 'W'], outputs=['Y'], strides=[args.strides, args.strides], pads=[args.pads, args.pads, args.pads, args.pads], kernel_shape=[args.kernel_shape, args.kernel_shape], group=args.group) #todo
    graph_def = onnx.helper.make_graph([node_def], 'onnx-averagepool-test', [X_info, W_info], [Y_info])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    ses = ort.InferenceSession(model_def.SerializeToString())
    res = ses.run(None, {'X': X, 'W': W})

    return res[0], model_def


def save_deployment_files(X, W, Y, model_def):
    deployment_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(deployment_dir, exist_ok=True)

    onnx.save(model_def, os.path.join(deployment_dir, "network.onnx"))
    np.savez(os.path.join(deployment_dir, "inputs.npz"), X=X, W=W)
    np.savez(os.path.join(deployment_dir, "outputs.npz"), Y=Y)


def main():
    args = parse_args()

    X, W = generate_input_data(args)
    Y, model_def = run_onnx_convtranspose(X, W, args)

    save_deployment_files(X, W, Y, model_def)

    print(f"Deployment files generated with [N:{args.N}, C:{args.C}, H:{args.H}, W:{args.W}]")


if __name__ == "__main__":
    main()
