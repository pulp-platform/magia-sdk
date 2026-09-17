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


def non_negative_int(value):
    try:
        ival = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid Integer number.")
    if ival < 0:
        raise argparse.ArgumentTypeError(f"Value must be non-negative ({value}).")
    return ival


def parse_args():
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for Conv test")

    parser.add_argument("N", type=positive_int, help="Batch size")
    parser.add_argument("C", type=positive_int, help="Number of input channels (C_in)")
    parser.add_argument("H", type=positive_int, help="Spatial height dimension")
    parser.add_argument("W", type=positive_int, help="Spatial width dimension")

    parser.add_argument("--strides", type=positive_int, default=1, help="Strides (default: 1)")
    parser.add_argument("--pads", type=non_negative_int, default=0, help="Padding (left=right=pad, default: 0)")
    parser.add_argument("--kernel_shape", type=positive_int, default=2, help="Kernel size (default: 2)")
    parser.add_argument("--group", type=positive_int, default=1, help="Number of groups (default: 1)")

    args = parser.parse_args()
    return args


def generate_input_data(args):
    X_shape = (args.N, args.C, args.H, args.W)
    W_shape = (args.C, args.C // args.group, args.kernel_shape, args.kernel_shape)
    B_shape = (args.C,)

    X = np.random.randn(*X_shape).astype(np.float16)
    W = np.random.randn(*W_shape).astype(np.float16)
    B = np.random.randn(*B_shape).astype(np.float16)

    return X, W, B


def get_output_shape(X_shape, args):
    N, C_in, H_in, W_in = X_shape
    C_out = C_in

    H_out = int(np.floor((H_in + 2 * args.pads - args.kernel_shape) / args.strides)) + 1
    W_out = int(np.floor((W_in + 2 * args.pads - args.kernel_shape) / args.strides)) + 1

    return (N, C_out, H_out, W_out)


def run_onnx_conv(X, W, B, args):
    Y_shape = get_output_shape(X.shape, args)

    X_info = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT16, X.shape)
    Y_info = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT16, Y_shape)
    W_initializer = onnx.helper.make_tensor('W', onnx.TensorProto.FLOAT16, W.shape, W.tobytes(), raw=True)
    B_initializer = onnx.helper.make_tensor('B', onnx.TensorProto.FLOAT16, B.shape, B.tobytes(), raw=True)

    opset = onnx.helper.make_operatorsetid("", 22)
    node_def = onnx.helper.make_node('Conv', inputs=['X', 'W', 'B'], outputs=['Y'], strides=[args.strides, args.strides], pads=[args.pads, args.pads, args.pads, args.pads], kernel_shape=[args.kernel_shape, args.kernel_shape], dilations=[1, 1],auto_pad='NOTSET',group=args.group)
    graph_def = onnx.helper.make_graph( nodes=[node_def],  name='onnx-conv2d-test',  inputs=[X_info], outputs=[Y_info],  initializer=[W_initializer, B_initializer])
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

    X, W, B = generate_input_data(args)
    Y, model_def = run_onnx_conv(X, W, B, args)

    save_deployment_files(X, Y, model_def)

    print(f"Deployment files generated with [N:{args.N}, C:{args.C}, H:{args.H}, W:{args.W}]")


if __name__ == "__main__":
    main()
