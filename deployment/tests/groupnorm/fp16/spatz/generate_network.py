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

    parser.add_argument("num_groups", type=positive_int, help="The number of groups. It should be a divisor of len")

    parser.add_argument("--epsilon", type=positive_float, default=1e-05, help="Epsilon value for BatchNorm (default: 1e-05)")

    args = parser.parse_args()
    if args.C % args.num_groups != 0:
        parser.error(f"The number of channels C ({args.C}) must be divisible by num_groups ({args.num_groups}).")

    return args


def generate_input_data(args):
    input_shape = (args.N, args.C, args.H, args.W)
    param_shape = (args.C, )

    X = np.random.randn(*input_shape).astype(np.float16)
    scale = (1.0 + 0.1 * np.random.randn(*param_shape)).astype(np.float16)
    B = (0.1 * np.random.randn(*param_shape)).astype(np.float16)

    return X, scale, B


# def run_onnx_groupnorm(X, scale, B, args):
#     X_info = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT16, X.shape)
#     scale_info = onnx.helper.make_tensor_value_info('Scale', onnx.TensorProto.FLOAT16, scale.shape)
#     B_info = onnx.helper.make_tensor_value_info('B', onnx.TensorProto.FLOAT16, B.shape)
#     Y_info = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT16, X.shape)

#     opset = onnx.helper.make_operatorsetid("", 21)
#     node_def = onnx.helper.make_node('GroupNormalization', ['X', 'Scale', 'B'], ['Y'], num_groups=args.num_groups, epsilon=args.epsilon)
#     graph_def = onnx.helper.make_graph([node_def], 'onnx-groupnorm-test', [X_info, scale_info, B_info], [Y_info])
#     model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

#     ses = ort.InferenceSession(model_def.SerializePartialToString())
#     res = ses.run(None, {'X': X, 'Scale': scale, 'B': B})

#     return res[0], model_def


# In this version scale and B are part of the node itself, not inputs
def run_onnx_groupnorm(X, scale, B, args):
    X_info = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT16, X.shape)
    Y_info = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT16, X.shape)

    scale_initializer = onnx.numpy_helper.from_array(scale, name='Scale')
    B_initializer = onnx.numpy_helper.from_array(B, name='B')

    opset = onnx.helper.make_operatorsetid("", 21)
    node_def = onnx.helper.make_node('GroupNormalization', ['X', 'Scale', 'B'], ['Y'], num_groups=args.num_groups, epsilon=args.epsilon)

    graph_def = onnx.helper.make_graph(
    nodes=[node_def], name='onnx-groupnorm-test', inputs=[X_info], outputs=[Y_info], initializer=[scale_initializer, B_initializer])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    ses = ort.InferenceSession(model_def.SerializePartialToString())
    res = ses.run(None, {'X': X})

    return res[0], model_def


# def save_deployment_files(X, scale, B, Y, model_def):
#     deployment_dir = os.path.dirname(os.path.abspath(__file__))
#     os.makedirs(deployment_dir, exist_ok=True)

#     onnx.save(model_def, os.path.join(deployment_dir, "network.onnx"))
#     np.savez(os.path.join(deployment_dir, "inputs.npz"), X=X, scale=scale, B=B)
#     np.savez(os.path.join(deployment_dir, "outputs.npz"), Y=Y)


# In this version scale and B are part of the node itself, not inputs
def save_deployment_files(X, Y, model_def):
    deployment_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(deployment_dir, exist_ok=True)

    onnx.save(model_def, os.path.join(deployment_dir, "network.onnx"))

    np.savez(os.path.join(deployment_dir, "inputs.npz"), X=X)
    np.savez(os.path.join(deployment_dir, "outputs.npz"), Y=Y)


def main():
    args = parse_args()

    X, scale, B = generate_input_data(args)
    Y, model_def = run_onnx_groupnorm(X, scale, B, args)

    # save_deployment_files(X, scale, B, Y, model_def)
    save_deployment_files(X, Y, model_def)

    print(f"Deployment files generated with [N:{args.N}, C:{args.C}, H:{args.H}, W:{args.W}]")


if __name__ == "__main__":
    main()
