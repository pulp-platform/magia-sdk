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


def parse_args():
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for ONNX Sub test")

    parser.add_argument("N", type=positive_int, help="Batch size")
    parser.add_argument("C", type=positive_int, help="Number of input channels")
    parser.add_argument("H", type=positive_int, help="Spatial height dimension")
    parser.add_argument("W", type=positive_int, help="Spatial width dimension")

    args = parser.parse_args()
    return args


def generate_input_data(args):
    shape = (args.N, args.C, args.H, args.W)

    A = np.random.randn(*shape).astype(np.float16)
    B = np.random.randn(*shape).astype(np.float16)

    return A, B


def run_onnx_sub(A, B):
    A_info = onnx.helper.make_tensor_value_info('A', onnx.TensorProto.FLOAT16, A.shape)
    B_info = onnx.helper.make_tensor_value_info('B', onnx.TensorProto.FLOAT16, B.shape)
    C_info = onnx.helper.make_tensor_value_info('C', onnx.TensorProto.FLOAT16, A.shape)

    opset = onnx.helper.make_operatorsetid("", 14)
    node_def = onnx.helper.make_node('Sub', ['A', 'B'], ['C'])
    graph_def = onnx.helper.make_graph([node_def], 'onnx-sub-test', [A_info, B_info], [C_info])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    sess = ort.InferenceSession(model_def.SerializeToString())
    res = sess.run(None, {'A': A, 'B': B})

    return res[0], model_def


def save_deployment_files(A, B, G, model_def):
    deployment_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(deployment_dir, exist_ok=True)

    onnx.save(model_def, os.path.join(deployment_dir, "network.onnx"))
    np.savez(os.path.join(deployment_dir, "inputs.npz"), A=A, B=B)
    np.savez(os.path.join(deployment_dir, "outputs.npz"), C=G)


def main():
    args = parse_args()

    A, B = generate_input_data(args)
    G, model_def = run_onnx_sub(A, B)

    save_deployment_files(A, B, G, model_def)

    print(f"Deployment files generated with [N:{args.N}, C:{args.C}, H:{args.H}, W:{args.W}]")

if __name__ == "__main__":
    main()
