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
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for MatMul test")

    parser.add_argument("N", type=positive_int, help="Batch size")
    parser.add_argument("C", type=positive_int, help="Number of input channels")

    parser.add_argument("M", type=positive_int, help="Rows of matrix A (Dimension M)")
    parser.add_argument("K", type=positive_int, help="Columns of A / Rows of B (Reduction dimension K)")
    parser.add_argument("O", type=positive_int, help="Columns of matrix B (Dimension O)")

    args = parser.parse_args()
    return args


def generate_input_data(args):
    A = np.random.randn(args.N, args.C, args.M, args.K).astype(np.float16)
    B = np.random.randn(args.N, args.C, args.K, args.O).astype(np.float16)

    return A, B


def run_onnx_matmul(A, B, args):
    Y_shape = (args.N, args.C, args.M, args.O)

    A_info = onnx.helper.make_tensor_value_info('A', onnx.TensorProto.FLOAT16, A.shape)
    B_info = onnx.helper.make_tensor_value_info('B', onnx.TensorProto.FLOAT16, B.shape)
    Y_info = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT16, Y_shape)

    opset = onnx.helper.make_operatorsetid("", 13)
    node_def = onnx.helper.make_node('MatMul', ['A', 'B'], ['Y'])
    graph_def = onnx.helper.make_graph([node_def], 'onnx-matmul4d-test', [A_info, B_info], [Y_info])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    ses = ort.InferenceSession(model_def.SerializeToString())
    res = ses.run(None, {'A': A, 'B': B})

    return res[0], model_def


def save_deployment_files(A, B, Y, model_def):
    deployment_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(deployment_dir, exist_ok=True)

    onnx.save(model_def, os.path.join(deployment_dir, "network.onnx"))

    np.savez(os.path.join(deployment_dir, "inputs.npz"), A=A, B=B)
    np.savez(os.path.join(deployment_dir, "outputs.npz"), Y=Y)


def main():
    args = parse_args()

    A, B = generate_input_data(args)
    Y, model_def = run_onnx_matmul(A, B, args)
    save_deployment_files(A, B, Y, model_def)

    print(f"Deployment files generated successfully for MatMul 4D w/ A={A.shape} B={B.shape}")


if __name__ == "__main__":
    main()
