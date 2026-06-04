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


def parse_perm(value_str):
    try:
        perm = [int(x) for x in value_str.split()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Permutation indices must be space-separated integers. Got: '{value_str}'")

    if len(perm) != 4:
        raise argparse.ArgumentTypeError(f"Transpose kernel strictly requires a 4D permutation vector. Got {len(perm)} elements.")

    if sorted(perm) != [0, 1, 2, 3]:
        raise argparse.ArgumentTypeError(f"Invalid permutation indices {perm}. Must be a valid rearrangement of [0, 1, 2, 3].")

    return perm


def parse_args():
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for Transpose test")

    parser.add_argument("N", type=positive_int, help="Batch size")
    parser.add_argument("C", type=positive_int, help="Number of input channels")
    parser.add_argument("H", type=positive_int, help="Spatial height dimension")
    parser.add_argument("W", type=positive_int, help="Spatial width dimension")

    parser.add_argument("--perm", type=parse_perm, default=[0, 2, 3, 1], help="Space-separated 4D permutation vector, e.g., '0 2 3 1' (default: 0 2 3 1)")

    args = parser.parse_args()
    return args


def generate_input_data(args):
    data_shape = (args.N, args.C, args.H, args.W)
    data = np.random.randn(*data_shape).astype(np.float16)

    return data


def run_onnx_transpose(data, perm):
    output_shape = [data.shape[p] for p in perm]

    data_info = onnx.helper.make_tensor_value_info('data', onnx.TensorProto.FLOAT16, data.shape)
    transposed_info = onnx.helper.make_tensor_value_info('transposed', onnx.TensorProto.FLOAT16, output_shape)

    opset = onnx.helper.make_operatorsetid("", 25)
    node_def = onnx.helper.make_node('Transpose', ['data'], ['transposed'], perm=perm)
    graph_def = onnx.helper.make_graph([node_def], 'onnx-transpose-test', [data_info], [transposed_info])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    sess = ort.InferenceSession(model_def.SerializePartialToString())
    res = sess.run(None, {'data': data})

    return res[0], model_def


def save_deployment_files(data, transposed, model_def):
    deployment_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(deployment_dir, exist_ok=True)

    onnx.save(model_def, os.path.join(deployment_dir, "network.onnx"))
    np.savez(os.path.join(deployment_dir, "inputs.npz"), data=data)
    np.savez(os.path.join(deployment_dir, "outputs.npz"), transposed=transposed)


def main():
    args = parse_args()

    data = generate_input_data(args)
    transposed, model_def = run_onnx_transpose(data, args.perm)

    save_deployment_files(data, transposed, model_def)

    print(f"Deployment files generated with input [N:{args.N}, C:{args.C}, H:{args.H}, W:{args.W}] - perm: {args.perm}")

if __name__ == "__main__":
    main()
