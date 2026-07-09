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

    if sorted(perm) != list(range(len(perm))):
        raise argparse.ArgumentTypeError(f"Invalid permutation indices {perm}. Must be a valid rearrangement of {list(range(len(perm)))}.")

    if perm[0] != 0:
        raise argparse.ArgumentTypeError(f"Transpose kernel strictly requires perm[0] == 0 (axis-0 sharding). Got {perm}.")

    return perm


def parse_args():
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for Transpose test")

    parser.add_argument("shape", type=positive_int, nargs='+', help="Input tensor shape as space-separated dimensions (rank = number of values)")

    parser.add_argument("--perm", type=parse_perm, default=None, help="Space-separated permutation vector with perm[0]=0, e.g. '0 2 1' (default: reverse all axes but axis 0)")

    args = parser.parse_args()

    if args.perm is None:
        args.perm = [0] + list(range(len(args.shape) - 1, 0, -1))

    if len(args.perm) != len(args.shape):
        parser.error(f"Permutation {args.perm} rank ({len(args.perm)}) must match input shape rank ({len(args.shape)}).")

    if args.shape[args.perm[-1]] % 2 != 0:
        parser.error(f"shape[perm[-1]] = {args.shape[args.perm[-1]]} (inner_len) must be even to keep the Spatz vector stores 4-byte aligned (RVV path).")

    return args


def generate_input_data(args):
    data_shape = tuple(args.shape)
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

    print(f"Deployment files generated with input shape {list(args.shape)} - perm: {args.perm}")


if __name__ == "__main__":
    main()
