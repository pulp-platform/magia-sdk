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
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for Gather test")

    parser.add_argument("N", type=positive_int, help="Batch size")
    parser.add_argument("C", type=positive_int, help="Number of input channels")
    parser.add_argument("H", type=positive_int, help="Spatial height dimension")
    parser.add_argument("W", type=positive_int, help="Spatial width dimension")

    parser.add_argument("--axis", type=int, default=1, help="Axis along which to gather (default: 1)")
    parser.add_argument("--index", type=int, default=0, help="Index to extract along the specified axis (default: 0)")

    args = parser.parse_args()
    return args


def generate_input_data(args):
    input_shape = (args.N, args.C, args.H, args.W)
    gather_dim_size = input_shape[args.axis]

    effective_index = args.index if args.index >= 0 else gather_dim_size + args.index
    assert 0 <= effective_index < gather_dim_size, f"Index {args.index} is out of bounds for axis {args.axis} with size {gather_dim_size}!"

    data = np.random.randn(*input_shape).astype(np.float16)
    indices = np.array([args.index], dtype=np.int64)

    return data, indices


def run_onnx_gather(data, indices, axis):
    output_shape = list(data.shape)
    output_shape[axis] = 1

    data_info = onnx.helper.make_tensor_value_info('data', onnx.TensorProto.FLOAT16, data.shape)
    output_info = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT16, output_shape)
    indices_initializer = onnx.helper.make_tensor('indices', onnx.TensorProto.INT64, indices.shape, indices.tolist())

    opset = onnx.helper.make_operatorsetid("", 13)
    node_def = onnx.helper.make_node('Gather', ['data', 'indices'], ['output'], axis=axis)
    graph_def = onnx.helper.make_graph([node_def], 'onnx-gather-test', inputs=[data_info], outputs=[output_info], initializer=[indices_initializer])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    sess = ort.InferenceSession(model_def.SerializePartialToString())
    res = sess.run(None, {'data': data})

    return res[0], model_def


def save_deployment_files(data, output, model_def):
    deployment_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(deployment_dir, exist_ok=True)

    onnx.save(model_def, os.path.join(deployment_dir, "network.onnx"))
    np.savez(os.path.join(deployment_dir, "inputs.npz"), data=data)
    np.savez(os.path.join(deployment_dir, "outputs.npz"), output=output)


def main():
    args = parse_args()

    data, indices = generate_input_data(args)
    output, model_def = run_onnx_gather(data, indices, args.axis)

    save_deployment_files(data, output, model_def)

    print(f"Deployment files generated with data [N:{args.N}, C:{args.C}, H:{args.H}, W:{args.W}] axis={args.axis}, index={args.index}")


if __name__ == "__main__":
    main()
