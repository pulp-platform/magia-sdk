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
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for ONNX Scatter test")

    parser.add_argument("N", type=positive_int, help="Batch size")
    parser.add_argument("C", type=positive_int, help="Number of input channels")
    parser.add_argument("H", type=positive_int, help="Spatial height dimension")
    parser.add_argument("W", type=positive_int, help="Spatial width dimension")

    parser.add_argument("--axis", type=int, default=1, help="Axis for scatter (default: 1)")

    args = parser.parse_args()
    return args


def generate_input_data(args):
    shape = (args.N, args.C, args.H, args.W)
    axis_dim = shape[args.axis]

    data = np.random.randn(*shape).astype(np.float16)
    indices = np.random.randint(low=0, high=axis_dim, size=shape).astype(np.int64)
    updates = np.random.randn(*shape).astype(np.float16)

    return data, indices, updates


def run_onnx_scatter(data, indices, updates, axis):
    data_info = onnx.helper.make_tensor_value_info('data', onnx.TensorProto.FLOAT16, data.shape)
    indices_info = onnx.helper.make_tensor_value_info('indices', onnx.TensorProto.INT64, indices.shape)
    updates_info = onnx.helper.make_tensor_value_info('updates', onnx.TensorProto.FLOAT16, updates.shape)
    output_info = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT16, data.shape)

    opset = onnx.helper.make_operatorsetid("", 11)
    node_def = onnx.helper.make_node('ScatterElements', inputs=['data', 'indices', 'updates'], outputs=['output'], axis=axis)
    graph_def = onnx.helper.make_graph([node_def], 'onnx-scatter-test', [data_info, indices_info, updates_info], [output_info])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    sess = ort.InferenceSession(model_def.SerializeToString())
    res = sess.run(None, {'data': data, 'indices': indices, 'updates': updates})

    return res[0], model_def


def save_deployment_files(data, indices, updates, output, model_def):
    deployment_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(deployment_dir, exist_ok=True)

    onnx.save(model_def, os.path.join(deployment_dir, "network.onnx"))
    np.savez(os.path.join(deployment_dir, "inputs.npz"), data=data, indices=indices, updates=updates)
    np.savez(os.path.join(deployment_dir, "outputs.npz"), output=output)


def main():
    args = parse_args()

    data, indices, updates = generate_input_data(args)
    output, model_def = run_onnx_scatter(data, indices, updates, args.axis)

    save_deployment_files(data, indices, updates, output, model_def)

    print(f"Deployment files generated with [N:{args.N}, C:{args.C}, H:{args.H}, W:{args.W}, axis:{args.axis}]")


if __name__ == "__main__":
    main()
