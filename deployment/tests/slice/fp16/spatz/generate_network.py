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
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for Slice test")

    parser.add_argument("N", type=positive_int, help="Batch size")
    parser.add_argument("C", type=positive_int, help="Number of input channels")
    parser.add_argument("H", type=positive_int, help="Spatial height dimension")
    parser.add_argument("W", type=positive_int, help="Spatial width dimension")

    parser.add_argument("--axis", type=int, default=1, help="Axis along which to slice")
    parser.add_argument("--start", type=int, default=0, help="Start index for slice")
    parser.add_argument("--end", type=int, default=-1, help="End index for slice")

    args = parser.parse_args()
    return args


def generate_input_data(args):
    input_shape = (args.N, args.C, args.H, args.W)

    data = np.random.randn(*input_shape).astype(np.float16)

    axis_val = args.axis
    dim_at_axis = input_shape[axis_val]

    start_val = args.start
    end_val = args.end if args.end != -1 else dim_at_axis

    starts = np.array([start_val], dtype=np.int64)
    ends = np.array([end_val], dtype=np.int64)
    axes = np.array([axis_val], dtype=np.int64)
    steps = np.array([1], dtype=np.int64)

    return data, starts, ends, axes, steps


def run_onnx_slice(data, starts, ends, axes, steps):
    data_info = onnx.helper.make_tensor_value_info('data', onnx.TensorProto.FLOAT16, data.shape)

    sliced_info = onnx.helper.make_tensor_value_info('sliced', onnx.TensorProto.FLOAT16, None)

    starts_initializer = onnx.helper.make_tensor(name='starts', data_type=onnx.TensorProto.INT64, dims=starts.shape, vals=starts.flatten().tolist())
    ends_initializer = onnx.helper.make_tensor(name='ends', data_type=onnx.TensorProto.INT64, dims=ends.shape, vals=ends.flatten().tolist())
    axes_initializer = onnx.helper.make_tensor(name='axes', data_type=onnx.TensorProto.INT64, dims=axes.shape, vals=axes.flatten().tolist())
    steps_initializer = onnx.helper.make_tensor(name='steps', data_type=onnx.TensorProto.INT64, dims=steps.shape, vals=steps.flatten().tolist())

    opset = onnx.helper.make_operatorsetid("", 13)
    node_def = onnx.helper.make_node('Slice', ['data', 'starts', 'ends', 'axes', 'steps'], ['sliced'])
    graph_def = onnx.helper.make_graph([node_def], 'onnx-slice-test', [data_info], [sliced_info], initializer=[starts_initializer, ends_initializer, axes_initializer, steps_initializer])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    sess = ort.InferenceSession(model_def.SerializeToString())
    res = sess.run(None, {'data': data})

    model_def.graph.output[0].type.tensor_type.shape.Clear()
    for dim in res[0].shape:
        model_def.graph.output[0].type.tensor_type.shape.dim.add().dim_value = dim

    return res[0], model_def


def save_deployment_files(data, sliced, model_def):
    deployment_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(deployment_dir, exist_ok=True)

    onnx.save(model_def, os.path.join(deployment_dir, "network.onnx"))
    np.savez(os.path.join(deployment_dir, "inputs.npz"), data=data)
    np.savez(os.path.join(deployment_dir, "outputs.npz"), sliced=sliced)


def main():
    args = parse_args()

    data, starts, ends, axes, steps = generate_input_data(args)

    sliced, model_def = run_onnx_slice(data, starts, ends, axes, steps)

    save_deployment_files(data, sliced, model_def)

    print(f"Deployment files generated with [N:{args.N}, C:{args.C}, H:{args.H}, W:{args.W}] on Axis:{args.axis}")


if __name__ == "__main__":
    main()
