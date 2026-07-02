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


def target_shape_list(value):
    try:
        cleaned_value = value.replace(",", " ").split()
        target = [int(x) for x in cleaned_value]

    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid shape list.")

    return target


def parse_args():
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for Reshape test")

    parser.add_argument("N", type=positive_int, help="Batch size")
    parser.add_argument("C", type=positive_int, help="Number of input channels")
    parser.add_argument("H", type=positive_int, help="Spatial height dimension")
    parser.add_argument("W", type=positive_int, help="Spatial width dimension")

    parser.add_argument("--target_shape", type=target_shape_list, default=[-1], help="Target shape for Reshape")

    args = parser.parse_args()
    return args


def generate_input_data(args):
    input_shape = (args.N, args.C, args.H, args.W)

    data = np.random.randn(*input_shape).astype(np.float16)

    total_elements = data.size
    target = list(args.target_shape)

    if -1 in target:
        idx = target.index(-1)
        target[idx] = 1
        prod = np.prod(target)
        target[idx] = int(total_elements // prod)

    for i, dim in enumerate(target):
        if dim == 0 and i < len(input_shape):
            target[i] = input_shape[i]

    shape = np.array(args.target_shape, dtype=np.int64)

    return data, shape


def run_onnx_reshape(data, shape):
    output_shape = np.reshape(data, [dim if dim != 0 else data.shape[i] for i, dim in enumerate(shape)]).shape

    data_info = onnx.helper.make_tensor_value_info('data', onnx.TensorProto.FLOAT16, data.shape)
    reshaped_info = onnx.helper.make_tensor_value_info('reshaped', onnx.TensorProto.FLOAT16, output_shape)

    shape_initializer = onnx.helper.make_tensor(name='shape', data_type=onnx.TensorProto.INT64, dims=shape.shape, vals=shape.flatten().tolist())

    opset = onnx.helper.make_operatorsetid("", 25)
    node_def = onnx.helper.make_node('Reshape', ['data', 'shape'], ['reshaped'])
    graph_def = onnx.helper.make_graph([node_def], 'onnx-reshape-test', [data_info], [reshaped_info], initializer=[shape_initializer])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    sess = ort.InferenceSession(model_def.SerializeToString())
    res = sess.run(None, {'data': data})

    return res[0], model_def


def save_deployment_files(data, reshaped, model_def):
    deployment_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(deployment_dir, exist_ok=True)

    onnx.save(model_def, os.path.join(deployment_dir, "network.onnx"))
    np.savez(os.path.join(deployment_dir, "inputs.npz"), data=data)
    np.savez(os.path.join(deployment_dir, "outputs.npz"), reshaped=reshaped)


def main():
    args = parse_args()

    data, shape = generate_input_data(args)

    reshaped, model_def = run_onnx_reshape(data, shape)

    save_deployment_files(data, reshaped, model_def)

    print(f"Deployment files generated with [N:{args.N}, C:{args.C}, H:{args.H}, W:{args.W}]")


if __name__ == "__main__":
    main()
