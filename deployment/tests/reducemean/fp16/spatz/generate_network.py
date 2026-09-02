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
    parser = argparse.ArgumentParser(
        description="Generator of Input Data and Golden Model for ReduceMean test"
    )
    parser.add_argument("N", type=positive_int, help="Batch size")
    parser.add_argument("C", type=positive_int, help="Number of input channels")
    parser.add_argument("H", type=positive_int, help="Spatial height dimension")
    parser.add_argument("W", type=positive_int, help="Spatial width dimension")

    parser.add_argument("--axis", type=int, default=1, help="Axis along which to reduce")
    parser.add_argument("--keepdims", type=int, default=0, help="Keep the reduced dimension or not.")

    args = parser.parse_args()
    return args


def generate_input_data(args):
    input_shape = (args.N, args.C, args.H, args.W)
    data = np.random.randn(*input_shape).astype(np.float16)
    return data


def run_onnx_reducemean(data, args):
    if args.keepdims:
        reduced_shape = list(data.shape)
        reduced_shape[args.axis] = 1
    else:
        reduced_shape = list(data.shape)
        del reduced_shape[args.axis]

    data_info = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT16, data.shape)
    reduced_info = onnx.helper.make_tensor_value_info("reduced", onnx.TensorProto.FLOAT16, tuple(reduced_shape))

    opset = onnx.helper.make_operatorsetid("", 13)
    node_def = onnx.helper.make_node("ReduceMean", ["data"], ["reduced"], axes=[args.axis], keepdims=args.keepdims)
    graph_def = onnx.helper.make_graph([node_def], "onnx-reducemean-test", [data_info], [reduced_info])
    model_def = onnx.helper.make_model(graph_def, producer_name="onnx-generator", opset_imports=[opset])

    sess = ort.InferenceSession(model_def.SerializeToString())
    res = sess.run(None, {"data": data})

    return res[0], model_def


def save_deployment_files(data, reduced, model_def):
    deployment_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(deployment_dir, exist_ok=True)

    onnx.save(model_def, os.path.join(deployment_dir, "network.onnx"))
    np.savez(os.path.join(deployment_dir, "inputs.npz"), data=data)
    np.savez(os.path.join(deployment_dir, "outputs.npz"), reduced=reduced)


def main():
    args = parse_args()
    data = generate_input_data(args)
    reduced, model_def = run_onnx_reducemean(data, args)
    save_deployment_files(data, reduced, model_def)

    print(f"Deployment files generated with [N:{args.N}, C:{args.C}, H:{args.H}, W:{args.W}] reduced on axis {args.axis}")


if __name__ == "__main__":
    main()
