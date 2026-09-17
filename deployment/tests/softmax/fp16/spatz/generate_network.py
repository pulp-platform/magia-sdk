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
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for Softmax test")

    parser.add_argument("N", type=positive_int, help="Batch size")
    parser.add_argument("C", type=positive_int, help="Number of input channels")
    parser.add_argument("H", type=positive_int, help="Spatial height dimension")
    parser.add_argument("W", type=positive_int, help="Spatial width dimension")

    parser.add_argument("--axis", type=int, default=-1, help="Axis along which to apply softmax (default: -1)")

    args = parser.parse_args()
    return args


def generate_input_data(args):
    input_shape = (args.N, args.C, args.H, args.W)
    input = np.random.randn(*input_shape).astype(np.float16)
    return input


def run_onnx_softmax(input, axis):
    input_info = onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT16, input.shape)
    output_info = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT16, input.shape)

    opset = onnx.helper.make_operatorsetid("", 13)
    node_def = onnx.helper.make_node('Softmax', ['input'], ['output'], axis=axis)
    graph_def = onnx.helper.make_graph([node_def], 'onnx-softmax-test', [input_info], [output_info])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    ses = ort.InferenceSession(model_def.SerializeToString())
    res = ses.run(None, {'input':input})

    return res[0], model_def


def save_deployment_files(input, output, model_def):
    deployment_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(deployment_dir, exist_ok=True)

    onnx.save(model_def, os.path.join(deployment_dir, "network.onnx"))
    np.savez(os.path.join(deployment_dir, "inputs.npz"), input=input)
    np.savez(os.path.join(deployment_dir, "outputs.npz"), output=output)


def main():
    args = parse_args()

    input = generate_input_data(args)
    output, model_def = run_onnx_softmax(input, args.axis)

    save_deployment_files(input, output, model_def)

    print(f"Deployment files generated with [N:{args.N}, C:{args.C}, H:{args.H}, W:{args.W}, axis:{args.axis}]")


if __name__ == "__main__":
    main()
