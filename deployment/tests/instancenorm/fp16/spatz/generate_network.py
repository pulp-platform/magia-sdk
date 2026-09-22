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


def positive_float(value):
    try:
        fvalue = float(value)

    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid Real number.")

    if fvalue <= 0:
        raise argparse.ArgumentTypeError(f"Epsilon must be positive ({value}).")

    return fvalue


def parse_args():
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for InstanceNormalization test")

    parser.add_argument("N", type=positive_int, help="Batch size")
    parser.add_argument("C", type=positive_int, help="Number of input channels")
    parser.add_argument("H", type=positive_int, help="Spatial height dimension")
    parser.add_argument("W", type=positive_int, help="Spatial width dimension")

    parser.add_argument("--epsilon", type=positive_float, default=1e-05, help="Epsilon value for InstanceNorm (default: 1e-05)")

    args = parser.parse_args()
    return args


def generate_input_data(args):
    input_shape = (args.N, args.C, args.H, args.W)
    param_shape = (args.C, )

    input = np.random.randn(*input_shape).astype(np.float16)
    scale = (1.0 + 0.1 * np.random.randn(*param_shape)).astype(np.float16)
    B = (0.1 * np.random.randn(*param_shape)).astype(np.float16)

    return input, scale, B


def run_onnx_instancenorm(input, scale, B, epsilon):
    input_info = onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT16, input.shape)
    output_info = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT16, input.shape)

    scale_initializer = onnx.numpy_helper.from_array(scale, name='scale')
    B_initializer = onnx.numpy_helper.from_array(B, name='B')

    opset = onnx.helper.make_operatorsetid("", 22)
    node_def = onnx.helper.make_node('InstanceNormalization', ['input', 'scale', 'B'], ['output'], epsilon=epsilon)
    graph_def = onnx.helper.make_graph(nodes=[node_def], name='onnx-instancenorm-test', inputs=[input_info], outputs=[output_info], initializer=[scale_initializer, B_initializer])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    ses = ort.InferenceSession(model_def.SerializePartialToString())
    res = ses.run(None, {'input': input})

    return res[0], model_def


def save_deployment_files(input, output, model_def):
    deployment_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(deployment_dir, exist_ok=True)

    onnx.save(model_def, os.path.join(deployment_dir, "network.onnx"))
    np.savez(os.path.join(deployment_dir, "inputs.npz"), input=input)
    np.savez(os.path.join(deployment_dir, "outputs.npz"), output=output)


def main():
    args = parse_args()

    input, scale, B = generate_input_data(args)
    output, model_def = run_onnx_instancenorm(input, scale, B, args.epsilon)

    save_deployment_files(input, output, model_def)

    print(f"Deployment files generated with [N:{args.N}, C:{args.C}, H:{args.H}, W:{args.W}]")


if __name__ == "__main__":
    main()
