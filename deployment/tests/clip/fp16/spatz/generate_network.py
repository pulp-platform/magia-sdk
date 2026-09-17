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
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for ONNX Clip test")

    parser.add_argument("N", type=positive_int, help="Batch size")
    parser.add_argument("C", type=positive_int, help="Number of input channels")
    parser.add_argument("H", type=positive_int, help="Spatial height dimension")
    parser.add_argument("W", type=positive_int, help="Spatial width dimension")

    args = parser.parse_args()
    return args


def generate_input_data(args):
    shape = (args.N, args.C, args.H, args.W)

    input = np.random.randn(*shape).astype(np.float16)
    idx = np.random.randint(0, input.size, size=2)
    val1 = input.flat[idx[0]]
    val2 = input.flat[idx[1]]
    val1, val2 = np.sort([val1, val2])
    min = np.array(val1, dtype=np.float16)
    max = np.array(val2, dtype=np.float16)

    return input, min, max


def run_onnx_clip(input, min, max):
    input_info = onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT16, input.shape)
    output_info = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT16, input.shape)
    min_initializer = onnx.numpy_helper.from_array(min, name='min')
    max_initializer = onnx.numpy_helper.from_array(max, name='max')

    opset = onnx.helper.make_operatorsetid("", 13)
    node_def = onnx.helper.make_node('Clip', ['input', 'min', 'max'], ['output'])
    graph_def = onnx.helper.make_graph([node_def], 'onnx-clip-test', inputs=[input_info], outputs=[output_info], initializer=[min_initializer, max_initializer])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    sess = ort.InferenceSession(model_def.SerializeToString())
    res = sess.run(None, {'input': input})

    return res[0], model_def

def save_deployment_files(input, output, model_def):
    deployment_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(deployment_dir, exist_ok=True)

    onnx.save(model_def, os.path.join(deployment_dir, "network.onnx"))
    np.savez(os.path.join(deployment_dir, "inputs.npz"), input=input)
    np.savez(os.path.join(deployment_dir, "outputs.npz"), output=output)


def main():
    args = parse_args()

    input, min, max = generate_input_data(args)
    output, model_def = run_onnx_clip(input, min, max)

    save_deployment_files(input, output, model_def)

    print(f"Deployment files generated with [N:{args.N}, C:{args.C}, H:{args.H}, W:{args.W}]")

if __name__ == "__main__":
    main()
