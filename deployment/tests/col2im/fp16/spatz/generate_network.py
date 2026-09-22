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


def non_negative_int(value):
    try:
        ival = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid Integer number.")

    if ival < 0:
        raise argparse.ArgumentTypeError(f"Value must be non-negative ({value}).")

    return ival


def parse_args():
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for Col2Im test")

    parser.add_argument("N", type=positive_int, help="Batch size")
    parser.add_argument("C", type=positive_int, help="Number of image channels")
    parser.add_argument("H_out", type=positive_int, help="Output spatial height dimension")
    parser.add_argument("W_out", type=positive_int, help="Output spatial width dimension")

    parser.add_argument("--kh", type=positive_int, default=3, help="Kernel/Block height (default: 3)")
    parser.add_argument("--kw", type=positive_int, default=3, help="Kernel/Block width (default: 3)")

    parser.add_argument("--pad_h", type=non_negative_int, default=0, help="Padding height (default: 0)")
    parser.add_argument("--pad_w", type=non_negative_int, default=0, help="Padding width (default: 0)")
    parser.add_argument("--stride_h", type=positive_int, default=1, help="Stride height (default: 1)")
    parser.add_argument("--stride_w", type=positive_int, default=1, help="Stride width (default: 1)")
    parser.add_argument("--dil_h", type=positive_int, default=1, help="Dilation height (default: 1)")
    parser.add_argument("--dil_w", type=positive_int, default=1, help="Dilation width (default: 1)")

    args = parser.parse_args()
    return args


def generate_input_data(args):
    L_h = (args.H_out + 2 * args.pad_h - args.dil_h * (args.kh - 1) - 1) // args.stride_h + 1
    L_w = (args.W_out + 2 * args.pad_w - args.dil_w * (args.kw - 1) - 1) // args.stride_w + 1

    if L_h <= 0 or L_w <= 0:
        raise ValueError("Invalid geometry: calculated column dimensions (L_h, L_w) must be positive.")

    L = L_h * L_w
    block_volume = args.kh * args.kw

    input_shape = (args.N, args.C * block_volume, L)

    input = np.random.randn(*input_shape).astype(np.float16)

    image_shape = np.array([args.H_out, args.W_out], dtype=np.int64)
    block_shape = np.array([args.kh, args.kw], dtype=np.int64)

    return input, image_shape, block_shape


def run_onnx_col2im(input, image_shape, block_shape, args):
    input_info = onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT16, input.shape)
    output_info = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT16, [args.N, args.C, args.H_out, args.W_out])

    image_shape_initializer = onnx.helper.make_tensor('image_shape', onnx.TensorProto.INT64, image_shape.shape, image_shape.tolist())
    block_shape_initializer = onnx.helper.make_tensor('block_shape', onnx.TensorProto.INT64, block_shape.shape, block_shape.tolist())

    pads = [args.pad_h, args.pad_w, args.pad_h, args.pad_w]
    strides = [args.stride_h, args.stride_w]
    dilations = [args.dil_h, args.dil_w]

    opset = onnx.helper.make_operatorsetid("", 18)
    node_def = onnx.helper.make_node('Col2Im', inputs=['input', 'image_shape', 'block_shape'], outputs=['output'], dilations=dilations, pads=pads, strides=strides)
    graph_def = onnx.helper.make_graph([node_def], 'onnx-col2im-test', [input_info], [output_info], initializer=[image_shape_initializer, block_shape_initializer])
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

    input, image_shape, block_shape = generate_input_data(args)
    output, model_def = run_onnx_col2im(input, image_shape, block_shape, args)

    save_deployment_files(input,  output, model_def)

    print(f"Deployment files generated with [N:{args.N}, C:{args.C}, H_out:{args.H_out}, W_out:{args.W_out}]")


if __name__ == "__main__":
    main()
