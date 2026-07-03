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
        raise argparse.ArgumentTypeError(f"Scale must be positive ({value}).")

    return fvalue


def parse_args():
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for Resize test")

    parser.add_argument("N", type=positive_int, help="Batch size")
    parser.add_argument("C", type=positive_int, help="Number of input channels")
    parser.add_argument("H", type=positive_int, help="Spatial height dimension")
    parser.add_argument("W", type=positive_int, help="Spatial width dimension")

    parser.add_argument("--scale_h", type=positive_float, default=2.0, help="Resize scale factor for Height")
    parser.add_argument("--scale_w", type=positive_float, default=2.0, help="Resize scale factor for Width")

    args = parser.parse_args()
    return args


def generate_input_data(args):
    input_shape = (args.N, args.C, args.H, args.W)

    X = np.random.randn(*input_shape).astype(np.float16)

    roi = np.array([], dtype=np.float32)
    scales = np.array([], dtype=np.float32)

    out_h = int(args.H * args.scale_h)
    out_w = int(args.W * args.scale_w)
    sizes = np.array([args.N, args.C, out_h, out_w], dtype=np.int64)

    return X, roi, scales, sizes


def run_onnx_resize(X, roi, scales, sizes):
    output_shape = tuple(sizes.tolist())

    X_info = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT16, X.shape)
    Y_info = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT16, output_shape)

    roi_initializer = onnx.helper.make_tensor(name='roi', data_type=onnx.TensorProto.FLOAT, dims=roi.shape, vals=roi.flatten().tolist())
    scales_initializer = onnx.helper.make_tensor(name='scales', data_type=onnx.TensorProto.FLOAT, dims=scales.shape, vals=scales.flatten().tolist())
    sizes_initializer = onnx.helper.make_tensor(name='sizes', data_type=onnx.TensorProto.INT64, dims=sizes.shape, vals=sizes.flatten().tolist())

    opset = onnx.helper.make_operatorsetid("", 23)
    node_def = onnx.helper.make_node('Resize', inputs=['X', 'roi', 'scales', 'sizes'], outputs=['Y'])
    graph_def = onnx.helper.make_graph([node_def], 'onnx-resize-test', [X_info], [Y_info], initializer=[roi_initializer, scales_initializer, sizes_initializer])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    sess = ort.InferenceSession(model_def.SerializeToString())
    res = sess.run(None, {'X': X})

    return res[0], model_def


def save_deployment_files(X, Y, model_def):
    deployment_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(deployment_dir, exist_ok=True)

    onnx.save(model_def, os.path.join(deployment_dir, "network.onnx"))
    np.savez(os.path.join(deployment_dir, "inputs.npz"), X=X)
    np.savez(os.path.join(deployment_dir, "outputs.npz"), Y=Y)


def main():
    args = parse_args()

    X, roi, scales, sizes = generate_input_data(args)

    Y, model_def = run_onnx_resize(X, roi, scales, sizes)

    save_deployment_files(X, Y, model_def)

    print(f"Deployment files generated with [N:{args.N}, C:{args.C}, H:{args.H}, W:{args.W}]")


if __name__ == "__main__":
    main()
