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
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for NCHW Concat test")

    parser.add_argument("N0", type=positive_int, help="Batch size for Input 0")
    parser.add_argument("C0", type=positive_int, help="Channels for Input 0")
    parser.add_argument("H0", type=positive_int, help="Height for Input 0")
    parser.add_argument("W0", type=positive_int, help="Width for Input 0")

    parser.add_argument("N1", type=positive_int, help="Batch size for Input 1")
    parser.add_argument("C1", type=positive_int, help="Channels for Input 1")
    parser.add_argument("H1", type=positive_int, help="Height for Input 1")
    parser.add_argument("W1", type=positive_int, help="Width for Input 1")

    parser.add_argument("--axis", type=int, default=1, help="Axis along which to concatenate (default: 1)")

    args = parser.parse_args()
    return args


def generate_input_data(args):
    if args.axis == 0:
        # Concat on N, CHW must match
        assert args.C0 == args.C1, "Axis=0: Channels must match (C0 == C1)!"
        assert args.H0 == args.H1, "Axis=0: Heights must match (H0 == H1)!"
        assert args.W0 == args.W1, "Axis=0: Widths must match (W0 == W1)!"
    elif args.axis == 1:
        # Concat on C, NHW must match
        assert args.N0 == args.N1, "Axis=1: Batches must match (N0 == N1)!"
        assert args.H0 == args.H1, "Axis=1: Heights must match (H0 == H1)!"
        assert args.W0 == args.W1, "Axis=1: Widths must match (W0 == W1)!"
    elif args.axis == 2:
        # Concat on H, NCW must match
        assert args.N0 == args.N1, "Axis=2: Batches must match (N0 == N1)!"
        assert args.C0 == args.C1, "Axis=2: Channels must match (C0 == C1)!"
        assert args.W0 == args.W1, "Axis=2: Widths must match (W0 == W1)!"
    elif args.axis == 3:
        # Concat on W, NCH must match
        assert args.N0 == args.N1, "Axis=3: Batches must match (N0 == N1)!"
        assert args.C0 == args.C1, "Axis=3: Channels must match (C0 == C1)!"
        assert args.H0 == args.H1, "Axis=3: Heights must match (H0 == H1)!"
    else:
        raise ValueError(f"Invalid axis {args.axis} for a 4D NCHW tensor.")

    input0_shape = (args.N0, args.C0, args.H0, args.W0)
    input1_shape = (args.N1, args.C1, args.H1, args.W1)

    input0 = np.random.randn(*input0_shape).astype(np.float16)
    input1 = np.random.randn(*input1_shape).astype(np.float16)

    return input0, input1


def run_onnx_concat(input0, input1, axis):
    input0_info = onnx.helper.make_tensor_value_info('input0', onnx.TensorProto.FLOAT16, input0.shape)
    input1_info = onnx.helper.make_tensor_value_info('input1', onnx.TensorProto.FLOAT16, input1.shape)

    output_shape = list(input0.shape)
    output_shape[axis] = input0.shape[axis] + input1.shape[axis]
    concat_result_info = onnx.helper.make_tensor_value_info('concat_result', onnx.TensorProto.FLOAT16, output_shape)

    opset = onnx.helper.make_operatorsetid("", 13)
    node_def = onnx.helper.make_node('Concat', ['input0', 'input1'], ['concat_result'], axis=axis)
    graph_def = onnx.helper.make_graph([node_def], 'onnx-concat-test', [input0_info, input1_info], [concat_result_info])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    sess = ort.InferenceSession(model_def.SerializeToString())
    res = sess.run(None, {'input0': input0, 'input1': input1})

    return res[0], model_def


def save_deployment_files(input0, input1, concat_result, model_def):
    deployment_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(deployment_dir, exist_ok=True)

    onnx.save(model_def, os.path.join(deployment_dir, "network.onnx"))
    np.savez(os.path.join(deployment_dir, "inputs.npz"), input0=input0, input1=input1)
    np.savez(os.path.join(deployment_dir, "outputs.npz"), concat_result=concat_result)


def main():
    args = parse_args()

    input0, input1 = generate_input_data(args)

    concat_result, model_def = run_onnx_concat(input0, input1, args.axis)

    save_deployment_files(input0, input1, concat_result, model_def)

    print(f"Deployment files generated with input0 [N:{args.N0}, C:{args.C0}, H:{args.H0}, W:{args.W0}] and input1 [N:{args.N1}, C:{args.C1}, H:{args.H1}, W:{args.W1}]")


if __name__ == "__main__":
    main()
