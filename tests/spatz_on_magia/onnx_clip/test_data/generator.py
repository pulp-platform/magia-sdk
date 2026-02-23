import os
import sys
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

def parse_args():
    if len(sys.argv) != 2:
        print("Error: missing argument <length>")
        print("Usage: python3 generator.py <length>")
        sys.exit(1)

    try:
        length = int(sys.argv[1])
    except ValueError:
        print("Error: <length> must be an integer")
        print("Usage: python3 generator.py <length>")
        sys.exit(1)

    return length

def generate_data(length):
    vals = (np.random.randn(2) * 5).astype(np.float16)
    val_min = np.array([np.min(vals)], dtype=np.float16)
    val_max = np.array([np.max(vals)], dtype=np.float16)

    a = (np.random.randn(length) * 10).astype(np.float16)

    return a, val_min, val_max

def run_onnx_clip(input_vec, val_min, val_max):
    a_info = helper.make_tensor_value_info('A', TensorProto.FLOAT16, input_vec.shape)
    min_info = helper.make_tensor_value_info('Min', TensorProto.FLOAT16, val_min.shape)
    max_info = helper.make_tensor_value_info('Max', TensorProto.FLOAT16, val_max.shape)
    out_info = helper.make_tensor_value_info('Output', TensorProto.FLOAT16, input_vec.shape)

    node_def = helper.make_node('Clip', ['A', 'Min', 'Max'], ['Output'])
    graph_def = helper.make_graph(
        [node_def],
        'onnx-clip-test',
        [a_info, min_info, max_info],
        [out_info]
    )

    model_def = helper.make_model(graph_def, producer_name='onnx-generator')
    sess = ort.InferenceSession(model_def.SerializeToString())
    res = sess.run(None, {'A': input_vec, 'Min': val_min, 'Max': val_max})

    return res[0]

def format_array(array):
    return "{ " + ", ".join(f"{x:f}f" for x in array) + " }"

def format_float(value):
    """Convert a float value into a C-style representation."""
    return f"{value:f}"

def generate_header_file(length, input_vec, min, max, expected, filename="data.h"):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    with open(filepath, "w") as f:
        f.write("/* Automatically generated header file for Spatz ONNX testing */\n")
        f.write("#ifndef DATA_H_\n")
        f.write("#define DATA_H_\n\n")

        f.write(f"#define LEN {length}\n\n")

        f.write(f"static const float16 input_vec[] = {format_array(input_vec)};\n")
        f.write(f"static const float16 min_val = {format_float(min.item())};\n")
        f.write(f"static const float16 max_val = {format_float(max.item())};\n")
        f.write(f"static const float16 expected[] = {format_array(expected)};\n\n")

        f.write("#endif  /* DATA_H_ */\n")

def main():
    length = parse_args()

    input, min, max = generate_data(length)
    expected = run_onnx_clip(input, min, max)

    generate_header_file(length, input, min, max, expected)

    print(f"File 'data.h' successfully generated with {length} elements.")

if __name__ == "__main__":
    main()
