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

def generate_operands(length):
    a = (np.random.randn(length) * 10).astype(np.float16)
    b = (np.random.randn(length) * 10).astype(np.float16)

    # avoid 0 and Inf or NaN
    epsilon = 1e-3
    a[a==0] += epsilon
    b[b==0] += epsilon
    a[~np.isfinite(a)] = 1.0
    b[~np.isfinite(a)] = 1.0

    return a, b


def run_onnx_div(vec_a, vec_b):
    a_info = helper.make_tensor_value_info('A', TensorProto.FLOAT16, vec_a.shape)
    b_info = helper.make_tensor_value_info('B', TensorProto.FLOAT16, vec_b.shape)
    out_info = helper.make_tensor_value_info('Quot', TensorProto.FLOAT16, vec_a.shape)

    node_def = helper.make_node('Div', ['A', 'B'], ['Quot'])
    graph_def = helper.make_graph(
        [node_def],
        'onnx-div-test',
        [a_info, b_info],
        [out_info]
    )
    model_def = helper.make_model(graph_def, producer_name='onnx-generator')

    sess = ort.InferenceSession(model_def.SerializeToString())
    res = sess.run(None, {'A': vec_a, 'B': vec_b})

    return res[0]

def format_array(array):
    return "{ " + ", ".join(f"{x:f}f" for x in array) + " }"

def generate_header_file(length, vec_a, vec_b, expected, filename="data.h"):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    with open(filepath, "w") as f:
        f.write("/* Automatically generated header file for Spatz ONNX testing */\n")
        f.write("#ifndef DATA_H_\n")
        f.write("#define DATA_H_\n\n")

        f.write(f"#define LEN {length}\n\n")

        f.write(f"static const float16 vec_a[] = {format_array(vec_a)};\n")
        f.write(f"static const float16 vec_b[] = {format_array(vec_b)};\n")
        f.write(f"static const float16 expected[] = {format_array(expected)};\n\n")

        f.write("#endif  /* DATA_H_ */\n")

def main():
    length = parse_args()

    vec_a, vec_b = generate_operands(length)
    expected = run_onnx_div(vec_a, vec_b)

    generate_header_file(length, vec_a, vec_b, expected)

    print(f"File 'data.h' successfully generated with {length} elements.")

if __name__ == "__main__":
    main()
