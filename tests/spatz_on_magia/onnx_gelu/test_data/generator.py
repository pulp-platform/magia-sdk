import os
import sys
import onnx
import numpy as np
import onnxruntime as ort

def parse_args():
    if len(sys.argv) != 2:
        print("Error: missing argmuent <length>")
        print("Usage: python3 generator.py <length>")
        sys.exit(1)

    try:
        length = int(sys.argv[1])
    except:
        print("Error: <length> must be an integer")
        print("Usage: python3 generator.py <length>")
        sys.exit(1)

    return length

def run_onnx_gelu(input):
    input_info = onnx.helper.make_tensor_value_info('I', onnx.TensorProto.FLOAT16, input.shape)
    output_info = onnx.helper.make_tensor_value_info('O', onnx.TensorProto.FLOAT16, input.shape)

    opset = onnx.helper.make_operatorsetid("", 20)
    node_def = onnx.helper.make_node('Gelu', ['I'], ['O'], approximate='tanh')
    graph_def = onnx.helper.make_graph([node_def], 'onnx-gelu-test', [input_info], [output_info])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])

    ses = ort.InferenceSession(model_def.SerializeToString())
    res = ses.run(None, {'I':input})

    return res[0]

def format_array(array):
    return "{ " + ", ".join(f"{x:f}f" for x in array) + " }"

def generate_header_file(length, input, expected, filename="data.h"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    with open(filepath, "w") as f:
        f.write(f"/* Automatically generated header file for Spatz ONNX testing */\n")
        f.write(f"#ifndef DATA_H_\n")
        f.write(f"#define DATA_H_\n\n")

        f.write(f"#define LEN {length}\n\n")

        f.write(f"static const float16 input_vec[] = {format_array(input)};\n")
        f.write(f"static const float16 expected_vec[] = {format_array(expected)};\n\n")

        f.write (f"#endif   /* DATA_H_ */\n")

def main():
    length = parse_args()

    input = (np.random.randn(length)).astype(np.float16)

    expected = run_onnx_gelu(input)

    generate_header_file(length, input, expected)

    print(f"File 'data.h' successfully generated with {length} elements.")

if __name__ == "__main__":
    main()
