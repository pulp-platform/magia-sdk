import argparse
import onnx
import os
import sys

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
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for ONNX Add test")

    parser.add_argument("N", type=positive_int, help="Vector size")

    args = parser.parse_args()
    return args


def generate_input_data(args):
    shape = args.N

    A = np.random.randn(shape).astype(np.float16)
    W = np.zeroes(shape * np.log2(shape)).astype(np.float16)
    G = np.random.randn(shape).astype(np.float16)

    return A, W, G


def run_fft(A, W, G):
    #TODO: Do the FFT on ABW and save on G lmao
    G = np.fft.fft(A)
    n = len(A)
    for s in range(1, np.log2(n)):
        m = 2**s
        w_m = np.exp((-2 * np.pi)j / m) 
        for t in range(0, (n -1), m):
            W[s * t] = 1
            for b in range(0, (m/2) - 1):
                W[s * t] = W[s * t] * w_m




def format_array(array):
    return "{ " + ", ".join(f"{x:f}f" for x in array.flatten()) + " }"


def generate_header_file(args, A, B, W, G, filename="data.h"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    with open(filepath, "w") as f:
        f.write("/* Automatically generated header file for Spatz ONNX testing */\n")
        f.write("#ifndef DATA_H_\n")
        f.write("#define DATA_H_\n\n")

        f.write(f"#define N {args.N}\n")
        f.write(f"#define TW {args.N - 1}\n")

        f.write(f"static const float16 A[] = {format_array(A)};\n")
        f.write(f"static const float16 W[] = {format_array(W)};\n")
        f.write(f"static const float16 G[] = {format_array(G)};\n\n")

        f.write("#endif  /* DATA_H_ */\n")


def main():
    args = parse_args()

    A, W, G = generate_input_data(args)
    run_fft(A, W, G)

    generate_header_file(args, A, W, G)

    print(f"File 'data.h' successfully generated with [N:{args.N}, C:{args.C}, H:{args.H}, W:{args.W}]")


if __name__ == "__main__":
    main()
