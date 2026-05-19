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

    IR = np.random.randn(shape).astype(np.float16)
    II = np.zeros(shape).astype(np.float16)
    WR = np.zeros((int)((shape / 2) * np.log2(shape))).astype(np.float16)
    WI = np.zeros((int)((shape / 2) * np.log2(shape))).astype(np.float16)
    GR = np.random.randn(shape).astype(np.float16)
    GI = np.random.randn(shape).astype(np.float16)

    return IR, II, WR, WI, GR, GI


def run_fft(IR, II, WR, WI, GR, GI):
    #TODO: Do the FFT on ABW and save on G lmao
    G = np.fft.fft(IR)
    GR = G.real
    GI = G.imag

    n = len(IR)

    for s in range((int)(np.log2(n))):
        for b in range((int)(n/2)):
            m = 2 ** (s + 1)
            k = b % (2 ** s)
            w = np.exp((-2 * np.pi) * 1j * (k / m))
            WR[(int)(s * (n/2) + b)] = w.real
            WI[(int)(s * (n/2) + b)] = w.imag

    return IR, II, WR, WI, GR, GI
            



def format_array(array):
    return "{ " + ", ".join(f"{x:.6f}f" for x in array.flatten()) + " }"


def generate_header_file(args, IR, II, WR, WI, GR, GI, filename="data.h"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    with open(filepath, "w") as f:
        f.write("/* Automatically generated header file for Spatz ONNX testing */\n")
        f.write("#ifndef DATA_H_\n")
        f.write("#define DATA_H_\n\n")

        f.write(f"#define VEC_LEN   {args.N}\n")
        f.write(f"#define TW_LEN    {int((args.N/2) * np.log2(args.N))}\n")
        f.write(f"#define LOG2_LEN  {int(np.log2(args.N))}\n")

        f.write(f"static const float16 IR[]      =   {format_array(IR)};\n")
        f.write(f"static const float16 II[]      =   {format_array(II)};\n")
        f.write(f"static const float16 WR[]      =   {format_array(WR)};\n")
        f.write(f"static const float16 WI[]      =   {format_array(WI)};\n")
        f.write(f"static const float16 GR[]      =   {format_array(GR)};\n")
        f.write(f"static const float16 GI[]      =   {format_array(GI)};\n\n")

        f.write("#endif  /* DATA_H_ */\n")


def main():
    args = parse_args()

    IR, II, WR, WI, GR, GI = generate_input_data(args)

    IR, II, WR, WI, GR, GI = run_fft(IR, II, WR, WI, GR, GI)

    generate_header_file(args, IR, II, WR, WI, GR, GI)

    print(f"File 'data.h' successfully generated with [N:{args.N}]")


if __name__ == "__main__":
    main()
