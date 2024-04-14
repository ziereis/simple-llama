import ctypes
import argparse
lib = ctypes.CDLL('build/libquantize.so')

lib.quantize_model.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.quantize_model.restype = None

def quantize_model(input_path: str, output_path: str):
    lib.quantize_model(input_path.encode(), output_path.encode())


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--bin", type=str, help="path to exported llama f32 weights", default= "bin/llama.bin")

  args = parser.parse_args()
  quantize_model(args.bin, "bin/llama_q8.bin")
