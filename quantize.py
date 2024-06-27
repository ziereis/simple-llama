import ctypes
import argparse

"""
This module provides functionality for quantizing LLaMA model weights to different levels (Q4, Q8).

The module defines a `QuantizationType` enumeration and functions to perform quantization using a shared library.
It supports command-line arguments to specify the input file, output file, and quantization level.

Classes and Functions:
- `QuantizationType`: Enumeration for quantization levels (Q8, Q4, NONE).
- `quant_q8(input_path: str, output_path: str)`: Quantizes the model weights to Q8.
- `quant_q4(input_path: str, output_path: str)`: Quantizes the model weights to Q4.


Example usage:
    python quantize.py --bin path/to/llama.bin --q 8

This will quantize the LLaMA model weights to Q8 and save the output to a new file.
"""


lib = ctypes.CDLL('build/libquantize.so')

class QuantizationType(ctypes.c_int):
  Q8 = 0
  Q4 = 1
  NONE = 2

lib.quant_model.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
lib.quant_model.restype = None

def quant_q8(input_path: str, output_path: str):
  lib.quant_model(input_path.encode(), output_path.encode(), QuantizationType.Q8)
def quant_q4(input_path: str, output_path: str):
  lib.quant_model(input_path.encode(), output_path.encode(), QuantizationType.Q4)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--bin", type=str, help="path to exported llama f32 weights", default= "bin/llama.bin")
  parser.add_argument("--q", type=int, help="quantization level, vaild values are 4 and 8", default=8)

  args = parser.parse_args()

  if (args.q == 8):
    quant_q8(args.bin, args.bin.replace(".bin", "_q8.bin"))
  elif (args.q== 4):
    quant_q4(args.bin, args.bin.replace(".bin", "_q4.bin"))
  else:
    print("Invalid quantization value")
    exit(1)
