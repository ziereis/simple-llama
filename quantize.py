import ctypes
import argparse
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
  parser.add_argument("--q", type=int, help="path to quantized llama weights", default=8)

  args = parser.parse_args()

  if (args.q == 8):
    quant_q8(args.bin, "bin/llama_q8.bin")
  elif (args.q== 4):
    quant_q4(args.bin, "bin/llama_q4.bin")
  else:
    print("Invalid quantization value")
    exit(1)
