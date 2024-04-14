import ctypes
import numpy as np

lib = ctypes.CDLL('build/libllama.so')

lib.QRuntime_new.argtypes = [ctypes.c_char_p]
lib.QRuntime_new.restype = ctypes.c_void_p

lib.QRuntime_delete.argtypes = [ctypes.c_void_p]
lib.QRuntime_delete.restype = None

lib.QRuntime_forward.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
lib.QRuntime_forward.restype = ctypes.POINTER(ctypes.c_float)


class QLLama:
  def __init__(self, model_path: str):
    self.model_path = model_path
    self.runtime = lib.QRuntime_new(model_path.encode())
    self.vocab_size = 32000
  def forward(self, tok: int, pos: int):
    res = lib.QRuntime_forward(self.runtime, tok, pos)
    return np.ctypeslib.as_array(res, (1, 32000))

  def __del__(self):
    lib.QRuntime_delete(self.runtime)


if __name__ == '__main__':
  llama = QLLama('bin/llama_q8.bin')
  print(llama.forward(100, 0))
