from abc import ABC, abstractmethod
import ctypes
import numpy as np
import os

"""
This module provides a runtime interface for the LLaMA model, allowing for different quantization levels 
(F32, Q8, Q4) and GPU support for Q4 models.

The module dynamically loads shared libraries for CPU and GPU operations and defines an abstract base class `Runtime` 
and its concrete implementations (`F32Runtime`, `Q8Runtime`, `Q4Runtime`, `DeviceQ4Runtime`). 
It also includes a factory function `CreateRuntime` to instantiate the appropriate runtime based on the model file 
and the target device.

Classes and Functions:
- `Runtime` (ABC): Abstract base class for model runtimes.
- `F32Runtime`: Implementation of `Runtime` for F32 quantization.
- `Q8Runtime`: Implementation of `Runtime` for Q8 quantization.
- `Q4Runtime`: Implementation of `Runtime` for Q4 quantization.
- `DeviceQ4Runtime`: Implementation of `Runtime` for Q4 quantization on GPU.
- `CreateRuntime`: Factory function to create the appropriate runtime instance based on the model file and device.

Example usage:
    llama = CreateRuntime('bin/llama_q4.bin', device="gpu")
    print(llama.forward(100, 0))
"""

# Load the library
lib = ctypes.CDLL('build/libllama.so')

if os.path.exists('build/libcu_llama.so'):
    cu_lib = ctypes.CDLL('build/libcu_llama.so')

    cu_lib.device_runtime_new_q4.argtypes = [ctypes.c_char_p]
    cu_lib.device_runtime_new_q4.restype = ctypes.c_void_p

    cu_lib.device_runtime_delete_q4.argtypes = [ctypes.c_void_p]
    cu_lib.device_runtime_delete_q4.restype = None

    cu_lib.device_runtime_forward_q4.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    cu_lib.device_runtime_forward_q4.restype = ctypes.POINTER(ctypes.c_float)


# Setup for runtime_new_f32
lib.runtime_new_f32.argtypes = [ctypes.c_char_p]
lib.runtime_new_f32.restype = ctypes.c_void_p

lib.runtime_delete_f32.argtypes = [ctypes.c_void_p]
lib.runtime_delete_f32.restype = None

lib.runtime_forward_f32.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
lib.runtime_forward_f32.restype = ctypes.POINTER(ctypes.c_float)

# Setup for runtime_new_Q8
lib.runtime_new_q8.argtypes = [ctypes.c_char_p]
lib.runtime_new_q8.restype = ctypes.c_void_p

lib.runtime_delete_q8.argtypes = [ctypes.c_void_p]
lib.runtime_delete_q8.restype = None

lib.runtime_forward_q8.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.runtime_forward_q8.restype = ctypes.POINTER(ctypes.c_float)  # Adjust if the data type differs

# Setup for runtime_new_Q4
lib.runtime_new_q4.argtypes = [ctypes.c_char_p]
lib.runtime_new_q4.restype = ctypes.c_void_p

lib.runtime_delete_q4.argtypes = [ctypes.c_void_p]
lib.runtime_delete_q4.restype = None

lib.runtime_forward_q4.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.runtime_forward_q4.restype = ctypes.POINTER(ctypes.c_float)

class Runtime(ABC):
    def __init__(self, filename: str):
        self.runtime = self._new_runtime(filename.encode())
        self.output_size = 32000

    @abstractmethod
    def _new_runtime(self, filename):
        pass

    @abstractmethod
    def _forward(self, handle, tok, pos):
        pass

    def forward(self, tok: int, pos: int):
        result_pointer = self._forward(self.runtime, tok, pos)
        return np.ctypeslib.as_array(result_pointer, (self.output_size,))

    def __del__(self):
        self._delete_runtime(self.runtime)

    @abstractmethod
    def _delete_runtime(self, handle):
        pass

class F32Runtime(Runtime):
    def _new_runtime(self, filename):
        return lib.runtime_new_f32(filename)

    def _forward(self, handle, tok, pos):
        return lib.runtime_forward_f32(handle, tok, pos)

    def _delete_runtime(self, handle):
        lib.runtime_delete_f32(handle)

class Q8Runtime(Runtime):
    def _new_runtime(self, filename):
        return lib.runtime_new_q8(filename)

    def _forward(self, handle, tok, pos):
        return lib.runtime_forward_q8(handle, tok, pos)

    def _delete_runtime(self, handle):
        lib.runtime_delete_q8(handle)

class Q4Runtime(Runtime):
    def _new_runtime(self, filename):
        return lib.runtime_new_q4(filename)

    def _forward(self, handle, tok, pos):
        return lib.runtime_forward_q4(handle, tok, pos)

    def _delete_runtime(self, handle):
        lib.runtime_delete_q4(handle)

class DeviceQ4Runtime(Runtime):
    def _new_runtime(self, filename):
        return cu_lib.device_runtime_new_q4(filename)

    def _forward(self, handle, tok, pos):
        return cu_lib.device_runtime_forward_q4(handle, tok, pos)

    def _delete_runtime(self, handle):
        pass
        cu_lib.device_runtime_delete_q4(handle)

def CreateRuntime(filename: str, device: str = 'cpu'):
  if (device.lower() == 'gpu'):
      if str.endswith(filename, 'q4.bin'):
        return DeviceQ4Runtime(filename)
      else:
        raise ValueError("Only Q4 models are supported on GPU for now.")
  else:
    if str.endswith(filename, 'llama.bin'):
        return F32Runtime(filename)
    elif str.endswith(filename, 'q8.bin'):
        return Q8Runtime(filename)
    elif str.endswith(filename, 'q4.bin'):
        return Q4Runtime(filename)
    else:
        raise ValueError(f'Unknown file type: {filename}')



if __name__ == '__main__':
  llama = CreateRuntime('bin/llama_q4.bin', device="gpu")
  print(llama.forward(100, 0))
