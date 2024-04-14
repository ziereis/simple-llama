#include "llama.h"


int main() {
  QRuntimeHandle handle = QRuntime_new("../bin/llama_q8.bin");
  float *out = QRuntime_forward(handle, 0, 0);
  QRuntime_delete(handle);
  return 0;
}
