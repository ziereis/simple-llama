#include "cu-llama.h"


int main() {
  void* rt = device_runtime_new_q4("../../../bin/llama_q4.bin");
  float *res = device_runtime_forward_q4(rt, 100, 0);
  device_runtime_delete_q4(rt);
}
