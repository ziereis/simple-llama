#include "cu-llama.h"
#include "stdio.h"


int main() {
  print_device_info();
  void *rt = device_runtime_new_q4("../../../bin/llama_q4.bin");
  Timer t;
  start_timer(&t);
  float *res = device_runtime_forward_q4(rt, 100, 0);
  stop_timer(&t);
  print_vec(res, 20, 0);
  printf("Time: %d\n", elapsed_time(&t));
  device_runtime_delete_q4(rt);
}
