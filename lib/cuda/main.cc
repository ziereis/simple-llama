#include "cu-llama.h"
#include "lib/llama.h"
#include "stdio.h"
#include "cu-ops.h"


int main() {
  print_device_info();
  void *rt = device_runtime_new_q4("../bin/llama_q4.bin");
  // double overall_time = 0;
  // Timer timer;
  // for (int i = 0; i < 10; i++) {
  //   start_timer(&timer);
  //   device_forward_q4((QLLamaRuntime*)rt, 100, 0);
  //   stop_timer(&timer);
  //   overall_time += elapsed_time(&timer);
  // }

  // printf("Time: %.4f\n", overall_time / 10);
  // printf("rms_time: %.4f\n", rms_time / 10);
  // printf("matvec_q4_time: %.4f\n", matvec_q4_time / 10);
  // printf("quantize_q4_time: %.4f\n", quantize_q4_time / 10);
  // printf("dequantize_q4_time: %.4f\n", dequantize_q4_time / 10);
  // printf("attention_time: %.4f\n", attention_time / 10);
  // printf("rotate_time: %.4f\n", rotate_time / 10);
  // printf("swiglu_time: %.4f\n", swiglu_time / 10);
  // printf("residual_time: %.4f\n", residual_time / 10);
  // printf("teste\n");



}
