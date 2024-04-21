#include "quantize.h"
#include <math.h>
#include <stdio.h>
#include <sys/mman.h>
#include <stdlib.h>
#include "llama.h"
#include "utils.h"
#include "string.h"
#include "ops.h"

float calc_error(f32 *orig, f32 *deq, u64 n) {
  f32 max_err = 0.0f;
  #pragma omp parallel for schedule(static) reduction(max : max_err)
  for (u64 i = 0; i < n; i++) {
    max_err = fmax(max_err, fabs(deq[i] - orig[i]));
  }
  return max_err;
}

i8 *q_buf = NULL;
f32 *deq_buf = NULL;
f32 *scales_buf = NULL;



void do_quant_q8(const char *layer_name, FILE *out, f32 *data, u64 n,
                 i32 group_size) {
  Timer timer;
  printf("Quantizing %s\n", layer_name);
  start_timer(&timer);
  quantize_q8(q_buf, scales_buf, data, n, group_size);
  dequantize_q8(deq_buf, q_buf, scales_buf, n, group_size);
  f32 max_error = calc_error(data, deq_buf, n);
  printf("Max error: %f\n", max_error);
  fwrite(q_buf, n, 1, out);
  fwrite(scales_buf, n / group_size * sizeof(f32), 1, out);
  stop_timer(&timer);
  printf("Time: %i ms\n", elapsed_time(&timer));
}

void do_quant_q4(const char *layer_name, FILE *out, f32 *data, u64 n,
                 i32 group_size) {
  Timer timer;
  printf("Quantizing %s\n", layer_name);
  start_timer(&timer);
  quantize_q4(q_buf, scales_buf, data, n, group_size);
  dequantize_q4(deq_buf, q_buf, scales_buf, n, group_size);
  f32 max_error = calc_error(data, deq_buf, n);
  printf("Max error: %f\n", max_error);
  fwrite(q_buf, n / 2, 1, out);
  fwrite(scales_buf, n / group_size * sizeof(f32), 1, out);
  stop_timer(&timer);
  printf("Time: %d ms\n", elapsed_time(&timer));
}

void quant_model(const char *in_file, const char *out_file,
                 QuantizationType qt) {
  if (qt == Q8) {
    printf("Quantizing to Q8\n");
  } else if (qt == Q4) {
    printf("Quantizing to Q4\n");
  } else {
    printf("Invalid quantization\n");
    exit(-1);
  }

  MappedFile in;
  if (!init(&in, in_file)) {
    printf("Failed to open file\n");
    exit(-1);
  }
  LLama m;
  if (!llama_init(&m, in.data, in.len)) {
    printf("Failed to initialize model\n");
    exit(-1);
  }


  int group_size = 128;
  u32 magic = 0x7fdd7f7f;
  u32 version = 4;

  FILE *out = fopen(out_file, "wb");
  fwrite(&magic, sizeof(u32), 1, out);
  fwrite(&version, sizeof(u32), 1, out);
  Params params = m.params;
  params.group_size = group_size;
  if (qt == Q8) {
    params.bitwidth = 8;
    fwrite(&params, sizeof(Params), 1, out);
  } else {
    params.bitwidth = 4;
    fwrite(&params, sizeof(Params), 1, out);
  }
  fseek(out, 256, SEEK_SET);

  printf("Writing norm weights non-quantized\n");
  fwrite(m.norm.data, SIZE_NORM_F32, 1, out);
  // using this as offset
  EncoderBlock l0 = m.layers[0];
  fwrite(l0.attention_norm.data, SIZE_NORM_LAYER_F32, 1, out);
  fwrite(l0.ffn_norm.data, SIZE_NORM_LAYER_F32, 1, out);
  printf("pos: %ld\n", ftell(out));
  q_buf = (i8 *)mmap(NULL, SIZE_WEIGHTS_H_LAYER_Q8, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  deq_buf = (f32 *)mmap(NULL, SIZE_WEIGHTS_H_LAYER_F32, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  scales_buf = (f32 *)mmap(NULL, SIZE_WEIGHTS_H_LAYER_F32 / group_size,
                  PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (qt == Q8) {
    do_quant_q8("token embeddings", out, m.tok_embeddings.data, COUNT_TOK_EMB,
                group_size);
    do_quant_q8("output weights", out, m.output_weights.data, COUNT_OUTPUT_WEIGHT, group_size);
    do_quant_q8("query weights", out, l0.wq.data, COUNT_WEIGHTS_LAYER,
                group_size);
    do_quant_q8("key weights", out, l0.wk.data, COUNT_WEIGHTS_LAYER,
                group_size);
    do_quant_q8("value weights", out, l0.wv.data, COUNT_WEIGHTS_LAYER,
                group_size);
    do_quant_q8("output weights", out, l0.wo.data, COUNT_WEIGHTS_LAYER,
                group_size);
    do_quant_q8("ffn w1", out, l0.w1.data, COUNT_WEIGHTS_H_LAYER, group_size);
    do_quant_q8("ffn w2", out, l0.w2.data, COUNT_WEIGHTS_H_LAYER, group_size);
    do_quant_q8("ffn w3", out, l0.w3.data, COUNT_WEIGHTS_H_LAYER, group_size);
  } else {
    do_quant_q4("token embeddings", out, m.tok_embeddings.data, COUNT_TOK_EMB,
                group_size);
    do_quant_q4("output weights", out, m.output_weights.data, COUNT_OUTPUT_WEIGHT, group_size);
    do_quant_q4("query weights", out, l0.wq.data, COUNT_WEIGHTS_LAYER,
                group_size);
    do_quant_q4("key weights", out, l0.wk.data, COUNT_WEIGHTS_LAYER,
                group_size);
    do_quant_q4("value weights", out, l0.wv.data, COUNT_WEIGHTS_LAYER,
                group_size);
    do_quant_q4("output weights", out, l0.wo.data, COUNT_WEIGHTS_LAYER,
                group_size);
    do_quant_q4("ffn w1", out, l0.w1.data, COUNT_WEIGHTS_H_LAYER, group_size);
    do_quant_q4("ffn w2", out, l0.w2.data, COUNT_WEIGHTS_H_LAYER, group_size);
    do_quant_q4("ffn w3", out, l0.w3.data, COUNT_WEIGHTS_H_LAYER, group_size);
  }

  munmap(q_buf, SIZE_WEIGHTS_H_LAYER_Q8);
  munmap(deq_buf, SIZE_WEIGHTS_H_LAYER_F32);
  munmap(scales_buf, SIZE_WEIGHTS_H_LAYER_F32 / group_size);
}
