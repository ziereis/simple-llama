#include <assert.h>
#include "llama.h"
#include "ops.h"
#include "quantize.h"
#include "utils.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>



static float tol = 0.1;


#define REQUIRE_EQ_APROX(a, b)                                                 \
  if (!approx_equal(a, b)) {                                                   \
    printf("failed: %s: %f != %s: %f in line: %d \n", #a, a, #b, b, __LINE__); \
    return 1;                                                                  \
  }

#define REQUIRE_EQ(a, b)                                                       \
  if (a != b) {                                                                \
    printf("failed: %s: %d != %s: %d in line: %d \n", #a, a, #b, b, __LINE__); \
    return 1;                                                                  \
  }

#define REQUIRE_NEQ(a, b)                                                      \
  if (a == b) {                                                                \
    printf("failed: %s: %d == %s: %d in line: %d \n", #a, a, #b, b, __LINE__); \
    return 1;                                                                  \
  }
#define REQUIRE_LT(a , b)                                                      \
  if (a >= b) {                                                                \
    printf("failed: %s: %f >= %s: %f in line: %d \n", #a, a, #b, b, __LINE__); \
    return 1;                                                                  \
  }



bool approx_equal(float a, float b) {
  return fabs(a - b) < tol;
}


int test_quantize_q8() {
  float in[] = {-8.5, 9.3, -1.5, 3.4, 5.6, -7.8, 9.1, -2.3, -4.5, 22.0, 0.0, 8.9};
  int n = 12;
  int g_size = 6;
  float scales[n / g_size];
  i8 out[n];
  quantize_q8(out, scales, in, n, g_size);
  float deq[n];
  dequantize_q8(deq, out, scales, n, g_size);
  f32 err = calc_error(in, deq, n);
  printf("err: %f\n", err);
  REQUIRE_LT(err, 0.1f);
  return 0;
}

int test_quantize_q4() {
  // for (i8 i = -8; i < 8; i++) {
  //   for (i8 j = -8; j < 8; j++) {
  //     i8 packed = pack_q4(i, j);
  //     Pair unpacked = unpack_q4(packed);
  //     REQUIRE_EQ(i, unpacked.left);
  //     REQUIRE_EQ(j, unpacked.right);
  //   }
  // }
  tol = 2;
  float in[] = {-8.5, 9.3,  -1.5, 3.4,  5.6, -7.8,
                9.1,  -2.3, -5.5, 22.0, 0.0, 8.9};
  int n = 12;
  int g_size = 4;
  float scales[ n / g_size];
  i8 out[n / 2];
  quantize_q4(out, scales,in, n, g_size);
  float deq[n];
  dequantize_q4(deq, out, scales, n, g_size);
  f32 err = calc_error(in, deq, n);
  REQUIRE_LT(err, 1.0f);
  return 0;
}

i32* generate_greedy(LLamaRuntime* rt, i32 *tokens, u32 n_toks, u32 max_toks) {
  i32 *res = malloc(max_toks * sizeof(i32));
  u32 pos = 0;
  for (; pos < n_toks; pos++) {
    float *logits = forward_f32(rt, tokens[pos], pos);
    res[pos] = tokens[pos];
  }
  for (; pos < max_toks; pos++) {
    float *logits = forward_f32(rt, res[pos - 1], pos);
    Matrix mat = {logits, {1, 32000}};
    softmax(mat.data, 32000);
    int max_idx = 0;
    float max_val = mat.data[0];
    for (int i = 1; i < 32000; i++) {
      if (mat.data[i] > max_val) {
        max_val = mat.data[i];
        max_idx = i;
      }
    }
    res[pos] = max_idx;
  }
  return res;
}

i32* generate_greedy_q8(QLLamaRuntime* rt, i32 *tokens, u32 n_toks, u32 max_toks) {
  i32 *res = malloc(max_toks * sizeof(i32));
  u32 pos = 0;
  for (; pos < n_toks; pos++) {
    float *logits = forward_q8(rt, tokens[pos], pos);
    res[pos] = tokens[pos];
  }
  for (; pos < max_toks; pos++) {
    float *logits = forward_q8(rt, res[pos - 1], pos);
    Matrix mat = {logits, {1, 32000}};
    softmax(mat.data , 32000);
    int max_idx = 0;
    float max_val = mat.data[0];
    for (int i = 1; i < 32000; i++) {
      if (mat.data[i] > max_val) {
        max_val = mat.data[i];
        max_idx = i;
      }
    }
    res[pos] = max_idx;
  }
  return res;
}

i32* generate_greedy_q4(QLLamaRuntime* rt, i32 *tokens, u32 n_toks, u32 max_toks) {
  i32 *res = malloc(max_toks * sizeof(i32));
  u32 pos = 0;
  for (; pos < n_toks; pos++) {
    float *logits = forward_q4(rt, tokens[pos], pos);
    res[pos] = tokens[pos];
  }
  for (; pos < max_toks; pos++) {
    float *logits = forward_q4(rt, res[pos - 1], pos);
    Matrix mat = {logits, {1, 32000}};
    softmax(mat.data, 32000);
    int max_idx = 0;
    float max_val = mat.data[0];
    for (int i = 1; i < 32000; i++) {
      if (mat.data[i] > max_val) {
        max_val = mat.data[i];
        max_idx = i;
      }
    }
    res[pos] = max_idx;
  }
  return res;
}

int test_forward_f32() {
  MappedFile in;
  init(&in, "../bin/llama.bin");
  LLama m;
  llama_init(&m, in.data, in.len);
  LLamaRuntime rt;
  runtime_init_f32(&rt, &m);


  i32 input[] = {6123, 5169, 948, 1171, 471, 263};

  i32 *res = generate_greedy(&rt, input, 6, 30);

  i32 expect[] = {6123, 5169, 948, 1171, 471, 263, 4824, 293, 391, 1058, 2113,
  278, 27813, 20604, 297, 29871, 29896, 29929, 29953, 29945, 363, 670, 664,
  373, 12101, 3546, 5964, 2926, 1199, 29889};

  for (int i = 0; i < 30; i++) {
    REQUIRE_EQ(expect[i], res[i]);
  }

  return 0;
}

int random_number(int min, int max) {
    return (rand() % (max - min + 1)) + min;
}

int test_matvec_mal() {
  Matrix w = new_matrix(4096, 4096);
  Matrix x = new_matrix(1, 4096);
  Matrix out = new_matrix(1, 4096);

  for (int i = 0; i < 4096; i++) {
    for (int j = 0; j < 4096; j++) {
      w.data[i * 4096 + j] = (f32)random_number(-100, 100) / 100.0f;
    }
    x.data[i] = (f32)random_number(-100, 100) / 100.0f;
  }

  matvec_mul(w.data, x.data, out.data, 4096, 4096);

  //print_vec(out.data, 20, 0);

  QMatrix qw = new_matrix_q8(4096, 4096, 128);
  QMatrix qx = new_matrix_q8(1, 4096, 128);
  Matrix qout = new_matrix(1, 4096);
  quantize_q8(qw.data, qw.scales, w.data, 4096 * 4096, 128);
  quantize_q8(qx.data, qx.scales, x.data,  4096, 128);
  matvec_mul_q8(qw.data, qw.scales, qx.data, qx.scales, qout.data, 4096, 4096, 128);

  print_vec(qout.data, 20, 0);

  tol = 0.5;

  for (int i = 0; i < 4096; i++) {
    REQUIRE_EQ_APROX(out.data[i], qout.data[i]);
  }

  QMatrix q4w = new_matrix_q4(4096, 4096, 128);
  QMatrix q4x = new_matrix_q4(1, 4096, 128);
  Matrix q4out = new_matrix(1, 4096);
  quantize_q4(q4w.data, q4w.scales, w.data, 4096 * 4096, 128);
  quantize_q4(q4x.data, q4x.scales, x.data, 4096, 128);

  f32* q4w_rec = malloc(4096 * 4096 * sizeof(f32));
  f32 *q4x_rec = malloc(4096 * sizeof(f32));
  dequantize_q4(q4w_rec, q4w.data, q4w.scales, 4096 * 4096, 128);
  dequantize_q4(q4x_rec, q4x.data, q4x.scales, 4096, 128);

  //print_vec(q4w_rec, 20, 0);


  float max_err = calc_error(w.data, q4w_rec, 4096 * 4096);
  printf("max err weights: %f\n", max_err);
  max_err = calc_error(x.data, q4x_rec, 4096);
  printf("max err x: %f\n", max_err);


  matvec_mul_q4(q4w.data, q4w.scales, q4x.data, q4x.scales, q4out.data, 4096, 4096, 128);

  print_vec(q4out.data, 20, 0);

  tol = 0.1;
  for (int i = 0; i < 4096; i++) {
    REQUIRE_EQ_APROX(out.data[i], q4out.data[i]);
  }


  return 0;
}




int test_forward_q8() {
  MappedFile in;
  init(&in, "../bin/llama_q8.bin");
  QLLama m;
  qllama_init(&m, in.data, in.len);
  QLLamaRuntime rt;
  runtime_init_q8(&rt, &m);

  i32 input[] = {6123, 5169, 948, 1171, 471, 263};


   i32 *res = generate_greedy_q8(&rt, input, 6, 30);

  i32 expect[] = {6123, 5169, 948, 1171, 471, 263, 4824, 293, 391, 1058, 2113,
  278, 27813, 20604, 297, 29871, 29896, 29929, 29953, 29945, 363, 670, 664,
  373, 12101, 3546, 5964, 2926, 1199, 29889};

  for (int i = 0; i < 30; i++) {
    printf("%d ", res[i]);
  }
  printf("\n");

  runtime_deinit_q8(&rt);

  return 0;
}


int test_forward_q4() {
  MappedFile in;
  init(&in, "../bin/llama_q4.bin");
  QLLama m;
  qllama_init(&m, in.data, in.len);
  QLLamaRuntime rt;
  runtime_init_q4(&rt, &m);

  i32 input[] = {6123, 5169, 948, 1171, 471, 263};


  i32 *res = generate_greedy_q4(&rt, input, 6, 30);

  i32 expect[] = {6123, 5169, 948, 1171, 471, 263, 4824, 293, 391, 1058, 2113,
  278, 27813, 20604, 297, 29871, 29896, 29929, 29953, 29945, 363, 670, 664,
  373, 12101, 3546, 5964, 2926, 1199, 29889};

  for (int i = 0; i < 30; i++) {
    printf("%d, ", res[i]);
  }
  printf("\n");

  return 0;
}

int main() {
  // int res = 0;
  // res = test_quantize_q8();
  // res = test_quantize_q4();
  // // res = test_matvec_mal();
  // Timer timer;
  // start_timer(&timer);
  // res = test_forward_q8();
  // stop_timer(&timer);
  // printf("Time: %d ms\n", elapsed_time(&timer));
  //res = test_matvec_mal();
  //quant_model("../bin/llama.bin", "../bin/llama_q4.bin", Q4);
  // QLLama m;
  // MappedFile in;
  // init(&in, "../bin/llama_q8.bin");
  // qllama_init(&m, in.data, in.len);
  // QLLamaRuntime rt;
  // runtime_init_q8(&rt, &m);
  QLLama dev_m;
  QLLama host_m;
  qllama_init_device(&dev_m, &host_m);
  return 0;
}
