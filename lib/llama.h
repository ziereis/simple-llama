#pragma once
#include "utils.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct __attribute__((__packed__)){
  i32 dim;
  i32 hidden_dim;
  i32 n_heads;
  i32 n_layers;
  i32 vocab_size;
  i32 max_seq_len;
  i32 group_size;
  i32 bitwidth;
} Params;


typedef struct  {
  Matrix attention_norm;
  QMatrix wq;
  QMatrix wk;
  QMatrix wv;
  QMatrix wo;
  Matrix ffn_norm;
  QMatrix w1;
  QMatrix w2;
  QMatrix w3;
} QEncoderBlock;

typedef struct  {
  Matrix attention_norm;
  Matrix wq;
  Matrix wk;
  Matrix wv;
  Matrix wo;
  Matrix ffn_norm;
  Matrix w1;
  Matrix w2;
  Matrix w3;
} EncoderBlock;

typedef struct {
  MappedFile file;
  u8 *data;
  u64 len;
  Params params;
  Matrix norm;
  QMatrix tok_embeddings;
  QMatrix output_weights;
  QEncoderBlock layers[32];
} QLLama;

typedef struct {
  MappedFile file;
  Params params;
  Matrix norm;
  Matrix tok_embeddings;
  Matrix output_weights;
  EncoderBlock layers[32];
} LLama;

typedef struct {
  Matrix q;
  Matrix f_x;
  QMatrix q_x;
  Matrix f_x_buf;
  QMatrix q_x_buf;
  Matrix f_x_buf2;
  QMatrix q_h_buf;
  Matrix f_h_buf;
  Matrix f_h_buf2;
  Matrix logits;
  Matrix logits_out; // only used in cuda rt
  Matrix attention;

  Matrix kcaches[32];
  Matrix vcaches[32];
  QLLama m;
} QLLamaRuntime;

typedef struct {
  Matrix q;
  Matrix x;
  Matrix x_buf;
  Matrix x_buf2;
  Matrix h_buf;
  Matrix h_buf2;
  Matrix logits;
  Matrix attention;

  Matrix kcaches[32];
  Matrix vcaches[32];
  LLama m;
} LLamaRuntime;


bool qllama_init(QLLama *m, u8* data, u64 size);
bool llama_init(LLama *m, u8 *data, u64 size);
void runtime_init_q8(QLLamaRuntime *rt, QLLama *m);
void runtime_init_f32(LLamaRuntime *rt, LLama *m);
void runtime_init_q4(QLLamaRuntime *rt, QLLama *m);

void runtime_deinit_f32(LLamaRuntime *rt);
void runtime_deinit_q8(QLLamaRuntime *rt);
void runtime_deinit_q4(QLLamaRuntime *rt);


f32 *forward_q8(QLLamaRuntime *rt, i32 tok, i32 pos);
f32 *forward_f32(LLamaRuntime *rt, i32 tok, i32 pos);
f32 *forward_q4(QLLamaRuntime *rt, i32 tok, i32 pos);


// for python bindings
void* runtime_new_f32(const char* filename);
void runtime_delete_f32(void* handle);
float *runtime_forward_f32(void *handle, int tok, int pos);

void* runtime_new_q8(const char* filename);
void runtime_delete_q8(void* handle);
float* runtime_forward_q8(void* handle, int tok, int pos);

void* runtime_new_q4(const char* filename);
void runtime_delete_q4(void* handle);
float *runtime_forward_q4(void *handle, int tok, int pos);

#ifdef __cplusplus
}
#endif
