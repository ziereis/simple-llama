#pragma once
#include "utils.h"


#ifdef BENCH
extern double rms_time;
extern double matvec_time;
extern double matvec_q8_time;
extern double matvec_q4_time;
extern double softmax_time;
extern double quantize_q8_time;
extern double dequantize_q8_time;
extern double quantize_q4_time;
extern double dequantize_q4_time;
extern double attention_time;
extern double rotate_time;
extern double swiglu_time;
extern double residual_time;
#endif


void rms_norm(f32* out, f32* x, f32* weights, i32 n);

void matvec_mul(f32* restrict weights, f32* restrict x, f32* restrict out, i32 m, i32 n);

void matvec_mul_q8(i8* restrict w, f32* restrict w_s, i8* restrict x, f32* restrict x_s, f32* restrict out, int m, int n, u32 group_size);

void matvec_mul_q4(i8* restrict w, f32* restrict w_s, i8* restrict x, f32* restrict x_s, f32* restrict out, int m, int n, u32 group_size);

void softmax(f32* restrict x, i32 n);

void quantize_q8(i8* restrict out, f32* restrict scales, f32* restrict in,
                 u64 n, i32 group_size);
void dequantize_q8(f32* restrict out, i8* restrict in, f32* restrict scales,
                   u64 n, i32 group_size);
void quantize_q4(i8* restrict out, f32* restrict scales, f32* restrict in,
                 u64 n, i32 group_size);
void dequantize_q4(f32* restrict out, i8* restrict in, f32* restrict scales,
                   u64 n, i32 group_size);

void compute_attention(f32* restrict att, f32* restrict q, f32* restrict kcache, f32* restrict vcache,
                       f32* restrict out, i32 pos, i32 n_heads, i32 head_dim, i32 max_seq_len);

void rotate_embeddings(f32* restrict x, i32 pos, i32 head_dim, i32 dim);

void swiglu(f32* restrict out, f32* restrict scales, i32 n);

void residual(f32* restrict out, f32* restrict in, i32 n);
