#pragma once
#include "utils.h"

void rms_norm(f32* out, f32* x, f32* weights, i32 n);

void matvec_mul(f32* weights, f32* x, f32* out, i32 m, i32 n);

void matvec_mul_q8(i8  * w, f32  * w_s, i8  * x, f32  * x_s, f32  * out, int m, int n, u32 group_size);

void matvec_mul_q4(i8  * w, f32  * w_s, i8  * x, f32  * x_s, f32  * out, int m, int n, u32 group_size);

void softmax(f32* x, i32 n);

void quantize_q8(i8  * out, f32  * scales, f32  * in,
                 u64 n, i32 group_size);
void dequantize_q8(f32  * out, i8  * in, f32  * scales,
                   u64 n, i32 group_size);
void quantize_q4(i8  * out, f32  * scales, f32  * in,
                 u64 n, i32 group_size);
void dequantize_q4(f32  * out, i8  * in, f32  * scales,
                   u64 n, i32 group_size);


void compute_attention(f32 * att, f32 * q, f32* kcache, f32* vcache,
                       f32* out, i32 pos, i32 n_heads, i32 head_dim, i32 max_seq_len);

void rotate_embeddings(f32* x, i32 pos, i32 head_dim, i32 dim);

void swiglu(f32* out, f32* scales, i32 n);

void residual(f32* out, f32* in, i32 n);
