#include "../utils.h"


#ifdef BENCH
#ifdef BENCH
extern double rms_time;
extern double matvec_q4_time;
extern double quantize_q4_time;
extern double dequantize_q4_time;
extern double attention_time;
extern double rotate_time;
extern double swiglu_time;
extern double residual_time;
#endif
#endif

void d_dequantize_q4(f32 *out, i8 *in, f32 *scales, u64 n, i32 group_size);
void d_rms_norm(f32 *out, f32 *x, f32 *weights, i32 n);
void d_quantize_q4(i8 *out, f32 *scales, f32 *in, u64 n, i32 group_size);
void d_matvec_mul_q4(i8 *__restrict__ w, f32 *__restrict__ w_s, i8 *__restrict__ x, f32 *__restrict__ x_s, f32 *out, int m,
                     int n, u32 group_size) ;
void d_rotate_embeddings(f32 *x, i32 pos, i32 head_dim, i32 dim);

void d_compute_attention(f32 *att, f32 *q, f32 *kcache, f32 *vcache, f32 *out,
                         i32 pos, i32 n_heads, i32 head_dim, i32 max_seq_len);
void d_residual(f32 *out, f32 *a, i32 n);
void d_swiglu(f32 *out, f32 *scales, i32 n);
