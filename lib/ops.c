#include "ops.h"
#include "string.h"
#include "utils.h"
#include <assert.h>
#include <math.h>

#ifdef BENCH
double rms_time = 0;
double matvec_time = 0;
double matvec_q8_time = 0;
double matvec_q4_time = 0;
double softmax_time = 0;
double quantize_q8_time = 0;
double dequantize_q8_time = 0;
double quantize_q4_time = 0;
double dequantize_q4_time = 0;
double attention_time = 0;
double rotate_time = 0;
double swiglu_time = 0;
double residual_time = 0;
#endif



void rms_norm(f32 *out, f32 *x, f32 *weights, i32 n) {
  #ifdef BENCH
  Timer t;
  start_timer(&t);
  #endif

  f32 sum = 0.0f;
  for (i32 i = 0; i < n; i++) {
    sum += x[i] * x[i];
  }
  sum /= n;
  sum += 1e-5f;
  sum = 1.0f / sqrt(sum);
  for (i32 i = 0; i < n; i++) {
    out[i] = weights[i] * x[i] * sum;
  }
  #ifdef BENCH
  stop_timer(&t);
  rms_time += elapsed_time(&t);
  #endif
}

#define AT_2D(mat, i, j) mat[i * n + j]

void matvec_mul(f32* weights, f32* x, f32* out, i32 m, i32 n) {
#ifdef BENCH
  Timer t;
  start_timer(&t);
#endif

#pragma omp parallel for schedule(static)
  for (i32 i = 0; i < m; i++) {
    f32 val = 0.0f;
    for (i32 j = 0; j < n; j++) {
      val += AT_2D(weights, i, j) * x[j];
    }
    out[i] = val;
  }
#ifdef BENCH
  stop_timer(&t);
  matvec_time += elapsed_time(&t);
#endif
}

void matvec_mul_q8(i8  * w, f32  * w_s, i8  * x, f32  * x_s, f32  * out, int m, int n, u32 group_size) {
#ifdef BENCH
  Timer t;
  start_timer(&t);
#endif

  u32 groups_per_row = n / group_size;

#pragma omp parallel for schedule(static)
  for (i32 i = 0; i < m; i++) {
    i8  * w_row = w + i * n;
    f32  * w_s_row = w_s + i * n / group_size;
    f32 faccum = 0.0f;
    for (u32 g_idx = 0; g_idx < groups_per_row; g_idx++) {
      i32 iaccum = 0;
      u32 start_idx = g_idx * group_size;
      u32 end_idx = (g_idx + 1) * group_size;
      for (u32 j = start_idx; j < end_idx; j++) {
        iaccum += ((i32)w_row[j]) * ((i32)x[j]);
      }
      faccum += ((float)iaccum) * w_s_row[g_idx] * x_s[g_idx];
    }
    out[i] = faccum;
  }
#ifdef BENCH
  stop_timer(&t);
  matvec_q8_time += elapsed_time(&t);
#endif
}


#define unpack_left(in) ((in >> 4) & 0x0f) - 8
#define unpack_right(in) (in & 0x0f) - 8
#define pack_q4(left, right) (((((u8)left) + 8) << 4) | (((u8)right) + 8))

void matvec_mul_q4(i8  * w, f32  * w_s, i8  * x, f32  * x_s, f32  * out, int m, int n, u32 group_size) {
#ifdef BENCH
  Timer t;
  start_timer(&t);
#endif

  u32 groups_per_row = n / group_size;

#pragma omp parallel for schedule(static)
  for (i32 i = 0; i < m; i++) {
    i8  * w_row = w + i * n / 2;
    f32  * w_s_row = w_s + i * n / group_size;
    f32 faccum = 0.0f;
    for (u32 g_idx = 0; g_idx < groups_per_row; g_idx++) {
      i32 iaccum = 0;
      u32 start_idx = g_idx * group_size / 2;
      u32 end_idx = (g_idx + 1) * group_size / 2;
      for (u32 j = start_idx; j < end_idx; j++) {
        u8 p_x = (u8)x[j];
        i32 x_left = unpack_left(p_x);
        i32 x_right = unpack_right(p_x);
        u8 p_w = (u8)w_row[j];
        i32 w_left = unpack_left(p_w);
        i32 w_right = unpack_right(p_w);
        iaccum += x_left * w_left + x_right * w_right;
      }
      faccum += ((float)iaccum) * w_s_row[g_idx] * x_s[g_idx];
    }
    out[i] = faccum;
  }
#ifdef BENCH
  stop_timer(&t);
  matvec_q4_time += elapsed_time(&t);
#endif
}

void softmax(f32* x, i32 n) {
#ifdef BENCH
  Timer t;
  start_timer(&t);
#endif

  f32 max_val = x[0];
  for (i32 i = 1; i < n; i++) {
    max_val = fmax(max_val, x[i]);
  }
  // exp and sum
  f32 sum = 0.0f;
  for (i32 i = 0; i < n; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // normalize
  for (i32 i = 0; i < n; i++) {
    x[i] /= sum;
  }
#ifdef BENCH
  stop_timer(&t);
  softmax_time += elapsed_time(&t);
#endif
}

void dequantize_q8(f32  * out, i8  * in, f32  * scales,
                   u64 n, i32 group_size) {
#ifdef BENCH
  Timer t;
  start_timer(&t);
#endif

  u64 n_groups = n / group_size;

#pragma omp parallel for schedule(static)
  for (u64 i = 0; i < n_groups; i++) {
    u64 start_idx = i * group_size;
    u64 end_idx = (i + 1) * (group_size);
    for (u64 j = start_idx; j < end_idx; j++) {
      out[j] = ((f32)(in[j])) * scales[i];
    }
  }
#ifdef BENCH
  stop_timer(&t);
  dequantize_q8_time += elapsed_time(&t);
#endif
}

void quantize_q8(i8  * out, f32  * scales, f32  * in,
                 u64 n, i32 group_size) {
#ifdef BENCH
  Timer t;
  start_timer(&t);
#endif

  u64 n_groups = n / group_size;
#pragma omp parallel for schedule(static)
  for (u64 i = 0; i < n_groups; i++) {
    u64 start_idx = i * group_size;
    u64 end_idx = (i + 1) * (group_size);

    f32 max = fabs(in[start_idx]);
    for (u64 j = start_idx; j < end_idx; j++) {
      max = fmax(max, fabs(in[j]));
    }
    f32 scale = max / 127.0f;

    for (u64 j = start_idx; j < end_idx; j++) {
      out[j] = (i8)round(in[j] / scale);
    }
    scales[i] = scale;
  }
#ifdef BENCH
  stop_timer(&t);
  quantize_q8_time += elapsed_time(&t);
#endif
}

void quantize_q4(i8  * out, f32  * scales, f32  * in,
                 u64 n, i32 group_size) {
#ifdef BENCH
  Timer t;
  start_timer(&t);
#endif

  assert(n % 2 == 0);
  assert(group_size % 2 == 0);
  u64 n_groups = n / group_size;
#pragma omp parallel for schedule(static)
  for (u64 i = 0; i < n_groups; i++) {
    u64 start_idx = i * group_size;
    u64 end_idx = (i + 1) * (group_size);

    f32 max = fabs(in[start_idx]);
    for (u64 j = start_idx; j < end_idx; j++) {
      max = fmax(max, fabs(in[j]));
    }
    f32 scale = max / 7.0f;

    for (u64 j = start_idx / 2; j < end_idx / 2; j++) {
      i8 left = (i8)round(in[j * 2] / scale);
      i8 right = (i8)round(in[(j * 2) + 1] / scale);
      out[j] = pack_q4(left, right);
    }
    scales[i] = scale;
  }
#ifdef BENCH
  stop_timer(&t);
  quantize_q4_time += elapsed_time(&t);
#endif
}

void dequantize_q4(f32  * out, i8  * in, f32  * scales,
                   u64 n, i32 group_size) {
#ifdef BENCH
  Timer t;
  start_timer(&t);
#endif

  u64 n_groups = n / group_size;
#pragma omp parallel for schedule(static)
  for (u64 i = 0; i < n_groups; i++) {
    u64 start_idx = i * group_size / 2;
    u64 end_idx = (i + 1) * (group_size / 2);
    for (u64 j = start_idx; j < end_idx; j++) {
        u8 p = (u8)in[j];
        i32 left = unpack_left(p);
        i32 right = unpack_right(p);
      out[j * 2] = ((f32)left) * scales[i];
      out[(j * 2) + 1] = ((f32)right) * scales[i];
    }
  }
#ifdef BENCH
  stop_timer(&t);
  dequantize_q4_time += elapsed_time(&t);
#endif
}

void compute_attention(f32 *att, f32 *q, f32 *kcache, f32 *vcache, f32 *out,
                           i32 pos, i32 n_heads, i32 head_dim,
                           i32 max_seq_len) {
#ifdef BENCH
  Timer t;
  start_timer(&t);
#endif

  int dim = n_heads * head_dim;
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < n_heads; i++) {
    f32  * q_head = q + i * head_dim;
    f32  * curr_att = att + i * max_seq_len;
    for (int t = 0; t <= pos; t++) {
      f32  * k_head = kcache + dim * t + i * head_dim;
      float score = 0;
      for (int j = 0; j < head_dim; j++) {
        score += q_head[j] * k_head[j];
      }
      score /= sqrtf(head_dim);
      curr_att[t] = score;
    }

    softmax(curr_att, pos + 1);

    f32  * out_head = out + i * head_dim;
    memset(out_head, 0, head_dim * sizeof(float));

    for (int t = 0; t <= pos; t++) {
      f32  * v_head = vcache + dim * t + i * head_dim;
      float score = curr_att[t];
      for (int j = 0; j < head_dim; j++) {
        out_head[j] += score * v_head[j];
      }
    }
  }
#ifdef BENCH
  stop_timer(&t);
  attention_time += elapsed_time(&t);
#endif
}

void rotate_embeddings(f32* x, i32 pos, i32 head_dim, i32 dim) {
#ifdef BENCH
  Timer t;
  start_timer(&t);
#endif

  for (i32 i = 0; i < dim; i += 2) {
    int idx = i % head_dim;
    float freq = 1.0f / powf(10000.0f, idx / (float)head_dim);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    float v0 = x[i];
    float v1 = x[i + 1];
    x[i] = v0 * fcr - v1 * fci;
    x[i + 1] = v0 * fci + v1 * fcr;
  }
#ifdef BENCH
  stop_timer(&t);
  rotate_time += elapsed_time(&t);
#endif
}

void swiglu(f32* out, f32* scales, i32 n) {
#ifdef BENCH
  Timer t;
  start_timer(&t);
#endif

  for (i32 i = 0; i < n; i++) {
    float val = out[i];
    val *= 1.0f / (1.0f + expf(-val));
    val *= scales[i];
    out[i] = val;
  }
#ifdef BENCH
  stop_timer(&t);
  swiglu_time += elapsed_time(&t);
#endif
}

void residual(f32* out, f32* in, i32 n) {
#ifdef BENCH
  Timer t;
  start_timer(&t);
#endif

  for (i32 i = 0; i < n; i++) {
    out[i] += in[i];
  }
#ifdef BENCH
  stop_timer(&t);
  residual_time += elapsed_time(&t);
#endif
}
