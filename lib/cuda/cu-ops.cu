#include "cu-ops.h"
#include <cassert>
#include <cfloat>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include "lib/utils.h"
#include "stdio.h"

/*
This file provides CUDA-based operations for the LLaMA model, specifically for handling Q4 quantization format.

The key functionalities include:
- Dequantization and quantization of Q4 format tensors.
- Root Mean Square (RMS) normalization.
- Matrix-vector multiplication for Q4 format.
- Rotation of embeddings.
- Computation of attention scores, applying softmax, and computing the attention output.
- Residual connections and Swish-Gated Linear Unit (SWIGLU) activation.

The file includes:
- CUDA kernel functions for dequantization, quantization, RMS normalization, matrix-vector multiplication, embedding rotation, attention computation, residual connections, and SWIGLU activation.
- Host functions to launch the CUDA kernels and synchronize the device.


Example usage:
    f32* output;
    d_dequantize_q4(output, input, scales, n, group_size);
    d_rms_norm(output, input, weights, n);
    d_matvec_mul_q4(weights, weight_scales, input, input_scales, output, m, n, group_size);
    d_compute_attention(att, q, kcache, vcache, output, pos, n_heads, head_dim, max_seq_len);
    d_residual(output, input, n);
    d_swiglu(output, scales, n);

This will perform various operations on Q4 quantized tensors using CUDA.
*/


#ifdef BENCH
double rms_time = 0;
double matvec_q4_time = 0;
double quantize_q4_time = 0;
double dequantize_q4_time = 0;
double attention_time = 0;
double rotate_time = 0;
double swiglu_time = 0;
double residual_time = 0;
#endif


#define unpack_left(in) ((in >> 4) & 0x0f) - 8
#define unpack_right(in) (in & 0x0f) - 8
#define pack_q4(left, right) (((((u8)left) + 8) << 4) | (((u8)right) + 8))


__global__ void deq_q4(f32 *out, i8 *in, f32 *scales, u64 n, i32 group_size) {
  u64 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n / group_size) {
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
}

void d_dequantize_q4(f32 *out, i8 *in, f32 *scales, u64 n, i32 group_size) {
  u64 n_groups = n / group_size;
#ifdef BENCH
  Timer t;
  start_timer(&t);
#endif
  deq_q4<<<n_groups, group_size>>>(out, in, scales, n, group_size);
#ifdef BENCH
  cudaDeviceSynchronize();
  stop_timer(&t);
  dequantize_q4_time += elapsed_time(&t);
#endif

}

__global__ void quant_q4(i8 *out, f32 *scales, f32 *in, u64 n, i32 group_size) {
  u64 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n / group_size) {
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
}

void d_quantize_q4(i8 *out, f32 *scales, f32 *in, u64 n, i32 group_size) {
  u64 n_groups = n / group_size;
  #ifdef BENCH
  Timer t;
  start_timer(&t);
  #endif
  quant_q4<<<n_groups, group_size>>>(out, scales, in, n, group_size);
  #ifdef BENCH
  cudaDeviceSynchronize();
  stop_timer(&t);
  quantize_q4_time += elapsed_time(&t);
  #endif
}


__global__ void sumSquares(float *result, float *x, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float partialSum[256]; // Adjust size based on block size

    float sum = 0;
    if (index < n) {
        sum = x[index] * x[index];
    }

    partialSum[threadIdx.x] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partialSum[threadIdx.x] += partialSum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (threadIdx.x == 0) {
        atomicAdd(result, partialSum[0]);
    }
}

__global__ void scale(float* out, float* x, float* weights, float sum, int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    out[index] = weights[index] * x[index] * sum;
  }
}


#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

void d_rms_norm(f32 *out, f32 *x, f32 *weights, i32 n) {
#ifdef BENCH
  Timer t;
  start_timer(&t);
#endif

  float *d_sum;
  cudaMalloc(&d_sum, sizeof(float));
  cudaMemset(d_sum, 0, sizeof(float));
  int block_size = 256;
  int grid_size = CEIL_DIV(n, block_size);
  sumSquares<<<grid_size, block_size>>>(d_sum, x, n);
  cudaDeviceSynchronize();
  float h_sum;
  cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_sum);
  h_sum /= n;
  h_sum += 1e-5f;
  h_sum = 1.0f / sqrt(h_sum);
  scale<<<grid_size, block_size>>>(out, x, weights, h_sum, n);
#ifdef BENCH
  cudaDeviceSynchronize();
  stop_timer(&t);
  rms_time += elapsed_time(&t);
#endif
}


template <int N, int Bdim, int g_size>
__global__ void matvec_mul_q4(i8 *__restrict__ w, f32 *__restrict__ w_s,
                              i8 *__restrict__ x, f32 *__restrict__ x_s,
                              f32 *out, int m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ char shared_x[N / 2];

  int copy_size = N / 2 / Bdim;
  int copy_start = threadIdx.x * copy_size;
  int copy_end = copy_start + copy_size;
  assert(copy_end <= N / 2);
  for (int j = copy_start; j < copy_end; j++) {
    shared_x[j] = x[j];
  }
  __syncthreads();

  constexpr int half_group = g_size / 2;

  constexpr int groups_per_row = N / g_size;
  if (i < m) {
    i8 * w_row = w + i * N / 2;
    f32 * w_s_row = w_s + i * groups_per_row;
    f32 faccum = 0.0f;
    #pragma unroll groups_per_row
    for (u32 g_idx = 0; g_idx < groups_per_row; g_idx++) {
      i32 iaccum = 0;
      u32 start_idx = g_idx * half_group;
      u32 end_idx = (g_idx + 1) * half_group;
      #pragma unroll half_group
      for (u32 j = start_idx; j < end_idx; j++) {
        u8 p_x = (u8)shared_x[j];
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
}



static constexpr int bdim = 128;
static constexpr int gdim = 128;
void d_matvec_mul_q4(i8 *w, f32 *w_s, i8 *x, f32 *x_s, f32 *out, int m,
                     int n, u32 group_size) {
  int grid_size = CEIL_DIV(m, bdim);
#ifdef BENCH
  Timer t;
  start_timer(&t);
#endif
  if (n == 11008) {
    matvec_mul_q4<11008, bdim, gdim><<<grid_size, bdim>>>(w, w_s, x, x_s, out, m);
  } else {
    matvec_mul_q4<4096, bdim, gdim><<<grid_size, bdim>>>(w, w_s, x, x_s, out, m);
  }
#ifdef BENCH
  cudaDeviceSynchronize();
  stop_timer(&t);
  matvec_q4_time += elapsed_time(&t);
#endif
}


__global__ void rot_emb(float* x, int pos, int head_dim, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim / 2) {
        int idx = (2 * i) % head_dim;
        float freq = 1.0f / powf(10000.0f, idx / (float)head_dim);
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);

        float v0 = x[2 * i];
        float v1 = x[2 * i + 1];
        x[2 * i] = v0 * fcr - v1 * fci;
        x[2 * i + 1] = v0 * fci + v1 * fcr;
    }
}

void d_rotate_embeddings(f32 *x, i32 pos, i32 head_dim, i32 dim) {
    int block_size = 256;
    int grid_size = CEIL_DIV(dim / 2, block_size);
#ifdef BENCH
    Timer t;
    start_timer(&t);
#endif

    rot_emb<<<grid_size, block_size>>>(x, pos, head_dim, dim);
#ifdef BENCH
  cudaDeviceSynchronize();
    stop_timer(&t);
    rotate_time += elapsed_time(&t);
#endif
}


__global__ void compute_scores_kernel(float *q, float *kcache, float *att, int pos, int n_heads, int head_dim, int max_seq_len) {
    int head = blockIdx.x;
    int t = threadIdx.x;
    if (head < n_heads && t <= pos) {
        float *q_head = q + head * head_dim;
        float *k_head = kcache + (t * n_heads * head_dim) + (head * head_dim);
        float score = 0.0f;

        for (int j = 0; j < head_dim; j++) {
            score += q_head[j] * k_head[j];
        }
        score /= sqrtf((float)head_dim);

        att[head * max_seq_len + t] = score;
    }
}

__global__ void apply_softmax_kernel(float *att, int pos, int n_heads, int max_seq_len) {
    int head = blockIdx.x;
    if (head < n_heads) {
        float *att_head = att + head * max_seq_len;
        float max_val = -FLT_MAX;
        for (int i = 0; i <= pos; i++) {
            max_val = max(max_val, att_head[i]);
        }
        float sum = 0.0f;
        for (int i = 0; i <= pos; i++) {
            att_head[i] = expf(att_head[i] - max_val);
            sum += att_head[i];
        }
        for (int i = 0; i <= pos; i++) {
            att_head[i] /= sum;
        }
    }
}

__global__ void compute_output_kernel(float *att, float *vcache, float *out, int pos, int n_heads, int head_dim, int max_seq_len) {
    int head = blockIdx.x;
    int j = threadIdx.x;
    if (head < n_heads && j < head_dim) {
        float sum = 0.0f;
        float *out_head = out + head * head_dim;
        for (int t = 0; t <= pos; t++) {
            float *v_head = vcache + (t * n_heads * head_dim) + (head * head_dim);
            sum += att[head * max_seq_len + t] * v_head[j];
        }
        out_head[j] = sum;
    }
}

void d_compute_attention(f32 *att, f32 *q, f32 *kcache, f32 *vcache, f32 *out,
                         i32 pos, i32 n_heads, i32 head_dim, i32 max_seq_len) {
dim3 blocks(n_heads);
dim3 threadsPerBlock(pos + 1);
#ifdef BENCH
Timer t;
start_timer(&t);
#endif

compute_scores_kernel<<<blocks, threadsPerBlock>>>(q, kcache, att, pos, n_heads, head_dim, max_seq_len);
cudaDeviceSynchronize();
apply_softmax_kernel<<<blocks, 1>>>(att, pos, n_heads, max_seq_len);
cudaDeviceSynchronize();
dim3 outputThreadsPerBlock(head_dim);
compute_output_kernel<<<blocks, outputThreadsPerBlock>>>(att, vcache, out, pos, n_heads, head_dim, max_seq_len);
cudaDeviceSynchronize();
#ifdef BENCH
stop_timer(&t);
attention_time += elapsed_time(&t);
#endif
}

__global__ void add(float *out, float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] += a[i];
    }
}

void d_residual(f32 *out, f32 *a, i32 n) {
    int block_size = 256;
    int grid_size = CEIL_DIV(n, block_size);
#ifdef BENCH
    Timer t;
    start_timer(&t);
#endif
    add<<<grid_size, block_size>>>(out, a, n);
#ifdef BENCH
cudaDeviceSynchronize();
    stop_timer(&t);
    residual_time += elapsed_time(&t);
#endif
}


__global__ void swig(f32* out, f32* scales,  int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float val = out[i];
    val *= 1.0f / (1.0f + expf(-val));
    val *= scales[i];
    out[i] = val;
  }
}

void d_swiglu(f32 *out, f32 *scales, i32 n) {
  int block_size = 256;
  int grid_size = CEIL_DIV(n, block_size);
#ifdef BENCH
  Timer t;
  start_timer(&t);
#endif
  swig<<<grid_size, block_size>>>(out, scales, n);
#ifdef BENCH
cudaDeviceSynchronize();
  stop_timer(&t);
  swiglu_time += elapsed_time(&t);
#endif
}
