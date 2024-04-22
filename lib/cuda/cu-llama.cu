#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include "assert.h"
extern "C" {
#include "../utils.h"
#include "../llama.h"
#include "../ops.h"
}
#include "stdio.h"
#include "cu-ops.h"
#include "algorithm"

void print_device_info() {
  cudaDeviceProp prop;
  int count = 0;

  cudaGetDeviceCount(&count);
  printf("Number of devices: %d\n", count);

  for (int i = 0; i < count; i++) {
    cudaGetDeviceProperties(&prop, i);
    printf("Device %d: %s\n", i, prop.name);
    printf("Global memory: %lu MB\n", prop.totalGlobalMem / (1024 * 1024));
    printf("Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Registers per block: %d\n", prop.regsPerBlock);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max block dimensions: %d x %d x %d\n", prop.maxThreadsDim[0],
           prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("FP16 support: %s\n", (prop.major > 5 || (prop.major == 5 && prop.minor >= 3)) ? "Yes" : "No");
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);

    // Approximate check for BF16 support: available in Ampere or newer architectures (Compute Capability 8.0+)
    if (prop.major >= 8) {
      printf("BF16 support: Yes\n");
    } else {
      printf("BF16 support: No\n");
    }
  }
}

QMatrix new_matrix_q4_device(i32 rows, i32 cols, i32 group_size) {
  i8 *data;
  cudaMalloc(&data, rows * cols * sizeof(i8));
  f32 *scales;
  cudaMalloc(&scales, rows * sizeof(f32));
  return (QMatrix){.scales = scales, .shape = {rows, cols}, .data = data};
}

Matrix new_matrix_device(i32 rows, i32 cols) {
  f32 *data;
  cudaMalloc(&data, rows * cols * sizeof(f32));
  return (Matrix){.data = data,.shape = {rows, cols}};
}

void copy_dtoh(Matrix device, Matrix host) {
  int n = host.shape[0] * host.shape[1];
  cudaMemcpy(host.data, device.data, n * sizeof(f32), cudaMemcpyDeviceToHost);
}
void copy_htod(Matrix host, Matrix device) {
  int n = host.shape[0] * host.shape[1];
  cudaMemcpy(device.data, host.data, n * sizeof(f32), cudaMemcpyHostToDevice);
}
void copy_dtoh_q4(QMatrix device, QMatrix host, int g_size) {
  int n = host.shape[0] * host.shape[1];
  cudaMemcpy(host.data, device.data, n, cudaMemcpyDeviceToHost);
  cudaMemcpy(host.scales, device.scales, n / g_size * sizeof(f32), cudaMemcpyDeviceToHost);
}
void copy_htod_q4(QMatrix host, QMatrix device, int g_size) {
  int n = host.shape[0] * host.shape[1];
  cudaMemcpy(device.data, host.data, n, cudaMemcpyHostToDevice);
  cudaMemcpy(device.scales, host.scales, n / g_size * sizeof(f32), cudaMemcpyHostToDevice);
}



void qllama_init_device(QLLama *d_m, QLLama *h_m) {
  assert(h_m->params.bitwidth == 4);
  d_m->file = h_m->file;
  d_m->params = h_m->params;
  int dim = h_m->params.dim;
  int h_dim = h_m->params.hidden_dim;
  int n_layers = h_m->params.n_layers;
  int vocab_size = h_m->params.vocab_size;
  int g_size = h_m->params.group_size;

  d_m->norm = new_matrix_device(1, dim);
  copy_htod(h_m->norm, d_m->norm);
  d_m->tok_embeddings = new_matrix_q4_device(vocab_size, dim, g_size);
  copy_htod_q4(h_m->tok_embeddings, d_m->tok_embeddings, g_size);
  d_m->output_weights = new_matrix_q4_device(vocab_size, dim, g_size);
  copy_htod_q4(h_m->output_weights, d_m->output_weights, g_size);

  for (int i = 0; i < n_layers; i++) {
    QEncoderBlock *h_layer = &h_m->layers[i];
    QEncoderBlock *d_layer = &d_m->layers[i];
    d_layer->ffn_norm = new_matrix_device(1, dim);
    copy_htod(h_layer->ffn_norm, d_layer->ffn_norm);
    d_layer->attention_norm = new_matrix_device(1, dim);
    copy_htod(h_layer->attention_norm, d_layer->attention_norm);
    d_layer->wq = new_matrix_q8(dim, dim, g_size);
    copy_htod_q4(h_layer->wq, d_layer->wq, g_size);
    d_layer->wk = new_matrix_q8(dim, dim, g_size);
    copy_htod_q4(h_layer->wk, d_layer->wk, g_size);
    d_layer->wv = new_matrix_q8(dim, dim, g_size);
    copy_htod_q4(h_layer->wv, d_layer->wv, g_size);
    d_layer->wo = new_matrix_q8(dim, dim, g_size);
    copy_htod_q4(h_layer->wo, d_layer->wo, g_size);
    d_layer->w1 = new_matrix_q8(h_dim, dim, g_size);
    copy_htod_q4(h_layer->w1, d_layer->w1, g_size);
    d_layer->w2 = new_matrix_q8(dim, h_dim, g_size);
    copy_htod_q4(h_layer->w2, d_layer->w2, g_size);
    d_layer->w3 = new_matrix_q8(dim, h_dim, g_size);
    copy_htod_q4(h_layer->w3, d_layer->w3, g_size);
  }
}

void device_runtime_init_q4(QLLamaRuntime *rt, QLLama *m) {
  int dim = m->params.dim;
  int h_dim = m->params.hidden_dim;
  int g_size = m->params.group_size;
  int n_layers = m->params.n_layers;

  rt->q = new_matrix_device(1, dim);
  rt->f_x = new_matrix_device(1, dim);
  rt->q_x = new_matrix_q4_device(1, dim, g_size);
  rt->f_x_buf = new_matrix_device(1, dim);
  rt->q_x_buf = new_matrix_q4_device(1, dim, g_size);
  rt->f_x_buf2 = new_matrix_device(1, dim);
  rt->q_h_buf = new_matrix_q4_device(1, h_dim, g_size);
  rt->f_h_buf = new_matrix_device(1, h_dim);
  rt->f_h_buf2 = new_matrix_device(1, dim);
  rt->logits = new_matrix_device(1, m->params.vocab_size);
  rt->attention = new_matrix_device(m->params.n_heads, m->params.max_seq_len);
  for (int i = 0; i < n_layers; i++) {
    rt->kcaches[i] = new_matrix_device(m->params.max_seq_len, dim);
    rt->vcaches[i] = new_matrix_device(m->params.max_seq_len, dim);
  }
  rt->m = *m;
}


void d_print_mat(Matrix m, int n ) {
  f32 *data = (f32 *)malloc(n * sizeof(f32));
  cudaMemcpy(data, m.data, n * sizeof(f32), cudaMemcpyDeviceToHost);
  print_vec(data, 20, 0);
  free(data);
}

#define unpack_left(in) ((in >> 4) & 0x0f) - 8
#define unpack_right(in) (in & 0x0f) - 8

void print_q4(QMatrix m, int n, int g_size) {
  int n_groups = m.shape[0] / g_size;
  for (int i = 0; i < std::min(n, n_groups); i++) {
    u64 start_idx = i * g_size / 2;
    u64 end_idx = (i + 1) * (g_size / 2);
    for (u64 j = start_idx; j < end_idx; j++) {
        u8 p = (u8)m.data[j];
        i32 left = unpack_left(p);
        i32 right = unpack_right(p);
        printf("%d %d", left, right);
    }
  }
  printf("\n");
}


f32* device_forward_q4(QLLamaRuntime *d_rt, QLLamaRuntime* h_rt, i32 tok, i32 pos) {
  QLLama *m = &d_rt->m;
  int dim = m->params.dim;
  int h_dim = m->params.hidden_dim;
  int g_size = m->params.group_size;
  int n_layers = m->params.n_layers;
  int vocab_size = m->params.vocab_size;
  int max_seq_len = m->params.max_seq_len;
  int n_heads = m->params.n_heads;

  QMatrix d_embedding = get_row_q4(d_rt->m.tok_embeddings, tok, g_size);
  QMatrix h_embedding = get_row_q4(h_rt->m.tok_embeddings, tok, g_size);
  print_q4(h_embedding, 1, g_size);

  d_dequantize_q4(d_rt->f_x.data, d_embedding.data, d_embedding.scales, dim, g_size);
  d_print_mat(d_rt->f_x, 10);
  dequantize_q4(h_rt->f_x.data, h_embedding.data, h_embedding.scales, dim,
                g_size);
  print_vec(h_rt->f_x.data, 10, 0);



  return nullptr;
}

int main() {
  // print_device_info();
  MappedFile in;
  init(&in, "../../bin/llama_q4.bin");
  QLLama h_m;
  qllama_init(&h_m, in.data, in.len);
  QLLama d_m;
  qllama_init_device(&d_m, &h_m);
  QLLamaRuntime d_rt;
  device_runtime_init_q4(&d_rt, &d_m);
  QLLamaRuntime h_rt;
  runtime_init_q4(&h_rt, &h_m);
  device_forward_q4(&d_rt, &h_rt, 6492, 0);
  // QMatrix m = new_matrix_q4_device(1, 1024, 32);
  // Matrix m2 = new_matrix_device(1, 1024);
  // d_dequantize_q4(m2.data, m.data, m.scales, 1024, 32);

  return 0;
}
