#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include "assert.h"
#include "stdio.h"
#include "cu-ops.h"
#include "cu-llama.h"
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

    if (prop.major >= 8) {
      printf("BF16 support: Yes\n");
    } else {
      printf("BF16 support: No\n");
    }
  }
}

QMatrix new_matrix_q4_device(i32 rows, i32 cols, i32 group_size) {
  i8 *data;
  cudaMalloc(&data, rows * cols * sizeof(i8) / 2);
  f32 *scales;
  cudaMalloc(&scales, rows * cols * sizeof(f32) / group_size);
  return (QMatrix){.scales = scales, .shape = {rows, cols}, .data = data};
}

void free_matrix_q4_device(QMatrix m) {
  cudaFree(m.data);
  cudaFree(m.scales);
}

Matrix new_matrix_device(i32 rows, i32 cols) {
  f32 *data;
  cudaMalloc(&data, rows * cols * sizeof(f32));
  return (Matrix){.data = data,.shape = {rows, cols}};
}

void free_matrix_device(Matrix m) {
  cudaFree(m.data);
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
  cudaMemcpy(host.data, device.data, n / 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(host.scales, device.scales, n / g_size * sizeof(f32), cudaMemcpyDeviceToHost);
}
void copy_htod_q4(QMatrix host, QMatrix device, int g_size) {
  int n = host.shape[0] * host.shape[1];
  cudaMemcpy(device.data, host.data, n / 2, cudaMemcpyHostToDevice);
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
    d_layer->wq = new_matrix_q4_device(dim, dim, g_size);
    copy_htod_q4(h_layer->wq, d_layer->wq, g_size);
    d_layer->wk = new_matrix_q4_device(dim, dim, g_size);
    copy_htod_q4(h_layer->wk, d_layer->wk, g_size);
    d_layer->wv = new_matrix_q4_device(dim, dim, g_size);
    copy_htod_q4(h_layer->wv, d_layer->wv, g_size);
    d_layer->wo = new_matrix_q4_device(dim, dim, g_size);
    copy_htod_q4(h_layer->wo, d_layer->wo, g_size);
    d_layer->w1 = new_matrix_q4_device(h_dim, dim, g_size);
    copy_htod_q4(h_layer->w1, d_layer->w1, g_size);
    d_layer->w2 = new_matrix_q4_device(dim, h_dim, g_size);
    copy_htod_q4(h_layer->w2, d_layer->w2, g_size);
    d_layer->w3 = new_matrix_q4_device(h_dim, dim, g_size);
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
  rt->f_h_buf2 = new_matrix_device(1, h_dim);
  rt->logits = new_matrix_device(1, m->params.vocab_size);
  rt->logits_out = new_matrix(1, m->params.vocab_size);
  rt->attention = new_matrix_device(m->params.n_heads, m->params.max_seq_len);
  for (int i = 0; i < n_layers; i++) {
    rt->kcaches[i] = new_matrix_device(m->params.max_seq_len, dim);
    rt->vcaches[i] = new_matrix_device(m->params.max_seq_len, dim);
  }
  rt->m = *m;
}

void device_runtime_deinit_q4(QLLamaRuntime *rt) {
  QLLama *m = &rt->m;
  free_matrix_device(m->norm);
  free_matrix_q4_device(m->tok_embeddings);
  free_matrix_q4_device(m->output_weights);
  for (int i = 0; i < m->params.n_layers; i++) {
    QEncoderBlock *layer = &m->layers[i];
    free_matrix_device(layer->ffn_norm);
    free_matrix_device(layer->attention_norm);
    free_matrix_q4_device(layer->wq);
    free_matrix_q4_device(layer->wk);
    free_matrix_q4_device(layer->wv);
    free_matrix_q4_device(layer->wo);
    free_matrix_q4_device(layer->w1);
    free_matrix_q4_device(layer->w2);
    free_matrix_q4_device(layer->w3);
  }
  free_matrix_device(rt->q);
  free_matrix_device(rt->f_x);
  free_matrix_q4_device(rt->q_x);
  free_matrix_device(rt->f_x_buf);
  free_matrix_q4_device(rt->q_x_buf);
  free_matrix_device(rt->f_x_buf2);
  free_matrix_q4_device(rt->q_h_buf);
  free_matrix_device(rt->f_h_buf);
  free_matrix_device(rt->f_h_buf2);
  free_matrix_device(rt->logits);
  free_matrix_device(rt->attention);
  for (int i = 0; i < rt->m.params.n_layers; i++) {
    free_matrix_device(rt->kcaches[i]);
    free_matrix_device(rt->vcaches[i]);
  }
  deinit(&rt->m.file);
}



void d_print_mat(Matrix m, int n ) {
  f32 *data = (f32 *)malloc(n * sizeof(f32));
  cudaMemcpy(data, m.data, n * sizeof(f32), cudaMemcpyDeviceToHost);
  print_vec(data, n, 0);
  free(data);
}

void d_print_vec(float *data, int n) {
  float *h_data = (float *)malloc(n * sizeof(float));
  cudaMemcpy(h_data, data, n * sizeof(float), cudaMemcpyDeviceToHost);
  print_vec(h_data, n, 0);
  free(h_data);
}

#define unpack_left(in) ((in >> 4) & 0x0f) - 8
#define unpack_right(in) (in & 0x0f) - 8

void print_q4(QMatrix m, int n, int g_size) {
  for (int i = 0; i < 1; i++) {
    u64 start_idx = i * g_size / 2;
    u64 end_idx = (i + 1) * std::min(g_size / 2, n);
    for (u64 j = start_idx; j < end_idx; j++) {
        u8 p = (u8)m.data[j];
        i32 left = unpack_left(p);
        i32 right = unpack_right(p);
        printf("%d %d", left, right);
    }
  }
  printf("\n");
}

void d_print_q4(QMatrix m, int n, int g_size) {
  QMatrix h_m = new_matrix_q4(m.shape[0], m.shape[1], g_size);
  copy_dtoh_q4(m, h_m, g_size);
  print_q4(h_m, n, g_size);
  free_matrix_q4(h_m);
}


void assert_approx(Matrix d_m, Matrix h_m) {
  assert(d_m.shape[0] == h_m.shape[0]);
  assert(d_m.shape[1] == h_m.shape[1]);
  int n = h_m.shape[0] * h_m.shape[1];
  float *h_data = (float *)malloc(n * sizeof(float));
  cudaDeviceSynchronize();
  cudaMemcpy(h_data, d_m.data, n * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) {
    if ( fabs(h_data[i] - h_m.data[i]) > 1e-2) {
      printf("Mismatch at %d: %f %f\n", i, h_data[i], h_m.data[i]);
    }
  }
}

f32* device_forward_q4(QLLamaRuntime *d_rt, i32 tok, i32 pos) {
  QLLama *m = &d_rt->m;
  int dim = m->params.dim;
  int h_dim = m->params.hidden_dim;
  int g_size = m->params.group_size;
  int n_layers = m->params.n_layers;
  int n_heads = m->params.n_heads;
  int head_dim = dim / n_heads;

  QMatrix d_embedding = get_row_q4(d_rt->m.tok_embeddings, tok, g_size);
  cudaDeviceSynchronize();
  d_dequantize_q4(d_rt->f_x.data, d_embedding.data, d_embedding.scales, dim, g_size);

  for (i32 l_id = 0; l_id < n_layers; l_id++) {
    QEncoderBlock *d_layer = &d_rt->m.layers[l_id];
    Matrix d_kcache = d_rt->kcaches[l_id];
    Matrix d_vcache = d_rt->vcaches[l_id];


    cudaDeviceSynchronize();
    d_rms_norm(d_rt->f_x_buf.data, d_rt->f_x.data, d_layer->attention_norm.data,
               dim);

    cudaDeviceSynchronize();
    d_quantize_q4(d_rt->q_x_buf.data, d_rt->q_x_buf.scales, d_rt->f_x_buf.data,
                  dim, g_size);
    // print_q4(h_rt->q_x_buf, 50, g_size);
    //print_vec(h_rt->q_x_buf.scales, 16, 0);
   //d_print_q4(d_rt->q_x_buf, , g_size);
    //d_print_vec(d_rt->q_x_buf.scales, 16);


    Matrix d_k = get_row(d_kcache, pos);
    Matrix d_v = get_row(d_vcache, pos);

    cudaDeviceSynchronize();
    d_matvec_mul_q4(d_layer->wq.data, d_layer->wq.scales, d_rt->q_x_buf.data,
                    d_rt->q_x_buf.scales, d_rt->q.data, d_layer->wq.shape[0],
                    d_layer->wq.shape[1], g_size);
    d_matvec_mul_q4(d_layer->wk.data, d_layer->wk.scales, d_rt->q_x_buf.data,
                  d_rt->q_x_buf.scales, d_k.data, d_layer->wk.shape[0],
                  d_layer->wk.shape[1], g_size);
    d_matvec_mul_q4(d_layer->wv.data, d_layer->wv.scales, d_rt->q_x_buf.data,
                  d_rt->q_x_buf.scales, d_v.data, d_layer->wv.shape[0],
                  d_layer->wv.shape[1], g_size);

    cudaDeviceSynchronize();
    d_rotate_embeddings(d_rt->q.data, pos, head_dim, dim);
    d_rotate_embeddings(d_k.data, pos, head_dim, dim);

    cudaDeviceSynchronize();
    d_compute_attention(d_rt->attention.data, d_rt->q.data, d_kcache.data,
                        d_vcache.data, d_rt->f_x_buf.data, pos, n_heads,
                        head_dim, d_rt->m.params.max_seq_len);


    cudaDeviceSynchronize();
    d_quantize_q4(d_rt->q_x_buf.data, d_rt->q_x_buf.scales, d_rt->f_x_buf.data,
                  dim, g_size);

    cudaDeviceSynchronize();
    d_matvec_mul_q4(d_layer->wo.data, d_layer->wo.scales, d_rt->q_x_buf.data,
                    d_rt->q_x_buf.scales, d_rt->f_x_buf2.data,
                    d_layer->wo.shape[0], d_layer->wo.shape[1], g_size);

    cudaDeviceSynchronize();
    d_residual(d_rt->f_x.data, d_rt->f_x_buf2.data, dim);

    cudaDeviceSynchronize();
    d_rms_norm(d_rt->f_x_buf.data, d_rt->f_x.data, d_layer->ffn_norm.data, dim);


    cudaDeviceSynchronize();
    d_quantize_q4(d_rt->q_x_buf.data, d_rt->q_x_buf.scales, d_rt->f_x_buf.data, dim,
                g_size);

    cudaDeviceSynchronize();
    d_matvec_mul_q4(d_layer->w1.data, d_layer->w1.scales, d_rt->q_x_buf.data,
                    d_rt->q_x_buf.scales, d_rt->f_h_buf.data,
                    d_layer->w1.shape[0], d_layer->w1.shape[1], g_size);
    d_matvec_mul_q4(d_layer->w3.data, d_layer->w3.scales, d_rt->q_x_buf.data,
                    d_rt->q_x_buf.scales, d_rt->f_h_buf2.data,
                    d_layer->w3.shape[0], d_layer->w3.shape[1], g_size);

    cudaDeviceSynchronize();
    d_swiglu(d_rt->f_h_buf.data, d_rt->f_h_buf2.data, h_dim);


    cudaDeviceSynchronize();
    d_quantize_q4(d_rt->q_h_buf.data, d_rt->q_h_buf.scales, d_rt->f_h_buf.data, h_dim,
            g_size);

    cudaDeviceSynchronize();
    d_matvec_mul_q4(d_layer->w2.data, d_layer->w2.scales, d_rt->q_h_buf.data,
                    d_rt->q_h_buf.scales, d_rt->f_x_buf.data,
                    d_layer->w2.shape[0], d_layer->w2.shape[1], g_size);

    cudaDeviceSynchronize();
    d_residual(d_rt->f_x.data, d_rt->f_x_buf.data, dim);
  }

  cudaDeviceSynchronize();
  d_rms_norm(d_rt->f_x.data, d_rt->f_x.data, d_rt->m.norm.data, dim);

  cudaDeviceSynchronize();
  d_quantize_q4(d_rt->q_x.data, d_rt->q_x.scales, d_rt->f_x.data, dim, g_size);

  cudaDeviceSynchronize();
  d_matvec_mul_q4(d_rt->m.output_weights.data, d_rt->m.output_weights.scales,
                d_rt->q_x.data, d_rt->q_x.scales, d_rt->logits.data,
                d_rt->m.output_weights.shape[0], d_rt->m.output_weights.shape[1],
                g_size);
  cudaDeviceSynchronize();
  copy_dtoh(d_rt->logits, d_rt->logits_out);
  return d_rt->logits_out.data;
}


void *device_runtime_new_q4(const char *filename) {
  MappedFile file;
  init(&file, filename);
  QLLama m;
  qllama_init(&m, file.data, file.len);
  QLLama m_d;
  qllama_init_device(&m_d, &m);
  QLLamaRuntime *rt = (QLLamaRuntime *)malloc(sizeof(QLLamaRuntime));
  device_runtime_init_q4(rt, &m_d);
  return rt;
}

void device_runtime_delete_q4(void *handle) {
  QLLamaRuntime *rt = (QLLamaRuntime *)handle;
  cudaDeviceSynchronize();
  device_runtime_deinit_q4(rt);
  cudaDeviceSynchronize();
  free(rt);
}

float* device_runtime_forward_q4(void *handle, int tok, int pos) {
  QLLamaRuntime *rt = (QLLamaRuntime *)handle;
  return device_forward_q4(rt, tok, pos);
}
