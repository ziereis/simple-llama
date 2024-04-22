#include "llama.h"
#include "assert.h"
#include "math.h"
#include "ops.h"
#include "stdio.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>

#include "llama.h"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>

bool qllama_init_device(QLLama *device, QLLama *host) {
  device->params = host->params;
}


void init_mat(Matrix *m, i32 rows, i32 cols, u8 *data, u64 pos) {
  m->shape[0] = rows;
  m->shape[1] = cols;
  m->data = (f32 *)(data + pos);
}

void init_qmat(QMatrix *m, i32 rows, i32 cols, u8 *data, u64 pos,
               u64 scales_pos) {
  m->shape[0] = rows;
  m->shape[1] = cols;
  m->data = (i8 *)(data + pos);
  m->scales = (f32 *)(data + scales_pos);
}

bool llama_init(LLama *m, u8 *data, u64 size) {
  m->file = (MappedFile){.data = data, .len = size};
  u64 pos = 0;
  if (*(u32 *)(data + pos) != 0x7fdd7f7f) {
    printf("Invalid magic\n");
    return false;
  }
  pos += sizeof(u32);
  u32 version = *(u32 *)(data + pos);
  if (version != 3) {
    printf("Invalid version %i \n", *(u32 *)(data + pos));
    return false;
  }
  pos += sizeof(i32);
  m->params = *(Params *)(data + pos);
  pos = 256;
  int dim = m->params.dim;
  int h_dim = m->params.hidden_dim;
  int n_layers = m->params.n_layers;
  printf("dim: %i\n", dim);
  printf("h_dim: %i\n", h_dim);
  printf("n_layers: %i\n", n_layers);

  u64 offset_norm = pos;
  u64 offset_tok_emb = offset_norm + SIZE_NORM_F32;
  u64 offset_output_weight = offset_tok_emb + SIZE_TOK_EMB_F32;
  u64 offset_att_norm = offset_output_weight + SIZE_OUTPUT_WEIGHT_F32;
  u64 offset_ffn_norm = offset_att_norm + SIZE_NORM_LAYER_F32;
  u64 offset_wq = offset_ffn_norm + SIZE_NORM_LAYER_F32;
  u64 offset_wk = offset_wq + SIZE_WEIGHTS_LAYER_F32;
  u64 offset_wv = offset_wk + SIZE_WEIGHTS_LAYER_F32;
  u64 offset_wo = offset_wv + SIZE_WEIGHTS_LAYER_F32;
  u64 offset_w1 = offset_wo + SIZE_WEIGHTS_LAYER_F32;
  u64 offset_w2 = offset_w1 + SIZE_WEIGHTS_H_LAYER_F32;
  u64 offset_w3 = offset_w2 + SIZE_WEIGHTS_H_LAYER_F32;
  u64 offset_end = offset_w3 + SIZE_WEIGHTS_H_LAYER_F32;

  assert(offset_end == size);

  init_mat(&m->norm, 1, dim, data, offset_norm);
  init_mat(&m->tok_embeddings, m->params.vocab_size, dim, data, offset_tok_emb);
  init_mat(&m->output_weights, m->params.vocab_size, dim, data,
           offset_output_weight);
  for (int i = 0; i < n_layers; i++) {
    EncoderBlock *layer = &m->layers[i];
    init_mat(&layer->attention_norm, 1, dim, data,
             offset_att_norm + i * SIZE_NORM_F32);
    init_mat(&layer->ffn_norm, 1, dim, data,
             offset_ffn_norm + i * SIZE_NORM_F32);
    init_mat(&layer->wq, dim, dim, data, offset_wq + i * SIZE_WEIGHTS_F32);
    init_mat(&layer->wk, dim, dim, data, offset_wk + i * SIZE_WEIGHTS_F32);
    init_mat(&layer->wv, dim, dim, data, offset_wv + i * SIZE_WEIGHTS_F32);
    init_mat(&layer->wo, dim, dim, data, offset_wo + i * SIZE_WEIGHTS_F32);
    init_mat(&layer->w1, h_dim, dim, data, offset_w1 + i * SIZE_WEIGHTS_H_F32);
    init_mat(&layer->w2, dim, h_dim, data, offset_w2 + i * SIZE_WEIGHTS_H_F32);
    init_mat(&layer->w3, h_dim, dim, data, offset_w3 + i * SIZE_WEIGHTS_H_F32);
  }
  return 1;
}

bool qllama_init(QLLama *m, u8 *data, u64 size) {
  m->file = (MappedFile){.data = data, .len = size};
  u64 pos = 0;
  if (*(u32 *)(data + pos) != 0x7fdd7f7f) {
    printf("Invalid magic\n");
    return false;
  }
  pos += sizeof(u32);
  u32 version = *(u32 *)(data + pos);
  if (version != 4) {
    printf("Invalid version %i \n", *(u32 *)(data + pos));
    return false;
  }
  pos += sizeof(i32);
  m->params = *(Params *)(data + pos);
  pos = 256;
  int dim = m->params.dim;
  int h_dim = m->params.hidden_dim;
  int n_layers = m->params.n_layers;
  printf("dim: %i\n", dim);
  printf("h_dim: %i\n", h_dim);
  printf("n_layers: %i\n", n_layers);

  int group_size = m->params.group_size;
  printf("bitwidth: %i\n", m->params.bitwidth);
  printf("group_size: %i\n", group_size);

  if (m->params.bitwidth == 8) {
    printf("loading q8\n");
    u64 offset_norm = pos;
    u64 offset_att_norm = offset_norm + SIZE_NORM_F32;
    u64 offset_ffn_norm = offset_att_norm + SIZE_NORM_LAYER_F32;

    u64 offset_tok_emb = offset_ffn_norm + SIZE_NORM_LAYER_F32;
    u64 offset_tok_emb_scales = offset_tok_emb + SIZE_TOK_EMB_Q8;
    u64 offset_output_weight =
        offset_tok_emb_scales + COUNT_TOK_EMB / group_size * 4;
    u64 offset_output_weight_scales =
        offset_output_weight + SIZE_OUTPUT_WEIGHT_Q8;
    u64 offset_wq =
        offset_output_weight_scales + COUNT_OUTPUT_WEIGHT / group_size * 4;
    u64 offset_wq_scales = offset_wq + SIZE_WEIGHTS_LAYER_Q8;
    u64 offset_wk = offset_wq_scales + COUNT_WEIGHTS_LAYER / group_size * 4;
    u64 offset_wk_scales = offset_wk + SIZE_WEIGHTS_LAYER_Q8;
    u64 offset_wv = offset_wk_scales + COUNT_WEIGHTS_LAYER / group_size * 4;
    u64 offset_wv_scales = offset_wv + SIZE_WEIGHTS_LAYER_Q8;
    u64 offset_wo = offset_wv_scales + COUNT_WEIGHTS_LAYER / group_size * 4;
    u64 offset_wo_scales = offset_wo + SIZE_WEIGHTS_LAYER_Q8;
    u64 offset_w1 = offset_wo_scales + COUNT_WEIGHTS_LAYER / group_size * 4;
    u64 offset_w1_scales = offset_w1 + SIZE_WEIGHTS_H_LAYER_Q8;
    u64 offset_w2 = offset_w1_scales + COUNT_WEIGHTS_H_LAYER / group_size * 4;
    u64 offset_w2_scales = offset_w2 + SIZE_WEIGHTS_H_LAYER_Q8;
    u64 offset_w3 = offset_w2_scales + COUNT_WEIGHTS_H_LAYER / group_size * 4;
    u64 offset_w3_scales = offset_w3 + SIZE_WEIGHTS_H_LAYER_Q8;
    u64 offset_end = offset_w3_scales + COUNT_WEIGHTS_H_LAYER / group_size * 4;

    printf("size: %lu\n", size);
    printf("end: %lu\n", offset_end);
    assert(offset_end == size);

    init_mat(&m->norm, 1, dim, data, offset_norm);
    init_qmat(&m->tok_embeddings, m->params.vocab_size, dim, data,
              offset_tok_emb, offset_tok_emb_scales);
    init_qmat(&m->output_weights, m->params.vocab_size, dim, data,
              offset_output_weight, offset_output_weight_scales);
    for (int i = 0; i < n_layers; i++) {
      QEncoderBlock *layer = &m->layers[i];
      init_mat(&layer->attention_norm, 1, dim, data,
               offset_att_norm + i * SIZE_NORM_F32);
      init_mat(&layer->ffn_norm, 1, dim, data,
               offset_ffn_norm + i * SIZE_NORM_F32);

      init_qmat(&layer->wq, dim, dim, data, offset_wq + i * SIZE_WEIGHTS_Q8,
                offset_wq_scales + i * COUNT_WEIGHTS / group_size * 4);
      init_qmat(&layer->wk, dim, dim, data, offset_wk + i * SIZE_WEIGHTS_Q8,
                offset_wk_scales + i * COUNT_WEIGHTS / group_size * 4);
      init_qmat(&layer->wv, dim, dim, data, offset_wv + i * SIZE_WEIGHTS_Q8,
                offset_wv_scales + i * COUNT_WEIGHTS / group_size * 4);
      init_qmat(&layer->wo, dim, dim, data, offset_wo + i * SIZE_WEIGHTS_Q8,
                offset_wo_scales + i * COUNT_WEIGHTS / group_size * 4);
      init_qmat(&layer->w1, h_dim, dim, data, offset_w1 + i * SIZE_WEIGHTS_H_Q8,
                offset_w1_scales + i * COUNT_WEIGHTS_H / group_size * 4);
      init_qmat(&layer->w2, dim, h_dim, data, offset_w2 + i * SIZE_WEIGHTS_H_Q8,
                offset_w2_scales + i * COUNT_WEIGHTS_H / group_size * 4);
      init_qmat(&layer->w3, h_dim, dim, data, offset_w3 + i * SIZE_WEIGHTS_H_Q8,
                offset_w3_scales + i * COUNT_WEIGHTS_H / group_size * 4);
    }
  } else {
    assert(m->params.bitwidth == 4);
    printf("loading q4\n");
    u64 offset_norm = pos;
    u64 offset_att_norm = offset_norm + SIZE_NORM_F32;
    u64 offset_ffn_norm = offset_att_norm + SIZE_NORM_LAYER_F32;

    u64 offset_tok_emb = offset_ffn_norm + SIZE_NORM_LAYER_F32;
    u64 offset_tok_emb_scales = offset_tok_emb + SIZE_TOK_EMB_Q4;
    u64 offset_output_weight =
        offset_tok_emb_scales + COUNT_TOK_EMB / group_size * 4;
    u64 offset_output_weight_scales =
        offset_output_weight + SIZE_OUTPUT_WEIGHT_Q4;
    u64 offset_wq =
        offset_output_weight_scales + COUNT_OUTPUT_WEIGHT / group_size * 4;
    u64 offset_wq_scales = offset_wq + SIZE_WEIGHTS_LAYER_Q4;
    u64 offset_wk = offset_wq_scales + COUNT_WEIGHTS_LAYER / group_size * 4;
    u64 offset_wk_scales = offset_wk + SIZE_WEIGHTS_LAYER_Q4;
    u64 offset_wv = offset_wk_scales + COUNT_WEIGHTS_LAYER / group_size * 4;
    u64 offset_wv_scales = offset_wv + SIZE_WEIGHTS_LAYER_Q4;
    u64 offset_wo = offset_wv_scales + COUNT_WEIGHTS_LAYER / group_size * 4;
    u64 offset_wo_scales = offset_wo + SIZE_WEIGHTS_LAYER_Q4;
    u64 offset_w1 = offset_wo_scales + COUNT_WEIGHTS_LAYER / group_size * 4;
    u64 offset_w1_scales = offset_w1 + SIZE_WEIGHTS_H_LAYER_Q4;
    u64 offset_w2 = offset_w1_scales + COUNT_WEIGHTS_H_LAYER / group_size * 4;
    u64 offset_w2_scales = offset_w2 + SIZE_WEIGHTS_H_LAYER_Q4;
    u64 offset_w3 = offset_w2_scales + COUNT_WEIGHTS_H_LAYER / group_size * 4;
    u64 offset_w3_scales = offset_w3 + SIZE_WEIGHTS_H_LAYER_Q4;
    u64 offset_end = offset_w3_scales + COUNT_WEIGHTS_H_LAYER / group_size * 4;

    printf("size: %lu\n", size);
    printf("end: %lu\n", offset_end);
    assert(offset_end == size);

    init_mat(&m->norm, 1, dim, data, offset_norm);
    init_qmat(&m->tok_embeddings, m->params.vocab_size, dim, data,
              offset_tok_emb, offset_tok_emb_scales);
    init_qmat(&m->output_weights, m->params.vocab_size, dim, data,
              offset_output_weight, offset_output_weight_scales);
    for (int i = 0; i < n_layers; i++) {
      QEncoderBlock *layer = &m->layers[i];
      init_mat(&layer->attention_norm, 1, dim, data,
               offset_att_norm + i * SIZE_NORM_F32);
      init_mat(&layer->ffn_norm, 1, dim, data,
               offset_ffn_norm + i * SIZE_NORM_F32);

      init_qmat(&layer->wq, dim, dim, data, offset_wq + i * SIZE_WEIGHTS_Q4,
                offset_wq_scales + i * COUNT_WEIGHTS / group_size * 4);
      init_qmat(&layer->wk, dim, dim, data, offset_wk + i * SIZE_WEIGHTS_Q4,
                offset_wk_scales + i * COUNT_WEIGHTS / group_size * 4);
      init_qmat(&layer->wv, dim, dim, data, offset_wv + i * SIZE_WEIGHTS_Q4,
                offset_wv_scales + i * COUNT_WEIGHTS / group_size * 4);
      init_qmat(&layer->wo, dim, dim, data, offset_wo + i * SIZE_WEIGHTS_Q4,
                offset_wo_scales + i * COUNT_WEIGHTS / group_size * 4);
      init_qmat(&layer->w1, h_dim, dim, data, offset_w1 + i * SIZE_WEIGHTS_H_Q4,
                offset_w1_scales + i * COUNT_WEIGHTS_H / group_size * 4);
      init_qmat(&layer->w2, dim, h_dim, data, offset_w2 + i * SIZE_WEIGHTS_H_Q4,
                offset_w2_scales + i * COUNT_WEIGHTS_H / group_size * 4);
      init_qmat(&layer->w3, h_dim, dim, data, offset_w3 + i * SIZE_WEIGHTS_H_Q4,
                offset_w3_scales + i * COUNT_WEIGHTS_H / group_size * 4);
    }
  }
  return 1;
}

void runtime_init_q8(QLLamaRuntime *rt, QLLama *m) {
  assert(m->params.bitwidth == 8);
  int dim = m->params.dim;
  int h_dim = m->params.hidden_dim;
  printf("dim: %i\n", dim);
  int g_size = m->params.group_size;
  rt->q = new_matrix(1, dim);
  rt->f_x = new_matrix(1, dim);
  rt->q_x = new_matrix_q8(1, dim, g_size);
  rt->f_x_buf = new_matrix(1, dim);
  rt->q_x_buf = new_matrix_q8(1, dim, g_size);
  rt->f_x_buf2 = new_matrix(1, dim);
  rt->q_h_buf = new_matrix_q8(1, h_dim, g_size);
  rt->f_h_buf = new_matrix(1, h_dim);
  rt->f_h_buf2 = new_matrix(1, h_dim);
  rt->logits = new_matrix(1, m->params.vocab_size);
  rt->attention = new_matrix(m->params.n_heads, m->params.max_seq_len);
  for (int i = 0; i < m->params.n_layers; i++) {
    rt->kcaches[i] = new_matrix(m->params.max_seq_len, dim);
    rt->vcaches[i] = new_matrix(m->params.max_seq_len, dim);
  }
  rt->m = *m;
}

void runtime_deinit_q8(QLLamaRuntime *rt) {
  free_matrix(rt->q);
  free_matrix(rt->f_x);
  free_matrix_q8(rt->q_x);
  free_matrix(rt->f_x_buf);
  free_matrix_q8(rt->q_x_buf);
  free_matrix(rt->f_x_buf2);
  free_matrix_q8(rt->q_h_buf);
  free_matrix(rt->f_h_buf);
  free_matrix(rt->f_h_buf2);
  free_matrix(rt->logits);
  free_matrix(rt->attention);
  for (int i = 0; i < rt->m.params.n_layers; i++) {
    free_matrix(rt->kcaches[i]);
    free_matrix(rt->vcaches[i]);
  }
  deinit(&rt->m.file);
}


void runtime_init_q4(QLLamaRuntime *rt, QLLama *m) {
  assert(m->params.bitwidth == 4);
  int dim = m->params.dim;
  int h_dim = m->params.hidden_dim;
  printf("dim: %i\n", dim);
  int g_size = m->params.group_size;
  rt->q = new_matrix(1, dim);
  rt->f_x = new_matrix(1, dim);
  rt->q_x = new_matrix_q4(1, dim, g_size);
  rt->f_x_buf = new_matrix(1, dim);
  rt->q_x_buf = new_matrix_q4(1, dim, g_size);
  rt->f_x_buf2 = new_matrix(1, dim);
  rt->q_h_buf = new_matrix_q4(1, h_dim, g_size);
  rt->f_h_buf = new_matrix(1, h_dim);
  rt->f_h_buf2 = new_matrix(1, h_dim);
  rt->logits = new_matrix(1, m->params.vocab_size);
  rt->attention = new_matrix(m->params.n_heads, m->params.max_seq_len);
  for (int i = 0; i < m->params.n_layers; i++) {
    rt->kcaches[i] = new_matrix(m->params.max_seq_len, dim);
    rt->vcaches[i] = new_matrix(m->params.max_seq_len, dim);
  }
  rt->m = *m;
}

void runtime_deinit_q4(QLLamaRuntime *rt) {
  free_matrix(rt->q);
  free_matrix(rt->f_x);
  free_matrix_q4(rt->q_x);
  free_matrix(rt->f_x_buf);
  free_matrix_q4(rt->q_x_buf);
  free_matrix(rt->f_x_buf2);
  free_matrix_q4(rt->q_h_buf);
  free_matrix(rt->f_h_buf);
  free_matrix(rt->f_h_buf2);
  free_matrix(rt->logits);
  free_matrix(rt->attention);
  for (int i = 0; i < rt->m.params.n_layers; i++) {
    free_matrix(rt->kcaches[i]);
    free_matrix(rt->vcaches[i]);
  }
  deinit(&rt->m.file);
}


void runtime_init_f32(LLamaRuntime *rt, LLama *m) {
  int dim = m->params.dim;
  int h_dim = m->params.hidden_dim;
  rt->q = new_matrix(1, dim);
  rt->x = new_matrix(1, dim);
  rt->x_buf = new_matrix(1, dim);
  rt->x_buf2 = new_matrix(1, dim);
  rt->h_buf = new_matrix(1, h_dim);
  rt->h_buf2 = new_matrix(1, h_dim);
  rt->logits = new_matrix(1, m->params.vocab_size);
  rt->attention = new_matrix(m->params.n_heads, m->params.max_seq_len);
  for (int i = 0; i < m->params.n_layers; i++) {
    rt->kcaches[i] = new_matrix(m->params.max_seq_len, dim);
    rt->vcaches[i] = new_matrix(m->params.max_seq_len, dim);
  }
  rt->m = *m;
}

void runtime_deinit_f32(LLamaRuntime *rt) {
  free_matrix(rt->q);
  free_matrix(rt->x);
  free_matrix(rt->x_buf);
  free_matrix(rt->x_buf2);
  free_matrix(rt->h_buf);
  free_matrix(rt->h_buf2);
  free_matrix(rt->logits);
  free_matrix(rt->attention);
  for (int i = 0; i < rt->m.params.n_layers; i++) {
    free_matrix(rt->kcaches[i]);
    free_matrix(rt->vcaches[i]);
  }
  deinit(&rt->m.file);
}


f32 *forward_f32(LLamaRuntime *rt, i32 tok, i32 pos) {
  int dim = rt->m.params.dim;
  int h_dim = rt->m.params.hidden_dim;
  int n_layers = rt->m.params.n_layers;
  int n_heads = rt->m.params.n_heads;
  int head_dim = dim / n_heads;

  Matrix embedding = get_row(rt->m.tok_embeddings, tok);
  memcpy(rt->x.data, embedding.data, dim * sizeof(float));

  for (i32 l_id = 0; l_id < n_layers; l_id++) {
#ifdef PRINT_DEBUG
    printf("layer: %i\n", l_id);
#endif
    EncoderBlock *layer = &rt->m.layers[l_id];
    Matrix kcache = rt->kcaches[l_id];
    Matrix vcache = rt->vcaches[l_id];

    rms_norm(rt->x_buf.data, rt->x.data, layer->attention_norm.data, dim);

#ifdef PRINT_DEBUG
    printf("attention norm:\n");
    print_vec(rt->x_buf.data, 10, 0);
    printf("----------------\n");
#endif

    Matrix k = get_row(kcache, pos);
    Matrix v = get_row(vcache, pos);

    matvec_mul(layer->wq.data, rt->x_buf.data, rt->q.data, dim, dim);
    matvec_mul(layer->wk.data, rt->x_buf.data, k.data, dim, dim);
    matvec_mul(layer->wv.data, rt->x_buf.data, v.data, dim, dim);

#ifdef PRINT_DEBUG
    printf("q\n");
    print_vec(rt->q.data, 10, 0);
    printf("----------------\n");
    printf("k\n");
    print_vec(k.data, 10, 0);
    printf("----------------\n");
    printf("v\n");
    print_vec(v.data, 10, 0);
    printf("----------------\n");
#endif

    rotate_embeddings(rt->q.data, pos, head_dim, dim);
    rotate_embeddings(k.data, pos, head_dim, dim);

    compute_attention(rt->attention.data, rt->q.data, kcache.data, vcache.data, rt->x_buf.data, pos,
                      n_heads, head_dim, rt->m.params.max_seq_len);

#ifdef PRINT_DEBUG
    printf("attention\n");
    print_vec(rt->x_buf.data, 10, 0);
    printf("----------------\n");
#endif

    matvec_mul(layer->wo.data, rt->x_buf.data, rt->x_buf2.data, dim, dim);

    residual(rt->x.data, rt->x_buf2.data, dim);

    rms_norm(rt->x_buf.data, rt->x.data, layer->ffn_norm.data, dim);

#ifdef PRINT_DEBUG
    printf("fnn norm:\n");
    print_vec(rt->x_buf.data, 10, 0);
    printf("----------------\n");
#endif

    matvec_mul(layer->w1.data, rt->x_buf.data, rt->h_buf.data, layer->w1.shape[0],
               layer->w1.shape[1]);
    matvec_mul(layer->w3.data, rt->x_buf.data, rt->h_buf2.data, layer->w3.shape[0],
               layer->w3.shape[1]);

#ifdef PRINT_DEBUG
    printf("hidden\n");
    print_vec(rt->h_buf.data, 10, 0);
    print_vec(rt->h_buf2.data, 10, 0);
    printf("----------------\n");
#endif

    swiglu(rt->h_buf.data, rt->h_buf2.data, h_dim);

    matvec_mul(layer->w2.data, rt->h_buf.data, rt->x_buf.data, layer->w2.shape[0],
               layer->w2.shape[1]);

#ifdef PRINT_DEBUG
    printf("after w2:\n");
    print_vec(rt->x_buf.data, 10, 0);
    printf("----------------\n");
#endif

    residual(rt->x.data, rt->x_buf.data, dim);
  }

  rms_norm(rt->x.data, rt->x.data, rt->m.norm.data, dim);

  matvec_mul(rt->m.output_weights.data, rt->x.data, rt->logits.data,
             rt->m.params.vocab_size, dim);

  return rt->logits.data;
}

f32 *forward_q8(QLLamaRuntime *rt, i32 tok, i32 pos) {
  int g_size = rt->m.params.group_size;
  int dim = rt->m.params.dim;
  int h_dim = rt->m.params.hidden_dim;
  int n_layers = rt->m.params.n_layers;
  int n_heads = rt->m.params.n_heads;
  int head_dim = dim / n_heads;

  QMatrix embedding = get_row_q8(rt->m.tok_embeddings, tok, g_size);

  dequantize_q8(rt->f_x.data, embedding.data, embedding.scales, dim, g_size);

#ifdef PRINT_DEBUG
  printf("embedding\n");
  print_vec(rt->f_x.data, 10, 0);
  printf("----------------\n");
#endif

  for (i32 l_id = 0; l_id < n_layers; l_id++) {
    QEncoderBlock *layer = &rt->m.layers[l_id];
    Matrix kcache = rt->kcaches[l_id];
    Matrix vcache = rt->vcaches[l_id];

    rms_norm(rt->f_x_buf.data, rt->f_x.data, layer->attention_norm.data, dim);

#ifdef PRINT_DEBUG
    printf("attention norm:\n");
    print_vec(rt->f_x_buf.data, 10, 0);
#endif

    quantize_q8(rt->q_x_buf.data, rt->q_x_buf.scales, rt->f_x_buf.data, dim,
                g_size);

    Matrix k = get_row(kcache, pos);
    Matrix v = get_row(vcache, pos);

    matvec_mul_q8(layer->wq.data, layer->wq.scales, rt->q_x_buf.data,
                  rt->q_x_buf.scales, rt->q.data, layer->wq.shape[0],
                  layer->wq.shape[1], g_size);
    matvec_mul_q8(layer->wk.data, layer->wk.scales, rt->q_x_buf.data,
                  rt->q_x_buf.scales, k.data, layer->wk.shape[0],
                  layer->wk.shape[1], g_size);
    matvec_mul_q8(layer->wv.data, layer->wv.scales, rt->q_x_buf.data,
                  rt->q_x_buf.scales, v.data, layer->wv.shape[0],
                  layer->wv.shape[1], g_size);

#ifdef PRINT_DEBUG
    printf("q\n");
    print_vec(rt->q.data, 10, 0);
    printf("----------------\n");
    printf("k\n");
    print_vec(k.data, 10, 0);
    printf("----------------\n");
    printf("v\n");
    print_vec(v.data, 10, 0);
    printf("----------------\n");
#endif

    rotate_embeddings(rt->q.data, pos, head_dim, dim);
    rotate_embeddings(k.data, pos, head_dim, dim);

    compute_attention(rt->attention.data, rt->q.data, kcache.data, vcache.data, rt->f_x_buf.data, pos,
                      n_heads, head_dim, rt->m.params.max_seq_len);

#ifdef PRINT_DEBUG
    printf("attention\n");
    print_vec(rt->f_x_buf.data, 10, 0);
    printf("----------------\n");
#endif

    quantize_q8(rt->q_x_buf.data, rt->q_x_buf.scales, rt->f_x_buf.data, dim,
                g_size);

    matvec_mul_q8(layer->wo.data, layer->wo.scales, rt->q_x_buf.data,
                  rt->q_x_buf.scales, rt->f_x_buf2.data, layer->wo.shape[0],
                  layer->wo.shape[1], g_size);

    residual(rt->f_x.data, rt->f_x_buf2.data, dim);

    rms_norm(rt->f_x_buf.data, rt->f_x.data, layer->ffn_norm.data, dim);

#ifdef PRINT_DEBUG
    printf("fnn norm:\n");
    print_vec(rt->f_x_buf.data, 10, 0);
    printf("----------------\n");
#endif

    quantize_q8(rt->q_x_buf.data, rt->q_x_buf.scales, rt->f_x_buf.data, dim,
                g_size);

    matvec_mul_q8(layer->w1.data, layer->w1.scales, rt->q_x_buf.data,
                  rt->q_x_buf.scales, rt->f_h_buf.data, layer->w1.shape[0],
                  layer->w1.shape[1], g_size);
    matvec_mul_q8(layer->w3.data, layer->w3.scales, rt->q_x_buf.data,
                  rt->q_x_buf.scales, rt->f_h_buf2.data, layer->w3.shape[0],
                  layer->w3.shape[1], g_size);

#ifdef PRINT_DEBUG
    printf("hidden\n");
    print_vec(rt->f_h_buf.data, 10, 0);
    print_vec(rt->f_h_buf2.data, 10, 0);
    printf("----------------\n");
#endif

    swiglu(rt->f_h_buf.data, rt->f_h_buf2.data, h_dim);

    quantize_q8(rt->q_h_buf.data, rt->q_h_buf.scales, rt->f_h_buf.data, h_dim,
                g_size);

    matvec_mul_q8(layer->w2.data, layer->w2.scales, rt->q_h_buf.data,
                  rt->q_h_buf.scales, rt->f_x_buf.data, layer->w2.shape[0],
                  layer->w2.shape[1], g_size);

#ifdef PRINT_DEBUG
    printf("after w2:\n");
    print_vec(rt->f_x_buf.data, 10, 0);
    printf("----------------\n");
#endif

    residual(rt->f_x.data, rt->f_x_buf.data, dim);
  }

  rms_norm(rt->f_x.data, rt->f_x.data, rt->m.norm.data, dim);

  quantize_q8(rt->q_x.data, rt->q_x.scales, rt->f_x.data, dim, g_size);

  matvec_mul_q8(rt->m.output_weights.data, rt->m.output_weights.scales,
                rt->q_x.data, rt->q_x.scales, rt->logits.data,
                rt->m.output_weights.shape[0], rt->m.output_weights.shape[1],
                g_size);

  return rt->logits.data;
}

f32 *forward_q4(QLLamaRuntime *rt, i32 tok, i32 pos) {
  int g_size = rt->m.params.group_size;
  int dim = rt->m.params.dim;
  int h_dim = rt->m.params.hidden_dim;
  int n_layers = rt->m.params.n_layers;
  int n_heads = rt->m.params.n_heads;
  int head_dim = dim / n_heads;

  QMatrix embedding = get_row_q4(rt->m.tok_embeddings, tok, g_size);

  dequantize_q4(rt->f_x.data, embedding.data, embedding.scales, dim, g_size);

#ifdef PRINT_DEBUG
  printf("embedding\n");
  print_vec(rt->f_x.data, 10, 0);
  printf("----------------\n");
#endif

  for (i32 l_id = 0; l_id < n_layers; l_id++) {
    QEncoderBlock *layer = &rt->m.layers[l_id];
    Matrix kcache = rt->kcaches[l_id];
    Matrix vcache = rt->vcaches[l_id];

    rms_norm(rt->f_x_buf.data, rt->f_x.data, layer->attention_norm.data, dim);

#ifdef PRINT_DEBUG
    printf("attention norm:\n");
    print_vec(rt->f_x_buf.data, 10, 0);
#endif

    quantize_q4(rt->q_x_buf.data, rt->q_x_buf.scales, rt->f_x_buf.data, dim,
                g_size);

    Matrix k = get_row(kcache, pos);
    Matrix v = get_row(vcache, pos);

    matvec_mul_q4(layer->wq.data, layer->wq.scales, rt->q_x_buf.data,
                  rt->q_x_buf.scales, rt->q.data, layer->wq.shape[0],
                  layer->wq.shape[1], g_size);
    matvec_mul_q4(layer->wk.data, layer->wk.scales, rt->q_x_buf.data,
                  rt->q_x_buf.scales, k.data, layer->wk.shape[0],
                  layer->wk.shape[1], g_size);
    matvec_mul_q4(layer->wv.data, layer->wv.scales, rt->q_x_buf.data,
                  rt->q_x_buf.scales, v.data, layer->wv.shape[0],
                  layer->wv.shape[1], g_size);

#ifdef PRINT_DEBUG
    printf("q\n");
    print_vec(rt->q.data, 10, 0);
    printf("----------------\n");
    printf("k\n");
    print_vec(k.data, 10, 0);
    printf("----------------\n");
    printf("v\n");
    print_vec(v.data, 10, 0);
    printf("----------------\n");
#endif

    rotate_embeddings(rt->q.data, pos, head_dim, dim);
    rotate_embeddings(k.data, pos, head_dim, dim);

    compute_attention(rt->attention.data, rt->q.data, kcache.data, vcache.data, rt->f_x_buf.data, pos,
                      n_heads, head_dim, rt->m.params.max_seq_len);

#ifdef PRINT_DEBUG
    printf("attention\n");
    print_vec(rt->f_x_buf.data, 10, 0);
    printf("----------------\n");
#endif

    quantize_q4(rt->q_x_buf.data, rt->q_x_buf.scales, rt->f_x_buf.data, dim,
                g_size);

    matvec_mul_q4(layer->wo.data, layer->wo.scales, rt->q_x_buf.data,
                  rt->q_x_buf.scales, rt->f_x_buf2.data, layer->wo.shape[0],
                  layer->wo.shape[1], g_size);

    residual(rt->f_x.data, rt->f_x_buf2.data, dim);

    rms_norm(rt->f_x_buf.data, rt->f_x.data, layer->ffn_norm.data, dim);

#ifdef PRINT_DEBUG
    printf("fnn norm:\n");
    print_vec(rt->f_x_buf.data, 10, 0);
    printf("----------------\n");
#endif

    quantize_q4(rt->q_x_buf.data, rt->q_x_buf.scales, rt->f_x_buf.data, dim,
                g_size);

    matvec_mul_q4(layer->w1.data, layer->w1.scales, rt->q_x_buf.data,
                  rt->q_x_buf.scales, rt->f_h_buf.data, layer->w1.shape[0],
                  layer->w1.shape[1], g_size);
    matvec_mul_q4(layer->w3.data, layer->w3.scales, rt->q_x_buf.data,
                  rt->q_x_buf.scales, rt->f_h_buf2.data, layer->w3.shape[0],
                  layer->w3.shape[1], g_size);

#ifdef PRINT_DEBUG
    printf("hidden\n");
    print_vec(rt->f_h_buf.data, 10, 0);
    print_vec(rt->f_h_buf2.data, 10, 0);
    printf("----------------\n");
#endif

    swiglu(rt->f_h_buf.data, rt->f_h_buf2.data, h_dim);

    quantize_q4(rt->q_h_buf.data, rt->q_h_buf.scales, rt->f_h_buf.data, h_dim,
                g_size);

    matvec_mul_q4(layer->w2.data, layer->w2.scales, rt->q_h_buf.data,
                  rt->q_h_buf.scales, rt->f_x_buf.data, layer->w2.shape[0],
                  layer->w2.shape[1], g_size);

#ifdef PRINT_DEBUG
    printf("after w2:\n");
    print_vec(rt->f_x_buf.data, 10, 0);
    printf("----------------\n");
#endif

    residual(rt->f_x.data, rt->f_x_buf.data, dim);
  }

  rms_norm(rt->f_x.data, rt->f_x.data, rt->m.norm.data, dim);

  quantize_q4(rt->q_x.data, rt->q_x.scales, rt->f_x.data, dim, g_size);

  matvec_mul_q4(rt->m.output_weights.data, rt->m.output_weights.scales,
                rt->q_x.data, rt->q_x.scales, rt->logits.data,
                rt->m.output_weights.shape[0], rt->m.output_weights.shape[1],
                g_size);

  return rt->logits.data;
}


void *runtime_new_f32(const char *filename) {
  MappedFile file;
  init(&file, filename);
  LLama m;
  llama_init(&m, file.data, file.len);
  LLamaRuntime *rt = malloc(sizeof(LLamaRuntime));
  runtime_init_f32(rt, &m);
  return rt;
}

void runtime_delete_f32(void *handle) {
  LLamaRuntime *rt = (LLamaRuntime *)handle;
  runtime_deinit_f32(rt);
  free(rt);
}

float* runtime_forward_f32(void *handle, int tok, int pos) {
  LLamaRuntime *rt = (LLamaRuntime *)handle;
  return forward_f32(rt, tok, pos);
}

void *runtime_new_q8(const char *filename) {
  MappedFile file;
  init(&file, filename);
  QLLama m;
  qllama_init(&m, file.data, file.len);
  QLLamaRuntime *rt = malloc(sizeof(QLLamaRuntime));
  runtime_init_q8(rt, &m);
  return rt;
}

void runtime_delete_q8(void *handle) {
  QLLamaRuntime *rt = (QLLamaRuntime *)handle;
  runtime_deinit_q8(rt);
  free(rt);
}

float* runtime_forward_q8(void *handle, int tok, int pos) {
  QLLamaRuntime *rt = (QLLamaRuntime *)handle;
  return forward_q8(rt, tok, pos);
}

void *runtime_new_q4(const char *filename) {
  MappedFile file;
  init(&file, filename);
  QLLama m;
  qllama_init(&m, file.data, file.len);
  QLLamaRuntime *rt = malloc(sizeof(QLLamaRuntime));
  runtime_init_q4(rt, &m);
  return rt;
}

void runtime_delete_q4(void *handle) {
  QLLamaRuntime *rt = (QLLamaRuntime *)handle;
  runtime_deinit_q4(rt);
  free(rt);
}

float* runtime_forward_q4(void *handle, int tok, int pos) {
  QLLamaRuntime *rt = (QLLamaRuntime *)handle;
  return forward_q4(rt, tok, pos);
}
