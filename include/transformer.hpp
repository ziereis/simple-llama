#pragma once
#include "tz-utils.hpp"
#include "matview.hpp"
#include <cinttypes>
#include <vector>

struct Params {
  i32 dim;
  i32 hidden_dim;
  i32 n_heads;
  i32 n_layers;
  i32 vocab_size;
  i32 max_seq_len;
} __attribute__((__packed__));

struct QuantizeParams {
  i32 group_size;
  i32 bitwidth;
}__attribute__((__packed__));

struct EncoderBlock {
  FMatrix attention_norm;
  FMatrix wq;
  FMatrix wk;
  FMatrix wv;
  FMatrix wo;
  FMatrix ffn_norm;
  FMatrix w1;
  FMatrix w2;
  FMatrix w3;

  void set_params(Params &params) {
    attention_norm.shape = {1, params.dim};
    wq.shape = {params.dim, params.dim};
    wk.shape = {params.dim, params.dim};
    wv.shape = {params.dim, params.dim};
    wo.shape = {params.dim, params.dim};
    ffn_norm.shape = {1, params.dim};
    w1.shape = {params.hidden_dim, params.dim};
    w2.shape = {params.dim, params.hidden_dim};
    w3.shape = {params.hidden_dim, params.dim};
  }
};

struct QEncoderBlock {
  FMatrix attention_norm;
  QMatrix wq;
  QMatrix wk;
  QMatrix wv;
  QMatrix wo;
  FMatrix ffn_norm;
  QMatrix w1;
  QMatrix w2;
  QMatrix w3;

  void set_params(Params &params, QuantizeParams &qparams) {
    attention_norm.shape = {1, params.dim};
    wq.shape = {params.dim, params.dim};
    wq.group_size = qparams.group_size;
    wk.shape = {params.dim, params.dim};
    wk.group_size = qparams.group_size;
    wv.shape = {params.dim, params.dim};
    wv.group_size = qparams.group_size;
    wo.shape = {params.dim, params.dim};
    wo.group_size = qparams.group_size;
    ffn_norm.shape = {1, params.dim};
    w1.shape = {params.hidden_dim, params.dim};
    w1.group_size = qparams.group_size;
    w2.shape = {params.dim, params.hidden_dim};
    w2.group_size = qparams.group_size;
    w3.shape = {params.hidden_dim, params.dim};
    w3.group_size = qparams.group_size;
  }
};



#define LLAMA_VALIDATE(COND, MSG)                                              \
  do {                                                                         \
    if (!(COND)) {                                                             \
      throw std::runtime_error(MSG);                                           \
    }                                                                          \
  } while (0)


class LLama {

public:
  void init(std::span<const u8> data) {
    tz::BinaryReader reader(data.data(), data.size());
    LLAMA_VALIDATE(reader.read<u32>() == 0x7fdd7f7f, "Invalid magic");
    LLAMA_VALIDATE(reader.read<i32>() == 1, "Invalid version");
    params = reader.read<Params>();
    // read pad
    reader.seek(256);
    for (auto &layer : layers) {
      layer.set_params(params);
    }
    for (auto &layer : layers) {
      layer.attention_norm.init(reader);
    }
    for (auto &layer : layers) {
      layer.ffn_norm.init(reader);
    }
    norm.shape = {1, params.dim};
    tok_embeddings.shape = {params.vocab_size, params.dim};
    norm.init(reader);
    tok_embeddings.init(reader);
    for (auto &layer : layers) {
      layer.wq.init(reader);
    }
    for (auto &layer : layers) {
      layer.wk.init(reader);
    }
    for (auto &layer : layers) {
      layer.wv.init(reader);
    }
    for (auto &layer : layers) {
      layer.wo.init(reader);
    }
    for (auto &layer : layers) {
      layer.w1.init(reader);
    }
    for (auto &layer : layers) {
      layer.w2.init(reader);
    }
    for (auto &layer : layers) {
      layer.w3.init(reader);
    }
    output_weights.shape = {params.vocab_size, params.dim};
    output_weights.init(reader);
    assert(reader.hasMore() == false);
  }

  public:
  Params params;
  FMatrix tok_embeddings;
  FMatrix norm;
  FMatrix output_weights;
  std::array<EncoderBlock, 32> layers;
};


void init_layer_q8(tz::BinaryReader& reader, std::span<QMatrix*> mats, i32 x_dim, i32 y_dim, i32 group_size) {
  auto w_sz = x_dim * y_dim * sizeof(i8) * mats.size();
  auto s_sz = (x_dim * y_dim / group_size) * sizeof(f32) * mats.size();
  auto bytes = reader.readChunk(w_sz + s_sz);
  i8 *w = (i8 *)bytes.data();
  f32 *s = (f32 *)(bytes.data() + w_sz);
  for (int i = 0; i < mats.size(); i++) {
    assert(mats[i]);
    mats[i]->data = w + i * (x_dim * y_dim);
    mats[i]->scales = s + i * (x_dim * y_dim / group_size);
    mats[i]->shape = {x_dim, y_dim};
    mats[i]->group_size = group_size;
  }
}


class QLLama {
  public:
  void init(std::span<const u8> data) {
    tz::BinaryReader reader(data.data(), data.size());
    LLAMA_VALIDATE(reader.read<u32>() == 0x7fdd7f7f, "Invalid magic");
    LLAMA_VALIDATE(reader.read<i32>() == 2, "Invalid version");
    params = reader.read<Params>();
    qparams = reader.read<QuantizeParams>();
    for (auto &layer : layers) {
      layer.set_params(params, qparams);
    }
    // read pad
    reader.seek(256);
    // first all norms as f32
    norm.shape = {1, params.dim};
    norm.init(reader);
    for (auto &layer : layers) {
      layer.attention_norm.init(reader);
    }
    for (auto &layer : layers) {
      layer.ffn_norm.init(reader);
    }
    // all quantized weights
    tok_embeddings.shape = {params.vocab_size, params.dim};
    tok_embeddings.group_size = qparams.group_size;
    tok_embeddings.init(reader);
    output_weights.shape = {params.vocab_size, params.dim};
    output_weights.group_size = qparams.group_size;
    output_weights.init(reader);

    std::vector<QMatrix*> wqs_init_list(layers.size());
    std::vector<QMatrix *> wks_init_list(layers.size());
    std::vector<QMatrix *> wvs_init_list(layers.size());
    std::vector<QMatrix *> wos_init_list(layers.size());
    std::vector<QMatrix *> w1s_init_list(layers.size());
    std::vector<QMatrix *> w2s_init_list(layers.size());
    std::vector<QMatrix *> w3s_init_list(layers.size());
    for (u32 i = 0; i < layers.size(); i++) {
      wqs_init_list[i] = &layers[i].wq;
      wks_init_list[i] = &layers[i].wk;
      wvs_init_list[i] = &layers[i].wv;
      wos_init_list[i] = &layers[i].wo;
      w1s_init_list[i] = &layers[i].w1;
      w2s_init_list[i] = &layers[i].w2;
      w3s_init_list[i] = &layers[i].w3;
    }
    init_layer_q8(reader, wqs_init_list, params.dim, params.dim,
                  qparams.group_size);
    init_layer_q8(reader, wks_init_list, params.dim, params.dim,
                  qparams.group_size);
    init_layer_q8(reader, wvs_init_list, params.dim, params.dim,
                  qparams.group_size);
    init_layer_q8(reader, wos_init_list, params.dim, params.dim,
                  qparams.group_size);
    init_layer_q8(reader, w1s_init_list, params.hidden_dim, params.dim,
                  qparams.group_size);
    init_layer_q8(reader, w2s_init_list, params.dim, params.hidden_dim,
                  qparams.group_size);
    init_layer_q8(reader, w3s_init_list, params.hidden_dim, params.dim,
                  qparams.group_size);
    assert(reader.hasMore() == false);
  }
public:
  Params params;
  QuantizeParams qparams;
  FMatrix norm;
  QMatrix tok_embeddings;
  QMatrix output_weights;
  std::array<QEncoderBlock, 32> layers;
};
