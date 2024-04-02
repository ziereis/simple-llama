#pragma once
#include "tz-utils.hpp"
#include "matview.hpp"

struct Params {
  i32 dim;
  i32 hidden_dim;
  i32 n_heads;
  i32 n_layers;
  i32 vocab_size;
  i32 max_seq_len;
} __attribute__((__packed__));


struct EncoderBlock {
  MatView attention_norm;
  MatView wq;
  MatView wk;
  MatView wv;
  MatView wo;
  MatView ffn_norm;
  MatView w1;
  MatView w2;
  MatView w3;

  void set_params(Params &params) {
    attention_norm.shape = {params.dim, 1};
    wq.shape = {params.dim, params.dim};
    wk.shape = {params.dim, params.dim};
    wv.shape = {params.dim, params.dim};
    wo.shape = {params.dim, params.dim};
    ffn_norm.shape = {params.dim, 1};
    w1.shape = {params.hidden_dim, params.dim};
    w2.shape = {params.dim, params.hidden_dim};
    w3.shape = {params.hidden_dim, params.dim};
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
    norm.shape = {params.dim, 1};
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
  MatView tok_embeddings;
  MatView norm;
  MatView output_weights;
  std::array<EncoderBlock, 32> layers;
};
