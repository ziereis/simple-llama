#include "include/transformer.hpp"
#include "include/tz-utils.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <ostream>
#include <vector>
#include <numeric>
#include "quantize.h"


struct Qresult {
  std::vector<i8> weights;
  std::vector<f32> scale;
  f32 max_error;
};

 void dequantize_q8(std::span<f32> result, std::span<i8> data, std::span<f32> scales) {
  int group_size = data.size() / scales.size();
  for (u32 i = 0; i < scales.size(); i++) {
    for (u32 j = i * group_size; j < (i + 1) * group_size; j++) {
      result[j] = static_cast<f32>(data[j]) * scales[i];
    }
  }
}

Qresult quantize_q8(std::span<f32> data, i32 group_size) {
  assert(data.size() % group_size == 0);
  std::vector<i8> weights(data.size());
  std::vector<f32> scales(data.size() / group_size);
  i32 n_groups = data.size() / group_size;
  #pragma omp parallel for schedule(static)
  for (i32 i = 0; i < n_groups; i++) {
    auto dbegin = data.begin() + i * group_size;
    auto dend = data.begin() + (i + 1) * group_size;
    f32 max = std::accumulate(dbegin, dend, 0.0f, [](f32 acc, f32 x) {
      return std::max(acc, std::abs(x));
    });
    f32 scale = max / 127.0f;
    auto wbegin = weights.begin() + i * group_size;
    std::transform(dbegin, dend, wbegin,
                   [scale](f32 x) { return static_cast<i8>(std::round(x / scale)); });
    scales[i] = scale;
  }
  std::vector<f32> recovered(data.size());
  f32 max_err = 0.0f;
  #pragma omp parallel for schedule(static) reduction(max : max_err)
  for (i32 i = 0; i < n_groups; i++) {
    f32 local_err = 0.0f;
    for (int j = i * group_size; j < (i + 1) * group_size; j++) {
      recovered[j] = static_cast<f32>(weights[j]) * scales[i];
      local_err = std::max(local_err, std::abs(data[j] - recovered[j]));
    }
    max_err = std::max(max_err, local_err);
  }
  return {std::move(weights), std::move(scales), max_err};
}

void quantize_model(const char* in_file, const char* out_file) {
  tz::MappedFile file(in_file);
  LLama m;
  m.init(file.asSpan());
  i32 group_size = 128;

  QuantizeParams qparams = {group_size, 8};

  u32 magic = 0x7fdd7f7f;
  u32 version = 2;
  std::ofstream out(out_file, std::ios::binary);
  out.write((char *)(&magic), sizeof(magic));
  out.write((char *)(&version), sizeof(version));
  out.write((char *)(&m.params), sizeof(Params));
  out.write((char *)(&qparams), sizeof(QuantizeParams));
  auto pad = 256 - out.tellp();
  // first all non quantized weights
  out.write(std::string(pad, 0).c_str(), pad);
  out.write((char *)(m.norm.data),
            m.norm.shape[0] * m.norm.shape[1] * sizeof(f32));
  for (auto &layer : m.layers) {
    out.write((char *)(layer.attention_norm.data),
              layer.attention_norm.shape[0] * layer.attention_norm.shape[1] *
                  sizeof(f32));
  }
  for (auto &layer : m.layers) {
    out.write((char *)(layer.ffn_norm.data),
              layer.ffn_norm.shape[0] * layer.ffn_norm.shape[1] * sizeof(f32));
  }
  // all quantized weights
  std::cout << "Quantizing tok_embeddings\n";
  m.tok_embeddings.dump_shape();
  Qresult qres =
      quantize_q8({m.tok_embeddings.data,
                   (u32)m.tok_embeddings.shape[0] * m.tok_embeddings.shape[1]},
                  group_size);
  out.write((char*) qres.weights.data(),qres.weights.size());
  out.write((char*) qres.scale.data(), qres.scale.size() * sizeof(f32));
  std::cout << "max error: " << qres.max_error << std::endl;

  std::cout << "Quantizing output_weights\n";
  m.output_weights.dump_shape();
  qres = quantize_q8({m.output_weights.data, (u32)m.output_weights.shape[0] *
                                                 m.output_weights.shape[1]},
                     group_size);
  out.write((char*) qres.weights.data(),qres.weights.size());
  out.write((char *)qres.scale.data(), qres.scale.size() * sizeof(f32));
  std::cout << "max error: " << qres.max_error << std::endl;

  // all weights for the layers are consecutive in memory so
  // we can quantize them all at once
  auto& l0 = m.layers[0];
  std::cout << "Quantizing Query weights\n";
  qres = quantize_q8({l0.wq.data, (u32)l0.wq.shape[0] * l0.wq.shape[1] * m.layers.size()},
                       group_size);
  out.write((char*) qres.weights.data(),qres.weights.size());
  out.write((char *)qres.scale.data(), qres.scale.size() * sizeof(f32));
  std::cout << "max error: " << qres.max_error << std::endl;

  std::cout << "Quantizing Key weights\n";
  qres = quantize_q8({l0.wk.data, (u32)l0.wk.shape[0] * l0.wk.shape[1] * m.layers.size()},
                     group_size);
  out.write((char*) qres.weights.data(),qres.weights.size());
  out.write((char *)qres.scale.data(), qres.scale.size() * sizeof(f32));
  std::cout << "max error: " << qres.max_error << std::endl;

  std::cout << "Quantizing Value weights\n";
  qres = quantize_q8(
      {l0.wv.data, (u32)l0.wv.shape[0] * l0.wv.shape[1] * m.layers.size()},
      group_size);
  out.write((char *)qres.weights.data(), qres.weights.size());
  out.write((char *)qres.scale.data(), qres.scale.size() * sizeof(f32));
  std::cout << "max error: " << qres.max_error << std::endl;

  std::cout << "Quantizing Output weights\n";
  qres = quantize_q8(
      {l0.wo.data, (u32)l0.wo.shape[0] * l0.wo.shape[1] * m.layers.size()},
      group_size);
  out.write((char *)qres.weights.data(), qres.weights.size());
  out.write((char *)qres.scale.data(), qres.scale.size() * sizeof(f32));
  std::cout << "max error: " << qres.max_error << std::endl;

  std::cout << "Quantizing FFN W1 weights\n";
  qres = quantize_q8(
      {l0.w1.data, (u32)l0.w1.shape[0] * l0.w1.shape[1] * m.layers.size()},
      group_size);
  out.write((char *)qres.weights.data(), qres.weights.size());
  out.write((char *)qres.scale.data(), qres.scale.size() * sizeof(f32));
  std::cout << "max error: " << qres.max_error << std::endl;

  std::cout << "Quantizing FFN W2 weights\n";
  qres = quantize_q8(
      {l0.w2.data, (u32)l0.w2.shape[0] * l0.w2.shape[1] * m.layers.size()},
      group_size);
  out.write((char *)qres.weights.data(), qres.weights.size());
  out.write((char *)qres.scale.data(), qres.scale.size() * sizeof(f32));
  std::cout << "max error: " << qres.max_error << std::endl;

  std::cout << "Quantizing FFN W3 weights\n";
  qres = quantize_q8(
      {l0.w3.data, (u32)l0.w3.shape[0] * l0.w3.shape[1] * m.layers.size()},
      group_size);
  out.write((char *)qres.weights.data(), qres.weights.size());
  out.write((char *)qres.scale.data(), qres.scale.size() * sizeof(f32));
  std::cout << "max error: " << qres.max_error << std::endl;

  out.close();
}
