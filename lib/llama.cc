#include "include/kernels-cpu.hpp"
#include "include/matview.hpp"
#include "include/transformer.hpp"
#include "vector"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <vector>
#include "llama.h"

// struct KVCache {
//   f32 *data;
//   i32 dim;
//   i32 head_dim;

//   MatView get_head(i32 head_idx, i32 t) {
//     return {data + t * dim + head_idx * head_dim, {1, head_dim}, false};
//   }
//   MatView get_embedding(i32 t) { return {data + t * dim, {1, dim}, false}; }

//   void init(i32 dim, i32 max_seq_len, i32 n_heads) {
//     this->dim = dim;
//     this->head_dim = dim / n_heads;
//     data = new f32[dim * max_seq_len];
//   }
// };

void rotate_embeddings(FMatrix x, i32 pos, i32 head_dim) {
  assert(x.shape[0] == 1);
  assert(x.shape[1] % head_dim == 0);
  for (i32 i = 0; i < x.shape[1]; i += 2) {
    auto idx = i % head_dim;
    float freq = 1.0f / powf(10000.0f, idx / (float)head_dim);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    float v0 = x(0, i);
    float v1 = x(0, i + 1);
    x(0, i) = v0 * fcr - v1 * fci;
    x(0, i + 1) = v0 * fci + v1 * fcr;
  }
}

class Runtime {

public:
  void init(std::span<const u8> data) {
    t.init(data);
    for (i32 i = 0; i < 32; i++) {
      kcaches[i] = FMatrix::New(t.params.max_seq_len, t.params.dim);
      vcaches[i] = FMatrix::New(t.params.max_seq_len, t.params.dim);
    }

    x = FMatrix::New(1, t.params.dim);
    x_buf = FMatrix::New(1, t.params.dim);
    x_buf2 = FMatrix::New(1, t.params.dim);
    q = FMatrix::New(1, t.params.dim);
    attention = FMatrix::New(t.params.n_heads, t.params.max_seq_len);
    hidden_buf = FMatrix::New(1, t.params.hidden_dim);
    hidden_buf2 = FMatrix::New(1, t.params.hidden_dim);
    logits = FMatrix::New(1, t.params.vocab_size);
  }

  FMatrix forward(i32 tok, i32 pos) {

    FMatrix embedding = t.tok_embeddings.get_row(tok);
    x.copy_from(embedding);


    for (i32 l_id = 0; l_id < t.params.n_layers; l_id++) {
      std::cout << "rt Layer " << l_id << std::endl;
      EncoderBlock &layer = t.layers[l_id];
      FMatrix &kcache = kcaches[l_id];
      FMatrix &vcache = vcaches[l_id];

      auto head_dim = t.params.dim / t.params.n_heads;

      FMatrix k = kcache.get_row(pos);
      FMatrix v = vcache.get_row(pos);

      rms_norm(x_buf, x, layer.attention_norm);


      matvec_mul(layer.wq, x_buf.t(), q.t());
      matvec_mul(layer.wk, x_buf.t(), k.t());
      matvec_mul(layer.wv, x_buf.t(), v.t());

      rotate_embeddings(q, pos, head_dim);
      rotate_embeddings(k, pos, head_dim);

#pragma omp parallel for schedule(static)
      for (int i = 0; i < t.params.n_heads; i++) {
        auto q_head = q.get_row_chunk(0, i, head_dim);
        auto attention_i = attention.get_row(i);
        for (int t = 0; t <= pos; t++) {
          auto k_head = kcache.get_row_chunk(t, i, head_dim);
          float score = dotprod(q_head.t(), k_head.t());
          score /= sqrt(head_dim);
          attention_i(0, t) = score;
        }

        softmax(attention_i, pos + 1);

        auto x_buf_head = x_buf.get_row_chunk(0, i, head_dim);
        memset(x_buf_head.data, 0, head_dim * sizeof(f32));

        for (int t = 0; t <= pos; t++) {
          auto v_head = vcache.get_row_chunk(t, i, head_dim);
          float score = attention_i(0, t);
          for (i32 j = 0; j < head_dim; j++) {
            x_buf_head(0, j) += score * v_head(0, j);
          }
        }
      }

      matvec_mul(layer.wo, x_buf.t(), x_buf2.t());

      for (i32 i = 0; i < t.params.dim; i++) {
        x(0, i) += x_buf2(0, i);
      }

      rms_norm(x_buf, x, layer.ffn_norm);

      matvec_mul(layer.w1, x_buf.t(), hidden_buf.t());
      matvec_mul(layer.w3, x_buf.t(), hidden_buf2.t());

      for (i32 i = 0; i < hidden_buf.shape[1]; i++) {
        float val = hidden_buf(0, i);
        val *= 1.0f / (1.0f + expf(-val));
        val *= hidden_buf2(0, i);
        hidden_buf(0, i) = val;
      }

      matvec_mul(layer.w2, hidden_buf.t(), x_buf.t());

      for (i32 i = 0; i < t.params.dim; i++) {
        x(0, i) += x_buf(0, i);
      }
    }

    rms_norm(x, x, t.norm);

    matvec_mul(t.output_weights, x.t(), logits.t());
    return logits;
  }

  std::vector<i32> generate(std::vector<i32> tokens, u32 max_toks) {
    std::vector<i32> result;
    u32 pos = 0;

    for (; pos < tokens.size(); pos++) {
      auto logits = forward(tokens[pos], pos);
      result.push_back(tokens[pos]);
    }
    for (; pos < max_toks; pos++) {
      auto logits = forward(result.back(), pos);
      softmax(logits, logits.shape[1]);

      auto next_token = std::distance(
          logits.data,
          std::max_element(logits.data, logits.data + logits.shape[1]));
      if (next_token == EOD_ID) {
        break;
      }
      result.push_back(next_token);
    }
    return result;
  }

private:
  static constexpr i32 EOD_ID = 2;
  FMatrix q;
  FMatrix x;
  FMatrix x_buf;
  FMatrix x_buf2;
  FMatrix attention;
  FMatrix hidden_buf;
  FMatrix hidden_buf2;
  FMatrix logits;
  std::array<FMatrix, 32> kcaches;
  std::array<FMatrix, 32> vcaches;
  LLama t;
};

void dequantize(FMatrix result, QMatrix in) {
  assert(result.shape == in.shape);
  for (u32 i = 0; i < in.get_n_groups(); i++) {
    for (u32 j = i * in.group_size; j < (i + 1) * in.group_size; j++) {
      result.data[j] = static_cast<f32>(in.data[j]) * in.scales[i];
    }
  }
}

void quantize(QMatrix result, FMatrix in) {
  assert(result.shape == in.shape);
#pragma omp parallel for schedule(static)
  for (u32 i = 0; i < result.get_n_groups(); i++) {
    auto dbegin = in.data + i * result.group_size;
    auto dend = in.data + (i + 1) * result.group_size;
    f32 max = std::accumulate(dbegin, dend, 0.0f, [](f32 acc, f32 x) {
      return std::max(acc, std::abs(x));
    });
    f32 scale = max / 127.0f;
    auto wbegin = result.data + i * result.group_size;
    std::transform(dbegin, dend, wbegin, [scale](f32 x) {
      return static_cast<i8>(std::round(x / scale));
    });
    result.scales[i] = scale;
  }
}


void check_none(f32 *begin, f32 *end) {
  for (auto it = begin; it != end; it++) {
    if (std::isnan(*it)) {
      throw std::runtime_error("nan scale");
    }
  }

}

class QRuntime {
public:
  void init(std::string filename) {
    file = tz::MappedFile(filename);
    auto data = file.asSpan();
    m.init(data);
    for (i32 i = 0; i < 32; i++) {
      kcaches[i] = FMatrix::New(m.params.max_seq_len, m.params.dim);
      vcaches[i] = FMatrix::New(m.params.max_seq_len, m.params.dim);
    }

    fq = FMatrix::New(1, m.params.dim);
    fx = FMatrix::New(1, m.params.dim);
    x = QMatrix::New(1, m.params.dim, m.qparams.group_size);
    x_buf = QMatrix::New(1, m.params.dim, m.qparams.group_size);
    fx_buf = FMatrix::New(1, m.params.dim);
    fx_buf2 = FMatrix::New(1, m.params.dim);
    attention = FMatrix::New(m.params.n_heads, m.params.max_seq_len);
    hidden_buf = QMatrix::New(1, m.params.hidden_dim, m.qparams.group_size);
    fhidden_buf = FMatrix::New(1, m.params.hidden_dim);
    fhidden_buf2 = FMatrix::New(1, m.params.hidden_dim);
    logits = FMatrix::New(1, m.params.vocab_size);
  }

  FMatrix forward(i32 tok, i32 pos) {


    QMatrix embedding = m.tok_embeddings.get_row(tok);
    x.copy_from(embedding);
    dequantize(fx, x);

    for (i32 l_id = 0; l_id < m.params.n_layers; l_id++) {
      QEncoderBlock &layer = m.layers[l_id];
      FMatrix kcache = kcaches[l_id];
      FMatrix vcache = vcaches[l_id];

      auto head_dim = m.params.dim / m.params.n_heads;

      FMatrix k = kcache.get_row(pos);
      FMatrix v = vcache.get_row(pos);

      rms_norm(fx_buf, fx, layer.attention_norm);

      quantize(x_buf, fx_buf);
      qmatvec_mul(layer.wq, x_buf.t(), fq.t());
      qmatvec_mul(layer.wk, x_buf.t(), k.t());
      qmatvec_mul(layer.wv, x_buf.t(), v.t());


      rotate_embeddings(fq, pos, head_dim);
      rotate_embeddings(k, pos, head_dim);

#pragma omp parallel for schedule(static)
      for (int i = 0; i < m.params.n_heads; i++) {
        auto q_head = fq.get_row_chunk(0, i, head_dim);
        auto attention_i = attention.get_row(i);
        for (int t = 0; t <= pos; t++) {
          auto k_head = kcache.get_row_chunk(t, i, head_dim);
          float score = dotprod(q_head.t(), k_head.t());
          score /= sqrt(head_dim);
          attention_i(0, t) = score;
        }

        softmax(attention_i, pos + 1);

        auto x_buf_head = fx_buf.get_row_chunk(0, i, head_dim);
        memset(x_buf_head.data, 0, head_dim * sizeof(f32));

        for (int t = 0; t <= pos; t++) {
          auto v_head = vcache.get_row_chunk(t, i, head_dim).t();
          float score = attention_i(0, t);
          for (i32 j = 0; j < head_dim; j++) {
            x_buf_head(0, j) += score * v_head(0, j);
          }
        }
      }

      quantize(x_buf, fx_buf);

      qmatvec_mul(layer.wo, x_buf.t(), fx_buf2.t());

      for (i32 i = 0; i < m.params.dim; i++) {
        fx(0, i) += fx_buf2(0, i);
      }

      rms_norm(fx_buf, fx, layer.ffn_norm);

      quantize(x_buf, fx_buf);

      qmatvec_mul(layer.w1, x_buf.t(), fhidden_buf.t());
      qmatvec_mul(layer.w3, x_buf.t(), fhidden_buf2.t());

      for (i32 i = 0; i < fhidden_buf.shape[1]; i++) {
        float val = fhidden_buf(0, i);
        val *= 1.0f / (1.0f + expf(-val));
        val *= fhidden_buf2(0, i);
        fhidden_buf(0, i) = val;
      }


      quantize(hidden_buf, fhidden_buf);

      qmatvec_mul(layer.w2, hidden_buf.t(), fx_buf.t());

      for (i32 i = 0; i < m.params.dim; i++) {
        fx(0, i) += fx_buf(0, i);
      }
    }

    rms_norm(fx, fx, m.norm);
    quantize(x, fx);
    qmatvec_mul(m.output_weights, x.t(), logits.t());
    return logits;
  }

  std::vector<i32> generate(std::vector<i32> tokens, u32 max_toks) {
    std::vector<i32> result;
    u32 pos = 0;

    for (; pos < tokens.size(); pos++) {
      auto logits = forward(tokens[pos], pos);
      result.push_back(tokens[pos]);
    }
    for (; pos < max_toks; pos++) {
      auto logits = forward(result.back(), pos);

      auto next_token = std::distance(
          logits.data,
          std::max_element(logits.data, logits.data + logits.shape[1]));
      if (next_token == EOD_ID) {
        break;
      }
      result.push_back(next_token);
    }
    return result;
  }

private:
  static constexpr i32 EOD_ID = 2;
  FMatrix fq;
  FMatrix fx;
  QMatrix x;
  FMatrix fx_buf;
  QMatrix x_buf;
  FMatrix fx_buf2;
  FMatrix attention;
  QMatrix hidden_buf;
  FMatrix fhidden_buf;
  FMatrix fhidden_buf2;
  FMatrix logits;

  std::array<FMatrix, 32> kcaches;
  std::array<FMatrix, 32> vcaches;
  QLLama m;
  tz::MappedFile file;
};

i32 get_version(std::span<const u8> data) {
  tz::BinaryReader reader(data.data(), data.size());
  LLAMA_VALIDATE(reader.read<u32>() == 0x7fdd7f7f, "Invalid magic");
  return reader.read<i32>();
}


extern "C" {

QRuntimeHandle QRuntime_new(const char* filename) {
  auto *rt = new QRuntime();
  rt->init(filename);
  return rt;
}

float* QRuntime_forward(QRuntimeHandle handle, i32 tok, i32 pos) {
    auto rt = static_cast<QRuntime*>(handle);
    auto logits = rt->forward(tok, pos);
    return logits.data;
}

void QRuntime_delete(QRuntimeHandle handle) {
    delete static_cast<QRuntime*>(handle);
}

}

// int main(int argc, char **argv) {
//   // std::string filename;
//   // std::vector<i32> tokens;

//   // filename = argv[1];

//   // for (int i = 2; i < argc; ++i) {
//   //   tokens.push_back(std::atoi(argv[i]));
//   // }

//   // tz::MappedFile data(filename);

//   // auto version = get_version(data.asSpan());
//   // if (version == 1) {
//   //   Runtime rt;

//   //   rt.init(data.asSpan());

//   //   auto res = rt.generate(tokens, 30);

//   //   for (auto &r : res) {
//   //     std::cout << r << " ";
//   //   }
//   // } else {
//   //   QRuntime rt;

//   //   rt.init(data.asSpan());

//   //   auto res = rt.generate(tokens, 30);

//   //   for (auto &r : res) {
//   //     std::cout << r << " ";
//   //   }
//   // }

//   // tz::MappedFile data("../bin/llama_q8.bin");
//   // QRuntime rt;
//   // rt.init(data.asSpan());

//   // auto res = rt.generate(
//   //     {3439, 17632, 1925, 29892, 278, 6368, 310, 14215, 537, 5922, 393, 29871},
//   //     50);

//   // for (auto &r : res) {
//   //   std::cout << r << " ";
//   // }

//   // tz::MappedFile data("../bin/llama.bin");
//   // Runtime rt;
//   // rt.init(data.asSpan());

//   // auto res = rt.generate(
//   //     {3439, 17632, 1925, 29892, 278, 6368, 310, 14215, 537, 5922, 393, 29871},
//   //     50);

//   // for (auto &r : res) {
//   //   std::cout << r << " ";
//   // }

//   // Runtime rt;

//   // rt.init(data.asSpan());

//   // auto res = rt.generate(
//   //     {3439, 17632, 1925, 29892, 278, 6368, 310, 14215, 537, 5922, 393,
//   //     29871}, 20);

//   // for (auto &r : res) {
//   //   std::cout << r << " ";
//   // }

//   // auto a = FMatrix::New(1, 10);
//   // auto b = FMatrix::New(1, 10);
//   // auto c = FMatrix::New(1, 1);

//   // for (int i = 0; i < 10; i++) {
//   //   a.data[i] = -1.f + 1/ (i + 1.0f);
//   //   b.data[i] = 1.0f;
//   // }

//   // matvec_mul(a, b.t(), c.t());

//   // std::cout << c(0, 0) << std::endl;

//   // auto aq = QMatrix::New(1, 10, 2);
//   // auto bq = QMatrix::New(1, 10, 2);
//   // auto cq = FMatrix::New(1, 1);

//   // quantize(aq, a);
//   // quantize(bq, b);

//   // qmatvec_mul(aq, bq.t(), c.t());

//   // std::cout << c(0, 0) << std::endl;

//   // a.dump();
//   // b.dump();
//   // aq.dump();
//   // bq.dump();



//   // tz::MappedFile data("../bin/llama_q8.bin");
//   // data.dump(20);
//   // QLLama m;
//   // m.init(data.asSpan());
// }
