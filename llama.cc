#include "include/kernels-cpu.hpp"
#include "include/transformer.hpp"
#include "vector"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <vector>

struct KVCache {
  f32 *data;
  i32 dim;
  i32 head_dim;

  MatView get_head(i32 head_idx, i32 t) {
    return {data + t * dim + head_idx * head_dim, {1, head_dim}, false};
  }
  MatView get_embedding(i32 t) { return {data + t * dim, {1, dim}, false}; }

  void init(i32 dim, i32 max_seq_len, i32 n_heads) {
    this->dim = dim;
    this->head_dim = dim / n_heads;
    data = new f32[dim * max_seq_len];
  }
};

void rotate_embeddings(MatView x, i32 pos, i32 head_dim) {
  assert(x.shape[1] == 1);
  assert(x.shape[0] % head_dim == 0);
  for (i32 i = 0; i < x.shape[0]; i += 2) {
    auto idx = i % head_dim;
    float freq = 1.0f / powf(10000.0f, idx / (float)head_dim);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    float v0 = x(i, 0);
    float v1 = x(i + 1, 0);
    x(i, 0) = v0 * fcr - v1 * fci;
    x(i + 1, 0) = v0 * fci + v1 * fcr;
  }
}

class Runtime {

public:
  void init(std::span<const u8> data) {
    t.init(data);
    for (i32 i = 0; i < 32; i++) {
      kcaches[i].init(t.params.dim, t.params.max_seq_len, t.params.n_heads);
      vcaches[i].init(t.params.dim, t.params.max_seq_len, t.params.n_heads);
    }

    x = MatView::New(t.params.dim, 1);
    x_buf = MatView::New(t.params.dim, 1);
    x_buf2 = MatView::New(t.params.dim, 1);
    q = MatView::New(t.params.dim, 1);
    attention = MatView::New(t.params.n_heads, t.params.max_seq_len);
    hidden_buf = MatView::New(t.params.hidden_dim, 1);
    hidden_buf2 = MatView::New(t.params.hidden_dim, 1);
    logits = MatView::New(t.params.vocab_size, 1);
  }

  MatView forward(i32 tok, i32 pos) {

    MatView embedding = t.tok_embeddings.get_row(tok).transpose();
    x.copy_from(embedding);

    for (i32 l_id = 0; l_id < t.params.n_layers; l_id++) {
      EncoderBlock &layer = t.layers[l_id];
      KVCache &kcache = kcaches[l_id];
      KVCache &vcache = vcaches[l_id];

      auto head_dim = t.params.dim / t.params.n_heads;

      auto k = kcache.get_embedding(pos).transpose();
      auto v = vcache.get_embedding(pos).transpose();

      rms_norm(x_buf, x, layer.attention_norm);
      matvec_mul(layer.wq, x_buf, q);
      matvec_mul(layer.wk, x_buf, k);
      matvec_mul(layer.wv, x_buf, v);

      rotate_embeddings(q, pos, head_dim);
      rotate_embeddings(k, pos, head_dim);

      #pragma omp parallel for schedule(static)
      for (int i = 0; i < t.params.n_heads; i++) {
        auto q_head_i = MatView{q.data + i * head_dim, {head_dim, 1}, false};
        auto attention_i = attention.get_row(i).transpose();
        for (int t = 0; t <= pos; t++) {
          auto k_head_i = kcache.get_head(i, t).transpose();
          float score = dotprod(q_head_i, k_head_i);
          score /= sqrt(head_dim);
          attention_i(t, 0) = score;
        }

        softmax(attention_i, pos + 1);

        auto x_buf_i = MatView{x_buf.data + i * head_dim, {head_dim, 1}, false};
        memset(x_buf_i.data, 0, head_dim * sizeof(f32));

        for (int t = 0; t <= pos; t++) {

          auto v_head_i = vcache.get_head(i, t).transpose();
          float score = attention_i(t, 0);
          for (i32 j = 0; j < head_dim; j++) {
            x_buf_i(j, 0) += score * v_head_i(j, 0);
          }
        }
      }

      matvec_mul(layer.wo, x_buf, x_buf2);

      for (i32 i = 0; i < t.params.dim; i++) {
        x(i, 0) += x_buf2(i, 0);
      }

      rms_norm(x_buf, x, layer.ffn_norm);

      matvec_mul(layer.w1, x_buf, hidden_buf);
      matvec_mul(layer.w3, x_buf, hidden_buf2);

      for (i32 i = 0; i < hidden_buf.shape[0]; i++) {
        float val = hidden_buf(i, 0);
        val *= 1.0f / (1.0f + expf(-val));
        val *= hidden_buf2(i, 0);
        hidden_buf(i, 0) = val;
      }

      matvec_mul(layer.w2, hidden_buf, x_buf);

      for (i32 i = 0; i < t.params.dim; i++) {
        x(i, 0) += x_buf(i, 0);
      }
    }

    rms_norm(x, x, t.norm);

    matvec_mul(t.output_weights, x, logits);
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
      softmax(logits, logits.shape[0]);

      auto next_token = std::distance(
          logits.data,
          std::max_element(logits.data, logits.data + logits.shape[0]));
      if (next_token == EOD_ID) {
        break;
      }
      result.push_back(next_token);
    }
    return result;
  }

private:
  static constexpr i32 EOD_ID = 2;
  MatView q;
  MatView x;
  MatView x_buf;
  MatView x_buf2;
  MatView attention;
  MatView hidden_buf;
  MatView hidden_buf2;
  MatView logits;
  std::array<KVCache, 32> kcaches;
  std::array<KVCache, 32> vcaches;
  LLama t;
};

int main(int argc, char **argv) {
  std::string filename;
  std::vector<i32> tokens;

  filename = argv[1];

  for (int i = 2; i < argc; ++i) {
    tokens.push_back(std::atoi(argv[i]));
  }

  tz::MappedFile data(filename);
  Runtime rt;

  rt.init(data.asSpan());

  auto res = rt.generate(tokens, 30);

  for (auto &r : res) {
    std::cout << r << " ";
  }
}
