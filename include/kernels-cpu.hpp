#pragma once
#include "matview.hpp"
#include <numeric>
#include <cmath>


void  __attribute__ ((noinline)) rms_norm(FMatrix out, FMatrix x, FMatrix weights) {
  assert(out.shape == x.shape);
  assert(x.shape[0] == 1);
  assert(x.shape == weights.shape);
  float sum = 0.0f;
  for (int i = 0; i < x.shape[1]; i++) {
    sum += x(0, i) * x(0, i);
  }
  sum /= x.shape[1];
  sum += 1e-5f;
  sum = 1.0f / (std::sqrt(sum));
  for (int i = 0; i < x.shape[1]; i++) {
    out(0, i) = weights(0, i) * (x(0, i) * sum);
  }
}

void __attribute__((noinline)) matvec_mul(FMatrix W, FMatrix x, FMatrix C) {
  assert(W.shape[1] == x.shape[0]);
  assert(W.shape[0] == C.shape[0]);
  assert(x.shape[1] == C.shape[1]);
  // x is always a column vector
  assert(x.shape[1] == 1);

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < W.shape[0]; i++) {
    float val = 0.0f;
    for (int j = 0; j < W.shape[1]; j++) {
      val += W(i, j) * x(j, 0);
    }
    C(i, 0) = val;
  }
}

void __attribute__((noinline)) qmatvec_mul(QMatrix W, QMatrix x, FMatrix C) {
  assert(W.shape[1] == x.shape[0]);
  assert(W.shape[0] == C.shape[0]);
  assert(x.shape[1] == C.shape[1]);
  // x is always a column vector
  assert(x.shape[1] == 1);
  assert(W.group_size == x.group_size);
  int g_size = W.group_size;
  #pragma omp parallel for schedule(static)
  for (i32 i = 0; i < W.shape[0]; i++) {
    auto w_row = W.get_row(i);
    float facc = 0.0f;
    for (u32 g = 0; g < w_row.get_n_groups(); g++) {
      i32 iacc = 0;
      for (u32 j = g * g_size; j < (g + 1) * g_size; j++) {
        iacc += ((i32)w_row(0, j)) * ((i32)x(j, 0));
      }
      facc += ((float)iacc) * w_row.scales[g] * x.scales[g];
    }
    C(i, 0) = facc;
  }
}



float __attribute__((noinline)) dotprod(FMatrix q, FMatrix k) {
  assert(q.shape == k.shape);
  assert(q.shape[1] == 1);

  float sum = 0.0f;

  for (int i = 0; i < q.shape[0]; i++) {
    sum += q(i, 0) * k(i, 0);
  }
  return sum;
}

void __attribute__ ((noinline)) softmax(FMatrix x, i32 limit) {
  assert(x.shape[0] == 1);
  assert(limit <= x.shape[1]);
    // find max value (for numerical stability)
    float max_val = x(0,0);
    for (int i = 1; i < limit; i++) {
        max_val = std::max(max_val, x(0, i));
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < limit; i++) {
        x(0,i) = expf(x(0,i) - max_val);
        sum += x(0,i);
    }
    // normalize
    for (int i = 0; i < limit; i++) {
        x(0,i) /= sum;
    }
}
