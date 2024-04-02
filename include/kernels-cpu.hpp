#pragma once
#include "matview.hpp"
#include <numeric>
#include <cmath>


void  __attribute__ ((noinline)) rms_norm(MatView out, MatView x, MatView weights) {
  assert(out.shape == x.shape);
  assert(x.shape[1] == 1);
  assert(x.shape == weights.shape);
  float sum = 0.0f;
  for (int i = 0; i < x.shape[0]; i++) {
    sum += x(i, 0) * x(i, 0);
  }
  sum /= x.shape[0];
  sum += 1e-5f;
  sum = 1.0f / (std::sqrt(sum));
  for (int i = 0; i < x.shape[0]; i++) {
    out(i, 0) = weights(i, 0) * (x(i, 0) * sum);
  }
}

void __attribute__((noinline)) matvec_mul(MatView W, MatView x, MatView C) {
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

float __attribute__((noinline)) dotprod(MatView q, MatView k) {
  assert(q.shape == k.shape);
  assert(q.shape[1] == 1);

  float sum = 0.0f;

  for (int i = 0; i < q.shape[0]; i++) {
    sum += q(i, 0) * k(i, 0);
  }
  return sum;
}

void __attribute__ ((noinline)) softmax(MatView x, i32 limit) {
  assert(x.shape[1] == 1);
  assert(limit <= x.shape[0]);
    // find max value (for numerical stability)
    float max_val = x(0,0);
    for (int i = 1; i < limit; i++) {
        max_val = std::max(max_val, x(i, 0));
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < limit; i++) {
        x(i,0) = expf(x(i,0) - max_val);
        sum += x(i,0);
    }
    // normalize
    for (int i = 0; i < limit; i++) {
        x(i,0) /= sum;
    }
}
