#pragma once
#include "utils.h"


typedef enum {
  Q8,
  Q4,
  NONE,
} QuantizationType;

void quant_model(const char* in_file, const char* out_file, QuantizationType qt);

float calc_error(f32 *orig, f32 *deq, u64 n);
