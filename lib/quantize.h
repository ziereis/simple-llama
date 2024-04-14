#ifndef QUANTIZE_H
#define QUANTIZE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void quantize_model(const char* in_file, const char* out_file);

#ifdef __cplusplus
}
#endif

#endif // QUANTIZE_H
