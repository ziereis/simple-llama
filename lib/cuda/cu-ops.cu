#include "cu-ops.h"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>


#define unpack_left(in) ((in >> 4) & 0x0f) - 8
#define unpack_right(in) (in & 0x0f) - 8
#define pack_q4(left, right) (((((u8)left) + 8) << 4) | (((u8)right) + 8))


__global__ void deq_q4(f32 *out, i8 *in, f32 *scales, u64 n, i32 group_size) {
  u64 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    u64 start_idx = i * group_size / 2;
    u64 end_idx = (i + 1) * (group_size / 2);
    for (u64 j = start_idx; j < end_idx; j++) {
      u8 p = (u8)in[j];
      i32 left = unpack_left(p);
      i32 right = unpack_right(p);
      out[j * 2] = ((f32)left) * scales[i];
      out[(j * 2) + 1] = ((f32)right) * scales[i];
    }
  }
}

void d_dequantize_q4(f32 *out, i8 *in, f32 *scales, u64 n, i32 group_size) {
  u64 n_groups = n / group_size;
  deq_q4<<<n_groups, group_size>>>(out, in, scales, n, group_size);
}
