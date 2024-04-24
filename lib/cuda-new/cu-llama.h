#include "../utils.h"
#include "../llama.h"


extern "C" {
void qllama_init_device(QLLama *d_m, QLLama *h_m);
void device_runtime_init_q4(QLLamaRuntime *rt, QLLama *m);
f32 *device_forward_q4(QLLamaRuntime *d_rt, i32 tok, i32 pos);
void device_runtime_deinit_q4(QLLamaRuntime *rt);

void* device_runtime_new_q4(const char* filename);
void device_runtime_delete_q4(void* handle);
float *device_runtime_forward_q4(void *handle, int tok, int pos);
}
