#ifdef __cplusplus
extern "C" {
#endif

typedef void* QRuntimeHandle;

QRuntimeHandle QRuntime_new(const char* filename);
void QRuntime_delete(QRuntimeHandle handle);
float* QRuntime_forward(QRuntimeHandle handle, int tok, int pos);

#ifdef __cplusplus
}
#endif
