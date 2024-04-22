CC=clang
CFLAGS=-g -fopenmp # Ensure flags are consistent for all compilations if needed
CUDA_PATH=/usr/local/cuda
LDFLAGS=-L$(CUDA_PATH)/lib64 -L/usr/lib/llvm-14/lib -lcudart -lcublas -lm
INCLUDES=-I$(CUDA_PATH)/include

SOURCES=cu-llama.cu cu-ops.cu ../utils.c ../llama.c ../ops.c
CU_SOURCES=$(filter %.cu,$(SOURCES))
C_SOURCES=$(filter %.c,$(SOURCES))

CU_OBJECTS=$(CU_SOURCES:.cu=.o)
C_OBJECTS=$(C_SOURCES:.c=.o)
OBJECTS=$(CU_OBJECTS) $(C_OBJECTS)

EXECUTABLE=cu-llama

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ -lm

%.o: %.cu
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(EXECUTABLE) $(OBJECTS)
