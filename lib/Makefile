CC=gcc
CFLAGS=-g -Wall -Wextra -std=gnu11 -fopenmp -Ofast -ffast-math -march=native

SRC_RUNTIME = llama.c ops.c utils.c test.c quantize.c
OBJ_RUNTIME = $(SRC_RUNTIME:.c=.o)

TARGET=test

all: $(TARGET)

$(TARGET): $(OBJ_RUNTIME)
	$(CC) $(CFLAGS) -o $@ $^ -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJ_RUNTIME)

.PHONY: all clean
