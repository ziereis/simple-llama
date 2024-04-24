#pragma once

#include <time.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef double f64;
typedef float f32;
typedef int64_t i64;
typedef int32_t i32;
typedef int16_t i16;
typedef int8_t i8;
typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;


#define DIM 4096ull
#define H_DIM 11008ull
#define N_LAYERS 32ull
#define VOCAB_SIZE 32000ull


#define COUNT_WEIGHTS DIM *DIM
#define COUNT_WEIGHTS_LAYER COUNT_WEIGHTS *N_LAYERS
#define COUNT_NORM DIM
#define COUNT_NORM_LAYER COUNT_NORM *N_LAYERS
#define COUNT_WEIGHTS_H H_DIM *DIM
#define COUNT_WEIGHTS_H_LAYER COUNT_WEIGHTS_H *N_LAYERS
#define COUNT_TOK_EMB VOCAB_SIZE *DIM
#define COUNT_OUTPUT_WEIGHT COUNT_TOK_EMB

#define SIZE_WEIGHTS_F32 COUNT_WEIGHTS * sizeof(f32)
#define SIZE_WEIGHTS_LAYER_F32 COUNT_WEIGHTS_LAYER * sizeof(f32)
#define SIZE_NORM_F32 COUNT_NORM * sizeof(f32)
#define SIZE_NORM_LAYER_F32 COUNT_NORM_LAYER * sizeof(f32)
#define SIZE_WEIGHTS_H_F32 COUNT_WEIGHTS_H * sizeof(f32)
#define SIZE_WEIGHTS_H_LAYER_F32 COUNT_WEIGHTS_H_LAYER * sizeof(f32)
#define SIZE_TOK_EMB_F32 COUNT_TOK_EMB * sizeof(f32)
#define SIZE_OUTPUT_WEIGHT_F32 COUNT_OUTPUT_WEIGHT * sizeof(f32)

#define SIZE_WEIGHTS_Q8 COUNT_WEIGHTS
#define SIZE_WEIGHTS_LAYER_Q8 COUNT_WEIGHTS_LAYER
#define SIZE_WEIGHTS_H_Q8 COUNT_WEIGHTS_H
#define SIZE_WEIGHTS_H_LAYER_Q8 COUNT_WEIGHTS_H_LAYER
#define SIZE_TOK_EMB_Q8 COUNT_TOK_EMB
#define SIZE_OUTPUT_WEIGHT_Q8 COUNT_OUTPUT_WEIGHT

#define SIZE_WEIGHTS_Q4 COUNT_WEIGHTS / 2
#define SIZE_WEIGHTS_LAYER_Q4 COUNT_WEIGHTS_LAYER / 2
#define SIZE_WEIGHTS_H_Q4 COUNT_WEIGHTS_H / 2
#define SIZE_WEIGHTS_H_LAYER_Q4 COUNT_WEIGHTS_H_LAYER / 2
#define SIZE_TOK_EMB_Q4 COUNT_TOK_EMB / 2
#define SIZE_OUTPUT_WEIGHT_Q4 COUNT_OUTPUT_WEIGHT / 2

typedef i32 Shape[2];


typedef struct {
  struct timespec start;
  struct timespec end;
} Timer;

void start_timer(Timer *t);
void stop_timer(Timer *t);
i32 elapsed_time(Timer *t);

typedef struct {
  i8 left;
  i8 right;
} Pair;

bool shape_eq(Shape lhs, Shape rhs);

//void print_q4(i8 *in, u64 n, u64 offset);
void print_vec(f32 *vec, u64 n, u64 offset);

typedef struct Matrix {
  f32 *data;
  Shape shape;
} Matrix;

Matrix new_matrix(i32 rows, i32 cols);
void free_matrix(Matrix m);
Matrix get_row(Matrix m, i32 i);
Matrix get_row_chunk(Matrix m, i32 row_idx, i32 chunk_idx, i32 chunk_size);

typedef struct {
  f32 *scales;
  Shape shape;
  i8 *data;
} QMatrix;

QMatrix new_matrix_q8(i32 rows, i32 cols, i32 group_size);
QMatrix new_matrix_q4(i32 rows, i32 cols, i32 group_size);
void free_matrix_q8(QMatrix m);
void free_matrix_q4(QMatrix m);
QMatrix get_row_q8(QMatrix m, i32 i, u32 group_size);
QMatrix get_row_q4(QMatrix m, i32 i, u32 group_size);


typedef struct {
  u8* data;
  u64  len;
} MappedFile;

bool init(MappedFile *file, const char *path);
void deinit(MappedFile *file);

#ifdef __cplusplus
}
#endif
