#include "utils.h"
#include "assert.h"
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <time.h>


bool
shape_eq(Shape lhs, Shape rhs) {
  return lhs[0] == rhs[0] && lhs[1] == rhs[1];
}

void start_timer(Timer *t) { clock_gettime(CLOCK_MONOTONIC, &t->start); }

void stop_timer(Timer *t) { clock_gettime(CLOCK_MONOTONIC, &t->end); }

double elapsed_time(Timer *t) {
    double delta_sec = (double)(t->end.tv_sec - t->start.tv_sec);
    double delta_nsec = (double)(t->end.tv_nsec - t->start.tv_nsec);
    return (delta_sec + (delta_nsec / 1e9)) * 1000;
}


Matrix new_matrix(i32 rows, i32 cols) {
  f32 *data = malloc(rows * cols * sizeof(f32));
  return (Matrix){.data = data, .shape = {rows, cols}};
}

void free_matrix(Matrix m) { free(m.data); }

QMatrix new_matrix_q8(i32 rows, i32 cols, i32 group_size) {
  i8 *data = malloc(rows * cols);
  f32 *scales = malloc(rows * cols / group_size * sizeof(f32));
  return (QMatrix){.data = data, .shape = {rows, cols}, .scales = scales};
}

void free_matrix_q8(QMatrix m) {
  free(m.data);
  free(m.scales);
}

QMatrix new_matrix_q4(i32 rows, i32 cols, i32 group_size) {
  i8 *data = malloc(rows * cols / 2);
  f32 *scales = malloc(rows * cols / group_size * sizeof(f32));
  return (QMatrix){.data = data, .shape = {rows, cols}, .scales = scales};
}

void free_matrix_q4(QMatrix m) {
  free(m.data);
  free(m.scales);
}

Matrix get_row(Matrix m, i32 i) {
  assert(i < m.shape[0]);
  return (Matrix){.data = m.data + i * m.shape[1], .shape = {1, m.shape[1]}};
}
Matrix get_row_chunk(Matrix m, i32 row_idx, i32 chunk_idx, i32 chunk_size) {
  assert(row_idx < m.shape[0]);
  assert(chunk_idx < m.shape[1] / chunk_size);
  return (Matrix){.data = m.data + row_idx * m.shape[1] + chunk_idx * chunk_size,
                  .shape = {1, chunk_size}};
}

QMatrix get_row_q8(QMatrix m, i32 i, u32 group_size) {
  assert(i < m.shape[0]);
  return (QMatrix){.data = m.data + i * m.shape[1],
                   .shape = {1, m.shape[1]},
                   .scales = m.scales + i * m.shape[1] / group_size};
}

QMatrix get_row_q4(QMatrix m, i32 i, u32 group_size) {
  assert(i < m.shape[0]);
  return (QMatrix){.data = m.data + i * m.shape[1] / 2,
                   .shape = {1, m.shape[1]},
                   .scales = m.scales + i * m.shape[1] / group_size};
}


// void print_q4(i8* in, u64 n, u64 offset) {
//   for(u64 i = 0; i < n; i++) {
//     Pair unpacked = unpack_q4(in[i + offset]);
//     printf("%d %d ", unpacked.left, unpacked.right);
//   }
// }


void print_vec(f32 *vec, u64 n, u64 offset) {
  for (u64 i = 0 ; i < n; i++) {
    printf("%f ", vec[offset + i]);
  }
  printf("\n");
}


QMatrix qmat_get_row(QMatrix m, i32 i, u32 group_size) {
    assert(i < m.shape[0]);
    return (QMatrix){.data = m.data + i * m.shape[1],
                   .shape = {1, m.shape[1]},
                   .scales = m.scales + i * m.shape[1] / group_size};
}

bool init(MappedFile *file, const char *path) {
  int fd = open(path, O_RDONLY);
  if (fd == -1) {
    printf("Failed to open file\n");
    return false;
  }
  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    close(fd);
    printf("Failed to get file size\n");
    return false;
  }
  file->len = sb.st_size;
  void *res = mmap(NULL, file->len, PROT_READ, MAP_PRIVATE, fd, 0);
  if (res == MAP_FAILED) {
    close(fd);
    printf("Failed to map file\n");
    return false;
  }
  file->data = (u8 *)res;
  return true;
  close(fd);
}

void deinit(MappedFile *file) {
  munmap(file->data, file->len);
}
