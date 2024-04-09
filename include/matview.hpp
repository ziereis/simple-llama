#pragma once
#include "tz-utils.hpp"
#include <cassert>
#include <cstring>
#include <sys/cdefs.h>

struct Shape {
  std::array<i32, 2> shape = {-1, -1};
  __always_inline friend bool operator==(const Shape &lhs, const Shape &rhs) {
    return lhs.shape == rhs.shape;
  }
  __always_inline friend bool operator!=(const Shape &lhs, const Shape &rhs) {
    return !(lhs == rhs);
  }
  __always_inline i32 &operator[](i32 i) {
    assert(i < 2);
    return shape[i];
  }
  __always_inline const i32 &operator[](i32 i) const {
    assert(i < 2);
    return shape[i];
  }
};


template <class T, class Derived> struct Matrix {
  T *data;
  Shape shape;
  virtual ~Matrix() = default;

  // fake transpose for 1d matrix
  Derived t() {
    assert(shape[0] != -1 && shape[1] != -1);
    auto m = *static_cast<Derived*>(this);
    m.shape = {shape[1], shape[0]};
    return m;
  }

  __always_inline T &operator()(i32 i, i32 j) const {
    return data[i * shape[1]+ j ];
  }

  std::span<T> get_data() {
    return {reinterpret_cast<T*>(data), shape[0] * shape[1]};
  }

  void dump_shape() const {
    std::cout << "MatView: " << shape[0] << "x" << shape[1] << std::endl;
  }

  virtual void dump(int limit = -1) const {
    dump_shape();
    auto x = limit == -1 ? shape[0] : std::min(limit,shape[0]);
    auto y = limit == -1 ? shape[1] : std::min(limit, shape[1]);
    for (i32 i = 0; i < x; i++) {
      for (i32 j = 0; j < y; j++) {
        std::cout << std::to_string((*this)(i, j)) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "------------------\n";
  }
};

struct FMatrix: public Matrix<f32, FMatrix> {

  void init(tz::BinaryReader &reader) {
    assert(this->shape[0] != -1 && this->shape[1] != -1);
    auto bytes = reader.readChunk(this->shape[0] * this->shape[1] * sizeof(f32));
    this->data = (f32 *)(bytes.data());
  }

  void copy_from(FMatrix &other) {
    assert(this->shape == other.shape);
    memcpy(this->data, other.data, this->shape[0] * this->shape[1] * sizeof(f32));
  }

  FMatrix get_row(i32 i) const {
    assert(i < shape[0]);
    FMatrix m;
    m.data = data + i * shape[1];
    m.shape = {1, shape[1]};
    return m;
  }

  FMatrix get_row_chunk(i32 i, i32 offset, i32 len) const {
    assert(i < shape[0]);
    FMatrix m;
    m.data = data + i * shape[1] + offset * len;
    m.shape = {1, len};
    return m;
  }

  static FMatrix New(i32 x_dim, i32 y_dim) {
    FMatrix m;
    m.data = reinterpret_cast<f32 *>(malloc(x_dim * y_dim * sizeof(f32)));
    m.shape = {x_dim, y_dim};
    return m;
  }
};


struct QMatrix : public Matrix<i8, QMatrix> {
  f32* scales = nullptr;
  u32 group_size = 0;

  u32 get_n_groups() const {
    return shape[0] * shape[1] / group_size;
  }

  void init(tz::BinaryReader &reader) {
    assert(this->shape[0] != -1 && this->shape[1] != -1);
    auto bytes = reader.readChunk(this->shape[0] * this->shape[1] * sizeof(i8));
    this->data = (i8 *)(bytes.data());
    auto scale_bytes = reader.readChunk(get_n_groups()* sizeof(f32));
    scales = (f32 *)(scale_bytes.data());
  }

  void copy_from(QMatrix &other) {
    assert(this->shape == other.shape);
    memcpy(this->data, other.data, this->shape[0] * this->shape[1] * sizeof(i8));
    memcpy(this->scales, other.scales, get_n_groups() * sizeof(f32));
  }

  QMatrix get_row(i32 i) const {
    assert(i < shape[0]);
    QMatrix m;
    m.data = data + i * shape[1];
    m.shape = {1, shape[1]};
    m.scales = scales + i * shape[1] / group_size;
    m.group_size = group_size;
    return m;
  }

  QMatrix get_row_chunk(i32 i, i32 offset, i32 len) const {
    assert(i < shape[0]);
    QMatrix m;
    m.data = data + i * shape[1] + offset * len;
    m.scales = scales + i * shape[1] / group_size + offset / group_size;
    m.shape = {1, len};
    m.group_size = group_size;
    return m;
  }
  static QMatrix New(i32 x_dim, i32 y_dim, i32 group_size) {
    QMatrix m;
    m.data = reinterpret_cast<i8*>(malloc(x_dim * y_dim * sizeof(i8)));
    auto scales_len = x_dim * y_dim / group_size;
    m.scales = reinterpret_cast<f32 *>(malloc(scales_len * sizeof(f32)));
    m.group_size = group_size;
    m.shape = {x_dim, y_dim};
    return m;
  }

  std::span<f32> get_scales() {
    return {scales, shape[0] * shape[1] / group_size};
  }

  void dump_scales(int limit = -1) const {
    std::cout << "Scales: ";
    std::cout << get_n_groups() << "\n";
    limit = limit == -1 ? get_n_groups() : std::min<int>(limit, get_n_groups());
    for (int i = 0; i < limit; i++) {
      std::cout << scales[i] << " ";
    }
    std::cout << "\n------------------\n";
  }

  void dump(int limit = -1) const override {
    dump_shape();
    Matrix<i8, QMatrix>::dump(limit);
    if (limit == -1) {
      limit = get_n_groups();
    } else {
      limit = std::min<int>(limit, get_n_groups());
    }
    std::cout << "Scales: ";
    std::cout << get_n_groups() << "\n";
    for (int i = 0; i < limit; i++) {
      std::cout << scales[i] << " ";
    }
    std::cout << "\n------------------\n";
  }

};
