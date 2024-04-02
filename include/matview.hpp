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

struct MatView{
  f32 *data;
  Shape shape;
  bool muteable = false;


  void copy_from(MatView &other) {
    assert(shape == other.shape);
    assert(muteable);
    memcpy(data, other.data, shape[0] * shape[1] * sizeof(f32));
  }

  MatView transpose() const {
    assert(shape[0] != -1 && shape[1] != -1);
    return {data, {shape[1], shape[0]}, muteable};
  }

  static MatView New(i32 x_dim, i32 y_dim) {
    auto data = reinterpret_cast<f32*>(malloc(x_dim * y_dim * sizeof(f32)));
    return {data, {x_dim, y_dim}, true};
  }

  void init(tz::BinaryReader &reader) {
    assert(shape[0] != -1 && shape[1] != -1);
    auto bytes = reader.readChunk(shape[0] * shape[1] * sizeof(f32));
    data = (f32 *)(bytes.data());
  }

  __always_inline MatView get_row(i32 i) const {
    assert(i < shape[0]);
    return {data + i * shape[1], 1, shape[1l], false};
  }

  __always_inline f32 &operator()(i32 i, i32 j) const {
    return data[i * shape[1]+ j ];
  }

  void dump_shape() const {
    std::cout << "MatView: " << shape[0] << "x" << shape[1] << std::endl;
  }
  void dump(int limit = -1) const {
    dump_shape();
    auto x = limit == -1 ? shape[0] : std::min(limit,shape[0]);
    auto y = limit == -1 ? shape[1] : std::min(limit, shape[1]);
    for (i32 i = 0; i < x; i++) {
      for (i32 j = 0; j < y; j++) {
        std::cout << (*this)(i, j) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "------------------\n";
  }
};
