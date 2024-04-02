#pragma once
#include <cstdint>
#include <fcntl.h>
#include <format>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "cassert"

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8 = int8_t;
using i32 = int32_t;
using i64 = int64_t;
using f32 = float;
using f64 = double;


namespace tz {

#define LOG_DEBUG(fmt, ...)                                                    \
  std::cout << std::format(fmt, __VA_ARGS__) << std::endl


class BinaryReader {
public:
  BinaryReader(const u8 *data, std::size_t size)
      : data(data), size(size), pos(0) {}

  BinaryReader() : data(nullptr), size(0), pos(0) {}

  template <class T> T read();
  template <typename T> T readIntLeb();
  std::string_view readStr();
  template <class T> T peek() const;
  void advance(std::size_t count);

  [[nodiscard]] bool hasMore() const;
  std::span<const u8> readChunk(std::size_t count);

  std::size_t tell() const { return pos; }
  void seek(std::size_t offset) {
    if (offset > size) {
      throw std::runtime_error("Invalid offset");
    }
    pos = offset;
  }

private:
  const uint8_t *data;
  std::size_t size;
  std::size_t pos;
};

template <typename T>
T BinaryReader::readIntLeb() {
  static_assert(std::is_integral<T>::value && !std::is_same<T, bool>::value,
                "T must be integral");
  using U = typename std::make_unsigned<T>::type;
  uint32_t shift = 0;
  U result = 0;
  while (true) {
    assert(shift < sizeof(T) * 8);
    uint8_t value = *reinterpret_cast<const uint8_t *>(data + pos);
    result |= static_cast<U>(value & 0x7f) << shift;
    shift += 7;
    pos++;
    if ((value & 0x80) == 0) {
      if constexpr (std::is_signed<T>::value) {
        if ((value & 0x40) && shift < sizeof(T) * 8) {
          result |= (~static_cast<U>(0)) << shift;
        }
      }
      break;
    }
  }
  assert(pos <= size);
  return static_cast<T>(result);
}


template <class T>
T BinaryReader::peek() const {
  if (pos + sizeof(T) > size) {
    throw std::runtime_error("Out of data");
  }
  return *reinterpret_cast<const T *>(data + pos);
}

template <class T>
T BinaryReader::read() {
  if (pos + sizeof(T) > size) {
    throw std::runtime_error("Out of data");
  }
  T value = *reinterpret_cast<const T *>(data + pos);
  pos += sizeof(T);
  return value;
}

std::string_view BinaryReader::readStr() {
  auto length = readIntLeb<uint32_t>();
  if (pos + length > size) {
    throw std::runtime_error("cant read string, out of data");
  }
  auto str =
      std::string_view(reinterpret_cast<const char *>(data + pos), length);
  pos += length;
  return str;
}


bool BinaryReader::hasMore() const { return pos < size; }

std::span<const u8> BinaryReader::readChunk(std::size_t count) {
  if (pos + count > size) {
    throw std::runtime_error("Out of data");
  }

  auto result = std::span<const u8>(data + pos, count);
  pos += count;
  return result;
}

void BinaryReader::advance(std::size_t count) {
  if (pos + count > size) {
    throw std::runtime_error("Out of data");
  }
  pos += count;
}


class NonCopyable {
public:
  NonCopyable() = default;
  NonCopyable(const NonCopyable &) = delete;
  NonCopyable &operator=(const NonCopyable &) = delete;
};

class NonMoveable {
public:
  NonMoveable() = default;
  NonMoveable(NonMoveable &&) = delete;
  NonMoveable &operator=(NonMoveable &&) = delete;
};

class MappedFile : NonCopyable {
public:
  MappedFile(std::string_view _filename) {
    // for null termination
    std::string filename = std::string(_filename);
    fd = open(filename.data(), O_RDONLY);
    if (fd == -1) {
      throw std::runtime_error("Failed to open file: " + filename);
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
      close(fd);
      throw std::runtime_error("Failed to get file size: " + filename);
    }

    length = sb.st_size;

    addr = mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
      close(fd);
      throw std::runtime_error("Failed to map file: " + filename);
    }
  }

  MappedFile(MappedFile &&other) noexcept {
    fd = other.fd;
    addr = other.addr;
    length = other.length;
    other.fd = -1;
    other.addr = nullptr;
    other.length = 0;
  }

  MappedFile &operator=(MappedFile &&other) noexcept {
    if (this != &other) {
      fd = other.fd;
      addr = other.addr;
      length = other.length;
      other.fd = -1;
      other.addr = nullptr;
      other.length = 0;
    }
    return *this;
  }

  ~MappedFile() {
    munmap(addr, length);
    close(fd);
  }

  void dump() const {
    for (std::size_t i = 0; i < length; i++) {
      printf("%02x ", data()[i]);
    }
    printf("\n");
  }

  const uint8_t *data() const { return static_cast<const uint8_t *>(addr); }

  std::size_t size() const { return length; }

  std::span<const u8> asSpan() const { return {data(), length}; }

private:
  int fd;
  void *addr;
  std::size_t length;
};

} // namespace tz
