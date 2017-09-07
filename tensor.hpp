#pragma once

#include <glog/logging.h>


namespace caffe2 {

template <class T, size_t kAlignment = 32>
struct AlignedAllocator {
  typedef T value_type;

  T* allocate(size_t n) {
    void* p;
    int r = posix_memalign(&p, kAlignment, n * sizeof(T));
    if (r == 0) return static_cast<T*>(p);
    throw std::bad_alloc();
  }

  void deallocate(void* p, size_t /*n*/) { free(p); }

  // std::allocator_traits generates needed rebind templates automatically
  // for allocators of type Allocator<U, Args> if Args are *type* arguments,
  // not size_t (see 20.6.8.1), so we have to write ours explicitly.
  template <class U>
  struct rebind {
    typedef AlignedAllocator<U, kAlignment> other;
  };
};

using TIndex = size_t;
class TensorCPU {
 public:
  size_t dim(size_t idx) const {
    CHECK_LT(idx, dims_.size());
    return dims_[idx];
  }
  std::vector<size_t> dims() const { return dims_; }
  void Resize(std::vector<size_t> dims) { dims_ = dims; }
  void Resize(size_t d0, size_t d1) { dims_ = {{d0, d1}}; }

  void Resize(size_t d0, size_t d1, size_t d2, size_t d3) { dims_ = {{d0, d1, d2, d3}}; }

  size_t size() const {
    size_t r = 1;
    for (auto d : dims_) {
      r *= d;
    }
    return r;
  }
  size_t ndim() const { return dims_.size(); }
  size_t size_to_dim(size_t idx) const {
    CHECK_LT(idx, dims_.size());
    size_t r = 1;
    for (auto i = 0; i < idx; ++i) {
      r *= dims_[i];
    }
    return r;
  }

  template <typename T>
  const T* data() const {
    CHECK_EQ(size() * sizeof(T), data_.size()) << dims_;
    return (const T*)data_.data();
  }

  template <typename T>
  T* mutable_data() {
    if (data_.size() != size() * sizeof(T)) {
      data_.resize(size() * sizeof(T));
    }
    CHECK_EQ(size() * sizeof(T), data_.size()) << dims_;
    return (T*)data_.data();
  }
  size_t capacity_{0};
  std::vector<size_t> dims_;
  std::vector<uint8_t, AlignedAllocator<uint8_t>> data_;
};

}
