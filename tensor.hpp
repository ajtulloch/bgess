#pragma once

#include <glog/logging.h>
#include <glog/stl_logging.h>

#include "eigen3/Eigen/Core"

#define CAFFE_ENFORCE_GT CHECK_GT
#define CAFFE_ENFORCE_EQ CHECK_EQ

namespace caffe2 {

// Common Eigen types that we will often use
template <typename T>
using EigenMatrixMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using EigenArrayMap =
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using EigenVectorMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenMatrixMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenVectorMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenVectorArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>;
using std::make_unique;
class CPUContext {};
class Workspace {};

namespace math {
template <class Context>
inline void CopyMatrix(const size_t item_size, const int M, const int N, const void* A,
                const int lda, void* B, const int ldb, Context* context);

template <>
inline void CopyMatrix<CPUContext>(const size_t itemsize, const int M, const int N,
                            const void* A, const int lda, void* B,
                            const int ldb, CPUContext* /*context*/) {
  if (lda == N && ldb == N) {
    // can coalese to a single memcpy of size M * N
    memcpy(static_cast<char*>(B), static_cast<const char*>(A),
           itemsize * N * M);
    return;
  }

  for (int i = 0; i < M; ++i) {
    memcpy(static_cast<char*>(B) + ldb * i * itemsize,
           static_cast<const char*>(A) + lda * i * itemsize, itemsize * N);
  }
}

}  // namespace math

template <class T, size_t kAlignment = 64>
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
  size_t dim32(size_t idx) const { return dim(idx); }

  std::vector<size_t> dims() const { return dims_; }
  void Resize(std::vector<size_t> dims) { dims_ = dims; }
  void ResizeLike(const TensorCPU& X) { Resize(X.dims()); }
  void Resize(size_t d0) { dims_ = {d0}; }
  void Resize(size_t d0, size_t d1) { dims_ = {{d0, d1}}; }
  void Resize(size_t d0, size_t d1, size_t d2, size_t d3) {
    dims_ = {{d0, d1, d2, d3}};
  }

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

  size_t nbytes() const { return data_.size(); }

  template <typename T>
  const T* data() const {
    CHECK_EQ(size() * sizeof(T), data_.size()) << dims_;
    return (const T*)data_.data();
  }

  void ShareExternalPointer(const uint8_t* data, size_t n) {
    CHECK_EQ(n, size());
    std::memcpy(mutable_data<uint8_t>(), data, n);
  }

  template <typename T>
  __attribute__((noinline)) T* mutable_data() {
    if (data_.size() != size() * sizeof(T)) {
      data_.resize(size() * sizeof(T));
    }
    CHECK_EQ(size() * sizeof(T), data_.size()) << dims_;
    return reinterpret_cast<T*>(data_.data());
  }
  size_t capacity_{0};
  std::vector<size_t> dims_;
  std::vector<uint8_t, AlignedAllocator<uint8_t>> data_;
};

}  // namespace caffe2
