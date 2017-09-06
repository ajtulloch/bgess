#include <immintrin.h>
#include <stdio.h>
#include <array>
#include <iomanip>
#include <iostream>

#include <glog/logging.h>

namespace caffe2 {

constexpr size_t kDefaultAlignment = 32;

inline size_t hsum(__m256i v) {
  size_t result = 0;
  result += static_cast<size_t>(_mm256_extract_epi64(v, 0));
  result += static_cast<size_t>(_mm256_extract_epi64(v, 1));
  result += static_cast<size_t>(_mm256_extract_epi64(v, 2));
  result += static_cast<size_t>(_mm256_extract_epi64(v, 3));
  return result;
};

template <size_t M, size_t N, size_t TileDepthBytes = 32>
inline void qgess_avx2(const uint8_t* A, const uint8_t* B,
                       float* C, const size_t Cstride,
                       const size_t QK) {
  static_assert(TileDepthBytes == 32, "");
  CHECK(QK < 1024);
  A = (const uint8_t*)__builtin_assume_aligned(A, kDefaultAlignment);
  B = (const uint8_t*)__builtin_assume_aligned(B, kDefaultAlignment);
  CHECK(QK % 32 == 0);

  const __m256i lookup = _mm256_setr_epi8(
      /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
      /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
      /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
      /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4,

      /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
      /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
      /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
      /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4);

  const __m256i low_mask = _mm256_set1_epi8(0x0f);

  const __m256i zero = _mm256_setzero_si256();

  __m256i Areg[M];

#define ITER                                                             \
  {                                                                      \
    for (size_t m = 0; m < M; ++m) {                                     \
      Areg[m] = _mm256_load_si256(reinterpret_cast<const __m256i*>(A)); \
      A += 32;                                                           \
    }                                                                    \
    for (size_t n = 0; n < N; ++n) {                                     \
      const auto Breg =                                                  \
          _mm256_load_si256(reinterpret_cast<const __m256i*>(B));       \
      B += 32;                                                           \
      for (size_t m = 0; m < M; ++m) {                                   \
        const __m256i vec = _mm256_xor_si256(Areg[m], Breg);             \
        const __m256i lo = _mm256_and_si256(vec, low_mask);              \
        const __m256i hi =                                               \
            _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask);       \
        const __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);         \
        const __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);         \
        local[m][n] = _mm256_add_epi8(local[m][n], popcnt1);             \
        local[m][n] = _mm256_add_epi8(local[m][n], popcnt2);             \
      }                                                                  \
    }                                                                    \
  }

  __m256i local[M][N];

  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      local[m][n] = _mm256_setzero_si256();
    }
  }

  size_t qk = 0;
  size_t QK256Unroll = (QK / 256) * 256;
  for (; qk < QK256Unroll; qk += 256) {
    ITER ITER ITER ITER ITER ITER ITER;
  }

  for (; qk < QK; qk += 32) {
    ITER;
  }

  __m256i acc[M][N];
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      acc[m][n] = _mm256_sad_epu8(local[m][n], zero);
    }
  }
#undef ITER

  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      C[Cstride * m + n] =
          static_cast<float>(ssize_t(QK * 8) - ssize_t(2 * hsum(acc[m][n])));
    }
  }
}

}  // namespace caffe2
