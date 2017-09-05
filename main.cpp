#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <stdio.h>

#include <Accelerate/Accelerate.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "benchmark/benchmark.h"
#include "yuxin.hpp"


constexpr size_t kDefaultAlignment = 64;

template <class T, size_t kAlignment = kDefaultAlignment>
struct AlignedAllocator {
  typedef T value_type;

  T* allocate(size_t n) {
    void* p;
    int r = posix_memalign(&p, kAlignment, n * sizeof(T));
    if (r == 0) return static_cast<T*>(p);
    // CHECK_EQ(r, ENOMEM);
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

namespace AVX2_harley_seal {

__attribute__((always_inline)) static __m256i popcount(const __m256i vec) {
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
  const __m256i lo = _mm256_and_si256(vec, low_mask);
  const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask);
  const __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
  const __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
  return _mm256_add_epi8(popcnt1, popcnt2);
}

static __attribute__((always_inline)) void CSA256(__m256i* h, __m256i* l,
                                                  __m256i a, __m256i b,
                                                  __m256i c) {
  __m256i u = a ^ b;
  *h = (a & b) | (u & c);
  *l = u ^ c;
}

uint64_t popcnt2(const __m256i* A, const __m256i* B, const uint64_t size) {
  // ASSUME: total redictuion is less than 8192 bits (since we accumulate up to
  // 2^8 in each 8 bit unit.)
  A = (const __m256i*)__builtin_assume_aligned(A, kDefaultAlignment);
  B = (const __m256i*)__builtin_assume_aligned(B, kDefaultAlignment);
  __m256i zero = _mm256_setzero_si256();
  __m256i total = _mm256_setzero_si256();
  __m256i local = _mm256_setzero_si256();
  __m256i ones = _mm256_setzero_si256();
  __m256i twos = _mm256_setzero_si256();
  __m256i fours = _mm256_setzero_si256();
  __m256i eights = _mm256_setzero_si256();
  __m256i twosA, twosB, foursA, foursB;

  const uint64_t limit = size - size % 8;
  uint64_t i = 0;

  for (; i < limit; i += 8) {
    CSA256(&twosA, &ones, ones, A[i + 0] ^ B[i + 0], A[i + 1] ^ B[i + 1]);
    CSA256(&twosB, &ones, ones, A[i + 2] ^ B[i + 2], A[i + 3] ^ B[i + 3]);
    CSA256(&foursA, &twos, twos, twosA, twosB);
    CSA256(&twosA, &ones, ones, A[i + 4] ^ B[i + 4], A[i + 5] ^ B[i + 5]);
    CSA256(&twosB, &ones, ones, A[i + 6] ^ B[i + 6], A[i + 7] ^ B[i + 7]);
    CSA256(&foursB, &twos, twos, twosA, twosB);
    CSA256(&eights, &fours, fours, foursA, foursB);
    local = _mm256_add_epi64(local, popcount(eights));
  }
  total = _mm256_sad_epu8(local, zero);
  total = _mm256_slli_epi64(total, 3);  // * 8
  total = _mm256_add_epi64(
      total, _mm256_slli_epi64(_mm256_sad_epu8(popcount(fours), zero),
                               2));  // += 4 * ...
  total = _mm256_add_epi64(
      total, _mm256_slli_epi64(_mm256_sad_epu8(popcount(twos), zero),
                               1));  // += 2 * ...
  total = _mm256_add_epi64(total, _mm256_sad_epu8(popcount(ones), zero));

  for (; i < size; i++) total = _mm256_add_epi64(total, popcount(A[i] ^ B[i]));

  return static_cast<uint64_t>(_mm256_extract_epi64(total, 0)) +
         static_cast<uint64_t>(_mm256_extract_epi64(total, 1)) +
         static_cast<uint64_t>(_mm256_extract_epi64(total, 2)) +
         static_cast<uint64_t>(_mm256_extract_epi64(total, 3));
}

uint64_t popcnt(const __m256i* A, const __m256i* B, const uint64_t size) {
  // ASSUME: size < 8192 (since we accumulate up to 2^8 in each 8 bit unit,
  // which is )
  A = (const __m256i*)__builtin_assume_aligned(A, kDefaultAlignment);
  B = (const __m256i*)__builtin_assume_aligned(B, kDefaultAlignment);
  __m256i zero = _mm256_setzero_si256();
  __m256i local = _mm256_setzero_si256();
  __m256i total = _mm256_setzero_si256();
  __m256i ones = _mm256_setzero_si256();
  __m256i twos = _mm256_setzero_si256();
  __m256i fours = _mm256_setzero_si256();
  __m256i eights = _mm256_setzero_si256();
  __m256i sixteens = _mm256_setzero_si256();
  __m256i twosA, twosB, foursA, foursB, eightsA, eightsB;

  const uint64_t limit = size - size % 16;
  uint64_t i = 0;

  for (; i < limit; i += 16) {
    CSA256(&twosA, &ones, ones, A[i + 0] ^ B[i + 0], A[i + 1] ^ B[i + 1]);
    CSA256(&twosB, &ones, ones, A[i + 2] ^ B[i + 2], A[i + 3] ^ B[i + 3]);
    CSA256(&foursA, &twos, twos, twosA, twosB);
    CSA256(&twosA, &ones, ones, A[i + 4] ^ B[i + 4], A[i + 5] ^ B[i + 5]);
    CSA256(&twosB, &ones, ones, A[i + 6] ^ B[i + 6], A[i + 7] ^ B[i + 7]);
    CSA256(&foursB, &twos, twos, twosA, twosB);
    CSA256(&eightsA, &fours, fours, foursA, foursB);
    CSA256(&twosA, &ones, ones, A[i + 8] ^ B[i + 8], A[i + 9] ^ B[i + 9]);
    CSA256(&twosB, &ones, ones, A[i + 10] ^ B[i + 10], A[i + 11] ^ B[i + 11]);
    CSA256(&foursA, &twos, twos, twosA, twosB);
    CSA256(&twosA, &ones, ones, A[i + 12] ^ B[i + 12], A[i + 13] ^ B[i + 13]);
    CSA256(&twosB, &ones, ones, A[i + 14] ^ B[i + 14], A[i + 15] ^ B[i + 15]);
    CSA256(&foursB, &twos, twos, twosA, twosB);
    CSA256(&eightsB, &fours, fours, foursA, foursB);
    CSA256(&sixteens, &eights, eights, eightsA, eightsB);
    local = _mm256_add_epi64(local, popcount(sixteens));
  }

  total = _mm256_sad_epu8(local, zero);
  total = _mm256_slli_epi64(total, 4);  // * 16
  total = _mm256_add_epi64(
      total, _mm256_slli_epi64(_mm256_sad_epu8(popcount(eights), zero),
                               3));  // += 8 * ...
  total = _mm256_add_epi64(
      total, _mm256_slli_epi64(_mm256_sad_epu8(popcount(fours), zero),
                               2));  // += 4 * ...
  total = _mm256_add_epi64(
      total, _mm256_slli_epi64(_mm256_sad_epu8(popcount(twos), zero),
                               1));  // += 2 * ...
  total = _mm256_add_epi64(total, _mm256_sad_epu8(popcount(ones), zero));

  for (; i < size; i++) total = _mm256_add_epi64(total, popcount(A[i] ^ B[i]));

  return static_cast<uint64_t>(_mm256_extract_epi64(total, 0)) +
         static_cast<uint64_t>(_mm256_extract_epi64(total, 1)) +
         static_cast<uint64_t>(_mm256_extract_epi64(total, 2)) +
         static_cast<uint64_t>(_mm256_extract_epi64(total, 3));
}

}  // namespace AVX2_harley_seal

uint64_t xnor_popcnt_AVX2_harley_seal(const uint8_t* A, const uint8_t* B,
                                      const size_t size) {
  assert(size % 32 == 0);
  uint64_t total =
      AVX2_harley_seal::popcnt((const __m256i*)A, (const __m256i*)B, size / 32);
  return total;
}

uint64_t xnor_popcnt_AVX2_harley_seal2(const uint8_t* A, const uint8_t* B,
                                       const size_t size) {
  assert(size % 32 == 0);
  uint64_t total = AVX2_harley_seal::popcnt2((const __m256i*)A,
                                             (const __m256i*)B, size / 32);
  return total;
}

__attribute__((noinline)) void xnor_popcnt_AVX2_lookup2x2(
    const uint8_t* A0, const uint8_t* A1, const uint8_t* B0, const uint8_t* B1,
    uint32_t* C, const size_t Cstride, const size_t n) {
  assert(n % 32 == 0);
  size_t i = 0;
  A0 = (const uint8_t*)__builtin_assume_aligned(A0, kDefaultAlignment);
  A1 = (const uint8_t*)__builtin_assume_aligned(A1, kDefaultAlignment);
  B0 = (const uint8_t*)__builtin_assume_aligned(B0, kDefaultAlignment);
  B1 = (const uint8_t*)__builtin_assume_aligned(B1, kDefaultAlignment);

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

  __m256i local00 = _mm256_setzero_si256();
  __m256i local01 = _mm256_setzero_si256();
  __m256i local10 = _mm256_setzero_si256();
  __m256i local11 = _mm256_setzero_si256();

#define ITER                                                         \
  {                                                                  \
    const __m256i A0vec =                                            \
        _mm256_load_si256(reinterpret_cast<const __m256i*>(A0 + i)); \
    const __m256i B0vec =                                            \
        _mm256_load_si256(reinterpret_cast<const __m256i*>(B0 + i)); \
    const __m256i A1vec =                                            \
        _mm256_load_si256(reinterpret_cast<const __m256i*>(A1 + i)); \
    const __m256i B1vec =                                            \
        _mm256_load_si256(reinterpret_cast<const __m256i*>(B1 + i)); \
    const __m256i vec00 = _mm256_xor_si256(A0vec, B0vec);            \
    const __m256i vec01 = _mm256_xor_si256(A0vec, B1vec);            \
    const __m256i vec10 = _mm256_xor_si256(A1vec, B0vec);            \
    const __m256i vec11 = _mm256_xor_si256(A1vec, B1vec);            \
    const __m256i lo00 = _mm256_and_si256(vec00, low_mask);          \
    const __m256i lo01 = _mm256_and_si256(vec01, low_mask);          \
    const __m256i lo10 = _mm256_and_si256(vec10, low_mask);          \
    const __m256i lo11 = _mm256_and_si256(vec11, low_mask);          \
    const __m256i hi00 =                                             \
        _mm256_and_si256(_mm256_srli_epi16(vec00, 4), low_mask);     \
    const __m256i hi01 =                                             \
        _mm256_and_si256(_mm256_srli_epi16(vec01, 4), low_mask);     \
    const __m256i hi10 =                                             \
        _mm256_and_si256(_mm256_srli_epi16(vec10, 4), low_mask);     \
    const __m256i hi11 =                                             \
        _mm256_and_si256(_mm256_srli_epi16(vec11, 4), low_mask);     \
    const __m256i popcntl00 = _mm256_shuffle_epi8(lookup, lo00);     \
    const __m256i popcntl01 = _mm256_shuffle_epi8(lookup, lo01);     \
    const __m256i popcntl10 = _mm256_shuffle_epi8(lookup, lo10);     \
    const __m256i popcntl11 = _mm256_shuffle_epi8(lookup, lo11);     \
    const __m256i popcnth00 = _mm256_shuffle_epi8(lookup, hi00);     \
    const __m256i popcnth01 = _mm256_shuffle_epi8(lookup, hi01);     \
    const __m256i popcnth10 = _mm256_shuffle_epi8(lookup, hi10);     \
    const __m256i popcnth11 = _mm256_shuffle_epi8(lookup, hi11);     \
    local00 = _mm256_add_epi8(local00, popcntl00);                   \
    local00 = _mm256_add_epi8(local00, popcnth00);                   \
    local01 = _mm256_add_epi8(local01, popcntl01);                   \
    local01 = _mm256_add_epi8(local01, popcnth01);                   \
    local10 = _mm256_add_epi8(local10, popcntl10);                   \
    local10 = _mm256_add_epi8(local10, popcnth10);                   \
    local11 = _mm256_add_epi8(local11, popcntl11);                   \
    local11 = _mm256_add_epi8(local11, popcnth11);                   \
    i += 32;                                                         \
  }

  while (i + 8 * 32 <= n) {
    ITER ITER ITER ITER ITER ITER ITER ITER;
  }

  while (i + 32 <= n) {
    ITER;
  }

  const auto acc00 = _mm256_sad_epu8(local00, _mm256_setzero_si256());
  const auto acc01 = _mm256_sad_epu8(local01, _mm256_setzero_si256());
  const auto acc10 = _mm256_sad_epu8(local10, _mm256_setzero_si256());
  const auto acc11 = _mm256_sad_epu8(local11, _mm256_setzero_si256());

#undef ITER
  // 4x32 each, [:128]
  auto r = [](__m256i v) -> uint32_t {
    uint32_t result = 0;
    result += static_cast<uint32_t>(_mm256_extract_epi64(v, 0));
    result += static_cast<uint32_t>(_mm256_extract_epi64(v, 1));
    result += static_cast<uint32_t>(_mm256_extract_epi64(v, 2));
    result += static_cast<uint32_t>(_mm256_extract_epi64(v, 3));
    return result;
  };
  C[0 * Cstride + 0] = r(acc00);
  C[0 * Cstride + 1] = r(acc01);
  C[1 * Cstride + 0] = r(acc10);
  C[1 * Cstride + 1] = r(acc11);
}

template <size_t M, size_t N>
__attribute__((noinline)) void xnor_popcnt_AVX2_lookupMxN(const uint8_t* A,
                                                          const uint8_t* B,
                                                          uint32_t* C,
                                                          const size_t Cstride,
                                                          const size_t qk) {
  A = (const uint8_t*)__builtin_assume_aligned(A, kDefaultAlignment);
  B = (const uint8_t*)__builtin_assume_aligned(B, kDefaultAlignment);
  assert(qk % 32 == 0);
  size_t i = 0;

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
      Areg[m] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(A)); \
      A += 32;                                                           \
    }                                                                    \
    for (size_t n = 0; n < N; ++n) {                                     \
      const auto Breg =                                                  \
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(B));       \
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
    i += 32;                                                             \
  }

  __m256i local[M][N];
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      local[m][n] = _mm256_setzero_si256();
    }
  }

  while (i + 8 * 32 <= qk) {
    ITER ITER ITER ITER ITER ITER ITER;
  }

  while (i + 32 <= qk) {
    ITER;
  }

  __m256i acc[M][N];
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      acc[m][n] = _mm256_sad_epu8(local[m][n], zero);
    }
  }
#undef ITER

  auto r = [](__m256i v) -> uint32_t {
    uint32_t result = 0;
    result += static_cast<uint32_t>(_mm256_extract_epi64(v, 0));
    result += static_cast<uint32_t>(_mm256_extract_epi64(v, 1));
    result += static_cast<uint32_t>(_mm256_extract_epi64(v, 2));
    result += static_cast<uint32_t>(_mm256_extract_epi64(v, 3));
    return result;
  };

  for (size_t m = 0; m < M; ++m) {
    // merge vectors
    // if (N % 8 == 0) {
    //   // static_assert(N % 8 == 0, "");
    //   for (size_t n = 0; n < N; n += 8) {
    //     const __m256i n01 = __builtin_shufflevector(
    //         (__v32qi)acc[m][n + 0], (__v32qi)acc[m][n + 1], 0, 1, 8, 9, 16,
    //         17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57, -1, -1, -1, -1, -1,
    //         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    //     const __m256i n23 = __builtin_shufflevector(
    //         (__v32qi)acc[m][n + 2], (__v32qi)acc[m][n + 3], 0, 1, 8, 9, 16,
    //         17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57, -1, -1, -1, -1, -1,
    //         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    //     const __m256i n45 = __builtin_shufflevector(
    //         (__v32qi)acc[m][n + 4], (__v32qi)acc[m][n + 5], 0, 1, 8, 9, 16,
    //         17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57, -1, -1, -1, -1, -1,
    //         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    //     const __m256i n67 = __builtin_shufflevector(
    //         (__v32qi)acc[m][n + 6], (__v32qi)acc[m][n + 7], 0, 1, 8, 9, 16,
    //         17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57, -1, -1, -1, -1, -1,
    //         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    //     const __m256i n0123 = __builtin_shufflevector(
    //         (__v32qi)n01, (__v32qi)n23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    //         12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
    //         44, 45, 46, 47);
    //     const __m256i n4567 = __builtin_shufflevector(
    //         (__v32qi)n45, (__v32qi)n67, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    //         12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
    //         44, 45, 46, 47);
    //     const __m256i n0011223344556677 = _mm256_hadd_epi16(n0123, n4567);
    //     const __m256i n0123456700000000 =
    //         _mm256_hadd_epi16(n0011223344556677, zero);
    //     const __m128i n01234567 = __builtin_shufflevector(
    //         (__v16hi)n0123456700000000, (__v16hi)n0123456700000000, 0, 1, 2,
    //         3, 4, 5, 6, 7);
    //     const __m256i n01234567v = _mm256_cvtepi16_epi32(n01234567);
    //     _mm256_store_si256((__m256i*)(C + Cstride * m + n), n01234567v);
    //     // acc[m][n]
    //   }
    // } else {
    for (size_t n = 0; n < N; ++n) {
      C[Cstride * m + n] = r(acc[m][n]);
    }
    // }
  }
  return;
}

std::uint64_t xnor_popcnt_AVX2_lookup(const uint8_t* A, const uint8_t* B,
                                      const size_t n) {
  assert(n % 32 == 0);
  size_t i = 0;

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

  __m256i acc = _mm256_setzero_si256();

#define ITER                                                                  \
  {                                                                           \
    const __m256i Avec =                                                      \
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(A + i));          \
    const __m256i Bvec =                                                      \
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(B + i));          \
    const __m256i vec = _mm256_xor_si256(Avec, Bvec);                         \
    const __m256i lo = _mm256_and_si256(vec, low_mask);                       \
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask); \
    const __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);                  \
    const __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);                  \
    local = _mm256_add_epi8(local, popcnt1);                                  \
    local = _mm256_add_epi8(local, popcnt2);                                  \
    i += 32;                                                                  \
  }

  while (i + 8 * 32 <= n) {
    __m256i local = _mm256_setzero_si256();
    ITER ITER ITER ITER ITER ITER ITER ITER acc =
        _mm256_add_epi64(acc, _mm256_sad_epu8(local, _mm256_setzero_si256()));
  }

  __m256i local = _mm256_setzero_si256();

  while (i + 32 <= n) {
    ITER;
  }

  acc = _mm256_add_epi64(acc, _mm256_sad_epu8(local, _mm256_setzero_si256()));

#undef ITER

  uint64_t result = 0;

  result += static_cast<uint64_t>(_mm256_extract_epi64(acc, 0));
  result += static_cast<uint64_t>(_mm256_extract_epi64(acc, 1));
  result += static_cast<uint64_t>(_mm256_extract_epi64(acc, 2));
  result += static_cast<uint64_t>(_mm256_extract_epi64(acc, 3));
  return result;
}

std::uint64_t popcnt_AVX2_lookup(const uint8_t* data, const size_t n) {
  assert(n % 32 == 0);
  size_t i = 0;

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

  __m256i acc = _mm256_setzero_si256();

#define ITER                                                                  \
  {                                                                           \
    const __m256i vec =                                                       \
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));       \
    const __m256i lo = _mm256_and_si256(vec, low_mask);                       \
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask); \
    const __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);                  \
    const __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);                  \
    local = _mm256_add_epi8(local, popcnt1);                                  \
    local = _mm256_add_epi8(local, popcnt2);                                  \
    i += 32;                                                                  \
  }

  while (i + 8 * 32 <= n) {
    __m256i local = _mm256_setzero_si256();
    ITER ITER ITER ITER ITER ITER ITER ITER acc =
        _mm256_add_epi64(acc, _mm256_sad_epu8(local, _mm256_setzero_si256()));
  }

  __m256i local = _mm256_setzero_si256();

  while (i + 32 <= n) {
    ITER;
  }

  acc = _mm256_add_epi64(acc, _mm256_sad_epu8(local, _mm256_setzero_si256()));

#undef ITER

  uint64_t result = 0;

  result += static_cast<uint64_t>(_mm256_extract_epi64(acc, 0));
  result += static_cast<uint64_t>(_mm256_extract_epi64(acc, 1));
  result += static_cast<uint64_t>(_mm256_extract_epi64(acc, 2));
  result += static_cast<uint64_t>(_mm256_extract_epi64(acc, 3));
  return result;
}

template <size_t M, size_t N>
__attribute__((noinline)) void qgess(const uint8_t* A, const uint8_t* B,
                                     uint32_t* C, size_t Cstride, size_t QK) {
  const size_t QK32Unroll = (QK / 32) * 32;
  const size_t QK8Unroll = (QK / 8) * 8;
  const size_t QK4Unroll = (QK / 4) * 4;
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      size_t acc = 0;
      size_t qk = 0;
      for (; qk < QK32Unroll; qk += 32) {
        acc += _popcnt64(*((uint64_t*)(&A[m * QK + qk + 0])) ^
                         *((uint64_t*)(&B[n * QK + qk + 0])));
        acc += _popcnt64(*((uint64_t*)(&A[m * QK + qk + 8])) ^
                         *((uint64_t*)(&B[n * QK + qk + 8])));
        acc += _popcnt64(*((uint64_t*)(&A[m * QK + qk + 16])) ^
                         *((uint64_t*)(&B[n * QK + qk + 16])));
        acc += _popcnt64(*((uint64_t*)(&A[m * QK + qk + 24])) ^
                         *((uint64_t*)(&B[n * QK + qk + 24])));
      }

      for (; qk < QK8Unroll; qk += 8) {
        acc += _popcnt64(*((uint64_t*)(&A[m * QK + qk])) ^
                         *((uint64_t*)(&B[n * QK + qk])));
      }
      for (; qk < QK4Unroll; ++qk) {
        acc += _popcnt32(*((uint32_t*)(&A[m * QK + qk])) ^
                         *((uint32_t*)(&B[n * QK + qk])));
      }
      for (; qk < QK; ++qk) {
        acc += _popcnt32(A[m * QK + qk] ^ B[n * QK + qk]);
      }
      C[m * Cstride + n] = acc;
    }
  }
}

__attribute__((noinline)) void qxnor_popcnt(const uint8_t* A, const uint8_t* B,
                                            uint32_t* C, size_t M, size_t N,
                                            size_t Cstride, size_t QK) {
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      size_t acc = 0;
      acc += xnor_popcnt_AVX2_lookup(&A[m * QK], &B[n * QK], QK);
      C[m * Cstride + n] = acc;
    }
  }
}

__attribute__((noinline)) void qxnor_popcnt_hs(const uint8_t* A,
                                               const uint8_t* B, uint32_t* C,
                                               size_t M, size_t N,
                                               size_t Cstride, size_t QK) {
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      size_t acc = 0;
      acc += xnor_popcnt_AVX2_harley_seal(&A[m * QK], &B[n * QK], QK);
      C[m * Cstride + n] = acc;
    }
  }
}

template <size_t MM, size_t NN>
__attribute__((noinline)) void qxnor_popcnt_mxn(const uint8_t* A,
                                                const uint8_t* B, uint32_t* C,
                                                size_t M, size_t N,
                                                size_t Cstride, size_t QK) {
  for (size_t m = 0; m < M; m += MM) {
    for (size_t n = 0; n < N; n += NN) {
      xnor_popcnt_AVX2_lookupMxN<MM, NN>(&A[m * QK], &B[n * QK],
                                         &C[m * Cstride + n], Cstride, QK);
    }
  }
}

__attribute__((noinline)) void qxnor_popcnt_hs2(const uint8_t* A,
                                                const uint8_t* B, uint32_t* C,
                                                size_t M, size_t N, size_t Cstride, size_t QK) {
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      size_t acc = 0;
      acc += xnor_popcnt_AVX2_harley_seal2(&A[m * QK], &B[n * QK], QK);
      C[m * Cstride + n] = acc;
    }
  }
}

__attribute__((noinline)) void qxnor_popcnt2x2(const uint8_t* A,
                                               const uint8_t* B, uint32_t* C,
                                               size_t M, size_t N,
                                               size_t Cstride, size_t QK) {
  for (size_t m = 0; m < M; m += 2) {
    for (size_t n = 0; n < N; n += 2) {
      xnor_popcnt_AVX2_lookup2x2(&A[m * QK], &A[(m + 1) * QK], &B[n * QK],
                                 &B[(n + 1) * QK], &C[m * Cstride + n], N, QK);
    }
  }
}

// TEST(bgess, qgess) {
//   constexpr size_t M = 20;
//   constexpr size_t N = 40;
//   std::vector<uint8_t, AlignedAllocator<uint8_t>> A(M * QK);
//   for (auto i = 0; i < A.size(); ++i) {
//     A[i] = rand();
//   }
//   std::vector<uint8_t, AlignedAllocator<uint8_t>> B(N * QK);
//   for (auto i = 0; i < B.size(); ++i) {
//     B[i] = rand();
//   }
//   std::vector<uint32_t> C(M * N);
//   std::vector<uint32_t> CR(M * N);
//   qgess<M, N>(A.data(), B.data(), C.data(), N, QK);
//   qgess<M, N>(A.data(), B.data(), CR.data(), N, QK);
//   for (auto i = 0; i < C.size(); ++i) {
//     CHECK_EQ(i, )
//   }
// }

template <size_t M, size_t N>
static void BM_qgess(benchmark::State& state) {
  size_t QK = state.range(0);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> A(M * QK);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> B(N * QK);
  std::vector<uint32_t> C(M * N);
  size_t iters = 0;
  while (state.KeepRunning()) {
    qgess<M, N>(A.data(), B.data(), C.data(), N, QK);
    ++iters;
  }
  state.counters["FLOPS"] = benchmark::Counter(2 * M * N * QK * 8 * iters,
                                               benchmark::Counter::kIsRate);
  // runXnorBenchmark();
}

template <size_t M, size_t N>
static void BM_qxnor_popcnt(benchmark::State& state) {
  size_t QK = state.range(0);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> A(M * QK);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> B(N * QK);
  std::vector<uint32_t> C(M * N);

  size_t iters = 0;
  while (state.KeepRunning()) {
    qxnor_popcnt(A.data(), B.data(), C.data(), M, N, N, QK);
    ++iters;
  }

  state.counters["FLOPS"] = benchmark::Counter(2 * M * N * QK * 8 * iters,
                                               benchmark::Counter::kIsRate);
}

template <size_t M, size_t N>
static void BM_qxnor_popcnt2x2(benchmark::State& state) {
  size_t QK = state.range(0);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> A(M * QK);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> B(N * QK);
  std::vector<uint32_t> C(M * N);

  size_t iters = 0;
  while (state.KeepRunning()) {
    qxnor_popcnt2x2(A.data(), B.data(), C.data(), M, N, N, QK);
    ++iters;
  }

  state.counters["FLOPS"] = benchmark::Counter(2 * M * N * QK * 8 * iters,
                                               benchmark::Counter::kIsRate);
}

template <size_t M, size_t N>
static void BM_qxnor_popcnt2x2_conv3x3(benchmark::State& state) {
  size_t QK = state.range(0);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> A(M * 3 * 3 * QK);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> B(N * 3 * 3 * QK);
  std::vector<uint32_t> C(M * N);

  size_t iters = 0;
  while (state.KeepRunning()) {
    qxnor_popcnt2x2(A.data(), B.data(), C.data(), M, N, N, QK * 3 * 3);
    ++iters;
  }

  state.counters["FLOPS"] = benchmark::Counter(
      2 * M * N * QK * 8 * 3 * 3 * iters, benchmark::Counter::kIsRate);
}

template <size_t M, size_t N>
static void BM_qxnor_popcnt_hs(benchmark::State& state) {
  size_t QK = state.range(0);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> A(M * QK);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> B(N * QK);
  std::vector<uint32_t> C(M * N);

  size_t iters = 0;
  while (state.KeepRunning()) {
    qxnor_popcnt_hs(A.data(), B.data(), C.data(), M, N, N, QK);
    ++iters;
  }

  state.counters["FLOPS"] = benchmark::Counter(2 * M * N * QK * 8 * iters,
                                               benchmark::Counter::kIsRate);
}

template <size_t M, size_t N>
static void BM_qxnor_popcnt_hs2(benchmark::State& state) {
  size_t QK = state.range(0);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> A(M * QK);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> B(N * QK);
  std::vector<uint32_t> C(M * N);

  size_t iters = 0;
  while (state.KeepRunning()) {
    qxnor_popcnt_hs2(A.data(), B.data(), C.data(), M, N, N, QK);
    ++iters;
  }

  state.counters["FLOPS"] = benchmark::Counter(2 * M * N * QK * 8 * iters,
                                               benchmark::Counter::kIsRate);
}

template <size_t M, size_t N, size_t MM, size_t NN>
static void BM_qxnor_popcnt_mxn(benchmark::State& state) {
  size_t QK = state.range(0);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> A(M * QK);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> B(N * QK);
  std::vector<uint32_t> C(M * N);

  size_t iters = 0;
  while (state.KeepRunning()) {
    qxnor_popcnt_mxn<MM, NN>(A.data(), B.data(), C.data(), M, N, N, QK);
    ++iters;
  }

  state.counters["FLOPS"] = benchmark::Counter(2 * M * N * QK * 8 * iters,
                                               benchmark::Counter::kIsRate);
}

template <size_t M, size_t N>
static void BM_yuxin_conv(benchmark::State& state) {
  size_t QK = state.range(0);
  BitTensor X(1, 1, M, QK * 8);
  BitTensor W(N, 1, 1, QK * 8);
  BitTensor Y(1, 1, M, N);
  size_t iters = 0;
  while (state.KeepRunning()) {
    const size_t ops = conv(W, X, Y);
    // if (ops != 2 * M * N * QK * 8) {
    //   std::cerr << "Ops: " << ops << ", expected: " << 2 * M * N * QK * 8 <<
    //   std::endl; throw std::runtime_error("Mismatch");
    // }
    ++iters;
  }

  state.counters["FLOPS"] = benchmark::Counter(2 * M * N * QK * 8 * iters,
                                               benchmark::Counter::kIsRate);
}

size_t divRoundUp(size_t a, size_t b) { return (a + (b - 1)) / b; }

template <size_t M, size_t N>
static void BM_yuxin_conv3x3(benchmark::State& state) {
  size_t QK = state.range(0);
  BitTensor X(1, M, M, QK * 8);
  BitTensor W(N, 3, 3, QK * 8);
  BitTensor Y(1, M, M, N);
  size_t iters = 0;
  while (state.KeepRunning()) {
    const size_t ops = conv(W, X, Y);
    // if (ops != 2 * M * M * N * QK * 8 * 3 * 3) {
    //   std::cerr << "Ops: " << ops << ", expected: " << 2 * M * M * N * QK * 8
    //   * 3 * 3 << std::endl; throw std::runtime_error("Mismatch");
    // }
    ++iters;
  }

  state.counters["FLOPS"] = benchmark::Counter(
      2 * M * M * N * QK * 8 * 3 * 3 * iters, benchmark::Counter::kIsRate);
}

template <size_t M, size_t N>
static void BM_sgemm(benchmark::State& state) {
  size_t QK = state.range(0);
  std::vector<float, AlignedAllocator<float>> A(M * QK * 8);
  std::vector<float, AlignedAllocator<float>> B(N * QK * 8);
  std::vector<float> C(M * N);

  size_t iters = 0;
  while (state.KeepRunning()) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, QK * 8, 1.0,
                A.data(), QK * 8, B.data(), QK * 8, 0.0, C.data(), N);
    ++iters;
  }

  state.counters["FLOPS"] = benchmark::Counter(2 * M * N * QK * 8 * iters,
                                               benchmark::Counter::kIsRate);
}


constexpr size_t kQKLowerBound = 64;
constexpr size_t kQKUpperBound = 16384;

BENCHMARK_TEMPLATE2(BM_qgess, 2, 2)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);

BENCHMARK_TEMPLATE2(BM_qxnor_popcnt, 2, 2)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE2(BM_qxnor_popcnt, 4, 4)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE2(BM_qxnor_popcnt, 8, 8)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE2(BM_qxnor_popcnt, 16, 16)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE2(BM_qxnor_popcnt, 32, 32)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE2(BM_qxnor_popcnt, 64, 64)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);

BENCHMARK_TEMPLATE2(BM_qxnor_popcnt_hs, 2, 2)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE2(BM_qxnor_popcnt_hs, 4, 4)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE2(BM_qxnor_popcnt_hs, 8, 8)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE2(BM_qxnor_popcnt_hs, 16, 16)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE2(BM_qxnor_popcnt_hs, 32, 32)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE2(BM_qxnor_popcnt_hs, 64, 64)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);

BENCHMARK_TEMPLATE2(BM_qxnor_popcnt_hs2, 2, 2)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE2(BM_qxnor_popcnt_hs2, 4, 4)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE2(BM_qxnor_popcnt_hs2, 8, 8)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE2(BM_qxnor_popcnt_hs2, 16, 16)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE2(BM_qxnor_popcnt_hs2, 32, 32)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE2(BM_qxnor_popcnt_hs2, 64, 64)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);

BENCHMARK_TEMPLATE2(BM_qxnor_popcnt2x2, 2, 2)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE2(BM_qxnor_popcnt2x2, 4, 4)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE2(BM_qxnor_popcnt2x2, 8, 8)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE2(BM_qxnor_popcnt2x2, 16, 16)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE2(BM_qxnor_popcnt2x2, 32, 32)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE2(BM_qxnor_popcnt2x2, 64, 64)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);

#define BENCHMARK_TEMPLATE4(n, a, b, c, d)               \
  BENCHMARK_PRIVATE_DECLARE(n) =                         \
      (::benchmark::internal::RegisterBenchmarkInternal( \
          new ::benchmark::internal::FunctionBenchmark(  \
              #n "<" #a "," #b "," #c "," #d ">", n<a, b, c, d>)))

BENCHMARK_TEMPLATE4(BM_qxnor_popcnt_mxn, 2, 2, 2, 2)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE4(BM_qxnor_popcnt_mxn, 4, 4, 2, 2)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE4(BM_qxnor_popcnt_mxn, 8, 8, 2, 2)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE4(BM_qxnor_popcnt_mxn, 16, 16, 2, 2)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE4(BM_qxnor_popcnt_mxn, 32, 32, 2, 2)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE4(BM_qxnor_popcnt_mxn, 64, 64, 2, 2)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);

BENCHMARK_TEMPLATE4(BM_qxnor_popcnt_mxn, 2, 256, 2, 2)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE4(BM_qxnor_popcnt_mxn, 2, 512, 2, 2)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE4(BM_qxnor_popcnt_mxn, 2, 1024, 2, 2)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);

BENCHMARK_TEMPLATE4(BM_qxnor_popcnt_mxn, 4, 4, 4, 4)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE4(BM_qxnor_popcnt_mxn, 8, 8, 4, 4)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE4(BM_qxnor_popcnt_mxn, 16, 16, 4, 4)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE4(BM_qxnor_popcnt_mxn, 32, 32, 4, 4)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE4(BM_qxnor_popcnt_mxn, 64, 64, 4, 4)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);

BENCHMARK_TEMPLATE4(BM_qxnor_popcnt_mxn, 3, 4, 3, 4)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE4(BM_qxnor_popcnt_mxn, 9, 8, 3, 4)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE4(BM_qxnor_popcnt_mxn, 15, 16, 3, 4)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE4(BM_qxnor_popcnt_mxn, 33, 32, 3, 4)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE4(BM_qxnor_popcnt_mxn, 66, 64, 3, 4)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);

BENCHMARK_TEMPLATE4(BM_qxnor_popcnt_mxn, 2, 3, 2, 3)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE4(BM_qxnor_popcnt_mxn, 8, 9, 2, 3)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE4(BM_qxnor_popcnt_mxn, 16, 15, 2, 3)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE4(BM_qxnor_popcnt_mxn, 32, 33, 2, 3)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE4(BM_qxnor_popcnt_mxn, 64, 66, 2, 3)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);

// BENCHMARK_TEMPLATE2(BM_yuxin_conv, 8, 8)
//     ->RangeMultiplier(2)
//     ->Range(kQKLowerBound, kQKUpperBound);
// BENCHMARK_TEMPLATE2(BM_yuxin_conv, 16, 16)
//     ->RangeMultiplier(2)
//     ->Range(kQKLowerBound, kQKUpperBound);
// BENCHMARK_TEMPLATE2(BM_yuxin_conv, 32, 32)
//     ->RangeMultiplier(2)
//     ->Range(kQKLowerBound, kQKUpperBound);
// BENCHMARK_TEMPLATE2(BM_yuxin_conv, 64, 64)
//     ->RangeMultiplier(2)
//     ->Range(kQKLowerBound, kQKUpperBound);

// BENCHMARK_TEMPLATE2(BM_qxnor_popcnt2x2_conv3x3, 16, 16)
//     ->RangeMultiplier(2)
//     ->Range(kQKLowerBound, kQKUpperBound / 2);
// BENCHMARK_TEMPLATE2(BM_qxnor_popcnt2x2_conv3x3, 32, 32)
//     ->RangeMultiplier(2)
//     ->Range(kQKLowerBound, kQKUpperBound / 2);
// BENCHMARK_TEMPLATE2(BM_qxnor_popcnt2x2_conv3x3, 64, 64)
//     ->RangeMultiplier(2)
//     ->Range(kQKLowerBound, kQKUpperBound / 2);

// BENCHMARK_TEMPLATE2(BM_yuxin_conv3x3, 16, 16)
//     ->RangeMultiplier(2)
//     ->Range(kQKLowerBound, kQKUpperBound / 2);
// BENCHMARK_TEMPLATE2(BM_yuxin_conv3x3, 32, 32)
//     ->RangeMultiplier(2)
//     ->Range(kQKLowerBound, kQKUpperBound / 2);
// BENCHMARK_TEMPLATE2(BM_yuxin_conv3x3, 64, 64)
//     ->RangeMultiplier(2)
//     ->Range(kQKLowerBound, kQKUpperBound / 2);

// BENCHMARK_TEMPLATE2(BM_qxnor_popcnt2x2, 256, 256)
//     ->RangeMultiplier(2)
//     ->Range(kQKLowerBound, kQKUpperBound);

BENCHMARK_TEMPLATE2(BM_sgemm, 32, 32)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE2(BM_sgemm, 64, 64)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  CHECK_EQ(RUN_ALL_TESTS(), 0);
  benchmark::Initialize(&argc, argv);

  benchmark::RunSpecifiedBenchmarks();
}

//BENCHMARK_MAIN();
