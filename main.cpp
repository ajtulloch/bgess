#include <Accelerate/Accelerate.h>
#include <immintrin.h>
#include <stdio.h>
#include <iomanip>
#include <iostream>
#include "benchmark/benchmark.h"

constexpr size_t kDefaultAlignment = 64;

template <class T, size_t kAlignment=kDefaultAlignment>
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

__attribute__((always_inline)) static __m256i popcount(const __m256i v) {
  __m256i lookup1 =
      _mm256_setr_epi8(4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8, 4, 5, 5,
                       6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8);

  __m256i lookup2 =
      _mm256_setr_epi8(4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0, 4, 3, 3,
                       2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);

  __m256i low_mask = _mm256_set1_epi8(0x0f);
  __m256i lo = v & low_mask;
  __m256i hi = _mm256_srli_epi16(v, 4) & low_mask;
  __m256i popcnt1 = _mm256_shuffle_epi8(lookup1, lo);
  __m256i popcnt2 = _mm256_shuffle_epi8(lookup2, hi);
  return _mm256_sad_epu8(popcnt1, popcnt2);
}

static __attribute__((always_inline)) void CSA256(__m256i* h, __m256i* l,
                                                  __m256i a, __m256i b,
                                                  __m256i c) {
  __m256i u = a ^ b;
  *h = (a & b) | (u & c);
  *l = u ^ c;
}

uint64_t popcnt2(const __m256i* A, const __m256i* B, const uint64_t size) {
  A = (const __m256i*)__builtin_assume_aligned(A, kDefaultAlignment);
  B = (const __m256i*)__builtin_assume_aligned(B, kDefaultAlignment);
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
    total = _mm256_add_epi64(total, popcount(sixteens));
  }

  total = _mm256_slli_epi64(total, 4);  // * 16
  total = _mm256_add_epi64(
      total, _mm256_slli_epi64(popcount(eights), 3));  // += 8 * ...
  total = _mm256_add_epi64(
      total, _mm256_slli_epi64(popcount(fours), 2));  // += 4 * ...
  total = _mm256_add_epi64(total,
                           _mm256_slli_epi64(popcount(twos), 1));  // += 2 * ...
  total = _mm256_add_epi64(total, popcount(ones));

  for (; i < size; i++) total = _mm256_add_epi64(total, popcount(A[i] ^ B[i]));

  return static_cast<uint64_t>(_mm256_extract_epi64(total, 0)) +
         static_cast<uint64_t>(_mm256_extract_epi64(total, 1)) +
         static_cast<uint64_t>(_mm256_extract_epi64(total, 2)) +
         static_cast<uint64_t>(_mm256_extract_epi64(total, 3));
}

uint64_t popcnt(const __m256i* A, const __m256i* B, const uint64_t size) {
  A = (const __m256i*)__builtin_assume_aligned(A, kDefaultAlignment);
  B = (const __m256i*)__builtin_assume_aligned(B, kDefaultAlignment);

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

    total = _mm256_add_epi64(total, popcount(sixteens));
  }

  total = _mm256_slli_epi64(total, 4);  // * 16
  total = _mm256_add_epi64(
      total, _mm256_slli_epi64(popcount(eights), 3));  // += 8 * ...
  total = _mm256_add_epi64(
      total, _mm256_slli_epi64(popcount(fours), 2));  // += 4 * ...
  total = _mm256_add_epi64(total,
                           _mm256_slli_epi64(popcount(twos), 1));  // += 2 * ...
  total = _mm256_add_epi64(total, popcount(ones));

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

__attribute__((noinline)) __m256i xnor_popcnt_AVX2_lookup2x2(const uint8_t* A0,
                                                             const uint8_t* A1,
                                                             const uint8_t* B0,
                                                             const uint8_t* B1,
                                                             const size_t n) {
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

  __m256i acc00 = _mm256_setzero_si256();
  __m256i acc01 = _mm256_setzero_si256();
  __m256i acc10 = _mm256_setzero_si256();
  __m256i acc11 = _mm256_setzero_si256();

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
    __m256i local00 = _mm256_setzero_si256();
    __m256i local01 = _mm256_setzero_si256();
    __m256i local10 = _mm256_setzero_si256();
    __m256i local11 = _mm256_setzero_si256();
    ITER ITER ITER ITER ITER ITER ITER ITER;
    acc00 = _mm256_add_epi64(acc00,
                             _mm256_sad_epu8(local00, _mm256_setzero_si256()));
    acc01 = _mm256_add_epi64(acc01,
                             _mm256_sad_epu8(local01, _mm256_setzero_si256()));
    acc10 = _mm256_add_epi64(acc10,
                             _mm256_sad_epu8(local10, _mm256_setzero_si256()));
    acc11 = _mm256_add_epi64(acc11,
                             _mm256_sad_epu8(local11, _mm256_setzero_si256()));
  }

  __m256i local00 = _mm256_setzero_si256();
  __m256i local01 = _mm256_setzero_si256();
  __m256i local10 = _mm256_setzero_si256();
  __m256i local11 = _mm256_setzero_si256();

  while (i + 32 <= n) {
    ITER;
  }
  acc00 =
      _mm256_add_epi64(acc00, _mm256_sad_epu8(local00, _mm256_setzero_si256()));
  acc01 =
      _mm256_add_epi64(acc01, _mm256_sad_epu8(local01, _mm256_setzero_si256()));
  acc10 =
      _mm256_add_epi64(acc10, _mm256_sad_epu8(local10, _mm256_setzero_si256()));
  acc11 =
      _mm256_add_epi64(acc11, _mm256_sad_epu8(local11, _mm256_setzero_si256()));

#undef ITER
  // 4x32 each, [:128]

  acc00 = _mm256_hadd_epi32(acc00, acc00);
  acc01 = _mm256_hadd_epi32(acc01, acc01);
  acc10 = _mm256_hadd_epi32(acc10, acc10);
  acc11 = _mm256_hadd_epi32(acc11, acc11);
  const auto acc0 = _mm256_hadd_epi32(acc00, acc01);
  const auto acc1 = _mm256_hadd_epi32(acc10, acc11);
  return _mm256_hadd_epi32(acc0, acc1);
}

// template <size_t M, size_t N>
// __attribute__((noinline)) __m256i xnor_popcnt_AVX2_lookup1x4(
//     const uint8_t *A, const uint8_t *B0, const uint8_t *B1, const uint8_t
//     *B2, const uint8_t *B3, const size_t qk) {
//   assert(qk % 32 == 0);
//   size_t i = 0;

//   const __m256i lookup = _mm256_setr_epi8(
//       /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
//       /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
//       /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
//       /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4,

//       /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
//       /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
//       /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
//       /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4);

//   const __m256i low_mask = _mm256_set1_epi8(0x0f);

//   __m256i acc0 = _mm256_setzero_si256();

//   //   while (i + 8 * 32 <= n) {
//   //     __m256i local00 = _mm256_setzero_si256();
//   //     __m256i local01 = _mm256_setzero_si256();
//   //     __m256i local10 = _mm256_setzero_si256();
//   //     __m256i local11 = _mm256_setzero_si256();
//   //     ITER ITER ITER ITER ITER ITER ITER ITER;
//   //     acc00 = _mm256_add_epi64(acc00,
//   //                              _mm256_sad_epu8(local00,
//   //                              _mm256_setzero_si256()));
//   //     acc01 = _mm256_add_epi64(acc01,
//   //                              _mm256_sad_epu8(local01,
//   //                              _mm256_setzero_si256()));
//   //     acc10 = _mm256_add_epi64(acc10,
//   //                              _mm256_sad_epu8(local10,
//   //                              _mm256_setzero_si256()));
//   //     acc11 = _mm256_add_epi64(acc11,
//   //                              _mm256_sad_epu8(local11,
//   //                              _mm256_setzero_si256()));
//   //   }

//   __m256i local00 = _mm256_setzero_si256();
//   __m256i local01 = _mm256_setzero_si256();
//   __m256i local10 = _mm256_setzero_si256();
//   __m256i local11 = _mm256_setzero_si256();

//   while (i + 32 <= qk) {
//     __m256i Avec[M];
//     __m256i Bvec[N];
//     for (size_t m = 0; m < M; ++m) {
//       Avec[m] =
//           _mm256_load_si256(reinterpret_cast<const __m256i *>(A + qk * m +
//           i));
//     }
//     for (size_t n = 0; n < N; ++n) {
//       Bvec[n] =
//           _mm256_load_si256(reinterpret_cast<const __m256i *>(B + qk * n +
//           i));
//     }
//     for (size_t m = 0; m < M; ++m) {
//     }
//     const __m256i vec00 = _mm256_xor_si256(A0vec, B0vec);
//     const __m256i vec01 = _mm256_xor_si256(A0vec, B1vec);
//     const __m256i vec10 = _mm256_xor_si256(A1vec, B0vec);
//     const __m256i vec11 = _mm256_xor_si256(A1vec, B1vec);
//     const __m256i lo00 = _mm256_and_si256(vec00, low_mask);
//     const __m256i lo01 = _mm256_and_si256(vec01, low_mask);
//     const __m256i lo10 = _mm256_and_si256(vec10, low_mask);
//     const __m256i lo11 = _mm256_and_si256(vec11, low_mask);
//     const __m256i hi00 =
//         _mm256_and_si256(_mm256_srli_epi16(vec00, 4), low_mask);
//     const __m256i hi01 =
//         _mm256_and_si256(_mm256_srli_epi16(vec01, 4), low_mask);
//     const __m256i hi10 =
//         _mm256_and_si256(_mm256_srli_epi16(vec10, 4), low_mask);
//     const __m256i hi11 =
//         _mm256_and_si256(_mm256_srli_epi16(vec11, 4), low_mask);
//     const __m256i popcntl00 = _mm256_shuffle_epi8(lookup, lo00);
//     const __m256i popcntl01 = _mm256_shuffle_epi8(lookup, lo01);
//     const __m256i popcntl10 = _mm256_shuffle_epi8(lookup, lo10);
//     const __m256i popcntl11 = _mm256_shuffle_epi8(lookup, lo11);
//     const __m256i popcnth00 = _mm256_shuffle_epi8(lookup, hi00);
//     const __m256i popcnth01 = _mm256_shuffle_epi8(lookup, hi01);
//     const __m256i popcnth10 = _mm256_shuffle_epi8(lookup, hi10);
//     const __m256i popcnth11 = _mm256_shuffle_epi8(lookup, hi11);
//     local00 = _mm256_add_epi8(local00, popcntl00);
//     local00 = _mm256_add_epi8(local00, popcnth00);
//     local01 = _mm256_add_epi8(local01, popcntl01);
//     local01 = _mm256_add_epi8(local01, popcnth01);
//     local10 = _mm256_add_epi8(local10, popcntl10);
//     local10 = _mm256_add_epi8(local10, popcnth10);
//     local11 = _mm256_add_epi8(local11, popcntl11);
//     local11 = _mm256_add_epi8(local11, popcnth11);
//     i += 32;
//   }
// }

// acc00 =
//     _mm256_add_epi64(acc00, _mm256_sad_epu8(local00,
//     _mm256_setzero_si256()));
// acc01 =
//     _mm256_add_epi64(acc01, _mm256_sad_epu8(local01,
//     _mm256_setzero_si256()));
// acc10 =
//     _mm256_add_epi64(acc10, _mm256_sad_epu8(local10,
//     _mm256_setzero_si256()));
// acc11 =
//     _mm256_add_epi64(acc11, _mm256_sad_epu8(local11,
//     _mm256_setzero_si256()));

// acc00 = _mm256_hadd_epi32(acc00, acc00);
// acc01 = _mm256_hadd_epi32(acc01, acc01);
// acc10 = _mm256_hadd_epi32(acc10, acc10);
// acc11 = _mm256_hadd_epi32(acc11, acc11);
// const auto acc0 = _mm256_hadd_epi32(acc00, acc01);
// const auto acc1 = _mm256_hadd_epi32(acc10, acc11);
// return _mm256_hadd_epi32(acc0, acc1);
// }

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
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      size_t acc = 0;
      for (size_t qk = 0; qk < QK; ++qk) {
        acc += __builtin_popcount(A[m * QK + qk] ^ B[n * QK + qk]);
      }
      C[m * Cstride + n] = acc;
    }
  }
}

template <size_t M, size_t N>
__attribute__((noinline)) void qxnor_popcnt(const uint8_t* A, const uint8_t* B,
                                            uint32_t* C, size_t Cstride,
                                            size_t QK) {
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      size_t acc = 0;
      acc += xnor_popcnt_AVX2_lookup(&A[m * QK], &B[n * QK], QK);
      C[m * Cstride + n] = acc;
    }
  }
}

template <size_t M, size_t N>
__attribute__((noinline)) void qxnor_popcnt_hs(const uint8_t* A,
                                               const uint8_t* B, uint32_t* C,
                                               size_t Cstride, size_t QK) {
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      size_t acc = 0;
      acc += xnor_popcnt_AVX2_harley_seal(&A[m * QK], &B[n * QK], QK);
      C[m * Cstride + n] = acc;
    }
  }
}

template <size_t M, size_t N>
__attribute__((noinline)) void qxnor_popcnt2x2(const uint8_t* A,
                                               const uint8_t* B, uint32_t* C,
                                               size_t Cstride, size_t QK) {
  for (size_t m = 0; m < M; m += 2) {
    for (size_t n = 0; n < N; n += 2) {
      const auto acc = xnor_popcnt_AVX2_lookup2x2(
          &A[m * QK], &A[(m + 1) * QK], &B[n * QK], &B[(n + 1) * QK], QK);
      C[(m + 0) * Cstride + (n + 0)] =
          static_cast<uint32_t>(_mm256_extract_epi64(acc, 0));
      C[(m + 0) * Cstride + (n + 1)] =
          static_cast<uint32_t>(_mm256_extract_epi64(acc, 1));
      C[(m + 1) * Cstride + (n + 0)] =
          static_cast<uint32_t>(_mm256_extract_epi64(acc, 2));
      C[(m + 1) * Cstride + (n + 1)] =
          static_cast<uint32_t>(_mm256_extract_epi64(acc, 3));
    }
  }
}

template <size_t M, size_t N>
__attribute__((noinline)) void qgess_blocked(const uint8_t* A, const uint8_t* B,
                                             uint32_t* C, size_t Cstride,
                                             size_t QK) {
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

  __m256i acc[M][N];
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      acc[m][n] = _mm256_setzero_si256();
    }
  }

  assert(QK % 32 == 0);
  for (size_t qk = 0; qk < (QK / 32) * 32; ++qk) {
    __m256i Areg[M];
    __m256i Breg[N];
    for (size_t m = 0; m < M; ++m) {
      Areg[m] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(A));
      A += 32;
    }

    for (size_t n = 0; n < N; ++n) {
      Breg[n] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(B));
      B += 32;
    }

    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        const __m256i AxorB = _mm256_xor_si256(Areg[m], Breg[n]);
        const __m256i lo = _mm256_and_si256(AxorB, low_mask);
        const __m256i hi =
            _mm256_and_si256(_mm256_srli_epi16(AxorB, 4), low_mask);
        const __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
        const __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
        __m256i local = _mm256_setzero_si256();
        local = _mm256_add_epi8(local, popcnt1);
        local = _mm256_add_epi8(local, popcnt2);
        acc[m][n] = _mm256_add_epi64(
            acc[m][n], _mm256_sad_epu8(local, _mm256_setzero_si256()));
      }
    }
  }

  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      const __m256i accmn = acc[m][n];
      uint64_t result = 0;
      result += static_cast<uint64_t>(_mm256_extract_epi64(accmn, 0));
      result += static_cast<uint64_t>(_mm256_extract_epi64(accmn, 1));
      result += static_cast<uint64_t>(_mm256_extract_epi64(accmn, 2));
      result += static_cast<uint64_t>(_mm256_extract_epi64(accmn, 3));
      C[m * Cstride + n] = result;
    }
  }
}

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
    qxnor_popcnt<M, N>(A.data(), B.data(), C.data(), N, QK);
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
    qxnor_popcnt2x2<M, N>(A.data(), B.data(), C.data(), N, QK);
    ++iters;
  }

  state.counters["FLOPS"] = benchmark::Counter(2 * M * N * QK * 8 * iters,
                                               benchmark::Counter::kIsRate);
}

template <size_t M, size_t N>
static void BM_qxnor_popcnt_hs(benchmark::State& state) {
  size_t QK = state.range(0);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> A(M * QK);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> B(N * QK);
  std::vector<uint32_t> C(M * N);

  size_t iters = 0;
  while (state.KeepRunning()) {
    qxnor_popcnt_hs<M, N>(A.data(), B.data(), C.data(), N, QK);
    ++iters;
  }

  state.counters["FLOPS"] = benchmark::Counter(2 * M * N * QK * 8 * iters,
                                               benchmark::Counter::kIsRate);
}

template <size_t M, size_t N>
static void BM_qgess_blocked(benchmark::State& state) {
  size_t QK = state.range(0);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> A(M * QK);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> B(N * QK);
  std::vector<uint32_t> C(M * N);

  size_t iters = 0;
  while (state.KeepRunning()) {
    qgess_blocked<M, N>(A.data(), B.data(), C.data(), N, QK);
    ++iters;
  }

  state.counters["FLOPS"] = benchmark::Counter(2 * M * N * QK * 8 * iters,
                                               benchmark::Counter::kIsRate);
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
constexpr size_t kQKUpperBound = 8192;

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

BENCHMARK_TEMPLATE2(BM_qxnor_popcnt2x2, 256, 256)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);

BENCHMARK_TEMPLATE2(BM_sgemm, 32, 32)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_TEMPLATE2(BM_sgemm, 64, 64)
    ->RangeMultiplier(2)
    ->Range(kQKLowerBound, kQKUpperBound);
BENCHMARK_MAIN();
