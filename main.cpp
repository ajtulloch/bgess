#include <immintrin.h>
#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <array>

#include <Accelerate/Accelerate.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <glog/stl_logging.h>
#include <gtest/gtest.h>

#include "benchmark/benchmark.h"
#include "yuxin.hpp"
#include "test.hpp"

#include "clean.hpp"

namespace caffe2 {

constexpr size_t k2b1bXBits = 2;

// How do we effiecintly quantize? One approach:

// - Load 256 floats of X. We fit 8 floats per vector, so 32 loads. Group in
// blocks of 4. On each load. - compare to form the bitmasts at each level. This
// is 8 distinct bit vectors, each of size 32bits - so 8 elements per bitmask.
// - Collapse these by a factor of 4 (3 blends), to get 32 elements per bitmask.
// - AND this with [1 << 0, 1 << 1, .., 1 << 7, 1 << 0, ...]
// - SAD this (8 wide addition) to get [1 << 0 & (b0) + 1 << 1 & (b1), ...]
// This is 4, 16 bit numbers - which is 4x8 bit numbers for us. This is 32 bits
// total.

// Repeat this 8 times. This gives us

void quantize2bStd(size_t QC, const float* __restrict__ Xdata,
                          float offset, float inter_center_distance,
                          std::array<uint8_t*, k2b1bXBits> XQdata) {
  size_t C = QC * 8;
  for (auto qc = 0; qc < QC; ++qc) {
    // compute the block in X.
    std::array<uint8_t, k2b1bXBits> p = {{0, 0}};
    for (auto b = 0; b < 8; ++b) {
      const auto c = qc * 8 + b;
      if (c < C) {
        float v = Xdata[qc * 8 + b];
        if (v < offset) {
          // zero'd already.
        } else if (v < offset + inter_center_distance) {
          p[0] |= 1 << b;
        } else if (v < offset + 2 * inter_center_distance) {
          p[1] |= 1 << b;
        } else {
          p[0] |= 1 << b;
          p[1] |= 1 << b;
        }
      }
    }
    for (auto i = 0; i < k2b1bXBits; ++i) {
      XQdata[i][qc] = p[i];
    }
  }
}

template <size_t TileSize, size_t TileDepthBytes>
void qpack_tiles(const uint8_t* __restrict__ Xdata,
                 uint8_t* __restrict__ XPdata, size_t M, size_t QK) {
  const size_t numTiles = divRoundUp(M, TileSize);
  const size_t numTilesDepth = divRoundUp(QK, TileDepthBytes);

  // Load L1 sized tiles per thread.
  // We read/write 2 * B * QK * TileSize bytes, so
  // B = C / (2 * QK * TileSize)
  for (size_t i = 0; i < numTiles; ++i) {
    for (size_t j = 0; j < numTilesDepth; ++j) {
      if (i != numTiles - 1 && j != numTilesDepth - 1) {
        // we have a full tile. Just memcpy.
        for (auto ii = 0; ii < TileSize; ++ii) {
          auto m = i * TileSize + ii;
          auto qk = j * TileDepthBytes;
          std::memcpy(
              &XPdata[TileDepthBytes * ii + TileDepthBytes * TileSize * j +
                      TileSize * TileDepthBytes * numTilesDepth * i],
              &Xdata[m * QK + qk], TileDepthBytes);
        }
      } else {
        for (size_t ii = 0; ii < TileSize; ++ii) {
          for (size_t jj = 0; jj < TileDepthBytes; ++jj) {
            size_t m = i * TileSize + ii;
            size_t qk = j * TileDepthBytes + jj;
            uint8_t pval = 0;
            if (m < M && qk < QK) {
              // get value from X
              pval = Xdata[m * QK + qk];
            }
            XPdata[jj + TileDepthBytes * ii + TileDepthBytes * TileSize * j +
                   TileSize * TileDepthBytes * numTilesDepth * i] = pval;
          }
        }
      }
    }
  }
}

void quantize2bAVX2(size_t QC, const float* __restrict__ Xdata, float offset,
                    float inter_center_distance,
                    std::array<uint8_t*, k2b1bXBits> XQdata) {
  CHECK_EQ(QC % 32, 0);
  const auto offset_plus_2_inter_center_distance =
      _mm256_set1_ps(offset + 2 * inter_center_distance);
  const auto offset_plus_inter_center_distance =
      _mm256_set1_ps(offset + inter_center_distance);
  const auto offset_ = _mm256_set1_ps(offset);
  const auto shifts = _mm256_setr_epi8(
      char(1 << 0), char(1 << 1), char(1 << 2), char(1 << 3), char(1 << 4),
      char(1 << 5), char(1 << 6), char(1 << 7), char(1 << 0), char(1 << 1),
      char(1 << 2), char(1 << 3), char(1 << 4), char(1 << 5), char(1 << 6),
      char(1 << 7), char(1 << 0), char(1 << 1), char(1 << 2), char(1 << 3),
      char(1 << 4), char(1 << 5), char(1 << 6), char(1 << 7), char(1 << 0),
      char(1 << 1), char(1 << 2), char(1 << 3), char(1 << 4), char(1 << 5),
      char(1 << 6), char(1 << 7));

  // Load 32 * 8 = 256 floats.
  for (size_t qc = 0; qc < QC; qc += 32) {
    __m256i ps0 = _mm256_setzero_si256();
    __m256i ps1 = _mm256_setzero_si256();

    // We need to load 256 floats here. Break into 8 blocks of 32 floats each.
    // Each block loads 32 floats.
    // Group into 8 blocks of 32 floats.
    for (size_t block = 0; block < 8; ++block) {
      __m256 x0 = _mm256_load_si256(
          reinterpret_cast<const __m256i*>(Xdata + qc * 8 + block * 32 + 0));
      __m256 x1 = _mm256_load_si256(
          reinterpret_cast<const __m256i*>(Xdata + qc * 8 + block * 32 + 0));
      __m256 x2 = _mm256_load_si256(
          reinterpret_cast<const __m256i*>(Xdata + qc * 8 + block * 32 + 0));
      __m256 x3 = _mm256_load_si256(
          reinterpret_cast<const __m256i*>(Xdata + qc * 8 + block * 32 + 0));

      auto join = [](__m256i a, __m256i b, __m256i c, __m256i d) {
        __m256i a_mask = _mm256_set1_epi32(0x000000ff);
        __m256i b_mask = _mm256_set1_epi32(0x0000ff00);
        __m256i c_mask = _mm256_set1_epi32(0x00ff0000);
        __m256i d_mask = _mm256_set1_epi32(0xff000000);
        return (a & a_mask) | (b & b_mask) | (c & c_mask) | (d & d_mask);
      };

      const auto x_geq_offset_plus_2_inter_center_distance =
          join(x0 >= offset_plus_2_inter_center_distance,
               x1 >= offset_plus_2_inter_center_distance,
               x2 >= offset_plus_2_inter_center_distance,
               x3 >= offset_plus_2_inter_center_distance);
      const auto x_geq_offset =
          join(x0 >= offset_, x1 >= offset_, x2 >= offset_, x3 >= offset_);
      const auto x_lt_offset_plus_inter_center_distance =
          join(x0 < offset_plus_inter_center_distance,
               x1 < offset_plus_inter_center_distance,
               x2 < offset_plus_inter_center_distance,
               x3 < offset_plus_inter_center_distance);

      const auto p1_mask = ~x_lt_offset_plus_inter_center_distance;
      const auto p0_mask =
                    (x_geq_offset & x_lt_offset_plus_inter_center_distance) |
                    x_geq_offset_plus_2_inter_center_distance;

      // Sum the 8-wide shifts.
      const auto p0 = _mm256_sad_epu8(shifts & p0_mask, _mm256_setzero_si256());
      const auto p1 = _mm256_sad_epu8(shifts & p1_mask, _mm256_setzero_si256());
      // TODO: shift into 4x32 bucket and add.
      ps0 = _mm256_add_epi8(ps0, p0);
      ps1 = _mm256_add_epi8(ps1, p0);
    }
    _mm256_store_si256((__m256i*)(XQdata[0] + 32), ps0);
    _mm256_store_si256((__m256i*)(XQdata[1] + 32), ps1);
  }
}

template <class T, size_t kAlignment = kDefaultAlignment>
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

namespace AVX2_harley_seal {

__attribute__((always_inline)) static __m256i popcount64(const __m256i vec) {
  __m256i lookup1 =
      _mm256_setr_epi8(4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8, 4, 5, 5,
                       6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8);

  __m256i lookup2 =
      _mm256_setr_epi8(4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0, 4, 3, 3,
                       2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);

  __m256i low_mask = _mm256_set1_epi8(0x0f);
  __m256i lo = vec & low_mask;
  __m256i hi = _mm256_srli_epi16(vec, 4) & low_mask;
  __m256i popcnt1 = _mm256_shuffle_epi8(lookup1, lo);
  __m256i popcnt2 = _mm256_shuffle_epi8(lookup2, hi);

  return _mm256_sad_epu8(popcnt1, popcnt2);
}

__attribute__((always_inline)) static __m256i popcount8(const __m256i vec) {
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
  // ASSUME: total redictuion is less than 8192 bits (since we accumulate up
  // to 2^8 in each 8 bit unit.)
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
    local = _mm256_add_epi8(local, popcount8(eights));
  }
  total = _mm256_sad_epu8(local, zero);
  total = _mm256_slli_epi64(total, 3);  // * 8
  total = _mm256_add_epi64(total, _mm256_slli_epi64(popcount64(fours),
                                                    2));  // += 4 * ...
  total = _mm256_add_epi64(
      total, _mm256_slli_epi64(popcount64(twos), 1));  // += 2 * ...
  total = _mm256_add_epi64(total, popcount64(ones));

  for (; i < size; i++)
    total = _mm256_add_epi64(total, popcount64(A[i] ^ B[i]));

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

  const uint64_t size16unroll = (size / 16) * 16;
  uint64_t i = 0;

  for (; i < size16unroll; i += 16) {
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
    local = _mm256_add_epi8(local, popcount8(sixteens));
  }

  total = _mm256_sad_epu8(local, zero);
  total = _mm256_slli_epi64(total, 4);  // * 16
  total = _mm256_add_epi64(
      total, _mm256_slli_epi64(popcount64(eights), 3));  // += 8 * ...
  total = _mm256_add_epi64(total, _mm256_slli_epi64(popcount64(fours),
                                                    2));  // += 4 * ...
  total = _mm256_add_epi64(
      total, _mm256_slli_epi64(popcount64(twos), 1));  // += 2 * ...
  total = _mm256_add_epi64(total, popcount64(ones));

  for (; i < size; i++) {
    total = _mm256_add_epi64(total, popcount64(A[i] ^ B[i]));
  }
  auto sum = static_cast<uint64_t>(_mm256_extract_epi64(total, 0)) +
             static_cast<uint64_t>(_mm256_extract_epi64(total, 1)) +
             static_cast<uint64_t>(_mm256_extract_epi64(total, 2)) +
             static_cast<uint64_t>(_mm256_extract_epi64(total, 3));
  return sum;
}

}  // namespace AVX2_harley_seal

__attribute__((noinline)) void xnor_popcnt_AVX2_lookup2x2(
    const uint8_t* A0, const uint8_t* A1, const uint8_t* B0, const uint8_t* B1,
    uint32_t* C, const size_t Cstride, const size_t n) {
  CHECK(n % 32 == 0);
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
  CHECK(qk % 32 == 0);
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
    for (size_t n = 0; n < N; ++n) {
      C[Cstride * m + n] = r(acc[m][n]);
    }
  }
  return;
}

std::uint64_t xnor_popcnt_AVX2_lookup(const uint8_t* A, const uint8_t* B,
                                      const size_t n) {
  CHECK(n % 32 == 0);
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
  CHECK(n % 32 == 0);
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

__attribute__((noinline)) void qgess(const uint8_t* A, const uint8_t* B,
                                     uint32_t* C, size_t M, size_t N,
                                     size_t Cstride, size_t QK) {
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

uint64_t xnor_popcnt_AVX2_harley_seal(const uint8_t* A, const uint8_t* B,
                                      const size_t size) {
  const auto size32Unroll = (size / 32) * 32;
  uint64_t total =
      AVX2_harley_seal::popcnt((const __m256i*)A, (const __m256i*)B, size / 32);
  total += xnor_popcnt_AVX2_lookup(A + size, B + size, size - size32Unroll);
  return total;
}

uint64_t xnor_popcnt_AVX2_harley_seal2(const uint8_t* A, const uint8_t* B,
                                       const size_t size) {
  const auto size32Unroll = (size / 32) * 32;
  uint64_t total = AVX2_harley_seal::popcnt2((const __m256i*)A,
                                             (const __m256i*)B, size / 32);
  total += xnor_popcnt_AVX2_lookup(A + size, B + size, size - size32Unroll);
  return total;
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

template <size_t MM, size_t NN>
__attribute__((noinline)) void qgemm_nt_avx2(const uint8_t* A, const uint8_t* B,
                                             float* C, size_t M, size_t N,
                                             size_t Cstride, size_t QK) {
  CHECK_EQ(M % MM, 0);
  CHECK_EQ(N % NN, 0);
  for (size_t m = 0; m < M; m += MM) {
    for (size_t n = 0; n < N; n += NN) {
      qgess_avx2<MM, NN>(&A[m * QK], &B[n * QK], &C[m * Cstride + n], Cstride,
                         QK);
    }
  }
}

__attribute__((noinline)) void qxnor_popcnt_hs2(const uint8_t* A,
                                                const uint8_t* B, uint32_t* C,
                                                size_t M, size_t N,
                                                size_t Cstride, size_t QK) {
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

TEST(BGess, qgess_1_0) {
  const size_t M = 20;
  const size_t N = 40;
  const size_t QK = 8;
  std::vector<uint8_t, AlignedAllocator<uint8_t>> A(M * QK);
  for (auto i = 0; i < A.size(); ++i) {
    A[i] = 0xff;
  }
  std::vector<uint8_t, AlignedAllocator<uint8_t>> B(N * QK);
  for (auto i = 0; i < B.size(); ++i) {
    B[i] = 0x00;
  }
  std::vector<uint32_t> C(M * N);
  qgess(A.data(), B.data(), C.data(), M, N, N, QK);
  for (auto i = 0; i < C.size(); ++i) {
    EXPECT_EQ(C.data()[i], QK * 8) << i << ", " << C.data()[i];
  }
}

TEST(BGess, qgess_lookup_1_0) {
  const size_t M = 20;
  const size_t N = 40;
  const size_t QK = 32;
  std::vector<uint8_t, AlignedAllocator<uint8_t>> A(M * QK);
  for (auto i = 0; i < A.size(); ++i) {
    A[i] = 0xff;
  }
  std::vector<uint8_t, AlignedAllocator<uint8_t>> B(N * QK);
  for (auto i = 0; i < B.size(); ++i) {
    B[i] = 0x00;
  }
  std::vector<uint32_t> C(M * N);
  std::vector<uint32_t> CR(M * N);
  qgess(A.data(), B.data(), C.data(), M, N, N, QK);
  qxnor_popcnt(A.data(), B.data(), CR.data(), M, N, N, QK);
  for (auto i = 0; i < C.size(); ++i) {
    EXPECT_EQ(C.data()[i], CR.data()[i]);
  }
}

TEST(BGess, qgess_mxn_2x2_1_0) {
  const size_t M = 20;
  const size_t N = 40;
  const size_t QK = 32;
  std::vector<uint8_t, AlignedAllocator<uint8_t>> A(M * QK);
  for (auto i = 0; i < A.size(); ++i) {
    A[i] = 0xff;
  }
  std::vector<uint8_t, AlignedAllocator<uint8_t>> B(N * QK);
  for (auto i = 0; i < B.size(); ++i) {
    B[i] = 0x00;
  }
  std::vector<uint32_t> C(M * N);
  std::vector<uint32_t> CR(M * N);
  qgess(A.data(), B.data(), C.data(), M, N, N, QK);
  qxnor_popcnt_mxn<2, 2>(A.data(), B.data(), CR.data(), M, N, N, QK);
  for (auto i = 0; i < C.size(); ++i) {
    EXPECT_EQ(C.data()[i], CR.data()[i]);
  }
}

TEST(BGess, qgess_hs_1_0) {
  const size_t M = 20;
  const size_t N = 40;
  const size_t QK = 32 * 32;
  std::vector<uint8_t, AlignedAllocator<uint8_t>> A(M * QK);
  for (auto i = 0; i < A.size(); ++i) {
    A[i] = 0xff;
  }
  std::vector<uint8_t, AlignedAllocator<uint8_t>> B(N * QK);
  for (auto i = 0; i < B.size(); ++i) {
    B[i] = 0x00;
  }
  std::vector<uint32_t> C(M * N);
  std::vector<uint32_t> CR(M * N);
  qgess(A.data(), B.data(), C.data(), M, N, N, QK);
  qxnor_popcnt_hs(A.data(), B.data(), CR.data(), M, N, N, QK);
  for (auto i = 0; i < C.size(); ++i) {
    EXPECT_EQ(C.data()[i], CR.data()[i]);
  }
}

TEST(BGess, qgess_hs2_1_0) {
  const size_t M = 20;
  const size_t N = 40;
  const size_t QK = 32 * 32;
  std::vector<uint8_t, AlignedAllocator<uint8_t>> A(M * QK);
  for (auto i = 0; i < A.size(); ++i) {
    A[i] = 0xff;
  }
  std::vector<uint8_t, AlignedAllocator<uint8_t>> B(N * QK);
  for (auto i = 0; i < B.size(); ++i) {
    B[i] = 0x00;
  }
  std::vector<uint32_t> C(M * N);
  std::vector<uint32_t> CR(M * N);
  qgess(A.data(), B.data(), C.data(), M, N, N, QK);
  qxnor_popcnt_hs2(A.data(), B.data(), CR.data(), M, N, N, QK);
  for (auto i = 0; i < C.size(); ++i) {
    EXPECT_EQ(C.data()[i], CR.data()[i]);
  }
}

void gemmTest(TIndex M, TIndex N, TIndex K) {
  auto X = genTensor11({M, K});
  auto W = genTensor11({N, K});
  CHECK_EQ(K % 8, 0);
  TensorCPU XQ, WQ, YQ, Y;
  {
    signQuantize(X, &XQ);
    signQuantize(W, &WQ);
    YQ.Resize(M, N);
    qgemm_nt_avx2<2, 2>(XQ.data<uint8_t>(), WQ.data<uint8_t>(),
                        YQ.mutable_data<float>(), M, N, N, K / 8);
  }
  {
    Y.Resize(M, N);
    gemmNT(M, N, K, X.data<float>(), W.data<float>(), Y.mutable_data<float>());
  }
  EXPECT_TRUE(Y.dims() == YQ.dims());
  for (auto i = 0; i < Y.size(); ++i) {
    EXPECT_NEAR(Y.data<float>()[i], YQ.data<float>()[i], 1e-3) << i;
  }
}

TEST(QConv, GemmTest) {
  gemmTest(2, 2, 256);
  gemmTest(16, 64, 256);
  gemmTest(24, 128, 256);
  gemmTest(32, 64, 256);
  gemmTest(40, 64, 256);
  gemmTest(64, 64, 256);
}


TEST(BGess, qgess_mxn_4x2_1_0) {
  const size_t M = 20;
  const size_t N = 40;
  const size_t QK = 32;
  std::vector<uint8_t, AlignedAllocator<uint8_t>> A(M * QK);
  for (auto i = 0; i < A.size(); ++i) {
    A[i] = 0xff;
  }
  std::vector<uint8_t, AlignedAllocator<uint8_t>> B(N * QK);
  for (auto i = 0; i < B.size(); ++i) {
    B[i] = 0x00;
  }
  std::vector<uint32_t> C(M * N);
  std::vector<uint32_t> CR(M * N);
  qgess(A.data(), B.data(), C.data(), M, N, N, QK);
  qxnor_popcnt_mxn<4, 2>(A.data(), B.data(), CR.data(), M, N, N, QK);
  for (auto i = 0; i < C.size(); ++i) {
    EXPECT_EQ(C.data()[i], CR.data()[i]);
  }
}

static void BM_qgess(benchmark::State& state) {
  size_t QK = state.range(0);
  size_t M = state.range(1);
  size_t N = state.range(2);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> A(M * QK);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> B(N * QK);
  std::vector<uint32_t> C(M * N);
  size_t iters = 0;
  while (state.KeepRunning()) {
    qgess(A.data(), B.data(), C.data(), M, N, N, QK);
    ++iters;
  }
  state.counters["FLOPS"] = benchmark::Counter(2 * M * N * QK * 8 * iters,
                                               benchmark::Counter::kIsRate);
  // runXnorBenchmark();
}

static void BM_qxnor_popcnt(benchmark::State& state) {
  size_t QK = state.range(0);
  size_t M = state.range(1);
  size_t N = state.range(2);
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

static void BM_qxnor_popcnt2x2(benchmark::State& state) {
  size_t QK = state.range(0);
  size_t M = state.range(1);
  size_t N = state.range(2);
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

template <size_t MM, size_t NN>
static void BM_qxnor_popcntmxn_conv3x3(benchmark::State& state) {
  size_t QK = state.range(0);
  size_t M = state.range(1);
  size_t N = state.range(2);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> A(M * 3 * 3 * QK);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> B(N * 3 * 3 * QK);
  std::vector<uint32_t> C(M * N);

  size_t iters = 0;
  while (state.KeepRunning()) {
    qxnor_popcnt_mxn<MM, NN>(A.data(), B.data(), C.data(), M, N, N, QK * 3 * 3);
    ++iters;
  }

  state.counters["FLOPS"] = benchmark::Counter(
      2 * M * N * QK * 8 * 3 * 3 * iters, benchmark::Counter::kIsRate);
}

static void BM_quantize2b1b(benchmark::State& state) {
  size_t QK = state.range(0);
  size_t M = state.range(1);
  size_t N = state.range(2);

  std::vector<float, AlignedAllocator<float>> X(M * QK * 8);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> XQ0(M * QK);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> XQ1(M * QK);

  size_t iters = 0;
  while (state.KeepRunning()) {
    for (size_t m = 0; m < M; ++m) {
      quantize2bStd(QK, X.data() + m * QK * 8, 0.5, 1.5,
                    std::array<uint8_t*, k2b1bXBits>{
                      {XQ0.data() + m * QK, XQ1.data() + m * QK}});
    }
    ++iters;
  }

  state.counters["FLOPS"] = benchmark::Counter(2 * M * N * QK * 8 * iters,
                                               benchmark::Counter::kIsRate);
  state.counters["GBS"] =
      benchmark::Counter(iters * 4 * M * QK * 8, benchmark::Counter::kIsRate);
}

static void BM_quantize2b1b_AVX2(benchmark::State& state) {
  size_t QK = state.range(0);
  size_t M = state.range(1);
  size_t N = state.range(2);

  std::vector<float, AlignedAllocator<float>> X(M * QK * 8);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> XQ0(M * QK);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> XQ1(M * QK);

  size_t iters = 0;
  while (state.KeepRunning()) {
    for (size_t m = 0; m < M; ++m) {
      quantize2bAVX2(QK, X.data() + m * QK * 8, 0.5, 1.5,
                     std::array<uint8_t*, k2b1bXBits>{
                         {XQ0.data() + m * QK, XQ1.data() + m * QK}});
    }
    ++iters;
  }

  state.counters["FLOPS"] = benchmark::Counter(2 * M * N * QK * 8 * iters,
                                               benchmark::Counter::kIsRate);
  state.counters["GBS"] = benchmark::Counter(
      iters * 4 * M * QK * 8, benchmark::Counter::kIsRate);
}

static void BM_qpack_tiles(benchmark::State& state) {
  size_t QK = state.range(0);
  size_t M = state.range(1);
  size_t N = state.range(2);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> A(M * QK);
  std::vector<uint8_t, AlignedAllocator<uint8_t>> B(divRoundUp(M, 2) * 2 *
                                                    divRoundUp(QK, 32) * 32);
  size_t iters = 0;
  while (state.KeepRunning()) {
    qpack_tiles<2, 32>(A.data(), B.data(), M, QK);
    ++iters;
  }

  state.counters["FLOPS"] = benchmark::Counter(2 * M * N * QK * 8 * iters,
                                               benchmark::Counter::kIsRate);
  state.counters["GBS"] =
      benchmark::Counter(iters * A.size(), benchmark::Counter::kIsRate);
}

static void BM_qxnor_popcnt_hs(benchmark::State& state) {
  size_t QK = state.range(0);
  size_t M = state.range(1);
  size_t N = state.range(2);
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

static void BM_qxnor_popcnt_hs2(benchmark::State& state) {
  size_t QK = state.range(0);
  size_t M = state.range(1);
  size_t N = state.range(2);
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

template <size_t MM, size_t NN>
static void BM_qxnor_popcnt_mxn(benchmark::State& state) {
  size_t QK = state.range(0);
  size_t M = state.range(1);
  size_t N = state.range(2);
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

static void BM_yuxin_conv(benchmark::State& state) {
  size_t QK = state.range(0);
  size_t M = state.range(1);
  size_t N = state.range(2);
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


static void BM_yuxin_conv3x3(benchmark::State& state) {
  size_t QK = state.range(0);
  size_t M = state.range(1);
  size_t N = state.range(2);
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

static void BM_sgemm(benchmark::State& state) {
  size_t QK = state.range(0);
  size_t M = state.range(1);
  size_t N = state.range(2);
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
constexpr size_t kQKUpperBound = 1024;

static void GessArguments(benchmark::internal::Benchmark* b) {
  for (int M = 2; M <= 64; M *= 4) {
    // for (int N = 2; N <= 64; N *= 2) {
    for (int QK = kQKLowerBound; QK <= kQKUpperBound; QK *= 2) {
      b->Args({QK, M, M});
    }
  }
}

BENCHMARK(BM_quantize2b1b)->Apply(GessArguments);
BENCHMARK(BM_quantize2b1b_AVX2)->Apply(GessArguments);
BENCHMARK(BM_qpack_tiles)->Apply(GessArguments);

BENCHMARK(BM_qgess)->Apply(GessArguments);
BENCHMARK(BM_qxnor_popcnt)->Apply(GessArguments);
BENCHMARK(BM_qxnor_popcnt_hs)->Apply(GessArguments);
BENCHMARK(BM_qxnor_popcnt_hs2)->Apply(GessArguments);
BENCHMARK(BM_qxnor_popcnt2x2)->Apply(GessArguments);
BENCHMARK(BM_yuxin_conv3x3)->Apply(GessArguments);
BENCHMARK_TEMPLATE2(BM_qxnor_popcntmxn_conv3x3, 2, 2)->Apply(GessArguments);
BENCHMARK_TEMPLATE2(BM_qxnor_popcnt_mxn, 2, 2)->Apply(GessArguments);
BENCHMARK_TEMPLATE2(BM_qxnor_popcnt_mxn, 1, 2)->Apply(GessArguments);
BENCHMARK_TEMPLATE2(BM_qxnor_popcnt_mxn, 2, 1)->Apply(GessArguments);

BENCHMARK(BM_sgemm)->Apply(GessArguments);

}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  CHECK_EQ(RUN_ALL_TESTS(), 0);
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}

// BENCHMARK_MAIN();
