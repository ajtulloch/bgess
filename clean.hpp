#include <glog/logging.h>
#include <immintrin.h>
#include <stdio.h>
#include <array>
#include <iomanip>
#include <iostream>

namespace caffe2 {

constexpr size_t kDefaultAlignment = 32;
constexpr size_t kTileDepthBytes = 32;
constexpr size_t kTileSize = 2;

template <size_t TileSize, size_t TileDepthBytes>
void qpack_tiles(const TensorCPU& X, size_t axis, TensorCPU* XP) {
  const size_t M = X.size_to_dim(axis);
  const size_t QK = X.size() / M;
  const size_t numTiles = divRoundUp(M, TileSize);
  const size_t numTilesDepth = divRoundUp(QK, TileDepthBytes);
  XP->Resize(numTiles, numTilesDepth, TileSize, TileDepthBytes);

  const auto* __restrict__ Xdata = X.data<uint8_t>();
  auto* __restrict__ XPdata = XP->mutable_data<uint8_t>();

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

inline size_t hsum(__m256i v) {
  size_t result = 0;
  result += static_cast<size_t>(_mm256_extract_epi64(v, 0));
  result += static_cast<size_t>(_mm256_extract_epi64(v, 1));
  result += static_cast<size_t>(_mm256_extract_epi64(v, 2));
  result += static_cast<size_t>(_mm256_extract_epi64(v, 3));
  return result;
};

template <size_t kUnrollM, size_t kUnrollN, size_t TileDepthBytes>
inline void qgess_packed(const uint8_t* __restrict__ A,
                         const uint8_t* __restrict__ B, float* __restrict__ C,
                         const size_t Cstride, const size_t QK) {
  static_assert(TileDepthBytes == 32, "");
  CHECK_LT(QK * 8, 8192);
  A = (const uint8_t*)__builtin_assume_aligned(A, kDefaultAlignment);
  B = (const uint8_t*)__builtin_assume_aligned(B, kDefaultAlignment);
  CHECK_EQ(QK % 32, 0);

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


#define ITER                                                            \
  {                                                                     \
    for (size_t m = 0; m < kUnrollM; ++m) {                             \
      Areg[m] = _mm256_load_si256(reinterpret_cast<const __m256i*>(A)); \
      A += 32;                                                          \
    }                                                                   \
                                                                        \
    for (size_t n = 0; n < kUnrollN; ++n) {                             \
      const auto Breg =                                                 \
          _mm256_load_si256(reinterpret_cast<const __m256i*>(B));       \
      B += 32;                                                          \
      for (size_t m = 0; m < kUnrollM; ++m) {                           \
        const __m256i vec = _mm256_xor_si256(Areg[m], Breg);            \
        const __m256i lo = _mm256_and_si256(vec, low_mask);             \
        const __m256i hi =                                              \
            _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask);      \
        const __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);        \
        const __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);        \
        local[m][n] = _mm256_add_epi8(local[m][n], popcnt1);            \
        local[m][n] = _mm256_add_epi8(local[m][n], popcnt2);            \
      }                                                                 \
    }                                                                   \
  }


  // Local accumulators. We count up to 8192 bits (256 bits per 8 bit lane, 32 8
  // bit lanes in a vector). This is 1024 bytes.

  __m256i Areg[kUnrollM];
  __m256i local[kUnrollM][kUnrollN];
  for (size_t m = 0; m < kUnrollM; ++m) {
    for (size_t n = 0; n < kUnrollN; ++n) {
      local[m][n] = _mm256_setzero_si256();
    }
  }

  size_t qk = 0;
  size_t QK256Unroll = (QK / 256) * 256;
  for (; qk < QK256Unroll; qk += 256) {
    ITER ITER ITER ITER ITER ITER ITER ITER;
  }

  for (; qk < QK; qk += 32) {
    ITER;
  }

  __m256i acc[kUnrollM][kUnrollN];
  for (size_t m = 0; m < kUnrollM; ++m) {
    for (size_t n = 0; n < kUnrollN; ++n) {
      acc[m][n] = _mm256_sad_epu8(local[m][n], zero);
    }
  }
#undef ITER

  for (size_t m = 0; m < kUnrollM; ++m) {
    for (size_t n = 0; n < kUnrollN; ++n) {
      C[Cstride * m + n] =
          static_cast<float>(ssize_t(QK * 8) - ssize_t(2 * hsum(acc[m][n])));
    }
  }
}

template <size_t TileSize, size_t TileDepthBytes>
inline void qgemm_nt_packed(const TensorCPU& A, const TensorCPU& B,
                            TensorCPU* C) {
  CHECK_EQ(A.ndim(), 4);
  CHECK_EQ(B.ndim(), 4);
  CHECK_EQ(A.dim(2), TileSize);
  CHECK_EQ(B.dim(2), TileSize);
  CHECK_EQ(A.dim(3), TileDepthBytes);
  CHECK_EQ(B.dim(3), TileDepthBytes);
  const size_t MT = A.dim(0);
  const size_t NT = B.dim(0);
  const size_t M = MT * TileSize;
  const size_t N = NT * TileSize;

  const size_t QKT = A.dim(1);
  const size_t K = QKT * 8 * TileDepthBytes;
  const size_t QK = K / 8;
  CHECK_EQ(A.dim(1), B.dim(1));
  C->Resize(M, N);
  const auto* Adata = A.data<uint8_t>();
  const auto* Bdata = B.data<uint8_t>();
  auto* Cdata = C->mutable_data<float>();

  CHECK_LT(K, std::pow(2, 16));
  CHECK_EQ(M % TileSize, 0);
  CHECK_EQ(N % TileSize, 0);
  const size_t MNumTiles = M / TileSize;
  const size_t NNumTiles = N / TileSize;

  for (size_t mTileIdx = 0; mTileIdx < MNumTiles; ++mTileIdx) {
    for (size_t nTileIdx = 0; nTileIdx < NNumTiles; ++nTileIdx) {
      // A layout: [M/TileSize][QK / TileDepth][TileSize][TileDepth]
      // C layout: [M/TileSize][TileSize][N/TileSize][TileSize]
      const auto* Ablock = &Adata[mTileIdx * QK * TileSize];
      const auto* Bblock = &Bdata[nTileIdx * QK * TileSize];
      auto* Cblock = &Cdata[mTileIdx * TileSize * N + nTileIdx * TileSize];
      const size_t Cstride = N;
      qgess_packed<TileSize, TileSize, TileDepthBytes>(Ablock, Bblock, Cblock,
                                                       Cstride, QK);
    }
  }
}

}  // namespace caffe2
