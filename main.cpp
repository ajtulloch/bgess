#include <immintrin.h>
#include <stdio.h>
#include <iostream>
#include "benchmark/benchmark.h"
#include <Accelerate/Accelerate.h>

std::uint64_t xor_popcnt_AVX2_lookup(const uint8_t *A, const uint8_t *B,
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
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(A + i));         \
    const __m256i Bvec =                                                      \
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(B + i));         \
    const __m256i vec = _mm256_xor_si256(Avec, Bvec);                        \
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

std::uint64_t popcnt_AVX2_lookup(const uint8_t *data, const size_t n) {
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
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + i));      \
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
__attribute__((noinline)) void qgess(const uint8_t *A, const uint8_t *B,
                                     uint32_t *C, size_t Cstride, size_t QK) {
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
__attribute__((noinline)) void qpopcnt(const uint8_t *A, const uint8_t *B,
                                       uint32_t *C, size_t Cstride, size_t QK) {
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      size_t acc = 0;
      acc += xor_popcnt_AVX2_lookup(&A[m * QK], &B[n * QK], QK);
      C[m * Cstride + n] = acc;
    }
  }
}

template <size_t M, size_t N>
static void BM_qgess(benchmark::State &state) {
  size_t QK = state.range(0);
  std::vector<uint8_t> A(M * QK);
  std::vector<uint8_t> B(N * QK);
  std::vector<uint32_t> C(M * N);
  size_t iters = 0;
  while (state.KeepRunning()) {
    qgess<M, N>(A.data(), B.data(), C.data(), N, QK);
    ++iters;
  }
  state.counters["FLOPS"] = benchmark::Counter(2 * M * N * QK * 8 * iters,
                                               benchmark::Counter::kIsRate);
}

template <size_t M, size_t N>
static void BM_qpopcnt(benchmark::State &state) {
  size_t QK = state.range(0);
  std::vector<uint8_t> A(M * QK);
  std::vector<uint8_t> B(N * QK);
  std::vector<uint32_t> C(M * N);

  size_t iters = 0;
  while (state.KeepRunning()) {
    qpopcnt<M, N>(A.data(), B.data(), C.data(), N, QK);
    ++iters;
  }

  state.counters["FLOPS"] =
      benchmark::Counter(2 * M * N * QK * 8 * iters, benchmark::Counter::kIsRate);
}

template <size_t M, size_t N>
static void BM_sgemm(benchmark::State &state) {
  size_t QK = state.range(0);
  std::vector<float> A(M * QK * 8);
  std::vector<float> B(N * QK * 8);
  std::vector<float> C(M * N);

  size_t iters = 0;
  while (state.KeepRunning()) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, QK * 8, 1.0,
                A.data(), QK * 8, B.data(), QK * 8, 0.0, C.data(), N);
    ++iters;
  }

  state.counters["FLOPS"] =
      benchmark::Counter(2 * M * N * QK * 8 * iters, benchmark::Counter::kIsRate);
}

// Register the function as a benchmark
BENCHMARK_TEMPLATE2(BM_qgess, 8, 8)->RangeMultiplier(8)->Range(16, 8192);
BENCHMARK_TEMPLATE2(BM_qgess, 128, 128)->RangeMultiplier(8)->Range(16, 8192);
BENCHMARK_TEMPLATE2(BM_qgess, 256, 256)->RangeMultiplier(8)->Range(16, 8192);

BENCHMARK_TEMPLATE2(BM_qpopcnt, 8, 8)->RangeMultiplier(8)->Range(16, 8192);
BENCHMARK_TEMPLATE2(BM_qpopcnt, 128, 128)->RangeMultiplier(8)->Range(16, 8192);
BENCHMARK_TEMPLATE2(BM_qpopcnt, 256, 256)->RangeMultiplier(8)->Range(16, 8192);

BENCHMARK_TEMPLATE2(BM_sgemm, 32, 32)->RangeMultiplier(8)->Range(16, 8192);
BENCHMARK_TEMPLATE2(BM_sgemm, 512, 512)->RangeMultiplier(8)->Range(16, 8192);
BENCHMARK_TEMPLATE2(BM_sgemm, 1024, 1024)->RangeMultiplier(8)->Range(16, 8192);
BENCHMARK_MAIN();
