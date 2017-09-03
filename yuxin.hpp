#pragma once

#include <x86intrin.h>
#include <cassert>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>

// A 4D tensor with the last dimension packed to 64bit integer.
class BitTensor {
 public:
  int dim[4];  // true (full) dimension
  int nr_c;
  uint64_t* ptr;

  BitTensor(int n, int h, int w, int c) : dim{n, h, w, c} {
    nr_c = (c - 1) / 64 + 1;
    sz_ = n * h * w * nr_c;
    offset_[0] = h * w * nr_c;
    offset_[1] = w * nr_c;
    offset_[2] = nr_c;
    ptr = new uint64_t[sz_]();
  }

  BitTensor(const BitTensor&) = delete;
  BitTensor& operator=(const BitTensor&) = delete;

  // access (x, y, z, 0)
  uint64_t& operator()(int x, int y, int z) {
    return ptr[x * offset_[0] + y * offset_[1] + z * offset_[2]];
  }
  const uint64_t& operator()(int x, int y, int z) const {
    return ptr[x * offset_[0] + y * offset_[1] + z * offset_[2]];
  }

  // access (x, y, z, w (packed dimension))
  uint64_t& operator()(int x, int y, int z, int w) {
    return ptr[x * offset_[0] + y * offset_[1] + z * offset_[2] + w];
  }
  const uint64_t& operator()(int x, int y, int z, int w) const {
    return ptr[x * offset_[0] + y * offset_[1] + z * offset_[2] + w];
  }

  int bufsize() const { return sz_; }
  int offset(int x) const { return offset_[x]; }

  void random() {
    std::default_random_engine generator;
    std::uniform_int_distribution<uint64_t> distribution(0, UINTMAX_MAX);
    for (int i = 0; i < dim[0]; ++i)
      for (int j = 0; j < dim[1]; ++j)
        for (int k = 0; k < dim[2]; ++k)
          for (int l = 0; l < nr_c; ++l) {
            auto& v = operator()(i, j, k, l);
            v = distribution(generator);
            if (l == nr_c - 1 && dim[3] % 64 > 0)
              v &= ((1ULL << (dim[3] % 64)) - 1);
          }
  }

  // fin: "0 1 0 1 1 ..." no new line, no size info
  void read_01_text(std::ifstream& fin) {
    for (int n = 0; n < dim[0]; ++n)
      for (int h = 0; h < dim[1]; ++h)
        for (int w = 0; w < dim[2]; ++w) {
          int p = 0, s = 0;
          uint64_t x;
          for (int c = 0; c < dim[3]; ++c) {
            fin >> x;
            operator()(n, h, w, p) |= x << s;
            if (++s == 64) s = 0, ++p;
          }
        }
  }

 private:
  int sz_, offset_[3];
};

void fc(const BitTensor& paraW, const BitTensor& input, BitTensor& output,
        bool binary_output) {
  int N = input.dim[0];
  // input: (n, h, w, ic)
  // output: (n, h', w', oc')
  assert(input.dim[0] == output.dim[0]);
  int M = output.offset(0);
  int total = input.dim[1] * input.dim[2] * input.dim[3];
  auto optr = &output(0, 0, 0, 0);
  for (int n = 0; n < N; ++n) {  // iterate over N
    auto wptr = &paraW(0, 0, 0, 0);
    for (int m = 0; m < M; ++m) {  // iterate over output_size/64
      auto& res = binary_output ? *(optr++) : *(optr);
      int limit =
          (!binary_output) ? 1 : (m < M - 1 ? 64 : (output.dim[3] % 64));
      for (int k = 0; k < std::min(64, limit); ++k) {  // iterate in uint64_t
        // int oc = m * 64 + k;
        auto iptr = &input(n, 0, 0, 0);
        int sum = 0;
        int S = paraW.offset(0);
        // sum += _mm_popcnt_u64((*(iptr++)) ^ (*(wptr++)));

        // beginning of popcnt kernel
        __m256i a, b, s = _mm256_setzero_si256();
        const __m256i vmask = _mm256_set1_epi8(0xf);
        const __m256i vpop =
            _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0,
                             1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
        const __m256i vzero = _mm256_setzero_si256();

#define INTRINSICS(i)                                 \
  a = _mm256_loadu_si256((__m256i const*)(iptr + i)); \
  b = _mm256_loadu_si256((__m256i const*)(wptr + i)); \
  b = _mm256_xor_si256(a, b);                         \
  a = _mm256_and_si256(b, vmask);                     \
  b = _mm256_srli_epi32(b, 4);                        \
  b = _mm256_and_si256(b, vmask);                     \
  b = _mm256_shuffle_epi8(vpop, b);                   \
  a = _mm256_shuffle_epi8(vpop, a);                   \
  a = _mm256_add_epi8(a, b);                          \
  r = _mm256_add_epi8(r, a);
        int i = 0;
        for (; i + 31 < S; i += 32) {
          __m256i r = _mm256_setzero_si256();
          INTRINSICS(0);
          INTRINSICS(4);
          INTRINSICS(8);
          INTRINSICS(12);
          INTRINSICS(16);
          INTRINSICS(20);
          INTRINSICS(24);
          INTRINSICS(28);
          s = _mm256_add_epi64(s, _mm256_sad_epu8(r, vzero));
          iptr += 32;
          wptr += 32;
        }
        __m256i r = _mm256_setzero_si256();
        for (; i + 3 < S; i += 4) {
          INTRINSICS(0);
          iptr += 4;
          wptr += 4;
        }
#undef INTRINSICS
        s = _mm256_add_epi64(s, _mm256_sad_epu8(r, vzero));
        for (; i < S; ++i) {
          uint64_t t = (*(iptr++)) ^ (*(wptr++));
          sum += _mm_popcnt_u64(t);
        }
        sum += static_cast<uint64_t>(_mm256_extract_epi64(s, 0));
        sum += static_cast<uint64_t>(_mm256_extract_epi64(s, 1));
        sum += static_cast<uint64_t>(_mm256_extract_epi64(s, 2));
        sum += static_cast<uint64_t>(_mm256_extract_epi64(s, 3));
        // end of popcnt kernel

        if (binary_output) {
          res |= (uint64_t)((total >> 1) >= sum) << k;
        } else {
          // printf("%d, %d\n", total, sum);
          *(optr++) = total - (sum << 1);
        }
      }
    }
  }
}

namespace {

// NOTICE: assume dst or data has enough length!

inline void bit_fill(uint64_t* dst, const uint64_t* src, int len, int offset) {
  int k = offset >> 6, r = offset & 63, t = 64 - r;
  dst += k;
  const uint64_t* end = src + len - 1;
  --src;
  for (;;) {
    if (++src == end) break;
    *(dst) |= *(src) << r;
    *(++dst) |= r ? (*(src) >> t) : 0;
  }
  *dst |= *(src) << r;
}

inline void bit_rshift(uint64_t* data, int len, int offset) {
  int k = offset >> 6, r = offset & 63, mask = r ? (1ULL << (64 - r)) - 1 : 0;
  __m128i m128 = _mm_set1_epi64x(mask);
  __m256i m256 = _mm256_set1_epi64x(mask);
  uint64_t *src = data + k, *dst = data, *end = data + len - k - 1;
  while (dst + 3 < end) {
    __m256i l = _mm256_loadu_si256((__m256i*)src),
            h = _mm256_loadu_si256((__m256i*)src + 1);
    l = _mm256_srli_epi64(l, r);
    h = _mm256_and_si256(h, m256);
    _mm256_storeu_si256((__m256i*)dst, _mm256_or_si256(h, l));
    dst += 4;
    src += 4;
  }
  while (dst + 1 < end) {
    __m128i l = _mm_loadu_si128((__m128i*)src),
            h = _mm_loadu_si128((__m128i*)src + 1);
    l = _mm_srli_epi64(l, r);
    h = _mm_and_si128(h, m128);
    _mm_storeu_si128((__m128i*)dst, _mm_or_si128(h, l));
    dst += 2;
    src += 2;
  }
  while (dst < end) {
    *(dst++) = (*(src) >> r) | (*(src + 1) & mask);
    src++;
  }
  *(dst++) = *(src) >> r;
  end = data + len;
  memset(dst, 0, sizeof(uint64_t) * (end - dst));
}
}  // namespace

// conv
__attribute__((noinline)) size_t conv(const BitTensor& paraW, const BitTensor& input, BitTensor& output) {
  //size_t ops = 0;
    int N = input.dim[0], H = input.dim[1], W = input.dim[2];
    int IC = input.dim[3], OC = output.dim[3];
    int KH = paraW.dim[1], KW = paraW.dim[2], KS = KH * KW;
    int ks_x_ic = KS * IC;
    int nr_ic = (IC - 1) / 64 + 1; // nr of u64 for i channels
    int nr = (ks_x_ic - 1) / 64 + 1; // nr of u64 in ker
    __m256i m_ks_x_ic = _mm256_set1_epi64x(KS * IC / 2 + 1);

    // input: (n, h, w, ic)
    // output: (n, h, w, oc)
    // W: (oc, h, w, ic)

    // ker [h, w, ic]
    uint64_t __ker[OC][nr];
    memset(__ker, 0, sizeof(__ker));
    const uint64_t *wptr = paraW.ptr;
    for (int oc = 0; oc < OC; ++oc) {
        for (int h = 0; h < KH; ++h)
            for (int w = 0; w < KW; ++w) {
                bit_fill(__ker[oc], wptr, nr_ic, h*KW*IC+w*IC);
                wptr += nr_ic;
            }
    }

    // transpose ker
    uint64_t ker[nr][OC];
    for (int c = 0; c < nr; ++c)
        for (int oc = 0; oc < OC; ++oc)
            ker[c][oc] = __ker[oc][c];

    // #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        uint64_t patch[W][nr];  // patch[w]: a khxkwxic patch in the current stripe starting from w
        memset(patch, 0, sizeof(patch));

        // load the first several rows from input into patch
        // & compute the first row of output
        uint64_t *optr = &output(i, 0, 0, 0);
        for (int y = 0; y < W-KW+1; ++y) {
            for (int h = 0; h < KH; ++h) {
                for (int w = 0; w < KW; ++w) {
                    bit_fill(patch[y], &input(i, h, y+w, 0), nr_ic, h*KW*IC+w*IC);
                    /*for (int c = 0; c < nr_ic; ++c) {
                        uint64_t val = input(i, h, y+w, c);
                        if (c == nr_ic - 1 && (IC & 63)) val &= (1ULL << (IC & 63)) - 1;
                        bit_fill(patch[y], val, h*KW*IC+w*IC+c*64);
                        //patch[y] |= input(i, h, y+w) << (h*KW*IC+w*IC);
                    }*/
                }
            }
            for (int oc = 0; oc < OC; ++oc) {
                //*optr |= static_cast<uint64_t>(int(KS*IC - (_mm_popcnt_u64(ker[oc] ^ patch[y]) << 1)) >= 0) << oc;
                if (oc && (oc & 63) == 0) ++optr;
                int cnt = 0, k = 0;
                #define UNROLL(i) \
                  cnt += _mm_popcnt_u64(__ker[oc][k+i] ^ patch[y][k+i]); \
                  //ops += 128;                                         \

                for (; k + 7 < nr; k += 8) {
                    UNROLL(0); UNROLL(1); UNROLL(2); UNROLL(3);
                    UNROLL(4); UNROLL(5); UNROLL(6); UNROLL(7);
                }
                for (; k < nr; ++k) UNROLL(0);
                #undef UNROLL
                *optr |= static_cast<uint64_t>(int(ks_x_ic - (cnt << 1)) >= 0) << (oc&63);
            }
            ++optr;
        }

        const uint64_t *iptr = &input(i, KH, 0, 0);
        int nr_stripe = (KW * IC - 1) / 64 + 1;
        for (int h = 1; h < H-KH+1; ++h) {
            uint64_t s[nr_stripe];  // load a input patch starting from (h, 0)
            memset(s, 0, sizeof(s));
            for (int w = 0; w < KW-1; ++w) {  // only load up to KW-1
                //s |= *(iptr++) << (w*IC+IC);
                bit_fill(s, iptr, nr_ic, w*IC+IC);
                iptr += nr_ic;
            }

            for (int w = 0; w < W-KW+1; ++w) {
                //s = (*(iptr++) << (KW*IC-IC)) | (s >> IC);
                // shift & load nr_ic params for one extra w
                bit_rshift(s, nr_stripe, IC);
                bit_fill(s, iptr, nr_ic, KW*IC-IC);
                iptr += nr_ic;

                //cur_patch = (cur_patch >> (KW*IC)) | ( s << ((KS-KW)*IC) );
                uint64_t *cur_patch = patch[w];
                bit_rshift(cur_patch, nr, KW*IC);
                bit_fill(cur_patch, s, nr_stripe, (KS-KW)*IC);

                int ls = 0;
                for (int oc_start = 0; oc_start < OC; oc_start += 16) {
                    __m256i sum[4];
                    sum[0] = _mm256_setzero_si256();
                    sum[1] = _mm256_setzero_si256();
                    sum[2] = _mm256_setzero_si256();
                    sum[3] = _mm256_setzero_si256();
                    for (int c = 0; c < nr; ++c) {
                        __m256i m_cur_patch = _mm256_set1_epi64x(cur_patch[c]);
                        auto ker_ptr = ker[c] + oc_start;

                        // popcnt kernel. accumulate popcnt(m_cur_patch ^ ker_ptr[i]) to s
                        #define INTRINSICS(i, s) {\
                            __m256i a, b;\
                            b = _mm256_loadu_si256((__m256i const*)(ker_ptr + i));\
                            b = _mm256_xor_si256(b, m_cur_patch);\
                            a = _mm256_and_si256(b, vmask);\
                            b = _mm256_srli_epi16(b, 4);\
                            b = _mm256_and_si256(b, vmask);\
                            b = _mm256_shuffle_epi8(vpop, b);\
                            a = _mm256_shuffle_epi8(vpop, a);\
                            a = _mm256_add_epi8(a, b);\
                            s = _mm256_add_epi8(s, a);\
                        }  // XXX have the risk of overflow using 8bit only. might need a sad
                            //s = _mm256_add_epi64(s, _mm256_sad_epu8(a, vzero));
                        //const __m256i vzero = _mm256_setzero_si256();
                        const __m256i vmask = _mm256_set1_epi8(0xf);
                        const __m256i vpop = _mm256_setr_epi8(
                                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
                            );
                        // FIXME: assume oc=4k
                        if (OC - oc_start >= 16) {
                            INTRINSICS(0, sum[0]);
                            INTRINSICS(4, sum[1]);
                            INTRINSICS(8, sum[2]);
                            INTRINSICS(12, sum[3]);
                        }
                        else if (OC - oc_start >= 12) {
                            INTRINSICS(0, sum[0]);
                            INTRINSICS(4, sum[1]);
                            INTRINSICS(8, sum[2]);
                        }
                        else if (OC - oc_start >= 8) {
                            INTRINSICS(0, sum[0]);
                            INTRINSICS(4, sum[1]);
                        }
                        else if (OC - oc_start >= 4) {
                            INTRINSICS(0, sum[0]);
                        }
                        #undef INTRINSICS
                    }

                    for (int oc = 0; oc < 4 && oc_start + (oc << 2) < OC; ++oc) {
                        if (ls == 64) ls = 0, ++optr;
                        const __m256i vzero = _mm256_setzero_si256();
                        sum[oc] = _mm256_sad_epu8(sum[oc], vzero);
                        sum[oc] = _mm256_cmpgt_epi64(m_ks_x_ic, sum[oc]);
                        *(optr) |= (uint64_t)_mm256_movemask_pd(_mm256_castsi256_pd(sum[oc])) << ls;
                        ls += 4;
                    }
                }
                ++optr;
            }
        }
    }
    //return ops;
    return 0;
}
