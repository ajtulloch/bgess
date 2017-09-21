#pragma once

#include "tensor.hpp"
#include "ulp.h"

namespace caffe2 {

// size_t divRoundUp(size_t a, size_t b) { return (a + (b - 1)) / b; }

// struct ConvArgs {
//   int stride_w{1};
//   int stride_h{1};
//   int pad_l{0};
//   int pad_t{0};
//   int pad_b{0};
//   int pad_r{0};
// };

// void signQuantize(const TensorCPU& X, TensorCPU* XQ) {
//   // CAFFE_ENFORCE_GT(X.ndim(), 1);
//   const auto N = X.size_to_dim(X.ndim() - 1);
//   auto C = X.size() / N;
//   const auto QC = divRoundUp(C, 8);
//   auto XQs = X.dims();
//   XQs[X.ndim() - 1] = QC;
//   XQ->Resize(XQs);
//   const float* Xdata = X.data<float>();
//   uint8_t* XQdata = XQ->mutable_data<uint8_t>();
//   for (auto n = 0; n < N; ++n) {
//     for (auto qc = 0; qc < QC; ++qc) {
//       // compute the block in X.
//       uint8_t p = 0;
//       for (auto b = 0; b < 8; ++b) {
//         const auto c = qc * 8 + b;
//         if (c < C) {
//           p |= (Xdata[c + C * n] > 0) << b;
//         }
//       }
//       XQdata[qc + QC * n] = p;
//     }
//   }
// }

void conv(const ConvArgs& args, const TensorCPU& X, const TensorCPU& W,
          const TensorCPU* b, TensorCPU* Y) {
  const auto N = X.dim(0);
  const auto IH = X.dim(1);
  const auto IW = X.dim(2);
  const auto KH = W.dim(1);
  const auto KW = W.dim(2);
  const auto IC = W.dim(3);
  Y->Resize(
      X.dim(0), (X.dim(1) - KH + args.pad_t + args.pad_b) / args.stride_h + 1,
      (X.dim(2) - KW + args.pad_l + args.pad_r) / args.stride_w + 1, W.dim(0));
  CHECK_EQ(W.dim(3), X.dim(3));
  const auto OH = Y->dim(1);
  const auto OW = Y->dim(2);
  const auto OC = Y->dim(3);

  const auto* Xdata = X.data<float>();
  const auto* Wdata = W.data<float>();
  auto* Ydata = Y->mutable_data<float>();
  for (auto n = 0; n < N; ++n) {
    for (auto oh = 0; oh < OH; ++oh) {
      for (auto ow = 0; ow < OW; ++ow) {
        for (auto oc = 0; oc < OC; ++oc) {
          float acc = b ? b->data<float>()[oc] : 0.0;
          for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
              for (int ic = 0; ic < IC; ++ic) {
                if (kh + args.stride_h * oh - args.pad_t < 0 ||
                    kh + args.stride_h * oh - args.pad_t >= IH ||
                    kw + args.stride_w * ow - args.pad_l < 0 ||
                    kw + args.stride_w * ow - args.pad_l >= IW) {
                  continue;
                }
                const auto x =
                    Xdata[ic + IC * (kw + args.stride_w * ow - args.pad_l) +
                          IC * IW * (kh + args.stride_h * oh - args.pad_t) +
                          n * IC * IW * IH];
                const auto w =
                    Wdata[ic + IC * kw + IC * KW * kh + IC * KW * KH * oc];
                acc += x * w;
              }
            }
          }
          Ydata[oc + OC * ow + OC * OW * oh + n * OC * OW * OH] = acc;
        }
      }
    }
  }
}

int randInt(int a, int b) {
  std::random_device rd;
  std::default_random_engine gen(rd());
  return std::uniform_int_distribution<int>(a, b)(gen);
}

TensorCPU genTensor11(std::vector<TIndex> shape) {
  TensorCPU r;
  r.Resize(shape);

  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_real_distribution<float> dis(0, 1);

  for (auto i = 0; i < r.size(); ++i) {
    r.mutable_data<float>()[i] = dis(gen) > 0.5 ? -1.0 : 1.0;
  };
  return r;
}

TensorCPU genTensorUniform11(std::vector<TIndex> shape) {
  TensorCPU r;
  r.Resize(shape);

  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_real_distribution<float> dis(-5.0, 5.0);

  for (auto i = 0; i < r.size(); ++i) {
    r.mutable_data<float>()[i] = dis(gen);
  };
  return r;
}

TensorCPU genTensor0123(std::vector<TIndex> shape) {
  TensorCPU r;
  r.Resize(shape);

  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_real_distribution<float> dis(0.1, 3.9);

  for (auto i = 0; i < r.size(); ++i) {
    r.mutable_data<float>()[i] = std::floor(dis(gen));
  };
  return r;
}

inline void gemmNT(int M, int N, int K, const float* A, const float* B,
                   float* C) {
  for (auto m = 0; m < M; ++m) {
    for (auto n = 0; n < N; ++n) {
      float acc = 0.0;
      for (auto k = 0; k < K; ++k) {
        acc += A[m * K + k] * B[n * K + k];
      }
      C[m * N + n] = acc;
    }
  }
}

inline void qgemmNT(int M, int N, int K, const uint8_t* A, const uint8_t* B,
                    float* C) {
  CHECK_EQ(K % 8, 0);
  const int QK = K / 8;
  for (auto m = 0; m < M; ++m) {
    for (auto n = 0; n < N; ++n) {
      float acc = 0.0;
      for (auto qk = 0; qk < QK; ++qk) {
        uint8_t mk = A[m * QK + qk];
        uint8_t nk = B[n * QK + qk];
        auto cnt = __builtin_popcount(mk ^ nk);
        acc += cnt;
      }
      C[m * N + n] = K - 2 * acc;
    }
  }
}

}  // namespace caffe2
