/* Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// bf16 / fp16 vectorized Mul fast paths.
//
// Mul is hit on every SwiGLU gating pass in tokenmixerlarge (and on every
// scale broadcast in MoE / GateValueScaling), so its bf16 throughput
// matters at the same scale as AddV2's. Structure mirrors
// musa_add_kernel.mu: contiguous vec8 (::uint4 = 8 elements per thread),
// scalar-broadcast scalar, and tail-vector-broadcast scalar paths.

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include <stdint.h>
#include <string.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/types.h"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

namespace {

constexpr int kMulThreadsPerBlock = 256;

static inline int64_t CeilDivMul(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

static inline bool IsAligned16Mul(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

__device__ __forceinline__ uint32_t mul_bf16_pair_packed(uint32_t a,
                                                          uint32_t b) {
  __mt_bfloat162 ap, bp;
  memcpy(&ap, &a, sizeof(ap));
  memcpy(&bp, &b, sizeof(bp));
  const float lo = __low2float(ap) * __low2float(bp);
  const float hi = __high2float(ap) * __high2float(bp);
  const __mt_bfloat162 prod = __floats2bfloat162_rn(lo, hi);
  uint32_t result;
  memcpy(&result, &prod, sizeof(result));
  return result;
}

__device__ __forceinline__ uint32_t mul_half_pair_packed(uint32_t a,
                                                          uint32_t b) {
  __half2 ap, bp;
  memcpy(&ap, &a, sizeof(ap));
  memcpy(&bp, &b, sizeof(bp));
  const __half2 prod = __hmul2(ap, bp);
  uint32_t result;
  memcpy(&result, &prod, sizeof(result));
  return result;
}

}  // namespace

extern "C" {

// ----- bf16 -----

__global__ void MulContiguousKernelBFloat16(const bfloat16* __restrict__ lhs,
                                             const bfloat16* __restrict__ rhs,
                                             bfloat16* __restrict__ output,
                                             int64_t size) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    const __mt_bfloat16 l =
        *reinterpret_cast<const __mt_bfloat16*>(&lhs[idx]);
    const __mt_bfloat16 r =
        *reinterpret_cast<const __mt_bfloat16*>(&rhs[idx]);
    const __mt_bfloat16 prod =
        __float2bfloat16(__bfloat162float(l) * __bfloat162float(r));
    *reinterpret_cast<__mt_bfloat16*>(&output[idx]) = prod;
  }
}

__global__ void MulContiguousKernelBFloat16Vec8(const ::uint4* __restrict__ lhs,
                                                 const ::uint4* __restrict__ rhs,
                                                 ::uint4* __restrict__ output,
                                                 int64_t vec_size) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < vec_size) {
    const ::uint4 l = lhs[idx];
    const ::uint4 r = rhs[idx];
    ::uint4 out;
    out.x = mul_bf16_pair_packed(l.x, r.x);
    out.y = mul_bf16_pair_packed(l.y, r.y);
    out.z = mul_bf16_pair_packed(l.z, r.z);
    out.w = mul_bf16_pair_packed(l.w, r.w);
    output[idx] = out;
  }
}

__global__ void MulScalarKernelBFloat16(const bfloat16* __restrict__ dense,
                                         const bfloat16* __restrict__ scalar,
                                         bfloat16* __restrict__ output,
                                         int64_t size) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    const __mt_bfloat16 s =
        *reinterpret_cast<const __mt_bfloat16*>(&scalar[0]);
    const float sf = __bfloat162float(s);
    const __mt_bfloat16 d =
        *reinterpret_cast<const __mt_bfloat16*>(&dense[idx]);
    const __mt_bfloat16 prod = __float2bfloat16(__bfloat162float(d) * sf);
    *reinterpret_cast<__mt_bfloat16*>(&output[idx]) = prod;
  }
}

__global__ void MulTailVectorKernelBFloat16(
    const bfloat16* __restrict__ dense,
    const bfloat16* __restrict__ tail_vector,
    bfloat16* __restrict__ output, int64_t size, int64_t width) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    const int64_t col = idx % width;
    const __mt_bfloat16 d =
        *reinterpret_cast<const __mt_bfloat16*>(&dense[idx]);
    const __mt_bfloat16 t =
        *reinterpret_cast<const __mt_bfloat16*>(&tail_vector[col]);
    const float prod_f = __bfloat162float(d) * __bfloat162float(t);
    const __mt_bfloat16 prod = __float2bfloat16(prod_f);
    *reinterpret_cast<__mt_bfloat16*>(&output[idx]) = prod;
  }
}

void LaunchMusaMulContiguousBFloat16(const void* lhs, const void* rhs,
                                      void* output, int64_t size,
                                      musaStream_t stream) {
  if (size <= 0) return;
  if (size >= 8 && IsAligned16Mul(lhs) && IsAligned16Mul(rhs) &&
      IsAligned16Mul(output)) {
    const int64_t vec_size = size / 8;
    const int64_t vec_blocks = CeilDivMul(vec_size, kMulThreadsPerBlock);
    MulContiguousKernelBFloat16Vec8<<<vec_blocks, kMulThreadsPerBlock, 0,
                                       stream>>>(
        reinterpret_cast<const ::uint4*>(lhs),
        reinterpret_cast<const ::uint4*>(rhs),
        reinterpret_cast<::uint4*>(output), vec_size);

    const int64_t tail = size - vec_size * 8;
    if (tail > 0) {
      const auto* l = reinterpret_cast<const bfloat16*>(lhs) + vec_size * 8;
      const auto* r = reinterpret_cast<const bfloat16*>(rhs) + vec_size * 8;
      auto* o = reinterpret_cast<bfloat16*>(output) + vec_size * 8;
      const int64_t tail_blocks = CeilDivMul(tail, kMulThreadsPerBlock);
      MulContiguousKernelBFloat16<<<tail_blocks, kMulThreadsPerBlock, 0,
                                     stream>>>(l, r, o, tail);
    }
    return;
  }
  const int64_t blocks = CeilDivMul(size, kMulThreadsPerBlock);
  MulContiguousKernelBFloat16<<<blocks, kMulThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const bfloat16*>(lhs),
      reinterpret_cast<const bfloat16*>(rhs),
      reinterpret_cast<bfloat16*>(output), size);
}

void LaunchMusaMulScalarBFloat16(const void* dense, const void* scalar,
                                  void* output, int64_t size,
                                  musaStream_t stream) {
  if (size <= 0) return;
  const int64_t blocks = CeilDivMul(size, kMulThreadsPerBlock);
  MulScalarKernelBFloat16<<<blocks, kMulThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const bfloat16*>(dense),
      reinterpret_cast<const bfloat16*>(scalar),
      reinterpret_cast<bfloat16*>(output), size);
}

void LaunchMusaMulTailVectorBFloat16(const void* dense,
                                      const void* tail_vector, void* output,
                                      int64_t size, int64_t width,
                                      musaStream_t stream) {
  if (size <= 0 || width <= 0 || size % width != 0) return;
  const int64_t blocks = CeilDivMul(size, kMulThreadsPerBlock);
  MulTailVectorKernelBFloat16<<<blocks, kMulThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const bfloat16*>(dense),
      reinterpret_cast<const bfloat16*>(tail_vector),
      reinterpret_cast<bfloat16*>(output), size, width);
}

// ----- fp16 -----

__global__ void MulContiguousKernelHalf(const Eigen::half* __restrict__ lhs,
                                         const Eigen::half* __restrict__ rhs,
                                         Eigen::half* __restrict__ output,
                                         int64_t size) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    const __half l = *reinterpret_cast<const __half*>(&lhs[idx]);
    const __half r = *reinterpret_cast<const __half*>(&rhs[idx]);
    const __half prod = __hmul(l, r);
    *reinterpret_cast<__half*>(&output[idx]) = prod;
  }
}

__global__ void MulContiguousKernelHalfVec8(const ::uint4* __restrict__ lhs,
                                             const ::uint4* __restrict__ rhs,
                                             ::uint4* __restrict__ output,
                                             int64_t vec_size) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < vec_size) {
    const ::uint4 l = lhs[idx];
    const ::uint4 r = rhs[idx];
    ::uint4 out;
    out.x = mul_half_pair_packed(l.x, r.x);
    out.y = mul_half_pair_packed(l.y, r.y);
    out.z = mul_half_pair_packed(l.z, r.z);
    out.w = mul_half_pair_packed(l.w, r.w);
    output[idx] = out;
  }
}

__global__ void MulScalarKernelHalf(const Eigen::half* __restrict__ dense,
                                     const Eigen::half* __restrict__ scalar,
                                     Eigen::half* __restrict__ output,
                                     int64_t size) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    const __half s = *reinterpret_cast<const __half*>(&scalar[0]);
    const __half d = *reinterpret_cast<const __half*>(&dense[idx]);
    *reinterpret_cast<__half*>(&output[idx]) = __hmul(d, s);
  }
}

__global__ void MulTailVectorKernelHalf(
    const Eigen::half* __restrict__ dense,
    const Eigen::half* __restrict__ tail_vector,
    Eigen::half* __restrict__ output, int64_t size, int64_t width) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    const int64_t col = idx % width;
    const __half d = *reinterpret_cast<const __half*>(&dense[idx]);
    const __half t = *reinterpret_cast<const __half*>(&tail_vector[col]);
    *reinterpret_cast<__half*>(&output[idx]) = __hmul(d, t);
  }
}

void LaunchMusaMulContiguousHalf(const void* lhs, const void* rhs, void* output,
                                  int64_t size, musaStream_t stream) {
  if (size <= 0) return;
  if (size >= 8 && IsAligned16Mul(lhs) && IsAligned16Mul(rhs) &&
      IsAligned16Mul(output)) {
    const int64_t vec_size = size / 8;
    const int64_t vec_blocks = CeilDivMul(vec_size, kMulThreadsPerBlock);
    MulContiguousKernelHalfVec8<<<vec_blocks, kMulThreadsPerBlock, 0,
                                   stream>>>(
        reinterpret_cast<const ::uint4*>(lhs),
        reinterpret_cast<const ::uint4*>(rhs),
        reinterpret_cast<::uint4*>(output), vec_size);

    const int64_t tail = size - vec_size * 8;
    if (tail > 0) {
      const auto* l = reinterpret_cast<const Eigen::half*>(lhs) + vec_size * 8;
      const auto* r = reinterpret_cast<const Eigen::half*>(rhs) + vec_size * 8;
      auto* o = reinterpret_cast<Eigen::half*>(output) + vec_size * 8;
      const int64_t tail_blocks = CeilDivMul(tail, kMulThreadsPerBlock);
      MulContiguousKernelHalf<<<tail_blocks, kMulThreadsPerBlock, 0, stream>>>(
          l, r, o, tail);
    }
    return;
  }
  const int64_t blocks = CeilDivMul(size, kMulThreadsPerBlock);
  MulContiguousKernelHalf<<<blocks, kMulThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const Eigen::half*>(lhs),
      reinterpret_cast<const Eigen::half*>(rhs),
      reinterpret_cast<Eigen::half*>(output), size);
}

void LaunchMusaMulScalarHalf(const void* dense, const void* scalar,
                              void* output, int64_t size, musaStream_t stream) {
  if (size <= 0) return;
  const int64_t blocks = CeilDivMul(size, kMulThreadsPerBlock);
  MulScalarKernelHalf<<<blocks, kMulThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const Eigen::half*>(dense),
      reinterpret_cast<const Eigen::half*>(scalar),
      reinterpret_cast<Eigen::half*>(output), size);
}

void LaunchMusaMulTailVectorHalf(const void* dense, const void* tail_vector,
                                  void* output, int64_t size, int64_t width,
                                  musaStream_t stream) {
  if (size <= 0 || width <= 0 || size % width != 0) return;
  const int64_t blocks = CeilDivMul(size, kMulThreadsPerBlock);
  MulTailVectorKernelHalf<<<blocks, kMulThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const Eigen::half*>(dense),
      reinterpret_cast<const Eigen::half*>(tail_vector),
      reinterpret_cast<Eigen::half*>(output), size, width);
}

}  // extern "C"

}  // namespace musa
}  // namespace tensorflow
