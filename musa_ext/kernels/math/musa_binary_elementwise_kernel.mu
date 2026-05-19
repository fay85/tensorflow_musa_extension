#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <stdint.h>
#include <string.h>

namespace tensorflow {
namespace musa {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kItemsPerThread = 4;
constexpr int kMaxBlocks = 4096;
constexpr int kMaxGridY = 65535;

static inline int64_t CeilDiv(int64_t x, int64_t y) { return (x + y - 1) / y; }

static inline int ClampBlocks(int64_t items, int64_t items_per_block) {
  int64_t blocks = CeilDiv(items, items_per_block);
  if (blocks < 1) return 1;
  return blocks > kMaxBlocks ? kMaxBlocks : static_cast<int>(blocks);
}

static inline int ClampGridY(int64_t rows) {
  if (rows < 1) return 1;
  return rows > kMaxGridY ? kMaxGridY : static_cast<int>(rows);
}

static inline bool IsAligned16(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

template <typename T>
struct AddOp {
  __device__ __forceinline__ T operator()(T lhs, T rhs) const {
    return lhs + rhs;
  }
};

template <typename T>
struct MulOp {
  __device__ __forceinline__ T operator()(T lhs, T rhs) const {
    return lhs * rhs;
  }
};

__global__ __launch_bounds__(kThreadsPerBlock) void AddFloat4ContiguousKernel(
    const float4* __restrict__ lhs, const float4* __restrict__ rhs,
    float4* __restrict__ out, int64_t vec_n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < vec_n; idx += stride) {
    const float4 l = lhs[idx];
    const float4 r = rhs[idx];
    out[idx] = make_float4(l.x + r.x, l.y + r.y, l.z + r.z, l.w + r.w);
  }
}

__global__ __launch_bounds__(kThreadsPerBlock) void MulFloat4ContiguousKernel(
    const float4* __restrict__ lhs, const float4* __restrict__ rhs,
    float4* __restrict__ out, int64_t vec_n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < vec_n; idx += stride) {
    const float4 l = lhs[idx];
    const float4 r = rhs[idx];
    out[idx] = make_float4(l.x * r.x, l.y * r.y, l.z * r.z, l.w * r.w);
  }
}

__global__ __launch_bounds__(kThreadsPerBlock) void AddFloat4ScalarKernel(
    const float4* __restrict__ dense, const float* __restrict__ scalar,
    float4* __restrict__ out, int64_t vec_n) {
  const float s = scalar[0];
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < vec_n; idx += stride) {
    const float4 d = dense[idx];
    out[idx] = make_float4(d.x + s, d.y + s, d.z + s, d.w + s);
  }
}

__global__ __launch_bounds__(kThreadsPerBlock) void MulFloat4ScalarKernel(
    const float4* __restrict__ dense, const float* __restrict__ scalar,
    float4* __restrict__ out, int64_t vec_n) {
  const float s = scalar[0];
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < vec_n; idx += stride) {
    const float4 d = dense[idx];
    out[idx] = make_float4(d.x * s, d.y * s, d.z * s, d.w * s);
  }
}

__global__ __launch_bounds__(kThreadsPerBlock) void AddTailVector2DFloatKernel(
    const float* __restrict__ dense, const float* __restrict__ tail_vector,
    float* __restrict__ out, int64_t rows, int64_t width, bool vector_on_left) {
  for (int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
    int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; col < width; col += stride) {
      const int64_t offset = row * width + col;
      const float d = dense[offset];
      const float v = tail_vector[col];
      out[offset] = vector_on_left ? v + d : d + v;
    }
  }
}

__global__ __launch_bounds__(kThreadsPerBlock) void MulTailVector2DFloatKernel(
    const float* __restrict__ dense, const float* __restrict__ tail_vector,
    float* __restrict__ out, int64_t rows, int64_t width, bool vector_on_left) {
  for (int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
    int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; col < width; col += stride) {
      const int64_t offset = row * width + col;
      const float d = dense[offset];
      const float v = tail_vector[col];
      out[offset] = vector_on_left ? v * d : d * v;
    }
  }
}

struct BFloat16AddOp {
  __device__ __forceinline__ __mt_bfloat16 operator()(__mt_bfloat16 lhs,
                                                      __mt_bfloat16 rhs) const {
    return __float2bfloat16(__bfloat162float(lhs) + __bfloat162float(rhs));
  }
};

struct BFloat16MulOp {
  __device__ __forceinline__ __mt_bfloat16 operator()(__mt_bfloat16 lhs,
                                                      __mt_bfloat16 rhs) const {
    return __float2bfloat16(__bfloat162float(lhs) * __bfloat162float(rhs));
  }
};

// bf16 / fp16 Mul fast paths: vec8 (::uint4) contiguous kernels with packed RNE
// multiply, plus scalar / tail-vector broadcast variants (SwiGLU-scale hot paths).

__device__ __forceinline__ uint32_t MulBf16PairPacked(uint32_t a, uint32_t b) {
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

__device__ __forceinline__ uint32_t MulHalfPairPacked(uint32_t a, uint32_t b) {
  __half2 ap, bp;
  memcpy(&ap, &a, sizeof(ap));
  memcpy(&bp, &b, sizeof(bp));
  const __half2 prod = __hmul2(ap, bp);
  uint32_t result;
  memcpy(&result, &prod, sizeof(result));
  return result;
}

__global__ __launch_bounds__(kThreadsPerBlock)
void MulBFloat16ContiguousVec8Kernel(const ::uint4* __restrict__ lhs,
                                     const ::uint4* __restrict__ rhs,
                                     ::uint4* __restrict__ output,
                                     int64_t vec_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < vec_size; idx += stride) {
    const ::uint4 l = lhs[idx];
    const ::uint4 r = rhs[idx];
    ::uint4 out;
    out.x = MulBf16PairPacked(l.x, r.x);
    out.y = MulBf16PairPacked(l.y, r.y);
    out.z = MulBf16PairPacked(l.z, r.z);
    out.w = MulBf16PairPacked(l.w, r.w);
    output[idx] = out;
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)
void MulHalfContiguousVec8Kernel(const ::uint4* __restrict__ lhs,
                                 const ::uint4* __restrict__ rhs,
                                 ::uint4* __restrict__ output,
                                 int64_t vec_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < vec_size; idx += stride) {
    const ::uint4 l = lhs[idx];
    const ::uint4 r = rhs[idx];
    ::uint4 out;
    out.x = MulHalfPairPacked(l.x, r.x);
    out.y = MulHalfPairPacked(l.y, r.y);
    out.z = MulHalfPairPacked(l.z, r.z);
    out.w = MulHalfPairPacked(l.w, r.w);
    output[idx] = out;
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)
void MulBFloat16ScalarKernel(const __mt_bfloat16* __restrict__ dense,
                             const __mt_bfloat16* __restrict__ scalar,
                             __mt_bfloat16* __restrict__ output, int64_t n) {
  const float sf = __bfloat162float(scalar[0]);
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < n; idx += stride) {
    output[idx] =
        __float2bfloat16(__bfloat162float(dense[idx]) * sf);
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)
void MulHalfScalarKernel(const half* __restrict__ dense,
                         const half* __restrict__ scalar, half* __restrict__ output,
                         int64_t n) {
  const __half s = *reinterpret_cast<const __half*>(scalar);
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < n; idx += stride) {
    const __half d = *reinterpret_cast<const __half*>(&dense[idx]);
    *reinterpret_cast<__half*>(&output[idx]) = __hmul(d, s);
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)
void MulBFloat16TailVectorKernel(const __mt_bfloat16* __restrict__ dense,
                                 const __mt_bfloat16* __restrict__ tail_vector,
                                 __mt_bfloat16* __restrict__ output, int64_t n,
                                 int64_t width) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < n; idx += stride) {
    const int64_t col = idx % width;
    output[idx] = __float2bfloat16(__bfloat162float(dense[idx]) *
                                   __bfloat162float(tail_vector[col]));
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)
void MulHalfTailVectorKernel(const half* __restrict__ dense,
                             const half* __restrict__ tail_vector,
                             half* __restrict__ output, int64_t n,
                             int64_t width) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < n; idx += stride) {
    const int64_t col = idx % width;
    const __half d = *reinterpret_cast<const __half*>(&dense[idx]);
    const __half t = *reinterpret_cast<const __half*>(&tail_vector[col]);
    *reinterpret_cast<__half*>(&output[idx]) = __hmul(d, t);
  }
}

template <typename T, typename Op>
__global__ __launch_bounds__(kThreadsPerBlock) void BinaryContiguousKernel(
    const T* __restrict__ lhs, const T* __restrict__ rhs, T* __restrict__ out,
    int64_t n, Op op) {
  int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
                kItemsPerThread;
  const int64_t stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x * kItemsPerThread;
  for (; idx < n; idx += stride) {
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
      const int64_t offset = idx + i;
      if (offset < n) {
        out[offset] = op(lhs[offset], rhs[offset]);
      }
    }
  }
}

template <typename T, typename Op>
__global__ __launch_bounds__(kThreadsPerBlock) void BinaryScalarKernel(
    const T* __restrict__ dense, const T* __restrict__ scalar,
    T* __restrict__ out, int64_t n, Op op, bool scalar_on_left) {
  const T scalar_value = scalar[0];
  int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
                kItemsPerThread;
  const int64_t stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x * kItemsPerThread;
  for (; idx < n; idx += stride) {
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
      const int64_t offset = idx + i;
      if (offset < n) {
        out[offset] = scalar_on_left ? op(scalar_value, dense[offset])
                                     : op(dense[offset], scalar_value);
      }
    }
  }
}

template <typename T, typename Op>
__global__ __launch_bounds__(kThreadsPerBlock) void BinaryTailVectorKernel(
    const T* __restrict__ dense, const T* __restrict__ tail_vector,
    T* __restrict__ out, int64_t n, int64_t width, Op op, bool vector_on_left) {
  int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
                kItemsPerThread;
  const int64_t stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x * kItemsPerThread;
  for (; idx < n; idx += stride) {
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
      const int64_t offset = idx + i;
      if (offset < n) {
        const T vector_value = tail_vector[offset % width];
        out[offset] = vector_on_left ? op(vector_value, dense[offset])
                                     : op(dense[offset], vector_value);
      }
    }
  }
}

template <typename T, typename Op>
void LaunchContiguousTyped(const T* lhs, const T* rhs, T* out, int64_t n,
                           musaStream_t stream, Op op) {
  if (n <= 0) return;
  const int blocks = ClampBlocks(n, kThreadsPerBlock * kItemsPerThread);
  BinaryContiguousKernel<T, Op>
      <<<blocks, kThreadsPerBlock, 0, stream>>>(lhs, rhs, out, n, op);
}

template <typename T, typename Op>
void LaunchScalarTyped(const T* dense, const T* scalar, T* out, int64_t n,
                       bool scalar_on_left, musaStream_t stream, Op op) {
  if (n <= 0) return;
  const int blocks = ClampBlocks(n, kThreadsPerBlock * kItemsPerThread);
  BinaryScalarKernel<T, Op><<<blocks, kThreadsPerBlock, 0, stream>>>(
      dense, scalar, out, n, op, scalar_on_left);
}

template <typename T, typename Op>
void LaunchTailVectorTyped(const T* dense, const T* tail_vector, T* out,
                           int64_t n, int64_t width, bool vector_on_left,
                           musaStream_t stream, Op op) {
  if (n <= 0 || width <= 0 || n % width != 0) return;
  const int blocks = ClampBlocks(n, kThreadsPerBlock * kItemsPerThread);
  BinaryTailVectorKernel<T, Op><<<blocks, kThreadsPerBlock, 0, stream>>>(
      dense, tail_vector, out, n, width, op, vector_on_left);
}

}  // namespace

extern "C" {

void LaunchMusaBinaryAddContiguousFloat(const float* lhs, const float* rhs,
                                        float* out, int64_t n,
                                        musaStream_t stream) {
  if (n >= 4 && (n % 4) == 0 && IsAligned16(lhs) && IsAligned16(rhs) &&
      IsAligned16(out)) {
    const int64_t vec_n = n / 4;
    const int blocks = ClampBlocks(vec_n, kThreadsPerBlock);
    AddFloat4ContiguousKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const float4*>(lhs),
        reinterpret_cast<const float4*>(rhs), reinterpret_cast<float4*>(out),
        vec_n);
    return;
  }
  LaunchContiguousTyped(lhs, rhs, out, n, stream, AddOp<float>());
}

void LaunchMusaBinaryAddScalarFloat(const float* dense, const float* scalar,
                                    float* out, int64_t n, bool scalar_on_left,
                                    musaStream_t stream) {
  if (n >= 4 && (n % 4) == 0 && IsAligned16(dense) && IsAligned16(out)) {
    const int64_t vec_n = n / 4;
    const int blocks = ClampBlocks(vec_n, kThreadsPerBlock);
    AddFloat4ScalarKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const float4*>(dense), scalar,
        reinterpret_cast<float4*>(out), vec_n);
    return;
  }
  LaunchScalarTyped(dense, scalar, out, n, scalar_on_left, stream,
                    AddOp<float>());
}

void LaunchMusaBinaryAddTailVectorFloat(const float* dense,
                                        const float* tail_vector, float* out,
                                        int64_t n, int64_t width,
                                        bool vector_on_left,
                                        musaStream_t stream) {
  if (n > 0 && width > 0 && n % width == 0 && width <= 4096) {
    const int64_t rows = n / width;
    const int x_blocks = ClampBlocks(width, kThreadsPerBlock);
    const dim3 grid(x_blocks, ClampGridY(rows), 1);
    AddTailVector2DFloatKernel<<<grid, kThreadsPerBlock, 0, stream>>>(
        dense, tail_vector, out, rows, width, vector_on_left);
    return;
  }
  LaunchTailVectorTyped(dense, tail_vector, out, n, width, vector_on_left,
                        stream, AddOp<float>());
}

void LaunchMusaBinaryMulContiguousFloat(const float* lhs, const float* rhs,
                                        float* out, int64_t n,
                                        musaStream_t stream) {
  if (n >= 4 && (n % 4) == 0 && IsAligned16(lhs) && IsAligned16(rhs) &&
      IsAligned16(out)) {
    const int64_t vec_n = n / 4;
    const int blocks = ClampBlocks(vec_n, kThreadsPerBlock);
    MulFloat4ContiguousKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const float4*>(lhs),
        reinterpret_cast<const float4*>(rhs), reinterpret_cast<float4*>(out),
        vec_n);
    return;
  }
  LaunchContiguousTyped(lhs, rhs, out, n, stream, MulOp<float>());
}

void LaunchMusaBinaryMulScalarFloat(const float* dense, const float* scalar,
                                    float* out, int64_t n, bool scalar_on_left,
                                    musaStream_t stream) {
  if (n >= 4 && (n % 4) == 0 && IsAligned16(dense) && IsAligned16(out)) {
    const int64_t vec_n = n / 4;
    const int blocks = ClampBlocks(vec_n, kThreadsPerBlock);
    MulFloat4ScalarKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const float4*>(dense), scalar,
        reinterpret_cast<float4*>(out), vec_n);
    return;
  }
  LaunchScalarTyped(dense, scalar, out, n, scalar_on_left, stream,
                    MulOp<float>());
}

void LaunchMusaBinaryMulTailVectorFloat(const float* dense,
                                        const float* tail_vector, float* out,
                                        int64_t n, int64_t width,
                                        bool vector_on_left,
                                        musaStream_t stream) {
  if (n > 0 && width > 0 && n % width == 0 && width <= 4096) {
    const int64_t rows = n / width;
    const int x_blocks = ClampBlocks(width, kThreadsPerBlock);
    const dim3 grid(x_blocks, ClampGridY(rows), 1);
    MulTailVector2DFloatKernel<<<grid, kThreadsPerBlock, 0, stream>>>(
        dense, tail_vector, out, rows, width, vector_on_left);
    return;
  }
  LaunchTailVectorTyped(dense, tail_vector, out, n, width, vector_on_left,
                        stream, MulOp<float>());
}

void LaunchMusaBinaryAddContiguousHalf(const half* lhs, const half* rhs,
                                       half* out, int64_t n,
                                       musaStream_t stream) {
  LaunchContiguousTyped(lhs, rhs, out, n, stream, AddOp<half>());
}

void LaunchMusaBinaryAddScalarHalf(const half* dense, const half* scalar,
                                   half* out, int64_t n, bool scalar_on_left,
                                   musaStream_t stream) {
  LaunchScalarTyped(dense, scalar, out, n, scalar_on_left, stream,
                    AddOp<half>());
}

void LaunchMusaBinaryAddTailVectorHalf(const half* dense,
                                       const half* tail_vector, half* out,
                                       int64_t n, int64_t width,
                                       bool vector_on_left,
                                       musaStream_t stream) {
  LaunchTailVectorTyped(dense, tail_vector, out, n, width, vector_on_left,
                        stream, AddOp<half>());
}

void LaunchMusaBinaryMulContiguousHalf(const half* lhs, const half* rhs,
                                       half* out, int64_t n,
                                       musaStream_t stream) {
  if (n <= 0) return;
  if (n >= 8 && IsAligned16(lhs) && IsAligned16(rhs) && IsAligned16(out)) {
    const int64_t vec_size = n / 8;
    const int blocks = ClampBlocks(vec_size, kThreadsPerBlock);
    MulHalfContiguousVec8Kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const ::uint4*>(lhs),
        reinterpret_cast<const ::uint4*>(rhs),
        reinterpret_cast<::uint4*>(out), vec_size);
    const int64_t tail = n - vec_size * 8;
    if (tail > 0) {
      const half* l = lhs + vec_size * 8;
      const half* r = rhs + vec_size * 8;
      half* o = out + vec_size * 8;
      const int tail_blocks = ClampBlocks(tail, kThreadsPerBlock * kItemsPerThread);
      LaunchContiguousTyped(l, r, o, tail, stream, MulOp<half>());
    }
    return;
  }
  LaunchContiguousTyped(lhs, rhs, out, n, stream, MulOp<half>());
}

void LaunchMusaBinaryMulScalarHalf(const half* dense, const half* scalar,
                                   half* out, int64_t n, bool scalar_on_left,
                                   musaStream_t stream) {
  if (n <= 0) return;
  if (scalar_on_left) {
    LaunchScalarTyped(dense, scalar, out, n, scalar_on_left, stream,
                     MulOp<half>());
    return;
  }
  const int blocks = ClampBlocks(n, kThreadsPerBlock);
  MulHalfScalarKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(dense, scalar,
                                                               out, n);
}

void LaunchMusaBinaryMulTailVectorHalf(const half* dense,
                                       const half* tail_vector, half* out,
                                       int64_t n, int64_t width,
                                       bool vector_on_left,
                                       musaStream_t stream) {
  if (n <= 0 || width <= 0 || n % width != 0) return;
  if (!vector_on_left) {
    const int blocks = ClampBlocks(n, kThreadsPerBlock);
    MulHalfTailVectorKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        dense, tail_vector, out, n, width);
    return;
  }
  LaunchTailVectorTyped(dense, tail_vector, out, n, width, vector_on_left, stream,
                        MulOp<half>());
}

void LaunchMusaBinaryAddContiguousBFloat16(const __mt_bfloat16* lhs,
                                           const __mt_bfloat16* rhs,
                                           __mt_bfloat16* out, int64_t n,
                                           musaStream_t stream) {
  LaunchContiguousTyped(lhs, rhs, out, n, stream, BFloat16AddOp());
}

void LaunchMusaBinaryAddScalarBFloat16(const __mt_bfloat16* dense,
                                       const __mt_bfloat16* scalar,
                                       __mt_bfloat16* out, int64_t n,
                                       bool scalar_on_left,
                                       musaStream_t stream) {
  LaunchScalarTyped(dense, scalar, out, n, scalar_on_left, stream,
                    BFloat16AddOp());
}

void LaunchMusaBinaryAddTailVectorBFloat16(const __mt_bfloat16* dense,
                                           const __mt_bfloat16* tail_vector,
                                           __mt_bfloat16* out, int64_t n,
                                           int64_t width, bool vector_on_left,
                                           musaStream_t stream) {
  LaunchTailVectorTyped(dense, tail_vector, out, n, width, vector_on_left,
                        stream, BFloat16AddOp());
}

void LaunchMusaBinaryMulContiguousBFloat16(const __mt_bfloat16* lhs,
                                           const __mt_bfloat16* rhs,
                                           __mt_bfloat16* out, int64_t n,
                                           musaStream_t stream) {
  if (n <= 0) return;
  if (n >= 8 && IsAligned16(lhs) && IsAligned16(rhs) && IsAligned16(out)) {
    const int64_t vec_size = n / 8;
    const int blocks = ClampBlocks(vec_size, kThreadsPerBlock);
    MulBFloat16ContiguousVec8Kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const ::uint4*>(lhs),
        reinterpret_cast<const ::uint4*>(rhs),
        reinterpret_cast<::uint4*>(out), vec_size);
    const int64_t tail = n - vec_size * 8;
    if (tail > 0) {
      const __mt_bfloat16* l = lhs + vec_size * 8;
      const __mt_bfloat16* r = rhs + vec_size * 8;
      __mt_bfloat16* o = out + vec_size * 8;
      const int tail_blocks =
          ClampBlocks(tail, kThreadsPerBlock * kItemsPerThread);
      LaunchContiguousTyped(l, r, o, tail, stream, BFloat16MulOp());
    }
    return;
  }
  LaunchContiguousTyped(lhs, rhs, out, n, stream, BFloat16MulOp());
}

void LaunchMusaBinaryMulScalarBFloat16(const __mt_bfloat16* dense,
                                       const __mt_bfloat16* scalar,
                                       __mt_bfloat16* out, int64_t n,
                                       bool scalar_on_left,
                                       musaStream_t stream) {
  if (n <= 0) return;
  if (scalar_on_left) {
    LaunchScalarTyped(dense, scalar, out, n, scalar_on_left, stream,
                     BFloat16MulOp());
    return;
  }
  const int blocks = ClampBlocks(n, kThreadsPerBlock);
  MulBFloat16ScalarKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      dense, scalar, out, n);
}

void LaunchMusaBinaryMulTailVectorBFloat16(const __mt_bfloat16* dense,
                                           const __mt_bfloat16* tail_vector,
                                           __mt_bfloat16* out, int64_t n,
                                           int64_t width, bool vector_on_left,
                                           musaStream_t stream) {
  if (n <= 0 || width <= 0 || n % width != 0) return;
  if (!vector_on_left) {
    const int blocks = ClampBlocks(n, kThreadsPerBlock);
    MulBFloat16TailVectorKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        dense, tail_vector, out, n, width);
    return;
  }
  LaunchTailVectorTyped(dense, tail_vector, out, n, width, vector_on_left,
                        stream, BFloat16MulOp());
}

void LaunchMusaBinaryAddContiguousInt32(const int32_t* lhs, const int32_t* rhs,
                                        int32_t* out, int64_t n,
                                        musaStream_t stream) {
  LaunchContiguousTyped(lhs, rhs, out, n, stream, AddOp<int32_t>());
}

void LaunchMusaBinaryAddScalarInt32(const int32_t* dense, const int32_t* scalar,
                                    int32_t* out, int64_t n,
                                    bool scalar_on_left, musaStream_t stream) {
  LaunchScalarTyped(dense, scalar, out, n, scalar_on_left, stream,
                    AddOp<int32_t>());
}

void LaunchMusaBinaryAddTailVectorInt32(const int32_t* dense,
                                        const int32_t* tail_vector,
                                        int32_t* out, int64_t n, int64_t width,
                                        bool vector_on_left,
                                        musaStream_t stream) {
  LaunchTailVectorTyped(dense, tail_vector, out, n, width, vector_on_left,
                        stream, AddOp<int32_t>());
}

void LaunchMusaBinaryMulContiguousInt32(const int32_t* lhs, const int32_t* rhs,
                                        int32_t* out, int64_t n,
                                        musaStream_t stream) {
  LaunchContiguousTyped(lhs, rhs, out, n, stream, MulOp<int32_t>());
}

void LaunchMusaBinaryMulScalarInt32(const int32_t* dense, const int32_t* scalar,
                                    int32_t* out, int64_t n,
                                    bool scalar_on_left, musaStream_t stream) {
  LaunchScalarTyped(dense, scalar, out, n, scalar_on_left, stream,
                    MulOp<int32_t>());
}

void LaunchMusaBinaryMulTailVectorInt32(const int32_t* dense,
                                        const int32_t* tail_vector,
                                        int32_t* out, int64_t n, int64_t width,
                                        bool vector_on_left,
                                        musaStream_t stream) {
  LaunchTailVectorTyped(dense, tail_vector, out, n, width, vector_on_left,
                        stream, MulOp<int32_t>());
}

void LaunchMusaBinaryAddContiguousInt64(const int64_t* lhs, const int64_t* rhs,
                                        int64_t* out, int64_t n,
                                        musaStream_t stream) {
  LaunchContiguousTyped(lhs, rhs, out, n, stream, AddOp<int64_t>());
}

void LaunchMusaBinaryAddScalarInt64(const int64_t* dense, const int64_t* scalar,
                                    int64_t* out, int64_t n,
                                    bool scalar_on_left, musaStream_t stream) {
  LaunchScalarTyped(dense, scalar, out, n, scalar_on_left, stream,
                    AddOp<int64_t>());
}

void LaunchMusaBinaryAddTailVectorInt64(const int64_t* dense,
                                        const int64_t* tail_vector,
                                        int64_t* out, int64_t n, int64_t width,
                                        bool vector_on_left,
                                        musaStream_t stream) {
  LaunchTailVectorTyped(dense, tail_vector, out, n, width, vector_on_left,
                        stream, AddOp<int64_t>());
}

void LaunchMusaBinaryMulContiguousInt64(const int64_t* lhs, const int64_t* rhs,
                                        int64_t* out, int64_t n,
                                        musaStream_t stream) {
  LaunchContiguousTyped(lhs, rhs, out, n, stream, MulOp<int64_t>());
}

void LaunchMusaBinaryMulScalarInt64(const int64_t* dense, const int64_t* scalar,
                                    int64_t* out, int64_t n,
                                    bool scalar_on_left, musaStream_t stream) {
  LaunchScalarTyped(dense, scalar, out, n, scalar_on_left, stream,
                    MulOp<int64_t>());
}

void LaunchMusaBinaryMulTailVectorInt64(const int64_t* dense,
                                        const int64_t* tail_vector,
                                        int64_t* out, int64_t n, int64_t width,
                                        bool vector_on_left,
                                        musaStream_t stream) {
  LaunchTailVectorTyped(dense, tail_vector, out, n, width, vector_on_left,
                        stream, MulOp<int64_t>());
}

}  // extern "C"

}  // namespace musa
}  // namespace tensorflow
