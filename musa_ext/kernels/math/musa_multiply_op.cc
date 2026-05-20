#include <cstdint>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {
namespace musa {

namespace {

// vec8 bf16/fp16 fast paths (musa_mul_kernel.mu) — no element-count cap.
extern "C" {
void LaunchMusaMulContiguousBFloat16(const void* lhs, const void* rhs,
                                     void* output, int64_t size,
                                     musaStream_t stream);
void LaunchMusaMulScalarBFloat16(const void* dense, const void* scalar,
                                 void* output, int64_t size,
                                 musaStream_t stream);
void LaunchMusaMulTailVectorBFloat16(const void* dense,
                                     const void* tail_vector, void* output,
                                     int64_t size, int64_t width,
                                     musaStream_t stream);
void LaunchMusaMulContiguousHalf(const void* lhs, const void* rhs, void* output,
                                 int64_t size, musaStream_t stream);
void LaunchMusaMulScalarHalf(const void* dense, const void* scalar,
                             void* output, int64_t size, musaStream_t stream);
void LaunchMusaMulTailVectorHalf(const void* dense, const void* tail_vector,
                                 void* output, int64_t size, int64_t width,
                                 musaStream_t stream);
}

// Small-tensor fast paths for float/int (musa_binary_elementwise_kernel.mu).
extern "C" {
void LaunchMusaBinaryMulContiguousFloat(const void* lhs, const void* rhs,
                                        void* output, int64_t size,
                                        musaStream_t stream);
void LaunchMusaBinaryMulScalarFloat(const void* dense, const void* scalar,
                                    void* output, int64_t size,
                                    bool scalar_on_left, musaStream_t stream);
void LaunchMusaBinaryMulTailVectorFloat(const void* dense,
                                        const void* tail_vector, void* output,
                                        int64_t size, int64_t width,
                                        bool vector_on_left,
                                        musaStream_t stream);
void LaunchMusaBinaryMulContiguousInt32(const void* lhs, const void* rhs,
                                        void* output, int64_t size,
                                        musaStream_t stream);
void LaunchMusaBinaryMulScalarInt32(const void* dense, const void* scalar,
                                    void* output, int64_t size,
                                    bool scalar_on_left, musaStream_t stream);
void LaunchMusaBinaryMulTailVectorInt32(const void* dense,
                                        const void* tail_vector, void* output,
                                        int64_t size, int64_t width,
                                        bool vector_on_left,
                                        musaStream_t stream);
void LaunchMusaBinaryMulContiguousInt64(const void* lhs, const void* rhs,
                                        void* output, int64_t size,
                                        musaStream_t stream);
void LaunchMusaBinaryMulScalarInt64(const void* dense, const void* scalar,
                                    void* output, int64_t size,
                                    bool scalar_on_left, musaStream_t stream);
void LaunchMusaBinaryMulTailVectorInt64(const void* dense,
                                        const void* tail_vector, void* output,
                                        int64_t size, int64_t width,
                                        bool vector_on_left,
                                        musaStream_t stream);
}

enum class MulFastPathResult {
  kNotHandled = 0,
  kLaunched,
  kFailed,
};

inline bool UseMulCustomKernelFastPath() {
  const char* env = std::getenv("MUSA_MUL_ENABLE_CUSTOM_KERNEL");
  if (env == nullptr || std::string(env).empty()) return true;
  const std::string value(env);
  return !(value == "0" || value == "false" || value == "FALSE" ||
           value == "off" || value == "OFF" || value == "no" || value == "NO");
}

bool SameShape(const TensorShape& lhs, const TensorShape& rhs) {
  if (lhs.dims() != rhs.dims()) return false;
  for (int i = 0; i < lhs.dims(); ++i) {
    if (lhs.dim_size(i) != rhs.dim_size(i)) return false;
  }
  return true;
}

bool IsTailVectorBroadcast(const Tensor& tensor,
                           const TensorShape& output_shape, int64_t* width) {
  if (output_shape.dims() <= 0) return false;
  const int64_t last_dim = output_shape.dim_size(output_shape.dims() - 1);
  if (last_dim <= 0 || tensor.NumElements() != last_dim || tensor.dims() == 0) {
    return false;
  }

  for (int i = tensor.dims() - 1; i >= 0; --i) {
    const int64_t dim = tensor.dim_size(i);
    if (i == tensor.dims() - 1) {
      if (dim != last_dim) return false;
      continue;
    }
    if (dim != 1) return false;
  }

  *width = last_dim;
  return true;
}

struct LowpMulLaunchers {
  using ContiguousFn = void (*)(const void*, const void*, void*, int64_t,
                                musaStream_t);
  using ScalarFn = ContiguousFn;
  using TailVectorFn = void (*)(const void*, const void*, void*, int64_t,
                                int64_t, musaStream_t);
  ContiguousFn contiguous;
  ScalarFn scalar;
  TailVectorFn tail_vector;
};

inline MulFastPathResult TryLaunchLowpMulFastPath(
    OpKernelContext* ctx, const Tensor& in0, const Tensor& in1,
    const TensorShape& output_shape, bool same_shape, Tensor* out,
    const LowpMulLaunchers& fns) {
  if (!UseMulCustomKernelFastPath()) return MulFastPathResult::kNotHandled;
  const int64_t output_elements = output_shape.num_elements();
  if (output_elements <= 0) return MulFastPathResult::kNotHandled;
  musaStream_t stream = GetMusaStreamByCtx(ctx);
  if (stream == nullptr) return MulFastPathResult::kNotHandled;

  const void* in0_ptr = in0.tensor_data().data();
  const void* in1_ptr = in1.tensor_data().data();
  void* out_ptr =
      const_cast<void*>(static_cast<const void*>(out->tensor_data().data()));

  bool launched = false;
  if (same_shape) {
    fns.contiguous(in0_ptr, in1_ptr, out_ptr, output_elements, stream);
    launched = true;
  } else if (in0.NumElements() == output_elements && in1.NumElements() == 1) {
    fns.scalar(in0_ptr, in1_ptr, out_ptr, output_elements, stream);
    launched = true;
  } else if (in1.NumElements() == output_elements && in0.NumElements() == 1) {
    fns.scalar(in1_ptr, in0_ptr, out_ptr, output_elements, stream);
    launched = true;
  } else if (in0.NumElements() == output_elements) {
    int64_t width = 0;
    if (IsTailVectorBroadcast(in1, output_shape, &width)) {
      fns.tail_vector(in0_ptr, in1_ptr, out_ptr, output_elements, width,
                      stream);
      launched = true;
    }
  } else if (in1.NumElements() == output_elements) {
    int64_t width = 0;
    if (IsTailVectorBroadcast(in0, output_shape, &width)) {
      fns.tail_vector(in1_ptr, in0_ptr, out_ptr, output_elements, width,
                      stream);
      launched = true;
    }
  }

  if (!launched) return MulFastPathResult::kNotHandled;

  const musaError_t launch_status = musaGetLastError();
  if (launch_status != musaSuccess) {
    ctx->CtxFailure(
        errors::Internal("MUSA Mul fast path launch failed (low-precision): ",
                         musaGetErrorString(launch_status)));
    return MulFastPathResult::kFailed;
  }
  return MulFastPathResult::kLaunched;
}

template <typename T>
struct MulFastPathLauncher {
  static constexpr bool kSupported = false;
  static void Contiguous(const void*, const void*, void*, int64_t,
                         musaStream_t) {}
  static void Scalar(const void*, const void*, void*, int64_t, bool,
                     musaStream_t) {}
  static void TailVector(const void*, const void*, void*, int64_t, int64_t,
                         bool, musaStream_t) {}
};

#define DEFINE_MUL_FAST_PATH_LAUNCHER(T, suffix)                              \
  template <>                                                                 \
  struct MulFastPathLauncher<T> {                                             \
    static constexpr bool kSupported = true;                                  \
    static void Contiguous(const void* lhs, const void* rhs, void* output,    \
                           int64_t size, musaStream_t stream) {               \
      LaunchMusaBinaryMulContiguous##suffix(lhs, rhs, output, size, stream);  \
    }                                                                         \
    static void Scalar(const void* dense, const void* scalar, void* output,   \
                       int64_t size, bool scalar_on_left,                     \
                       musaStream_t stream) {                                 \
      LaunchMusaBinaryMulScalar##suffix(dense, scalar, output, size,          \
                                        scalar_on_left, stream);              \
    }                                                                         \
    static void TailVector(const void* dense, const void* tail_vector,        \
                           void* output, int64_t size, int64_t width,         \
                           bool vector_on_left, musaStream_t stream) {        \
      LaunchMusaBinaryMulTailVector##suffix(dense, tail_vector, output, size, \
                                            width, vector_on_left, stream);   \
    }                                                                         \
  };

DEFINE_MUL_FAST_PATH_LAUNCHER(float, Float)
DEFINE_MUL_FAST_PATH_LAUNCHER(int32, Int32)
DEFINE_MUL_FAST_PATH_LAUNCHER(int64, Int64)

#undef DEFINE_MUL_FAST_PATH_LAUNCHER

// Small-tensor cap for float/int binary-elementwise fast paths only.
constexpr int64_t kMulCustomFastPathMaxElements = 8192;

template <typename T>
bool ShouldUseMulCustomKernelFastPath(const Tensor& in0, const Tensor& in1,
                                      const TensorShape& output_shape,
                                      bool same_shape) {
  const int64_t output_elements = output_shape.num_elements();
  if (output_elements <= 0 || output_elements > kMulCustomFastPathMaxElements) {
    return false;
  }
  if (same_shape) {
    return true;
  }
  if (in0.NumElements() == output_elements && in1.NumElements() == 1) {
    return true;
  }
  if (in1.NumElements() == output_elements && in0.NumElements() == 1) {
    return true;
  }
  int64_t width = 0;
  if (in0.NumElements() == output_elements &&
      IsTailVectorBroadcast(in1, output_shape, &width)) {
    return true;
  }
  if (in1.NumElements() == output_elements &&
      IsTailVectorBroadcast(in0, output_shape, &width)) {
    return true;
  }
  return false;
}

template <typename T>
MulFastPathResult TryLaunchMulFastPath(OpKernelContext* ctx, const Tensor& in0,
                                       const Tensor& in1,
                                       const TensorShape& output_shape,
                                       bool same_shape, Tensor* out) {
  if (!MulFastPathLauncher<T>::kSupported ||
      !ShouldUseMulCustomKernelFastPath<T>(in0, in1, output_shape,
                                           same_shape)) {
    return MulFastPathResult::kNotHandled;
  }

  const int64_t output_elements = output_shape.num_elements();
  if (output_elements <= 0) return MulFastPathResult::kNotHandled;

  const void* in0_ptr = in0.tensor_data().data();
  const void* in1_ptr = in1.tensor_data().data();
  void* out_ptr = const_cast<char*>(out->tensor_data().data());
  musaStream_t stream = GetMusaStreamByCtx(ctx);
  if (stream == nullptr) return MulFastPathResult::kNotHandled;

  bool launched = false;
  if (same_shape) {
    MulFastPathLauncher<T>::Contiguous(in0_ptr, in1_ptr, out_ptr,
                                       output_elements, stream);
    launched = true;
  } else if (in0.NumElements() == output_elements && in1.NumElements() == 1) {
    MulFastPathLauncher<T>::Scalar(in0_ptr, in1_ptr, out_ptr, output_elements,
                                   false, stream);
    launched = true;
  } else if (in1.NumElements() == output_elements && in0.NumElements() == 1) {
    MulFastPathLauncher<T>::Scalar(in1_ptr, in0_ptr, out_ptr, output_elements,
                                   true, stream);
    launched = true;
  } else if (in0.NumElements() == output_elements) {
    int64_t width = 0;
    if (IsTailVectorBroadcast(in1, output_shape, &width)) {
      MulFastPathLauncher<T>::TailVector(in0_ptr, in1_ptr, out_ptr,
                                         output_elements, width, false, stream);
      launched = true;
    }
  } else if (in1.NumElements() == output_elements) {
    int64_t width = 0;
    if (IsTailVectorBroadcast(in0, output_shape, &width)) {
      MulFastPathLauncher<T>::TailVector(in1_ptr, in0_ptr, out_ptr,
                                         output_elements, width, true, stream);
      launched = true;
    }
  }

  if (!launched) return MulFastPathResult::kNotHandled;

  auto launch_status = musaGetLastError();
  if (launch_status != musaSuccess) {
    ctx->CtxFailure(errors::Internal("MUSA Mul fast path launch failed: ",
                                     musaGetErrorString(launch_status)));
    return MulFastPathResult::kFailed;
  }

  return MulFastPathResult::kLaunched;
}

template <>
MulFastPathResult TryLaunchMulFastPath<bfloat16>(
    OpKernelContext* ctx, const Tensor& in0, const Tensor& in1,
    const TensorShape& output_shape, bool same_shape, Tensor* out) {
  // SwiGLU gating in tokenmixerlarge: same-shape Mul of two bf16 activations.
  // MoE Gate-Value-Scaling: scalar Mul. RMSNorm scale: tail-vector Mul.
  // Uncapped vec8 path with fp32-intermediate math (legacy tf2.15.1 parity).
  return TryLaunchLowpMulFastPath(
      ctx, in0, in1, output_shape, same_shape, out,
      LowpMulLaunchers{LaunchMusaMulContiguousBFloat16,
                       LaunchMusaMulScalarBFloat16,
                       LaunchMusaMulTailVectorBFloat16});
}

template <>
MulFastPathResult TryLaunchMulFastPath<Eigen::half>(
    OpKernelContext* ctx, const Tensor& in0, const Tensor& in1,
    const TensorShape& output_shape, bool same_shape, Tensor* out) {
  return TryLaunchLowpMulFastPath(
      ctx, in0, in1, output_shape, same_shape, out,
      LowpMulLaunchers{LaunchMusaMulContiguousHalf, LaunchMusaMulScalarHalf,
                       LaunchMusaMulTailVectorHalf});
}

inline bool MulFastPathEligible(const Tensor& in0, const Tensor& in1,
                                const TensorShape& output_shape,
                                bool same_shape) {
  if (output_shape.num_elements() <= 0) return false;
  if (same_shape) return true;
  const int64_t n = output_shape.num_elements();
  if (in0.NumElements() == n && in1.NumElements() == 1) return true;
  if (in1.NumElements() == n && in0.NumElements() == 1) return true;
  int64_t width = 0;
  if (in0.NumElements() == n && IsTailVectorBroadcast(in1, output_shape, &width)) {
    return true;
  }
  if (in1.NumElements() == n && IsTailVectorBroadcast(in0, output_shape, &width)) {
    return true;
  }
  return false;
}

}  // namespace

template <typename T>
class MusaMultiplyOp : public MusaOpKernel {
 public:
  using MusaOpKernel::MusaOpKernel;

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    BCast bcast(BCast::Vec(in0.shape().dim_sizes()),
                BCast::Vec(in1.shape().dim_sizes()));

    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes for Mul: ", in0.shape().DebugString(),
                    " and ", in1.shape().DebugString()));

    TensorShape output_shape = BCast::ToShape(bcast.output_shape());
    const bool same_shape = SameShape(in0.shape(), in1.shape());

    Tensor* output = nullptr;
    const bool fast_path_possible =
        MulFastPathEligible(in0, in1, output_shape, same_shape) &&
        ((MulFastPathLauncher<T>::kSupported &&
          ShouldUseMulCustomKernelFastPath<T>(in0, in1, output_shape,
                                              same_shape)) ||
         std::is_same<T, bfloat16>::value || std::is_same<T, Eigen::half>::value);
    if (fast_path_possible) {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    } else if (in0.shape() == output_shape) {
      const std::vector<int> forwardable_input_indices = {0};
      OP_REQUIRES_OK(
          ctx, ctx->forward_input_or_allocate_output(forwardable_input_indices,
                                                     0, output_shape, &output));
    } else if (in1.shape() == output_shape) {
      const std::vector<int> forwardable_input_indices = {1};
      OP_REQUIRES_OK(
          ctx, ctx->forward_input_or_allocate_output(forwardable_input_indices,
                                                     0, output_shape, &output));
    } else {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    }

    if (output->NumElements() == 0) return;

    const auto fast_path_status = TryLaunchMulFastPath<T>(
        ctx, in0, in1, output_shape, same_shape, output);
    if (fast_path_status != MulFastPathResult::kNotHandled) {
      return;
    }

    MUSA_OP_REQUIRES_MUDNN_HANDLE(ctx);
    auto& handle = GetHandleByCtx(ctx);

    mBinary binary_op;
    binary_op.SetMode(::musa::dnn::Binary::Mode::MUL);

    mTensor mt_in0 = CreateMTensor(in0, format_);
    mTensor mt_in1 = CreateMTensor(in1, format_);
    mTensor mt_out = CreateMTensor(*output, format_);

    auto status = binary_op.Run(handle, mt_out, mt_in0, mt_in1);

    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal("MUSA Multiply execution failed. Status: ",
                                 (int)status));
  }
};

#define REGISTER_MUSA_MULTIPLY(TYPE)                        \
  REGISTER_KERNEL_BUILDER(                                  \
      Name("Mul").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaMultiplyOp<TYPE>);

REGISTER_MUSA_MULTIPLY(float);
REGISTER_MUSA_MULTIPLY(Eigen::half);
REGISTER_MUSA_MULTIPLY(bfloat16);
REGISTER_MUSA_MULTIPLY(int32);
REGISTER_MUSA_MULTIPLY(int64);

#undef REGISTER_MUSA_MULTIPLY

}  // namespace musa
}  // namespace tensorflow
