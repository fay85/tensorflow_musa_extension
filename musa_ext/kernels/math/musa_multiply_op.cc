#include <cstdlib>
#include <string>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/bcast.h"
#include "../utils_op.h"

// Plain-C bf16/fp16 vectorized Mul fast-path launchers (defined in
// musa_mul_kernel.mu). Used only when both inputs land on shapes the
// fast paths support (same-shape contiguous, scalar broadcast, or
// tail-vector broadcast); everything else still falls through to muDNN
// Binary below.
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

namespace tensorflow {
namespace musa {

namespace {

inline bool UseMulCustomKernelFastPath() {
  // Same kill-switch convention as AddV2: set =0 to force everything through
  // muDNN, useful for A/B numerical regression hunts.
  const char* env = std::getenv("MUSA_MUL_ENABLE_CUSTOM_KERNEL");
  if (env == nullptr || std::string(env).empty()) return true;
  const std::string value(env);
  return !(value == "0" || value == "false" || value == "FALSE" ||
           value == "off" || value == "OFF" || value == "no" || value == "NO");
}

inline bool MulSameShape(const TensorShape& lhs, const TensorShape& rhs) {
  if (lhs.dims() != rhs.dims()) return false;
  for (int i = 0; i < lhs.dims(); ++i) {
    if (lhs.dim_size(i) != rhs.dim_size(i)) return false;
  }
  return true;
}

// Mirror of IsTailVectorBroadcast in musa_add_op.cc: accepts [C], [1, C],
// [1, 1, C], ... as broadcasts over the last dim of `output_shape`.
inline bool MulIsTailVectorBroadcast(const Tensor& tensor,
                                      const TensorShape& output_shape,
                                      int64_t* width) {
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

enum class MulFastPathResult { kNotHandled = 0, kLaunched, kFailed };

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
    if (MulIsTailVectorBroadcast(in1, output_shape, &width)) {
      fns.tail_vector(in0_ptr, in1_ptr, out_ptr, output_elements, width,
                      stream);
      launched = true;
    }
  } else if (in1.NumElements() == output_elements) {
    int64_t width = 0;
    if (MulIsTailVectorBroadcast(in0, output_shape, &width)) {
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
MulFastPathResult TryLaunchMulFastPath(OpKernelContext* /*ctx*/,
                                         const Tensor& /*in0*/,
                                         const Tensor& /*in1*/,
                                         const TensorShape& /*output_shape*/,
                                         bool /*same_shape*/, Tensor* /*out*/) {
  return MulFastPathResult::kNotHandled;
}

template <>
MulFastPathResult TryLaunchMulFastPath<bfloat16>(
    OpKernelContext* ctx, const Tensor& in0, const Tensor& in1,
    const TensorShape& output_shape, bool same_shape, Tensor* out) {
  // SwiGLU gating in tokenmixerlarge: same-shape Mul of two bf16 activations.
  // MoE Gate-Value-Scaling: scalar Mul. RMSNorm scale: tail-vector Mul.
  // All three shapes hit the custom kernel below.
  return TryLaunchLowpMulFastPath(
      ctx, in0, in1, output_shape, same_shape, out,
      LowpMulLaunchers{
          LaunchMusaMulContiguousBFloat16,
          LaunchMusaMulScalarBFloat16,
          LaunchMusaMulTailVectorBFloat16,
      });
}

template <>
MulFastPathResult TryLaunchMulFastPath<Eigen::half>(
    OpKernelContext* ctx, const Tensor& in0, const Tensor& in1,
    const TensorShape& output_shape, bool same_shape, Tensor* out) {
  return TryLaunchLowpMulFastPath(ctx, in0, in1, output_shape, same_shape, out,
                                   LowpMulLaunchers{
                                       LaunchMusaMulContiguousHalf,
                                       LaunchMusaMulScalarHalf,
                                       LaunchMusaMulTailVectorHalf,
                                   });
}

}  // namespace

template <typename T>
class MusaMultiplyOp : public MusaOpKernel {
 public:
  using MusaOpKernel::MusaOpKernel;

  // Multiply is element-wise and computationally lightweight
  // Mark as inexpensive to enable inline scheduling
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

    Tensor* output = nullptr;
    if (in0.shape() == output_shape) {
      const std::vector<int> forwardable_input_indices = {0};
      OP_REQUIRES_OK(
          ctx, ctx->forward_input_or_allocate_output(
                   forwardable_input_indices, 0, output_shape, &output));
    } else if (in1.shape() == output_shape) {
      const std::vector<int> forwardable_input_indices = {1};
      OP_REQUIRES_OK(
          ctx, ctx->forward_input_or_allocate_output(
                   forwardable_input_indices, 0, output_shape, &output));
    } else {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    }

    if (output->NumElements() == 0) return;

    // bf16 / fp16 custom fast path. Falls through to muDNN Binary below for
    // unhandled shapes (general broadcasts) or unhandled dtypes (fp32, int*).
    const bool same_shape = MulSameShape(in0.shape(), output_shape) &&
                            MulSameShape(in1.shape(), output_shape);
    const MulFastPathResult fp =
        TryLaunchMulFastPath<T>(ctx, in0, in1, output_shape, same_shape, output);
    if (fp == MulFastPathResult::kLaunched) return;
    if (fp == MulFastPathResult::kFailed) return;  // ctx already failed

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
