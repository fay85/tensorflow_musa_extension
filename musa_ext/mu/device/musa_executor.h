#ifndef TENSORFLOW_MUSA_MU_DEVICE_MUSA_EXECUTOR_H_
#define TENSORFLOW_MUSA_MU_DEVICE_MUSA_EXECUTOR_H_

#include <memory>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "musa_device.h"
#include "musa_event.h"
#include "musa_memcpy.h"
#include "musa_memset.h"
#include "musa_stream.h"
#include "xla/stream_executor/stream_executor_internal.h"
namespace stream_executor {
namespace musa {

inline absl::Status FromMusaStatus(mStatus s) {
  if (s == mStatus::SUCCESS) {
    return absl::Status();
  }
  return absl::InternalError("MUSA Operation Failed");
}

class MusaExecutor : public internal::StreamExecutorInterface {
 public:
  // TF 2.15 removed PluginConfig entirely from StreamExecutor's surface
  // (the plugin registry was redesigned around the C-API PluggableDevice
  // path).  The constructor is therefore a no-arg default.
  MusaExecutor() = default;
  ~MusaExecutor() override {}

  absl::Status Init(int device_ordinal, DeviceOptions device_options) override {
    device_ordinal_ = device_ordinal;
    return absl::Status();
  }

  // TF 2.15 Stream(executor) constructor invokes this internally and owns
  // the returned StreamInterface; callers can no longer inject a pre-built
  // MusaStream the way they could in TF 2.6.  To preserve the existing
  // MUSA architecture (where MusaDevice creates the compute musaStream_t
  // up front and then binds muDNN / muBLAS to it), MusaDeviceContext stages
  // its caller-owned handle here via SetPendingStreamHandle() right before
  // constructing the SE Stream — the next GetStreamImplementation() call
  // consumes the staged handle and wraps it, instead of creating a fresh
  // stream.  This is a single-threaded setup hand-off; it is NOT a general
  // mechanism for thread-safe stream reuse.
  void SetPendingStreamHandle(musaStream_t h) { pending_stream_handle_ = h; }

  std::unique_ptr<internal::StreamInterface> GetStreamImplementation()
      override {
    if (pending_stream_handle_ != nullptr) {
      musaStream_t h = pending_stream_handle_;
      pending_stream_handle_ = nullptr;
      return std::make_unique<MusaStream>(h);
    }
    musaStream_t h;
    musaError_t err = musaStreamCreate(&h);
    if (err != musaSuccess) {
      LOG(ERROR) << "musaStreamCreate failed: " << musaGetErrorString(err);
      return nullptr;
    }
    return std::make_unique<MusaStream>(h);
  }

  std::unique_ptr<internal::EventInterface> CreateEventImplementation()
      override {
    return std::make_unique<MusaEvent>();
  }

  std::unique_ptr<internal::KernelInterface> CreateKernelImplementation()
      override {
    return nullptr;
  }

  // NOTE: TF 2.15 removed internal::TimerInterface and the AllocateTimer /
  // StartTimer / StopTimer pure virtuals — timing is now done via Event
  // primitives directly. We therefore do NOT override GetTimerImplementation.

  DeviceMemoryBase Allocate(uint64_t size, int64_t memory_space) override {
    if (size == 0) {
      return DeviceMemoryBase(nullptr, 0);
    }
    musaSetDevice(device_ordinal_);
    void* ptr = nullptr;
    musaError_t err = musaMalloc(&ptr, size);
    if (err != musaSuccess) {
      LOG(ERROR) << "MusaExecutor::Allocate failed for " << size
                 << " bytes: " << musaGetErrorString(err);
      return DeviceMemoryBase(nullptr, 0);
    }
    return DeviceMemoryBase(ptr, size);
  }

  void* GetSubBuffer(DeviceMemoryBase* parent, uint64_t offset,
                     uint64_t size) override {
    return reinterpret_cast<char*>(parent->opaque()) + offset;
  }

  void Deallocate(DeviceMemoryBase* mem) override {
    if (mem && mem->opaque()) {
      musaSetDevice(device_ordinal_);
      musaError_t err = musaFree(mem->opaque());
      if (err != musaSuccess) {
        LOG(ERROR) << "MUSA Deallocate failed: " << musaGetErrorString(err);
      }
    }
  }

  bool HostMemoryRegister(void* mem, uint64_t size) override { return true; }
  bool HostMemoryUnregister(void* mem) override { return true; }

  void* HostMemoryAllocate(uint64_t size) override { return nullptr; }
  void HostMemoryDeallocate(void* mem) override {}

  absl::Status SynchronousMemZero(DeviceMemoryBase* location,
                                  uint64_t size) override {
    mHandle h;

    return FromMusaStatus(
        tensorflow::musa::Memset(h, location->opaque(), size, 0));
  }

  absl::Status SynchronousMemSet(DeviceMemoryBase* location, int value,
                                 uint64_t size) override {
    mHandle h;
    return FromMusaStatus(tensorflow::musa::Memset(
        h, location->opaque(), size, static_cast<uint8_t>(value)));
  }

  absl::Status SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                 const void* host_src,
                                 uint64_t size) override {
    // H2D
    return FromMusaStatus(
        tensorflow::musa::MusaMemcpyH2D(gpu_dst->opaque(), host_src, size));
  }

  absl::Status SynchronousMemcpy(void* host_dst,
                                 const DeviceMemoryBase& gpu_src,
                                 uint64_t size) override {
    // D2H
    return FromMusaStatus(
        tensorflow::musa::MusaMemcpyD2H(host_dst, gpu_src.opaque(), size));
  }

  absl::Status SynchronousMemcpyDeviceToDevice(DeviceMemoryBase* gpu_dst,
                                               const DeviceMemoryBase& gpu_src,
                                               uint64_t size) override {
    // D2D
    return FromMusaStatus(tensorflow::musa::MusaMemcpyD2D(
        gpu_dst->opaque(), gpu_src.opaque(), size));
  }

  musaStream_t GetMusaStream(Stream* stream) {
    auto* musa_stream_impl = static_cast<MusaStream*>(stream->implementation());

    return musa_stream_impl->GetStream();
  }

  // D2D Async
  bool MemcpyDeviceToDevice(Stream* stream, DeviceMemoryBase* gpu_dst,
                            const DeviceMemoryBase& gpu_src,
                            uint64_t size) override {
    auto status = tensorflow::musa::MusaMemcpyAsyncD2D(
        gpu_dst->opaque(), gpu_src.opaque(), size, GetMusaStream(stream));
    return status == mStatus::SUCCESS;
  }

  // H2D Async
  bool Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst, const void* host_src,
              uint64_t size) override {
    auto status = tensorflow::musa::MusaMemcpyAsyncH2D(
        gpu_dst->opaque(), host_src, size, GetMusaStream(stream));
    return status == mStatus::SUCCESS;
  }

  // D2H Async
  bool Memcpy(Stream* stream, void* host_dst, const DeviceMemoryBase& gpu_src,
              uint64_t size) override {
    auto status = tensorflow::musa::MusaMemcpyAsyncD2H(
        host_dst, gpu_src.opaque(), size, GetMusaStream(stream));
    return status == mStatus::SUCCESS;
  }

  // MemZero Async
  absl::Status MemZero(Stream* stream, DeviceMemoryBase* location,
                       uint64_t size) override {
    mHandle h;
    h.SetStream(GetMusaStream(stream));
    return FromMusaStatus(
        tensorflow::musa::Memset(h, location->opaque(), size, 0));
  }

  // Memset (single-byte pattern) async.  Defaulted in 2.15 to "Not
  // implemented"; we wire it through to the muDNN Memset path so callers that
  // dispatch via this overload (e.g. some XLA paths) succeed.
  absl::Status Memset(Stream* stream, DeviceMemoryBase* location,
                      uint8 pattern, uint64_t size) override {
    mHandle h;
    h.SetStream(GetMusaStream(stream));
    return FromMusaStatus(tensorflow::musa::Memset(
        h, location->opaque(), size, static_cast<uint8_t>(pattern)));
  }

  // Memset32 Async
  absl::Status Memset32(Stream* stream, DeviceMemoryBase* location,
                        uint32_t pattern, uint64_t size) override {
    mHandle h;
    h.SetStream(GetMusaStream(stream));
    return FromMusaStatus(
        tensorflow::musa::Memset32(h, location->opaque(), size, pattern));
  }

  absl::Status BlockHostUntilDone(Stream* stream) override {
    internal::StreamInterface* implementation = stream->implementation();
    auto* musa_stream = static_cast<MusaStream*>(implementation);
    return musa_stream->BlockHostUntilDone_DEBUG(stream);
  }

  // TF 2.15 signature change: callback type is now
  // absl::AnyInvocable<absl::Status() &&> (rvalue-only call op), not
  // std::function<absl::Status()>.  AnyInvocable supports move-only callables
  // and is consumed on its single invocation.
  bool HostCallback(
      Stream* stream,
      absl::AnyInvocable<absl::Status() &&> callback) override {
    musaStream_t musa_stream = GetMusaStream(stream);
    auto* heap_cb =
        new absl::AnyInvocable<absl::Status() &&>(std::move(callback));
    musaError_t err = musaLaunchHostFunc(
        musa_stream,
        [](void* user_data) {
          auto* cb =
              static_cast<absl::AnyInvocable<absl::Status() &&>*>(user_data);
          // AnyInvocable's call operator is rvalue-qualified — consume it.
          (void)std::move (*cb)();
          delete cb;
        },
        heap_cb);
    if (err != musaSuccess) {
      LOG(WARNING) << "MusaExecutor::HostCallback failed: "
                   << musaGetErrorString(err);
      delete heap_cb;
      return false;
    }
    return true;
  }

  // NOTE: TF 2.15 removed AllocateTimer / DeallocateTimer / StartTimer /
  // StopTimer / PlatformDeviceCount from StreamExecutorInterface — do NOT
  // re-add overrides for them.

  absl::Status EnablePeerAccessTo(StreamExecutorInterface* other) override {
    return absl::Status();
  }
  bool CanEnablePeerAccessTo(StreamExecutorInterface* other) override {
    return false;
  }

  absl::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override {
    internal::DeviceDescriptionBuilder builder;
    builder.set_name("MUSA Device");
    return builder.Build();
  }

  bool SynchronizeAllActivity() override { return true; }
  bool DeviceMemoryUsage(int64_t* free, int64_t* total) const override {
    return false;
  }
  bool AllocateStream(Stream* stream) override { return true; }
  void DeallocateStream(Stream* stream) override {}
  bool CreateStreamDependency(Stream* dependent, Stream* other) override {
    // Create an event on 'other' stream and wait on 'dependent' stream
    musaEvent_t event;
    musaError_t err = musaEventCreateWithFlags(&event, musaEventDisableTiming);
    if (err != musaSuccess) {
      LOG(ERROR) << "CreateStreamDependency: musaEventCreate failed: "
                 << musaGetErrorString(err);
      return false;
    }

    musaStream_t other_stream = GetMusaStream(other);
    err = musaEventRecord(event, other_stream);
    if (err != musaSuccess) {
      LOG(ERROR) << "CreateStreamDependency: musaEventRecord failed: "
                 << musaGetErrorString(err);
      musaEventDestroy(event);
      return false;
    }

    musaStream_t dependent_stream = GetMusaStream(dependent);
    err = musaStreamWaitEvent(dependent_stream, event, 0);
    if (err != musaSuccess) {
      LOG(ERROR) << "CreateStreamDependency: musaStreamWaitEvent failed: "
                 << musaGetErrorString(err);
      musaEventDestroy(event);
      return false;
    }

    // Event can be destroyed after wait is queued
    err = musaEventDestroy(event);
    if (err != musaSuccess) {
      LOG(WARNING) << "CreateStreamDependency: musaEventDestroy failed: "
                   << musaGetErrorString(err);
    }

    return true;
  }

  absl::Status AllocateEvent(Event* event) override {
    auto* musa_event = static_cast<MusaEvent*>(event->implementation());
    if (!musa_event) {
      return absl::InternalError("Invalid event implementation");
    }
    if (!musa_event->Init()) {
      return absl::InternalError("Failed to initialize MUSA event");
    }
    return absl::Status();
  }

  absl::Status DeallocateEvent(Event* event) override {
    auto* musa_event = static_cast<MusaEvent*>(event->implementation());
    if (musa_event && musa_event->handle()) {
      musaEventDestroy(musa_event->handle());
    }
    return absl::Status();
  }

  absl::Status RecordEvent(Stream* stream, Event* event) override {
    auto* musa_event = static_cast<MusaEvent*>(event->implementation());
    if (!musa_event || !musa_event->handle()) {
      return absl::InternalError("Invalid event");
    }
    musaStream_t mstream = GetMusaStream(stream);
    musaError_t err = musaEventRecord(musa_event->handle(), mstream);
    if (err != musaSuccess) {
      return absl::InternalError("musaEventRecord failed");
    }
    return absl::Status();
  }

  absl::Status WaitForEvent(Stream* stream, Event* event) override {
    auto* musa_event = static_cast<MusaEvent*>(event->implementation());
    if (!musa_event || !musa_event->handle()) {
      return absl::InternalError("Invalid event");
    }
    musaStream_t mstream = GetMusaStream(stream);
    musaError_t err = musaStreamWaitEvent(mstream, musa_event->handle(), 0);
    if (err != musaSuccess) {
      return absl::InternalError("musaStreamWaitEvent failed");
    }
    return absl::Status();
  }

  Event::Status PollForEventStatus(Event* event) override {
    auto* musa_event = static_cast<MusaEvent*>(event->implementation());
    if (!musa_event || !musa_event->handle()) {
      return Event::Status::kError;
    }
    musaError_t err = musaEventQuery(musa_event->handle());
    if (err == musaSuccess) return Event::Status::kComplete;
    if (err == musaErrorNotReady) return Event::Status::kPending;
    return Event::Status::kError;
  }

 private:
  int device_ordinal_ = -1;
  // See SetPendingStreamHandle() above.  Always nullptr except for the
  // brief window between MusaDeviceContext::MusaDeviceContext() staging
  // its handle and the Stream(executor) constructor consuming it.
  musaStream_t pending_stream_handle_ = nullptr;
};

}  // namespace musa
}  // namespace stream_executor

#endif
