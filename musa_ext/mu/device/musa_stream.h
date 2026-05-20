#ifndef TENSORFLOW_MUSA_MU1_DEVICE_MUSA_STREAM_H_
#define TENSORFLOW_MUSA_MU1_DEVICE_MUSA_STREAM_H_

#include <musa_runtime.h>

// In TF 2.10+ `xla/stream_executor/platform/port.h` no longer exists.
// It previously provided `port::Status` (the `stream_executor::port` namespace
// alias). We migrated to `absl::Status` directly; transitive availability
// through stream.h is reliable in 2.15 but include the absl header
// explicitly so this file builds even if SE's transitive include graph
// changes again in a future TF.
#include "absl/status/status.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace musa {

class MusaStream : public internal::StreamInterface {
 public:
  explicit MusaStream(musaStream_t stream) : musa_stream_(stream) {}
  ~MusaStream() override {}
  musaStream_t GetStream() const { return musa_stream_; }

  absl::Status BlockHostUntilDone_DEBUG(Stream* stream) {
    musaError_t result = musaStreamSynchronize(musa_stream_);
    if (result != musaSuccess) {
      return absl::InternalError("Sync Failed");
    }
    return absl::Status();
  }

  void* GpuStreamHack() override { return (void*)musa_stream_; }
  void** GpuStreamMemberHack() override {
    return reinterpret_cast<void**>(&musa_stream_);
  }

 private:
  musaStream_t musa_stream_;
};

}  // namespace musa
}  // namespace stream_executor

#endif