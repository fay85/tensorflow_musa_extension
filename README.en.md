# TensorFlow MUSA Extension

TensorFlow MUSA Extension provides TensorFlow support for Moore Threads MUSA GPUs. It packages MUSA device registration, operator kernels, and graph optimization as the `tensorflow_musa` Python package.

> **Branch: `tf_2.15.1_pluggable`**
>
> This branch targets **TensorFlow 2.15.1** with the **PluggableDevice (StreamExecutor C API)** default path, including mixed bf16 training accuracy fixes (uncapped vec8 `Mul` for SwiGLU-scale tensors, pageable D2H staging, etc.).
>
> - **TF ≥ 2.10** is required for PluggableDevice; this branch **builds and tests against TF 2.15.1 only**.
> - For the **legacy `MusaDevice` path** (no PluggableDevice), use the separate tree `tensorflow_musa_extension_2.15.1` (`tf2.15.1` branch).
> - For **TF 2.6.1**, use the `main` branch.

## Features

- Registers MUSA via PluggableDevice / `SE_InitPlugin` (default path).
- Provides MUSA implementations for common TensorFlow operators and selected fusion paths.
- Loads the runtime plugin with `import tensorflow_musa` (`load_op_library` + `load_pluggable_device_library`).
- Debugging and environment-variable guidance is available in [docs/DEBUG_GUIDE.md](docs/DEBUG_GUIDE.md).

## Requirements

- **TensorFlow == 2.15.1** (same version at build and runtime; official wheel uses `_GLIBCXX_USE_CXX11_ABI=1`)
- Moore Threads MUSA SDK, installed at `/usr/local/musa` by default
- CMake 3.10 or newer
- GCC/G++ compatible with the TF 2.15.1 wheel ABI
- Python ≥ 3.9 (match your TF 2.15.1 wheel)
- NumPy 1.19.0 or newer

The built wheel must match the TensorFlow version and Python environment used at build time.

## Build and install

```bash
git clone <repository-url>
cd tensorflow_musa_extension
git checkout tf_2.15.1_pluggable

pip install tensorflow==2.15.1
./build.sh wheel
pip install --force-reinstall dist/tensorflow_musa-*.whl --no-deps
```

To build only the plugin shared library during development:

```bash
pip install tensorflow==2.15.1
./build.sh
```

The build scripts verify that `tf.__version__` is **2.15.1**. To override the pin (not recommended):

```bash
export TENSORFLOW_MUSA_TARGET_TF=2.15.1
```

## TensorFlow version and compile-time macros (developers)

When building against **TF 2.15.1**, version macros come from the installed TensorFlow headers (`tensorflow/core/public/version.h`). You do **not** need to pass a TF 2.10 flag manually. Key thresholds in the source:

| Condition | Meaning (TF 2.15.1 build) |
|-----------|---------------------------|
| `TF_MINOR_VERSION >= 10` | PluggableDevice path: `SE_InitPlugin`, `MusaKernelRuntimeView`, `tensorflow_accelerator_device_info` |
| `TF_MINOR_VERSION < 10` | Legacy C++ `MusaDevice` / `DeviceFactory::Register` (**not compiled on this branch**) |
| `TF_MINOR_VERSION >= 15` | SE plugin `mem_zero` / `memset` / `memset32` callbacks |

CMake picks **CXX11 ABI** from the installed TF wheel (1 for TF 2.15.1).

Optional legacy device path for A/B testing (still on TF 2.15.1):

```bash
export TENSORFLOW_MUSA_USE_LEGACY_DEVICE=1
python your_script.py
```

## Quick check

```python
import tensorflow as tf
import tensorflow_musa as tf_musa

print(tf.__version__)  # expect 2.15.1
print(tf.config.list_physical_devices("MUSA"))
```

Run a simple operation on MUSA:

```python
import tensorflow as tf
import tensorflow_musa

with tf.device("/device:MUSA:0"):
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    y = tf.matmul(x, x)

print(y)
```

## Tests

Run the operator test suite from an installed wheel:

```bash
MUSA_VISIBLE_DEVICES=0 python test/test_runner.py
```

Run PluggableDevice compliance tests:

```bash
MUSA_VISIBLE_DEVICES=0 python test/test_runner.py --pattern "pluggable_*_test.py" --detail
```

## Debugging

See [docs/DEBUG_GUIDE.md](docs/DEBUG_GUIDE.md) for logging, GraphDef dumps, telemetry, and runtime debugging environment variables.

## Contributing

Issues and pull requests are welcome. Please include tests for new operators or behavior changes.

## License

Apache License 2.0
