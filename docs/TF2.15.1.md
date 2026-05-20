# TensorFlow 2.15.1 variant (`tf2.15.1` branch)

This branch targets **TensorFlow 2.15.1** on the **legacy `MusaDevice` path** (no PluggableDevice / StreamExecutor). It is based on commit `6667530` (2026-05-14), **before** the PluggableDevice merge (`a4e6597`) and the accuracy regressions observed on `main` with tokenmixerlarge.

## Why this branch exists

- `origin/tf2.15.1` was removed from the remote; this branch recreates a TF 2.15.1–compatible tree from:
  - **Base:** `6667530` — legacy device, pre–PluggableDevice `main`
  - **Ports from local `origin/tf2.15.1`:** ABI/build (`CXX11_ABI=1`), `MusaAdam` / `MusaResourceApplyAdamMixed`, vec8 bf16/fp16 `Mul`, sparse Adam fast path, Grappler `MusaAdamMixedFusion`

## Requirements

- TensorFlow **== 2.15.1** (official wheel uses `_GLIBCXX_USE_CXX11_ABI=1`)
- MUSA SDK at `/usr/local/musa` (or set `MUSA_PATH`)
- GCC/G++ compatible with the TF 2.15.1 wheel ABI
- Python ≥ 3.9 (match your TF 2.15.1 wheel)

## Build

```bash
git checkout tf2.15.1
pip install tensorflow==2.15.1
./build.sh wheel
pip install dist/tensorflow_musa-*.whl --force-reinstall --no-deps
```

Release-only `.so`:

```bash
./build.sh release
# -> build/libmusa_plugin.so
```

## Usage (tokenmixerlarge / mixed bf16)

```python
import tensorflow as tf
import tensorflow_musa  # loads libmusa_plugin.so + legacy MusaDevice

from tensorflow_musa import MusaAdam  # or: from tensorflow_musa._optimizers import _make_musa_adam_class; MusaAdam = _make_musa_adam_class()
```

In `train.py`, prefer the wheel path over raw `tf.load_library` only:

```bash
pip install dist/tensorflow_musa-*.whl --no-deps
python train.py --backend musa --precision mixed_bf16 ...
```

(`--lib_path` is optional when `import tensorflow_musa` runs first.)

## What is intentionally **not** in this branch

- PluggableDevice / `SE_InitPlugin` default path (`a4e6597` and later `main`)
- Multi-TF unified `main` wheel (build is pinned to 2.15.1 only)

## Mul fast path

bf16/fp16 SwiGLU `Mul` uses dedicated vec8 kernels (`musa_mul_kernel.mu`). Disable for muDNN-only comparison:

```bash
export MUSA_MUL_ENABLE_CUSTOM_KERNEL=0
```

## Relation to `main`

| | `tf2.15.1` (this branch) | `main` (after `a4e6597`) |
|--|--------------------------|---------------------------|
| Device | Legacy `MusaDevice` | PluggableDevice SE default |
| TF versions | 2.15.1 only | 2.6.1 + 2.15.1 (wheel matrix) |
| Mixed Adam | Yes (`MusaAdam`) | Yes (`MusaAdam`, merged via #257) |
| Accuracy (tokenmixer) | Baseline for regression work | Reported lower AUC vs old tf2.15.1 plugin |
