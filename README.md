# TensorFlow MUSA Extension

TensorFlow MUSA Extension 是面向摩尔线程（Moore Threads）MUSA GPU 的 TensorFlow 插件。它将 MUSA 设备注册、算子内核和图优化能力打包为 `tensorflow_musa` Python 包。

> **分支说明：`tf_2.15.1_pluggable`**
>
> 本分支面向 **TensorFlow 2.15.1 + PluggableDevice（StreamExecutor C API）** 默认路径，并包含针对 mixed bf16 训练精度的修复（SwiGLU 等大尺寸 `Mul` 的 vec8 快速路径、pageable D2H 同步等）。
>
> - 需要 **TF ≥ 2.10** 才能使用 PluggableDevice；本分支 **仅支持并测试 TF 2.15.1**。
> - 若需 **legacy `MusaDevice` 路径**（无 PluggableDevice），请使用独立仓库/分支 `tensorflow_musa_extension_2.15.1`（`tf2.15.1`）。
> - 若需 **TF 2.6.1**，请使用 `main` 分支。

## 主要特性

- 通过 PluggableDevice / `SE_InitPlugin` 将 MUSA 注册为 TensorFlow 设备（默认路径）。
- 提供 TensorFlow 常用算子和部分融合路径的 MUSA 实现。
- 通过 `import tensorflow_musa` 自动加载运行时插件（`load_op_library` + `load_pluggable_device_library`）。
- 调试和环境变量说明见 [docs/DEBUG_GUIDE.md](docs/DEBUG_GUIDE.md)。

## 环境要求

- **TensorFlow == 2.15.1**（构建与运行均需一致；官方 wheel 使用 `_GLIBCXX_USE_CXX11_ABI=1`）
- Moore Threads MUSA SDK，默认安装路径为 `/usr/local/musa`
- CMake 3.10 或更新版本
- 与 TF 2.15.1 wheel ABI 兼容的 GCC/G++
- Python ≥ 3.9（与所安装的 TF 2.15.1 wheel 匹配）
- NumPy 1.19.0 或更新版本

生成的 wheel **必须**与构建时使用的 TensorFlow 版本和 Python 环境匹配。请在与目标运行环境相同的 TF/Python 下分别构建 wheel。

## 构建与安装

```bash
git clone <repository-url>
cd tensorflow_musa_extension
git checkout tf_2.15.1_pluggable

pip install tensorflow==2.15.1
./build.sh wheel
pip install --force-reinstall dist/tensorflow_musa-*.whl --no-deps
```

开发时如只需构建插件动态库：

```bash
pip install tensorflow==2.15.1
./build.sh
```

构建脚本会校验已安装的 `tf.__version__` 是否为 **2.15.1**。如需临时覆盖（不推荐），可设置：

```bash
export TENSORFLOW_MUSA_TARGET_TF=2.15.1
```

## TensorFlow 版本与编译宏（供开发者参考）

本分支在 **TF 2.15.1** 下编译时，预处理宏来自已安装的 TensorFlow 头文件（`tensorflow/core/public/version.h`），**不需要**手动指定 TF 2.10。代码中的版本分界含义如下：

| 条件 | 作用（TF 2.15.1 构建时） |
|------|---------------------------|
| `TF_MINOR_VERSION >= 10` | 启用 PluggableDevice 路径：`SE_InitPlugin`、`MusaKernelRuntimeView`、`tensorflow_accelerator_device_info` |
| `TF_MINOR_VERSION < 10` | 启用 legacy C++ `MusaDevice` / `DeviceFactory::Register`（**本分支不编译此路径**） |
| `TF_MINOR_VERSION >= 15` | 注册 SE 插件的 `mem_zero` / `memset` / `memset32` 回调 |

CMake 会根据已安装 TF wheel 自动选择 **CXX11 ABI**（2.15.1 为 `_GLIBCXX_USE_CXX11_ABI=1`），无需额外配置。

可选：使用 legacy C++ 设备路径进行对比测试（仍须在 TF 2.15.1 下）：

```bash
export TENSORFLOW_MUSA_USE_LEGACY_DEVICE=1
python your_script.py
```

## 快速验证

```python
import tensorflow as tf
import tensorflow_musa as tf_musa

print(tf.__version__)  # 期望 2.15.1
print(tf.config.list_physical_devices("MUSA"))
```

在 MUSA 设备上运行一个简单算子：

```python
import tensorflow as tf
import tensorflow_musa

with tf.device("/device:MUSA:0"):
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    y = tf.matmul(x, x)

print(y)
```

## 测试

从已安装的 wheel 运行算子测试：

```bash
MUSA_VISIBLE_DEVICES=0 python test/test_runner.py
```

运行 PluggableDevice 合规性测试：

```bash
MUSA_VISIBLE_DEVICES=0 python test/test_runner.py --pattern "pluggable_*_test.py" --detail
```

## 调试

日志、GraphDef dump、遥测和运行时调试环境变量请参考 [docs/DEBUG_GUIDE.md](docs/DEBUG_GUIDE.md)。

## 参与贡献

欢迎提交 Issue 和 Pull Request。新增算子或行为变更请附带测试。

## 许可证

Apache License 2.0
