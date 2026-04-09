# ExecuTorch Backend for Torch-TensorRT

Export Torch-TensorRT compiled models to ExecuTorch `.pte` format. TensorRT
engines are serialized into the `.pte` file and executed at runtime via
ExecuTorch's delegation mechanism — no Python or libtorch needed at inference.

## Prerequisites

- **NVIDIA GPU** with CUDA support
- **CUDA** 12.6+ ([install guide](https://developer.nvidia.com/cuda-downloads))
- **TensorRT** 10.3+ ([install guide](https://developer.nvidia.com/tensorrt))
- **Python** 3.10+

## Setup

```bash
python -m venv .venv && source .venv/bin/activate

# Pick the CUDA wheel index matching your toolkit version:
#   CUDA 12.6 → cu126, CUDA 12.8 → cu128
PYTHON_ONLY=1 pip install --extra-index-url https://download.pytorch.org/whl/cu126 -e ".[executorch]"
```

Set `LD_LIBRARY_PATH` (Linux) so the TensorRT and CUDA shared libraries are
found at runtime (required in every new shell session). If `libcudart.so.12` is
not found, add your CUDA toolkit's `lib64` directory as well:

```bash
source .venv/bin/activate
TRT_LIBS=$(python -c "import os, tensorrt_libs; print(os.path.dirname(tensorrt_libs.__file__))")
export LD_LIBRARY_PATH=$TRT_LIBS${CUDA_HOME:+:$CUDA_HOME/lib64}:$LD_LIBRARY_PATH
```

Verify:

```bash
python -c "import torch; import torch_tensorrt; import executorch; print('OK')"
```

## Quickstart

### 1. Export a model to `.pte`

```python
import torch
import torch_tensorrt

class MyModel(torch.nn.Module):
    def forward(self, x):
        return x + 1

model = MyModel().eval().cuda()
inputs = [torch.randn(1, 3, 224, 224).cuda()]

trt_model = torch_tensorrt.compile(model, inputs=inputs, min_block_size=1)
torch_tensorrt.save(trt_model, "model.pte", output_format="executorch", inputs=inputs)
```

### 2. Build and run the C++ runner

Running a TRT-delegated `.pte` requires the C++ ExecuTorch runtime with the
TRT backend compiled in.

#### Build ExecuTorch

```bash
git clone --depth 1 --recurse-submodules https://github.com/pytorch/executorch.git && cd executorch
mkdir cmake-out && cd cmake-out
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
         -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
         -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
         -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON
cmake --build . --parallel
cmake --install . --prefix install
```

#### Build the runner

Create a `CMakeLists.txt` for the runner (adjust paths as needed):

```cmake
cmake_minimum_required(VERSION 3.29)
project(runner)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

# ExecuTorch (built from source above)
set(EXECUTORCH_SOURCE_DIR "" CACHE PATH "Path to executorch source root")
set(EXECUTORCH_BUILD_DIR "${EXECUTORCH_SOURCE_DIR}/cmake-out" CACHE PATH
    "Path to executorch build directory")
# cmake --install puts configs under install/lib or install/lib64
list(APPEND CMAKE_PREFIX_PATH
    "${EXECUTORCH_BUILD_DIR}/install/lib/cmake/ExecuTorch"
    "${EXECUTORCH_BUILD_DIR}/install/lib64/cmake/ExecuTorch")
find_package(executorch REQUIRED)
# Backend headers are not part of the install — include the source tree parent
# so that <executorch/runtime/backend/interface.h> resolves correctly
include_directories(${EXECUTORCH_SOURCE_DIR}/..)

# TensorRT
set(TENSORRT_ROOT "" CACHE PATH "Path to TensorRT installation")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS "${TENSORRT_ROOT}/include")
find_library(NVINFER_LIB nvinfer HINTS "${TENSORRT_ROOT}/lib")
if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIB)
  message(FATAL_ERROR "TensorRT not found. Set -DTENSORRT_ROOT=<path>")
endif()

# TensorRT backend (from torch_tensorrt source)
set(TENSORRT_EXECUTORCH_DIR "" CACHE PATH "Path to py/torch_tensorrt/executorch/csrc")
add_library(trt_backend STATIC
    ${TENSORRT_EXECUTORCH_DIR}/TensorRTBackend.cpp)
target_include_directories(trt_backend PRIVATE
    ${TENSORRT_EXECUTORCH_DIR}
    ${TENSORRT_INCLUDE_DIR})
target_compile_options(trt_backend PRIVATE -frtti -fexceptions)
target_link_libraries(trt_backend PRIVATE
    executorch_core
    CUDA::cudart
    ${NVINFER_LIB})

# Runner executable
add_executable(runner main.cpp)
target_compile_options(runner PRIVATE -frtti -fexceptions)
target_link_libraries(runner PRIVATE
    executorch
    portable_ops_lib
    extension_data_loader
    extension_module_static
    extension_tensor
    $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_backend>
    CUDA::cudart
    ${NVINFER_LIB})
```

Write `main.cpp`:

```cpp
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <cuda_runtime.h>

using namespace executorch::extension;

int main(int argc, char* argv[]) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model.pte>\n", argv[0]); return 1; }
    Module module(argv[1], Module::LoadMode::Mmap);

    void* input = nullptr;
    cudaMallocManaged(&input, 1 * 3 * 224 * 224 * sizeof(float));
    auto tensor = from_blob(
        input, {1, 3, 224, 224}, executorch::aten::ScalarType::Float,
        [](void* p) { cudaFree(p); });

    const auto result = module.forward(tensor);

    if (result.ok()) {
        const auto output = result->at(0).toTensor().const_data_ptr<float>();
        printf("output[0] = %f\n", output[0]);
    }

    return 0;
}
```

Build and run:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_SOURCE_DIR=/path/to/executorch \
    -DTENSORRT_ROOT=/path/to/TensorRT \
    -DTENSORRT_EXECUTORCH_DIR=/path/to/TensorRT/py/torch_tensorrt/executorch/csrc
cmake --build . --parallel
./runner model.pte
```

> **Note:** The `.pte` file contains a TRT engine serialized for a specific GPU
> architecture. Running it on a different GPU than the one it was compiled on may
> fail. Recompile the model when targeting a different device.

## Compile options

All standard `torch_tensorrt.compile()` options work:

```python
trt_model = torch_tensorrt.compile(
    model, inputs=inputs,
    enabled_precisions={torch.float16},    # FP16 inference
    workspace_size=1 << 30,                # 1 GB TRT workspace
    min_block_size=5,                      # min ops per TRT partition
)
torch_tensorrt.save(trt_model, "model.pte", output_format="executorch", inputs=inputs)
```

### Extra partitioners

Route unsupported ops to other backends (e.g. XNNPACK) instead of slow
portable CPU kernels. TensorRT is always included automatically:

```python
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

torch_tensorrt.save(
    trt_model, "model.pte",
    output_format="executorch",
    inputs=inputs,
    partitioners=[XnnpackPartitioner()],
)
```

## ExecuTorch-native API

The high-level API above is one-shot: compile, lower, save. If you need to
inspect the graph between steps or customize the ExecuTorch lowering, use the
step-by-step API instead. `to_trt()` is a convenience wrapper that handles
TRT compilation and engine conversion in one call, returning an
`ExportedProgram` you then lower to ExecuTorch yourself:

```python
import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from torch_tensorrt.executorch import to_trt, TensorRTPartitioner, get_edge_compile_config

model = MyModel().eval().cuda()
inputs = (torch.randn(1, 3, 224, 224).cuda(),)

ep = export(model, inputs)

edge = to_edge_transform_and_lower(
    to_trt(ep, inputs, workspace_size=1 << 30),
    compile_config=get_edge_compile_config(),
    partitioner=[TensorRTPartitioner()],
)
pte = edge.to_executorch()

with open("model.pte", "wb") as f:
    pte.write_to_file(f)
```

`to_trt()` accepts the same options as `dynamo.compile()` (e.g. `workspace_size`,
`enabled_precisions`). Any extra keyword arguments not handled directly are
forwarded to `dynamo.compile()`. Note that `to_trt()`
defaults to `min_block_size=1` and `require_full_compilation=True`, which
differ from `dynamo.compile()` defaults. `get_edge_compile_config()` disables
Edge IR validation and Edge ops, both of which must be off because TRT engine
ops do not have out-variants.

Add extra partitioners in the `to_edge_transform_and_lower()` call:

```python
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

edge = to_edge_transform_and_lower(
    to_trt(ep, inputs),
    compile_config=get_edge_compile_config(),
    partitioner=[TensorRTPartitioner(), XnnpackPartitioner()],
)
```

## Running tests

Run all ExecuTorch tests:

```bash
pytest tests/py/dynamo/executorch/ -v
```

Run only unit tests (no TensorRT engine compilation):

```bash
pytest tests/py/dynamo/executorch/ -m unit -v
```

Tests require `executorch` and `torch_tensorrt`. Integration tests also require a
CUDA GPU; unit tests (`-m unit`) run without one.

## Further reading

- [Torch-TensorRT documentation](https://pytorch.org/TensorRT/)
- [ExecuTorch documentation](https://pytorch.org/executorch/)
