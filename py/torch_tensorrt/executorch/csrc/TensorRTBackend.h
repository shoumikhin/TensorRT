#pragma once

#include <cuda_runtime.h>
#include <NvInfer.h>

#include <executorch/runtime/backend/interface.h>

#include <memory>
#include <string>
#include <vector>

namespace torch_tensorrt {
namespace executorch {

struct TrtDeleter {
  template <typename T>
  void operator()(T* p) const {
    delete p;
  }
};
template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter>;

class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override;
};

struct InputProfileBounds {
  nvinfer1::Dims min;
  nvinfer1::Dims max;
};

struct EngineHandle {
  TrtLogger logger;
  TrtUniquePtr<nvinfer1::IRuntime> runtime;
  TrtUniquePtr<nvinfer1::ICudaEngine> engine;
  TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx;
  cudaStream_t stream = nullptr;
  std::vector<std::string> input_binding_names;
  std::vector<std::string> output_binding_names;
  std::vector<InputProfileBounds> input_profile_bounds;
  size_t num_inputs = 0;
  size_t num_outputs = 0;
  int device_id = 0;
  bool unified_memory = false;

  // Cached GPU buffers for outputs (allocated on first execute, reused)
  std::vector<void*> cached_output_ptrs;
  std::vector<size_t> cached_output_sizes;

  // Cached GPU buffers for host inputs on discrete GPU (grow-only, reused)
  std::vector<void*> cached_input_ptrs;
  std::vector<size_t> cached_input_sizes;

  ~EngineHandle() {
    cudaSetDevice(device_id);
    if (stream) {
      cudaStreamSynchronize(stream);
    }
    for (void* p : cached_output_ptrs) {
      if (p) cudaFree(p);
    }
    for (void* p : cached_input_ptrs) {
      if (p) cudaFree(p);
    }
    exec_ctx.reset();
    engine.reset();
    runtime.reset();
    if (stream) {
      cudaStreamDestroy(stream);
      stream = nullptr;
    }
  }
};

class TensorRTBackend final
    : public ::executorch::runtime::BackendInterface {
 public:
  bool is_available() const override;

  ::executorch::runtime::Result<::executorch::runtime::DelegateHandle*> init(
      ::executorch::runtime::BackendInitContext& context,
      ::executorch::runtime::FreeableBuffer* processed,
      ::executorch::runtime::ArrayRef<::executorch::runtime::CompileSpec>
          compile_specs) const override;

  ::executorch::runtime::Error execute(
      ::executorch::runtime::BackendExecutionContext& context,
      ::executorch::runtime::DelegateHandle* handle,
      ::executorch::runtime::Span<::executorch::runtime::EValue*> args) const override;

  void destroy(::executorch::runtime::DelegateHandle* handle) const override;
};

} // namespace executorch
} // namespace torch_tensorrt
