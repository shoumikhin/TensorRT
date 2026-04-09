#include "TensorRTBackend.h"
#include "TensorRTBlobHeader.h"

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/platform/log.h>

#include <cstring>
#include <vector>

namespace torch_tensorrt {
namespace executorch {

using namespace ::executorch::runtime;
using ::executorch::aten::SizesType;

void TrtLogger::log(Severity severity, const char* msg) noexcept {
  if (severity <= Severity::kERROR) {
    ET_LOG(Error, "TensorRT: %s", msg);
  } else if (severity == Severity::kWARNING) {
    ET_LOG(Info, "TensorRT warning: %s", msg);
  }
}

// Heavyweight: creates a full TRT runtime to probe availability.
// Matches the pattern expected by ExecuTorch backend registration.
bool TensorRTBackend::is_available() const {
  static TrtLogger logger;
  std::unique_ptr<nvinfer1::IRuntime> rt(nvinfer1::createInferRuntime(logger));
  return rt != nullptr;
}

Result<DelegateHandle*> TensorRTBackend::init(
    BackendInitContext& context,
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs) const {
  auto* handle =
      context.get_runtime_allocator()->allocateInstance<EngineHandle>();
  if (!handle) {
    return Error::MemoryAllocationFailed;
  }
  new (handle) EngineHandle();

  // Parse TR01 blob
  TensorRTBlobHeader header;
  if (!TensorRTBlobHeader::parse(
          processed->data(), processed->size(), header)) {
    ET_LOG(Error, "Failed to parse TensorRT blob header");
    handle->~EngineHandle();
    return Error::InvalidProgram;
  }

  handle->input_binding_names = std::move(header.input_binding_names);
  handle->output_binding_names = std::move(header.output_binding_names);
  handle->num_inputs = handle->input_binding_names.size();
  handle->num_outputs = handle->output_binding_names.size();
  handle->device_id = header.device_id;

  // Set CUDA device and create stream
  auto cuda_err = cudaSetDevice(handle->device_id);
  if (cuda_err != cudaSuccess) {
    ET_LOG(
        Error,
        "cudaSetDevice(%d) failed: %s",
        handle->device_id,
        cudaGetErrorString(cuda_err));
    handle->~EngineHandle();
    return Error::InvalidProgram;
  }

  cuda_err = cudaStreamCreate(&handle->stream);
  if (cuda_err != cudaSuccess) {
    ET_LOG(
        Error,
        "cudaStreamCreate failed: %s",
        cudaGetErrorString(cuda_err));
    handle->~EngineHandle();
    return Error::InvalidProgram;
  }

  // Detect unified memory (Jetson)
  int is_integrated = 0;
  cuda_err = cudaDeviceGetAttribute(
      &is_integrated, cudaDevAttrIntegrated, handle->device_id);
  if (cuda_err != cudaSuccess) {
    ET_LOG(
        Info,
        "cudaDeviceGetAttribute(cudaDevAttrIntegrated) failed: %s, assuming discrete GPU",
        cudaGetErrorString(cuda_err));
  }
  handle->unified_memory = (is_integrated != 0);

  // Create TRT runtime and deserialize engine
  handle->runtime.reset(nvinfer1::createInferRuntime(handle->logger));
  if (!handle->runtime) {
    ET_LOG(Error, "Failed to create TensorRT runtime");
    handle->~EngineHandle();
    return Error::InvalidProgram;
  }

  const void* engine_data =
      TensorRTBlobHeader::engine_data(processed->data(), header);
  handle->engine.reset(handle->runtime->deserializeCudaEngine(
      engine_data, header.engine_size));
  if (!handle->engine) {
    ET_LOG(Error, "Failed to deserialize TensorRT engine");
    handle->~EngineHandle();
    return Error::InvalidProgram;
  }

  handle->exec_ctx.reset(handle->engine->createExecutionContext());
  if (!handle->exec_ctx) {
    ET_LOG(Error, "Failed to create TensorRT execution context");
    handle->~EngineHandle();
    return Error::InvalidProgram;
  }

  // V1 limitation: reject shape tensor inputs
  for (const auto& name : handle->input_binding_names) {
    if (handle->engine->isShapeInferenceIO(name.c_str())) {
      ET_LOG(
          Error,
          "Shape tensor input '%s' not supported in V1",
          name.c_str());
      handle->~EngineHandle();
      return Error::InvalidProgram;
    }
  }

  // Cache optimization profile bounds per input for validation
  for (size_t i = 0; i < handle->num_inputs; i++) {
    InputProfileBounds bounds;
    bounds.min = handle->engine->getProfileShape(
        handle->input_binding_names[i].c_str(), 0,
        nvinfer1::OptProfileSelector::kMIN);
    bounds.max = handle->engine->getProfileShape(
        handle->input_binding_names[i].c_str(), 0,
        nvinfer1::OptProfileSelector::kMAX);
    if (bounds.min.nbDims < 0 || bounds.max.nbDims < 0) {
      ET_LOG(
          Error,
          "getProfileShape failed for input '%s'",
          handle->input_binding_names[i].c_str());
      handle->~EngineHandle();
      return Error::InvalidProgram;
    }
    handle->input_profile_bounds.push_back(bounds);
  }

  // Free processed data — no longer needed after deserialization
  processed->Free();

  return handle;
}

Error TensorRTBackend::execute(
    BackendExecutionContext& context,
    DelegateHandle* delegate_handle,
    Span<EValue*> args) const {
  auto* handle = static_cast<EngineHandle*>(delegate_handle);
  auto* exec_ctx = handle->exec_ctx.get();
  auto stream = handle->stream;
  size_t num_args = args.size();

  if (num_args != handle->num_inputs + handle->num_outputs) {
    ET_LOG(
        Error,
        "TRT execute: expected %zu args (%zu in + %zu out), got %zu",
        handle->num_inputs + handle->num_outputs,
        handle->num_inputs,
        handle->num_outputs,
        num_args);
    return Error::InvalidProgram;
  }

  auto cuda_err = cudaSetDevice(handle->device_id);
  ET_CHECK_OR_RETURN_ERROR(
      cuda_err == cudaSuccess,
      InvalidProgram,
      "cudaSetDevice(%d) failed: %s",
      handle->device_id,
      cudaGetErrorString(cuda_err));

  // Initialize cached input buffers on first call
  if (handle->cached_input_ptrs.empty()) {
    handle->cached_input_ptrs.resize(handle->num_inputs, nullptr);
    handle->cached_input_sizes.resize(handle->num_inputs, 0);
  }

  // Bind inputs
  for (size_t i = 0; i < handle->num_inputs; i++) {
    auto& tensor = args[i]->toTensor();
    const auto& name = handle->input_binding_names[i];
    void* bind_ptr = nullptr;

    if (tensor.nbytes() == 0) {
      // Zero-size tensor: use a 1-byte cached buffer
      if (handle->cached_input_sizes[i] == 0) {
        auto alloc_err = cudaMalloc(&handle->cached_input_ptrs[i], 1);
        if (alloc_err != cudaSuccess) {
          return Error::MemoryAllocationFailed;
        }
        handle->cached_input_sizes[i] = 1;
      }
      bind_ptr = handle->cached_input_ptrs[i];
    } else {
      cudaPointerAttributes attrs{};
      cudaError_t err =
          cudaPointerGetAttributes(&attrs, tensor.const_data_ptr());
      if (err != cudaSuccess) cudaGetLastError(); // clear sticky error

      if (handle->unified_memory) {
        // On integrated GPUs (Jetson), all host memory is CUDA-accessible.
        // Bind the host pointer directly — no allocation or copy needed.
        bind_ptr = tensor.mutable_data_ptr();
      } else if (err == cudaSuccess &&
          (attrs.type == cudaMemoryTypeDevice ||
           attrs.type == cudaMemoryTypeManaged)) {
        bind_ptr = tensor.mutable_data_ptr();
      } else {
        // Host input on discrete GPU — use cached GPU staging buffer
        size_t needed = tensor.nbytes();
        if (needed > handle->cached_input_sizes[i]) {
          if (handle->cached_input_ptrs[i]) {
            cudaFree(handle->cached_input_ptrs[i]);
          }
          auto alloc_err =
              cudaMalloc(&handle->cached_input_ptrs[i], needed);
          if (alloc_err != cudaSuccess) {
            handle->cached_input_ptrs[i] = nullptr;
            handle->cached_input_sizes[i] = 0;
            return Error::MemoryAllocationFailed;
          }
          handle->cached_input_sizes[i] = needed;
        }
        bind_ptr = handle->cached_input_ptrs[i];
        auto cpy_err = cudaMemcpyAsync(
            bind_ptr,
            tensor.const_data_ptr(),
            tensor.nbytes(),
            cudaMemcpyHostToDevice,
            stream);
        if (cpy_err != cudaSuccess) {
          ET_LOG(
              Error,
              "H2D cudaMemcpyAsync failed for input '%s': %s",
              name.c_str(),
              cudaGetErrorString(cpy_err));
          return Error::InvalidProgram;
        }
      }
    }
    if (!exec_ctx->setTensorAddress(name.c_str(), bind_ptr)) {
      ET_LOG(
          Error,
          "TRT setTensorAddress failed for input '%s'",
          name.c_str());
      return Error::InvalidProgram;
    }
  }

  // Set input shapes with profile bounds validation
  for (size_t i = 0; i < handle->num_inputs; i++) {
    auto& tensor = args[i]->toTensor();
    nvinfer1::Dims dims;
    dims.nbDims = tensor.dim();
    if (dims.nbDims > nvinfer1::Dims::MAX_DIMS) {
      ET_LOG(
          Error,
          "Input '%s' has %d dims, exceeds TRT max of %d",
          handle->input_binding_names[i].c_str(),
          dims.nbDims,
          nvinfer1::Dims::MAX_DIMS);
      return Error::InvalidArgument;
    }
    for (int d = 0; d < dims.nbDims; d++) {
      dims.d[d] = static_cast<int64_t>(tensor.sizes()[d]);
    }

    const auto& bounds = handle->input_profile_bounds[i];
    if (dims.nbDims != bounds.min.nbDims) {
      ET_LOG(
          Error,
          "Input '%s' has %d dims but engine profile expects %d",
          handle->input_binding_names[i].c_str(),
          dims.nbDims,
          bounds.min.nbDims);
      return Error::InvalidArgument;
    }
    for (int d = 0; d < dims.nbDims; d++) {
      if (dims.d[d] < bounds.min.d[d] || dims.d[d] > bounds.max.d[d]) {
        ET_LOG(
            Error,
            "Input '%s' dim %d has value %lld, outside profile bounds [%lld..%lld]",
            handle->input_binding_names[i].c_str(),
            d,
            static_cast<long long>(dims.d[d]),
            static_cast<long long>(bounds.min.d[d]),
            static_cast<long long>(bounds.max.d[d]));
        return Error::InvalidArgument;
      }
    }

    if (!exec_ctx->setInputShape(
            handle->input_binding_names[i].c_str(), dims)) {
      ET_LOG(
          Error,
          "TRT setInputShape failed for '%s'",
          handle->input_binding_names[i].c_str());
      return Error::InvalidArgument;
    }
  }

  // Infer output shapes
  int32_t const io_size =
      static_cast<int32_t>(handle->engine->getNbIOTensors());
  std::vector<char const*> unresolved(io_size);
  int32_t nbUnresolved =
      exec_ctx->inferShapes(io_size, unresolved.data());
  if (nbUnresolved != 0) {
    ET_LOG(Error, "TRT inferShapes failed for %d tensors", nbUnresolved);
    return Error::InvalidProgram;
  }

  // Bind outputs
  // Track which outputs need D2H copy (index -> cached buffer ptr)
  std::vector<std::pair<size_t, void*>> outputs_needing_copy;

  // Initialize cached output buffers on first call
  if (handle->cached_output_ptrs.empty()) {
    handle->cached_output_ptrs.resize(handle->num_outputs, nullptr);
    handle->cached_output_sizes.resize(handle->num_outputs, 0);
  }

  for (size_t i = 0; i < handle->num_outputs; i++) {
    auto& tensor = args[handle->num_inputs + i]->toTensor();
    const auto& name = handle->output_binding_names[i];

    nvinfer1::Dims trt_dims = exec_ctx->getTensorShape(name.c_str());

    SizesType new_sizes[nvinfer1::Dims::MAX_DIMS];
    for (int d = 0; d < trt_dims.nbDims; d++) {
      new_sizes[d] = static_cast<SizesType>(trt_dims.d[d]);
    }
    auto resize_err = resize_tensor(
        tensor,
        {new_sizes, static_cast<size_t>(trt_dims.nbDims)});
    if (resize_err != Error::Ok) {
      ET_LOG(Error, "resize_tensor failed for output '%s'", name.c_str());
      return resize_err;
    }

    void* bind_ptr = nullptr;
    if (tensor.nbytes() == 0) {
      // Zero-size tensor: use a 1-byte cached buffer
      if (handle->cached_output_sizes[i] == 0) {
        auto alloc_err = cudaMalloc(&handle->cached_output_ptrs[i], 1);
        if (alloc_err != cudaSuccess) {
          return Error::MemoryAllocationFailed;
        }
        handle->cached_output_sizes[i] = 1;
      }
      bind_ptr = handle->cached_output_ptrs[i];
    } else {
      cudaPointerAttributes attrs{};
      cudaError_t err =
          cudaPointerGetAttributes(&attrs, tensor.const_data_ptr());
      if (err != cudaSuccess) cudaGetLastError();

      if (handle->unified_memory) {
        // On integrated GPUs (Jetson), bind the host pointer directly.
        bind_ptr = tensor.mutable_data_ptr();
      } else if (err == cudaSuccess &&
          (attrs.type == cudaMemoryTypeDevice ||
           attrs.type == cudaMemoryTypeManaged)) {
        bind_ptr = tensor.mutable_data_ptr();
      } else {
        // Output is host memory on discrete GPU — use cached GPU buffer
        size_t needed = tensor.nbytes();
        if (needed > handle->cached_output_sizes[i]) {
          if (handle->cached_output_ptrs[i]) {
            cudaFree(handle->cached_output_ptrs[i]);
          }
          auto alloc_err =
              cudaMalloc(&handle->cached_output_ptrs[i], needed);
          if (alloc_err != cudaSuccess) {
            handle->cached_output_ptrs[i] = nullptr;
            handle->cached_output_sizes[i] = 0;
            return Error::MemoryAllocationFailed;
          }
          handle->cached_output_sizes[i] = needed;
        }
        bind_ptr = handle->cached_output_ptrs[i];
        outputs_needing_copy.push_back({i, bind_ptr});
      }
    }
    if (!exec_ctx->setTensorAddress(name.c_str(), bind_ptr)) {
      ET_LOG(
          Error,
          "TRT setTensorAddress failed for output '%s'",
          name.c_str());
      return Error::InvalidProgram;
    }
  }

  // Execute
  if (!exec_ctx->enqueueV3(stream)) {
    ET_LOG(Error, "TRT enqueueV3 failed");
    return Error::InvalidProgram;
  }

  // Copy outputs from device staging buffers to host tensors (discrete GPU only)
  for (auto& [idx, dev_ptr] : outputs_needing_copy) {
    auto& tensor = args[handle->num_inputs + idx]->toTensor();
    auto cpy_err = cudaMemcpyAsync(
        tensor.mutable_data_ptr(),
        dev_ptr,
        tensor.nbytes(),
        cudaMemcpyDeviceToHost,
        stream);
    if (cpy_err != cudaSuccess) {
      ET_LOG(
          Error,
          "D2H cudaMemcpyAsync failed for output %zu: %s",
          idx,
          cudaGetErrorString(cpy_err));
      return Error::InvalidProgram;
    }
  }

  // Synchronize
  cuda_err = cudaStreamSynchronize(stream);
  if (cuda_err != cudaSuccess) {
    ET_LOG(
        Error,
        "cudaStreamSynchronize failed: %s",
        cudaGetErrorString(cuda_err));
    return Error::InvalidProgram;
  }

  return Error::Ok;
}

void TensorRTBackend::destroy(DelegateHandle* handle) const {
  // Explicitly destroy the handle's resources. The handle memory itself
  // is owned by ExecuTorch's arena allocator and freed when the program is unloaded.
  static_cast<EngineHandle*>(handle)->~EngineHandle();
}

namespace {
TensorRTBackend& get_backend() {
  static TensorRTBackend backend;
  return backend;
}
// NOLINTNEXTLINE(facebook-hte-NonPodStaticDeclaration)
const Backend backend_id{"TensorRTBackend", &get_backend()};
static const auto registered = register_backend(backend_id);
} // namespace

} // namespace executorch
} // namespace torch_tensorrt
