/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Implementation of the slim, libtorch-free ETRTEngine handle used by
 * the portable ExecuTorch backend (see ETRTEngine.h for rationale).
 *
 * Zero includes from `core/runtime/` or `core/util/` (i.e. no
 * core::runtime::TRTEngine, no util/prelude.h, no torch/c10/at headers).
 * The wire format constants below MUST stay in sync with
 *   core/runtime/runtime.h::SerializedInfoIndex
 */
#include "core/runtime/executorch/ETRTEngine.h"

#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <executorch/runtime/platform/log.h>

namespace torch_tensorrt {
namespace executorch_backend {

namespace {

// Wire-format slot indices — MUST mirror core/runtime/runtime.h::SerializedInfoIndex.
// Canonical source of truth: third-party/torch_tensorrt/2.12/core/runtime/runtime.h
constexpr size_t SERIALIZATION_LEN = 12;
constexpr size_t ABI_TARGET_IDX = 0;
constexpr size_t NAME_IDX = 1;
constexpr size_t DEVICE_IDX = 2;
constexpr size_t ENGINE_IDX = 3;
constexpr size_t INPUT_BINDING_NAMES_IDX = 4;
constexpr size_t OUTPUT_BINDING_NAMES_IDX = 5;
constexpr const char* ABI_VERSION = "9";
constexpr char BINDING_DELIM = '%';

template <typename T>
std::shared_ptr<T> make_trt(T* ptr) {
  return std::shared_ptr<T>(ptr, TrtDeleter{});
}

std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    if (!item.empty()) {
      out.push_back(std::move(item));
    }
  }
  return out;
}

int parse_device_id(const std::string& device_info_str) {
  // device_info format produced by RTDevice::serialize():
  //   "<id>%<sm_major>%<sm_minor>%<dla_core>%<device_type>%<dev_name>"
  // We only need the first field.
  auto pos = device_info_str.find('%');
  std::string head = (pos == std::string::npos)
      ? device_info_str
      : device_info_str.substr(0, pos);
  if (head.empty()) {
    return 0;
  }
  try {
    return std::stoi(head);
  } catch (...) {
    return 0;
  }
}

class SimpleLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity == Severity::kINTERNAL_ERROR || severity == Severity::kERROR) {
      ET_LOG(Error, "[TRT] %s", msg);
    } else if (severity == Severity::kWARNING) {
      ET_LOG(Info, "[TRT] %s", msg);
    }
  }
};

SimpleLogger& get_logger() {
  static SimpleLogger l;
  return l;
}

} // namespace

void ETRTEngine::verify_serialization_fmt(const std::vector<std::string>& info) {
  if (info.size() != SERIALIZATION_LEN) {
    throw std::runtime_error(
        std::string(
            "TRT serialized blob has wrong number of fields: expected ") +
        std::to_string(SERIALIZATION_LEN) + ", got " +
        std::to_string(info.size()));
  }
  if (info[ABI_TARGET_IDX] != ABI_VERSION) {
    throw std::runtime_error(
        std::string("TRT serialized blob ABI version mismatch: expected ") +
        ABI_VERSION + ", got " + info[ABI_TARGET_IDX]);
  }
}

ETRTEngine::ETRTEngine(std::vector<std::string> serialized_info) {
  verify_serialization_fmt(serialized_info);

  name = serialized_info[NAME_IDX];

  device_id = parse_device_id(serialized_info[DEVICE_IDX]);
  cudaError_t err = cudaSetDevice(device_id);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("cudaSetDevice(") + std::to_string(device_id) +
        ") failed: " + cudaGetErrorString(err));
  }

  rt = make_trt(nvinfer1::createInferRuntime(get_logger()));
  if (!rt) {
    throw std::runtime_error("createInferRuntime failed");
  }

  const std::string& engine_bytes = serialized_info[ENGINE_IDX];
  cuda_engine = make_trt(
      rt->deserializeCudaEngine(engine_bytes.data(), engine_bytes.size()));
  if (!cuda_engine) {
    throw std::runtime_error("deserializeCudaEngine failed");
  }

  exec_ctx = make_trt(cuda_engine->createExecutionContext());
  if (!exec_ctx) {
    throw std::runtime_error("createExecutionContext failed");
  }

  in_binding_names =
      split(serialized_info[INPUT_BINDING_NAMES_IDX], BINDING_DELIM);
  out_binding_names =
      split(serialized_info[OUTPUT_BINDING_NAMES_IDX], BINDING_DELIM);

  // Validate binding names against the engine
  for (const auto& nm : in_binding_names) {
    if (cuda_engine->getTensorIOMode(nm.c_str()) !=
        nvinfer1::TensorIOMode::kINPUT) {
      throw std::runtime_error(
          std::string("Input binding name '") + nm +
          "' not found or not an input in engine");
    }
  }
  for (const auto& nm : out_binding_names) {
    if (cuda_engine->getTensorIOMode(nm.c_str()) !=
        nvinfer1::TensorIOMode::kOUTPUT) {
      throw std::runtime_error(
          std::string("Output binding name '") + nm +
          "' not found or not an output in engine");
    }
  }

  num_io = {in_binding_names.size(), out_binding_names.size()};

  // Per-engine non-blocking CUDA stream owned by the backend (no libtorch).
  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    stream = nullptr;
    throw std::runtime_error(
        std::string("cudaStreamCreateWithFlags failed: ") +
        cudaGetErrorString(err));
  }
}

ETRTEngine::~ETRTEngine() {
  if (stream != nullptr) {
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    stream = nullptr;
  }
  // shared_ptrs destroy in reverse declaration order:
  // exec_ctx → cuda_engine → rt. TRT lifetime contract satisfied.
}

} // namespace executorch_backend
} // namespace torch_tensorrt
