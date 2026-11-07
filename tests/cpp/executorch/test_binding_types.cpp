/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// A tensor whose type differs from its engine binding would be read or written as the binding's type. A float64 model
// compiled without truncate_double gives such a program: its engine binds float32 while the program keeps float64.

#include "torch_tensorrt/executorch/EngineHandle.h"
#include "torch_tensorrt/executorch/TensorRTBackend.h"

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <vector>

namespace torch_tensorrt {
namespace executorch_backend {
namespace {

using namespace ::executorch::aten;
using namespace ::executorch::runtime;

constexpr int32_t kElements = 4;
// Room for the widest type a test passes.
constexpr size_t kBufferBytes = kElements * sizeof(double);

// y = 2x + 1 on float32.
std::vector<uint8_t> build_blob() {
  static TRTLogger logger;
  TRTUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  TRTUniquePtr<nvinfer1::INetworkDefinition> network(builder ? builder->createNetworkV2(0) : nullptr);
  TRTUniquePtr<nvinfer1::IBuilderConfig> config(builder ? builder->createBuilderConfig() : nullptr);
  if (!network || !config) {
    return {};
  }
  auto* input = network->addInput("input_0", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {kElements}});
  static const float scale = 2.0f;
  static const float bias = 1.0f;
  auto* multiplier = network->addConstant({1, {1}}, {nvinfer1::DataType::kFLOAT, &scale, 1});
  auto* addend = network->addConstant({1, {1}}, {nvinfer1::DataType::kFLOAT, &bias, 1});
  auto* doubled =
      network->addElementWise(*input, *multiplier->getOutput(0), nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
  auto* out =
      network->addElementWise(*doubled, *addend->getOutput(0), nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
  out->setName("output_0");
  network->markOutput(*out);
  TRTUniquePtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  if (!plan) {
    return {};
  }
  const std::string metadata =
      R"({"io_bindings":[{"name":"input_0","is_input":true},{"name":"output_0","is_input":false}],"device_id":0})";
  const uint32_t metadata_offset = 32;
  const auto metadata_size = static_cast<uint32_t>(metadata.size());
  const uint32_t engine_offset = (metadata_offset + metadata_size + 15) / 16 * 16;
  const uint64_t engine_size = plan->size();
  std::vector<uint8_t> blob(engine_offset + engine_size, 0);
  std::memcpy(blob.data(), "TR01", 4);
  std::memcpy(blob.data() + 4, &metadata_offset, sizeof(metadata_offset));
  std::memcpy(blob.data() + 8, &metadata_size, sizeof(metadata_size));
  std::memcpy(blob.data() + 12, &engine_offset, sizeof(engine_offset));
  std::memcpy(blob.data() + 16, &engine_size, sizeof(engine_size));
  std::memcpy(blob.data() + metadata_offset, metadata.data(), metadata.size());
  std::memcpy(blob.data() + engine_offset, plan->data(), plan->size());
  return blob;
}

class BindingTypesTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    runtime_init();
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      return;
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    blob_ = build_blob();
  }

  void SetUp() override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      const char* required = std::getenv("TORCHTRT_EXECUTORCH_REQUIRE_CUDA");
      if (required && std::strcmp(required, "1") == 0) {
        FAIL() << "This run requires CUDA; no binding type test could execute";
      }
      GTEST_SKIP() << "No CUDA device; binding types are not covered";
    }
    ASSERT_FALSE(blob_.empty());
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&input_, kBufferBytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&output_, kBufferBytes), cudaSuccess);
    const std::vector<float> ones(kElements, 1.0f);
    ASSERT_EQ(cudaMemcpy(input_, ones.data(), kElements * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(output_, 0, kBufferBytes), cudaSuccess);
    BackendInitContext context(&allocator_, nullptr, nullptr, nullptr);
    FreeableBuffer processed(blob_.data(), blob_.size(), nullptr);
    auto result = backend_.init(context, &processed, {});
    ASSERT_TRUE(result.ok());
    handle_ = result.get();
  }

  void TearDown() override {
    if (handle_ != nullptr) {
      backend_.destroy(handle_);
    }
    cudaFree(input_);
    cudaFree(output_);
  }

  Error execute(ScalarType input_type, ScalarType output_type) {
    SizesType sizes[] = {kElements};
    TensorImpl in(input_type, 1, sizes, input_);
    TensorImpl out(output_type, 1, sizes, output_);
    EValue in_value{Tensor(&in)};
    EValue out_value{Tensor(&out)};
    EValue* args[] = {&in_value, &out_value};
    BackendExecutionContext context;
    const Error err = backend_.execute(context, handle_, {args, 2});
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    return err;
  }

  std::vector<float> output() {
    std::vector<float> values(kElements);
    EXPECT_EQ(cudaMemcpy(values.data(), output_, kElements * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    return values;
  }

  static std::vector<uint8_t> blob_;
  TensorRTBackend backend_;
  alignas(EngineHandle) uint8_t arena_[4096]{};
  MemoryAllocator allocator_{sizeof(arena_), arena_};
  DelegateHandle* handle_ = nullptr;
  void* input_ = nullptr;
  void* output_ = nullptr;
};

std::vector<uint8_t> BindingTypesTest::blob_;

TEST_F(BindingTypesTest, AnInputOfAnotherTypeIsRefusedBeforeTheEngineRuns) {
  EXPECT_EQ(execute(ScalarType::Double, ScalarType::Float), Error::InvalidProgram);
  // Same width as the binding's float32, so the check is on the type and not the size.
  EXPECT_EQ(execute(ScalarType::Int, ScalarType::Float), Error::InvalidProgram);
  EXPECT_EQ(output(), std::vector<float>(kElements, 0.0f));
  // A refused call leaves the handle usable.
  ASSERT_EQ(execute(ScalarType::Float, ScalarType::Float), Error::Ok);
  EXPECT_EQ(output(), std::vector<float>(kElements, 3.0f));
}

TEST_F(BindingTypesTest, AnOutputOfAnotherTypeIsRefusedBeforeTheEngineRuns) {
  EXPECT_EQ(execute(ScalarType::Float, ScalarType::Double), Error::InvalidProgram);
  EXPECT_EQ(output(), std::vector<float>(kElements, 0.0f));
}

} // namespace
} // namespace executorch_backend
} // namespace torch_tensorrt
