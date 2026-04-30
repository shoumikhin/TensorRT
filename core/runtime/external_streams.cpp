/*
 * Copyright (c) NVIDIA Corporation.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "core/runtime/external_streams.h"

#include <map>
#include <mutex>

namespace torch_tensorrt {
namespace core {
namespace runtime {

namespace {

std::mutex& registry_mutex() {
  static std::mutex m;
  return m;
}

std::map<int64_t, cudaStream_t>& registry() {
  static std::map<int64_t, cudaStream_t> r;
  return r;
}

} // namespace

void register_external_stream(int64_t device_id, cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  registry()[device_id] = stream;
}

void unregister_external_stream(int64_t device_id) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  registry().erase(device_id);
}

std::optional<cudaStream_t> get_external_stream(int64_t device_id) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto& r = registry();
  auto it = r.find(device_id);
  if (it == r.end()) {
    return std::nullopt;
  }
  return it->second;
}

void clear_external_streams() {
  std::lock_guard<std::mutex> lock(registry_mutex());
  registry().clear();
}

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
