/*
 * Copyright (c) NVIDIA Corporation.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <optional>

namespace torch_tensorrt {
namespace core {
namespace runtime {

// Externally-managed stream registry.
//
// Allows callers to install a CUDA stream (per device) that the runtime
// will use as the engine stream for enqueueV3, instead of pulling one
// from torch's internal stream pool.
//
// Primary motivation: callers that want enqueueV3 to run inside a CUDA
// Green Context (cuda 12.4+) for SM partitioning. They create the stream
// themselves via cuGreenCtxStreamCreate and register it here. The runtime
// then wraps it via at::cuda::getStreamFromExternal() at execute() time.
//
// Lifetime: the caller owns the registered stream. They must keep it alive
// until they unregister it (or destroy the underlying green context),
// and must call unregister_external_stream() before destroying the stream.
//
// Thread safety: all functions are thread-safe.

void register_external_stream(int64_t device_id, cudaStream_t stream);

void unregister_external_stream(int64_t device_id);

std::optional<cudaStream_t> get_external_stream(int64_t device_id);

void clear_external_streams();

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
