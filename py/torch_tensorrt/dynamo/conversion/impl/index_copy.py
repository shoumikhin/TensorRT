# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""TensorRT conversion for ``aten.index_copy.default``.

Mirrors the structure of ``slice_scatter`` but the eligibility check is
gated by a validator (declared in ``aten_ops_converters.py``) and there
are two registered converters for the same op:

* ``aten_ops_index_copy_kv`` — HIGH priority, validator-gated. Fires only
  for a write that ``lift_mutated_buffers`` marked to update in place, when
  the input is a 4-D static floating-point cache, ``dim=2``, and the source
  has ``shape[2] == 1`` (single-position write, the common
  streaming-decoder pattern) or is written at consecutive positions
  ``start + arange(n)`` (a prefill chunk, ``n`` possibly dynamic). Emits
  ``IKVCacheUpdateLayer`` whose output is aliased to the cache input.

* ``aten_ops_index_copy_fallback``: STANDARD priority. Fires for every
  other case, static or dynamic. It refuses an int8 write, a float64 write
  without ``truncate_double``, and a write that ``compile`` marks because
  it would hand an engine a size of 0, which no engine profile takes.
  ``compile`` marks it only when no other op that can run in an engine
  reads its empty operands, for example a number of positions exported as
  ``Dim("T", max=8)``, whose smallest value is 0 even when an ``Input``
  spec starts at 1, since refit sees only the program. Export
  ``Dim("T", min=1, max=8)`` to keep the write in the engine. Implements
  the general semantics via scatter (equivalent to what the torch
  decomposition would produce).

Apart from those, no graph break is introduced for non-KV cases; they just
take a less-efficient TRT path.

The two functions here implement the bodies. Registration with the
validator lives next to the other aten converters for discoverability.
"""

from __future__ import annotations

import logging
from typing import Optional

import tensorrt as trt
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    get_positive_dim,
    set_layer_name,
)
from torch_tensorrt.dynamo.conversion.impl import select
from torch_tensorrt.dynamo.conversion.impl.shape import shape as get_shape
from torch_tensorrt.dynamo.conversion.impl.shuffle import reshape
from torch_tensorrt.dynamo.conversion.impl.slice_scatter import (
    emit_kv_cache_update_layer,
)
from torch_tensorrt.dynamo.utils import DYNAMIC_DIM

logger = logging.getLogger(__name__)


def index_copy_kv(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: int,
    index: TRTTensor,
    src: TRTTensor,
) -> TRTTensor:
    """KV-cache fast path. Caller (the validator) has already verified
    that this case is KV-eligible — 4-D static cache, dim=2, source
    rank=4 whose seq dim is either 1 or written at consecutive positions.

    ``index`` is the position tensor: shape ``[s_update]``. A single
    position is used directly as the write start. For several,
    ``lift_mutated_buffers`` proved they are consecutive and ascending, so the
    first one is the start. Positions are not checked against the cache:
    TensorRT ignores the updates that fall outside it, without an error.
    KVCacheUpdate's ``writeIndices`` arg expects shape
    ``[batch]``, so we broadcast/repeat the single index value across the
    batch dimension.
    """
    cache_shape = tuple(input.shape)
    batch = cache_shape[0]

    # KVCacheUpdate accepts int32 / int64 writeIndices; TRT auto-promotes
    # but be explicit to avoid surprises across version drift.
    if index.dtype != trt.int32:
        index = cast_trt_tensor(ctx, index, trt.int32, name + "_index_to_int32")

    # writeIndices shape must be [batch]. For batch=1 with index shape [1]
    # this is already correct. For batch > 1, broadcast: emit a constant
    # of shape [batch] filled with the single index value at runtime via
    # a gather/expand pattern. The validator's job is to allow only cases
    # where this is well-defined; for now we restrict to batch == 1 (see
    # the validator). A run of several positions is written from its first.
    write_indices = index
    if src.shape[2] != 1:
        start_layer = ctx.net.add_slice(index, start=(0,), shape=(1,), stride=(1,))
        set_layer_name(start_layer, target, name + "_write_start", source_ir)
        write_indices = start_layer.get_output(0)

    # When batch > 1, we'd need to broadcast `index` (shape [1]) to
    # [batch]. The validator currently keeps us in the batch==1 case.
    if isinstance(batch, int) and batch > 1:
        # Defensive: shouldn't happen if the validator is correct, but
        # fall back rather than emit a wrong layer.
        logger.debug(
            "index_copy_kv: batch > 1 not yet supported for runtime indices; "
            "falling back to scatter"
        )
        return index_copy_fallback(ctx, target, source_ir, name, input, dim, index, src)

    out = emit_kv_cache_update_layer(ctx, name, input, src, write_indices)
    if out is None:
        # KV emission failed (e.g. input not a direct network input);
        # fall through to scatter so correctness is preserved.
        return index_copy_fallback(ctx, target, source_ir, name, input, dim, index, src)
    return out


def index_copy_fallback(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: int,
    index: TRTTensor,
    src: TRTTensor,
) -> TRTTensor:
    """General-purpose ``index_copy``: scatter ``src`` into ``input`` at
    positions in ``index`` along ``dim``. Equivalent to the standard
    torch decomposition: build a broadcast index tensor of the same
    shape as ``src`` with ``index`` placed along ``dim`` and call
    ``scatter``. Positions are not checked, static or dynamic: a negative position
    counts from the end and one past the end is dropped, where PyTorch raises.
    """
    rank = len(input.shape)
    # A 0-d input and source are one element each, which scatter takes as they are.
    if rank == 0:
        return select.scatter(
            ctx, target, source_ir, name + "_fallback_scatter", input, dim, index, src
        )
    dim = get_positive_dim(dim, rank)
    # A 0-d source next to a 1-D input is one element too, as a source of shape (1,).
    if len(src.shape) == 0:
        src = reshape(ctx, target, source_ir, name + "_src_1d", src, (1,))
    src_shape = tuple(src.shape)

    # Reshape `index` (1-D, length matches src.shape[dim]) so it broadcasts
    # over the remaining dims of `src`. A dynamic length stays DYNAMIC_DIM, which
    # the shuffle resolves from the index itself.
    reshape_to = [1] * rank
    reshape_to[dim] = src_shape[dim]

    shuffle = ctx.net.add_shuffle(index)
    shuffle.reshape_dims = trt.Dims(reshape_to)
    reshaped_index = shuffle.get_output(0)

    # Broadcast/expand to src's shape. ``broadcast_to`` semantics: numpy
    # array first then to TRT.
    if DYNAMIC_DIM not in src_shape:
        # Static case: just expand via a slice with broadcast strides.
        expand_layer = ctx.net.add_slice(
            reshaped_index,
            start=tuple(0 for _ in range(rank)),
            shape=src_shape,
            stride=tuple(0 if i != dim else 1 for i in range(rank)),
        )
        index_broadcast = expand_layer.get_output(0)
    else:
        runtime_src_shape = [
            (
                get_shape(
                    ctx, target, source_ir, name + f"_src_shape_{axis}", src, axis
                )
                if size == DYNAMIC_DIM
                else size
            )
            for axis, size in enumerate(src_shape)
        ]
        index_broadcast = impl.slice.expand(
            ctx,
            target,
            source_ir,
            name + "_index_expand",
            reshaped_index,
            tuple(runtime_src_shape),
        )

    return select.scatter(
        ctx,
        target,
        source_ir,
        name + "_fallback_scatter",
        input,
        dim,
        index_broadcast,
        src,
    )
