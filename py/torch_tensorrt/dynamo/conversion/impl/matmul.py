# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

import tensorrt as trt
import torch
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt import _enums
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    broadcast,
    cast_trt_tensor,
    get_trt_tensor,
    set_layer_name,
)


def matrix_multiply(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    other: TRTTensor,
    input_matrix_op: trt.MatrixOperation = trt.MatrixOperation.NONE,
    other_matrix_op: trt.MatrixOperation = trt.MatrixOperation.NONE,
) -> TRTTensor:
    if not isinstance(input, trt.ITensor):
        input = get_trt_tensor(ctx, input, f"{name}_input")
    if not isinstance(other, trt.ITensor):
        other = get_trt_tensor(
            ctx,
            other,
            f"{name}_other",
            dtype=_enums.dtype._from(input.dtype).to(torch.dtype),
        )

    preset_diff = 0

    if len(input.shape) == 1:
        preset_diff -= 1
        input_matrix_op = trt.MatrixOperation.VECTOR

    if len(other.shape) == 1:
        preset_diff += 1
        other_matrix_op = trt.MatrixOperation.VECTOR

    input, other = broadcast(
        ctx, input, other, f"{name}_input", f"{name}_other", preset_diff
    )
    if input.dtype in (trt.int32, trt.int64) and other.dtype in (trt.int32, trt.int64):
        return _integer_matrix_multiply(
            ctx,
            target,
            source_ir,
            name,
            input,
            other,
            input_matrix_op,
            other_matrix_op,
        )

    # Get the original input dtype
    input_dtype = _enums.dtype._from(input.dtype).to(torch.dtype)

    if (
        ctx.net.get_flag(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        and ctx.compilation_settings.use_fp32_acc
        and input_dtype == torch.float16
    ):
        input = cast_trt_tensor(ctx, input, torch.float32, f"{name}_input_casted")
        other = cast_trt_tensor(ctx, other, torch.float32, f"{name}_other_casted")

    matmul_layer = ctx.net.add_matrix_multiply(
        input, input_matrix_op, other, other_matrix_op
    )
    matmul_output = matmul_layer.get_output(0)

    if (
        ctx.net.get_flag(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        and ctx.compilation_settings.use_fp32_acc
        and input_dtype == torch.float16
    ):
        matmul_output = cast_trt_tensor(
            ctx, matmul_output, input_dtype, f"{name}_output_casted"
        )

    set_layer_name(matmul_layer, target, name, source_ir)
    return matmul_output


def _integer_matrix_multiply(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    other: TRTTensor,
    input_matrix_op: trt.MatrixOperation,
    other_matrix_op: trt.MatrixOperation,
) -> TRTTensor:
    """Multiply integer matrices exactly, as int64 products summed over the inner dim.

    TensorRT matrix multiply takes no integers, and float32 or TF32 would round them.
    TensorRT fuses the multiply and the sum, so memory stays low, but this is much
    slower than a float matrix multiply, and the build fails once M x K x N passes
    2**40.
    """

    def as_matrix(value: TRTTensor, op: trt.MatrixOperation, side: str) -> TRTTensor:
        value = cast_trt_tensor(ctx, value, trt.int64, f"{name}_{side}_int64")
        rank = len(value.shape)
        if op == trt.MatrixOperation.VECTOR:
            # A vector on the left is one row, a vector on the right is one column.
            dim = rank - 1 if side == "input" else rank
            return impl.unsqueeze.unsqueeze(
                ctx, target, source_ir, f"{name}_{side}_matrix", value, dim
            )
        if op == trt.MatrixOperation.TRANSPOSE:
            permutation = list(range(rank))
            permutation[-2:] = [rank - 1, rank - 2]
            return impl.permutation.permute(
                ctx, target, source_ir, f"{name}_{side}_transpose", value, permutation
            )
        return value

    lhs = as_matrix(input, input_matrix_op, "input")
    rhs = as_matrix(other, other_matrix_op, "other")
    rank = len(lhs.shape)
    lhs = impl.unsqueeze.unsqueeze(ctx, target, source_ir, f"{name}_lhs", lhs, rank)
    rhs = impl.unsqueeze.unsqueeze(ctx, target, source_ir, f"{name}_rhs", rhs, rank - 2)
    products = impl.elementwise.mul(
        ctx, target, source_ir, f"{name}_products", lhs, rhs
    )
    output = impl.reduce.sum(
        ctx, target, source_ir, f"{name}_sum", products, rank - 1, False
    )
    if other_matrix_op == trt.MatrixOperation.VECTOR:
        output = impl.squeeze.squeeze(
            ctx, target, source_ir, f"{name}_other_vector", output, -1
        )
    if input_matrix_op == trt.MatrixOperation.VECTOR:
        output = impl.squeeze.squeeze(
            ctx,
            target,
            source_ir,
            f"{name}_input_vector",
            output,
            -1 if other_matrix_op == trt.MatrixOperation.VECTOR else -2,
        )

    # An int32 engine tensor can hold an int64 value, such as an argmax, so the result
    # dtype comes from the graph. The cast to int32 wraps like PyTorch.
    node = ctx.current_node
    result_dtype = getattr(
        node.meta.get("val") if node is not None else None, "dtype", None
    )
    if result_dtype == torch.int32 or (
        result_dtype is None and input.dtype == other.dtype == trt.int32
    ):
        return cast_trt_tensor(ctx, output, trt.int32, f"{name}_int32")
    return output
