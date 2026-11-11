# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Literal, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import tensorrt as trt
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    get_trt_tensor,
    has_dynamic_shape,
    set_layer_name,
)
from torch_tensorrt.dynamo.conversion.impl.cat import (
    unify_and_concat_trt_tensors as unify_trt_shape_tensors,
)
from torch_tensorrt.dynamo.conversion.impl.shape import (
    get_shape_with_dynamic_shape,
)


def upsample(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    size: Optional[Sequence[int]],
    scale_factor: Optional[Sequence[float]],
    mode: str,
    align_corners: bool,
) -> TRTTensor:
    layer = ctx.net.add_resize(input)

    if scale_factor is not None:
        layer.scales = [1.0, 1.0] + list(scale_factor)
    else:
        shape = list(input.shape)[:2]
        if size is not None:
            shape += list(size)
        if has_dynamic_shape(shape):
            shape = get_shape_with_dynamic_shape(
                ctx, target, source_ir, name, shape, input
            )
            layer.set_input(1, shape)
        else:
            trt_shape = unify_trt_shape_tensors(
                ctx, target, name, shape, concat_axis=0, force_trt_output=False
            )
            if isinstance(trt_shape, list):
                layer.shape = trt_shape
            else:
                layer.set_input(1, trt_shape)

    if mode == "nearest":
        layer.resize_mode = trt.InterpolationMode.NEAREST
        layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ASYMMETRIC
    elif mode in ("linear", "bilinear", "trilinear"):
        layer.resize_mode = trt.InterpolationMode.LINEAR
        layer.coordinate_transformation = (
            trt.ResizeCoordinateTransformation.ALIGN_CORNERS
            if align_corners
            else trt.ResizeCoordinateTransformation.HALF_PIXEL
        )
    elif mode == "bicubic":
        layer.resize_mode = trt.InterpolationMode.CUBIC
        layer.coordinate_transformation = (
            trt.ResizeCoordinateTransformation.ALIGN_CORNERS
            if align_corners
            else trt.ResizeCoordinateTransformation.HALF_PIXEL
        )

    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


def _antialias_taps(
    input_size: int,
    output_size: int,
    align_corners: bool,
    scale: Optional[float],
    mode: Literal["bilinear", "bicubic"],
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Returns ATen's antialiased resize on one axis as (output_size, taps) arrays: the
    input index of each tap, its weight, and whether it is inside the output's window.
    """
    # Every step rounds to float32 as ATen's CUDA kernel does, including the fused
    # multiply-adds its compiler emits, so a following round to uint8 agrees with PyTorch.
    f32 = np.float32

    def fma(a: Any, b: Any, c: Any) -> Any:
        # float64 holds the product of two float32 values exactly: one rounding, as an FMA.
        return (np.float64(a) * b + c).astype(f32)

    if align_corners:
        ratio = (
            f32(input_size - 1) / f32(output_size - 1) if output_size > 1 else f32(0)
        )
    elif scale is not None and scale > 0:
        ratio = f32(1.0 / scale)
    else:
        ratio = f32(input_size) / f32(output_size)
    # A downscale stretches the filter over every input pixel it covers.
    stretch = ratio >= 1.0
    support = f32(1.0 if mode == "bilinear" else 2.0) * (ratio if stretch else f32(1))
    invscale = f32(1.0 / float(ratio)) if stretch else f32(1)
    half = np.arange(output_size, dtype=f32) + f32(0.5)
    start = np.maximum((fma(ratio, half, -support) + f32(0.5)).astype(np.int64), 0)
    end = np.minimum(
        (fma(ratio, half, support) + f32(0.5)).astype(np.int64), input_size
    )
    count = end - start
    taps = np.arange(count.max())
    in_window = taps < count[:, None]
    indices = np.minimum(start[:, None] + taps, input_size - 1)
    start_minus_center = fma(-ratio, half, start)[:, None]
    x = np.abs((taps.astype(f32) + start_minus_center + f32(0.5)) * invscale)
    if mode == "bilinear":
        weights = np.maximum(f32(1) - x, f32(0))
    else:
        a = f32(-0.5)
        weights = np.where(
            x < 1,
            fma(fma(a + 2, x, -(a + 3)) * x, x, f32(1)),
            np.where(x < 2, fma(fma(x - 5, x, f32(8)), x, f32(-4)) * a, f32(0)),
        )
    weights *= in_window
    # Summed tap by tap, in order, like the kernel (numpy's sum is pairwise).
    total = weights[:, 0].copy()
    for tap in range(1, weights.shape[1]):
        total += weights[:, tap]
    return indices, weights / total[:, None], in_window


def upsample_antialias(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    size: Sequence[int],
    align_corners: bool,
    scales: Sequence[Optional[float]],
    mode: Literal["bilinear", "bicubic"],
) -> TRTTensor:
    # TensorRT's resize has no antialiasing. As in ATen, each output pixel is a weighted
    # sum of the input pixels in its window, along the width and then the height. The
    # sums run in float32 and are rounded to the input dtype once, at the end.
    output = cast_trt_tensor(
        ctx, input, trt.float32, f"{name}_float32", target, source_ir
    )
    for axis in (3, 2):
        indices, weights, in_window = _antialias_taps(
            input.shape[axis],
            size[axis - 2],
            align_corners,
            scales[axis - 2],
            mode,
        )
        # All taps in one gather and one reduce: per-tap gathers and adds get fused
        # across chained resizes, and the engine build can then take minutes.
        gather = ctx.net.add_gather(
            output,
            get_trt_tensor(ctx, indices.astype(np.int32), f"{name}_indices_{axis}"),
            axis,
        )
        set_layer_name(gather, target, f"{name}_gather_{axis}", source_ir)
        # The gather swaps the resized axis for the (output, taps) axes of the indices.
        shape = (1,) * axis + weights.shape + (1,) * (3 - axis)
        taps = gather.get_output(0)
        if not in_window.all():
            # A tap past a window's end must add nothing, even for an infinite pixel.
            # Masking the pixel rather than the product lets the multiply and the sum
            # fuse, as they do in PyTorch's kernel.
            taps = impl.condition.where(
                ctx,
                target,
                source_ir,
                f"{name}_in_window_{axis}",
                taps,
                np.zeros((1,) * 5, dtype=np.float32),
                in_window.reshape(shape),
            )
        weighted = impl.elementwise.mul(
            ctx,
            target,
            source_ir,
            f"{name}_weighted_{axis}",
            taps,
            weights.reshape(shape).astype(np.float32),
        )
        output = impl.reduce.sum(
            ctx, target, source_ir, f"{name}_sum_{axis}", weighted, axis + 1, False
        )
    return cast_trt_tensor(ctx, output, input.dtype, f"{name}_cast", target, source_ir)
