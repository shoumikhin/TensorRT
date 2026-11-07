# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestAddConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (2, 1)),
            ("3d", (2, 1, 2)),
        ]
    )
    def test_add_tensor(self, _, shape):
        class add(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.add.Tensor(lhs_val, rhs_val)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            add(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), 1),
            ("3d", (2, 1, 2), 2.0),
        ]
    )
    def test_add_tensor_alpha(self, _, shape, alpha):
        class add(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.add.Tensor(lhs_val, rhs_val, alpha=alpha)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            add(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), 1.0),
            ("3d", (2, 1, 2), 2),
        ]
    )
    def test_add_scalar(self, _, shape, scalar):
        class add(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.add.Tensor(lhs_val, scalar)

        inputs = [torch.randn(shape)]
        self.run_test(
            add(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), 1.0, 1.0),
            ("3d", (2, 1, 2), 2, 2),
        ]
    )
    def test_add_scalar_alpha(self, _, shape, scalar, alpha):
        class add(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.add.Tensor(lhs_val, scalar, alpha=alpha)

        inputs = [torch.randn(shape)]
        self.run_test(
            add(),
            inputs,
        )

    def test_add_int8_buffer(self):
        # TensorRT reads an INT8 constant as quantized data, so a plain int8 weight
        # must not become one.
        class add(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "offsets", torch.arange(-128, 128, dtype=torch.int8).reshape(16, 16)
                )

            def forward(self, x):
                return torch.ops.aten.add.Tensor(self.offsets, x)

        inputs = [torch.randn(16, 16)]
        self.run_test(
            add(),
            inputs,
            rtol=0,
            atol=0,
        )

    def test_add_int8_buffer_int8_output_wraps(self):
        class add(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "offsets", torch.tensor([100, -100, 120, 5], dtype=torch.int8)
                )

            def forward(self, x):
                return torch.ops.aten.add.Tensor(self.offsets, x.to(torch.int8))

        model = add().cuda().eval()
        inputs = (torch.tensor([100.0, -100.0, 120.0, 5.0]).cuda(),)
        trt_model = torch_tensorrt.dynamo.compile(
            torch.export.export(model, inputs), arg_inputs=inputs, min_block_size=1
        )
        out = trt_model(*inputs)
        self.assertEqual(out.dtype, torch.int8)
        torch.testing.assert_close(
            out.cpu(), torch.tensor([-56, 56, -16, 10], dtype=torch.int8)
        )

    @parameterized.expand(
        [
            ("python_int", lambda x, c: x + 100),
            ("python_bool", lambda x, c: x + True),
            ("buffer", lambda x, c: torch.ops.aten.add.Tensor(c, x)),
            ("sub", lambda x, c: x - 100),
            ("rsub", lambda x, c: 100 - x),
            ("mul", lambda x, c: x * c),
            ("cat", lambda x, c: torch.cat([x, c]) + torch.cat([c, x])),
            ("where", lambda x, c: torch.where(x > 0, c, x) + x),
            ("index_select", lambda x, c: torch.index_select(c, 0, x.long() % 4) + x),
        ]
    )
    def test_int8_result_wraps_around_inside_the_engine(self, _, compute):
        # TensorRT has no plain int8 constant, so c and Python scalars become int32
        # constants. The float32 cast reads the int8 result inside the engine.
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "c", torch.tensor([100, -100, 120, 2], dtype=torch.int8)
                )

            def forward(self, x):
                return compute(x.to(torch.int8), self.c).to(torch.float32)

        model = Model().cuda().eval()
        inputs = (torch.tensor([100.0, -100.0, 127.0, 5.0]).cuda(),)
        trt_model = torch_tensorrt.dynamo.compile(
            torch.export.export(model, inputs), arg_inputs=inputs, min_block_size=1
        )
        torch.testing.assert_close(trt_model(*inputs), model(*inputs), rtol=0, atol=0)

    @parameterized.expand(
        [
            ("2d", (2, 4), lambda y: y.float()),
            ("3d", (2, 1, 4), lambda y: y.float()),
            ("int8_output", (2, 4), lambda y: y),
            ("sum", (2, 4), lambda y: y.sum(1).float()),
            ("bool", (2, 4), lambda y: y.bool().float()),
            ("reshape", (2, 4), lambda y: y.reshape(-1).float()),
            ("slice", (2, 4), lambda y: y[:, 1:3].float()),
            ("cat", (2, 4), lambda y: torch.cat([y, y]).float()),
            # The middle of the chain overflows int8, and only its end is wrapped.
            ("chain", (2, 4), lambda y: ((y * 3 + 100) * 5 - 7).float()),
        ]
    )
    def test_wrapped_int8_result_feeds_other_ops(self, _, shape, consume):
        # -128 * 2 wraps to 0 in int8, which the bool case reads.
        class Model(nn.Module):
            def forward(self, x):
                return consume(x.to(torch.int8) * 2)

        model = Model().cuda().eval()
        values = torch.tensor([100.0, -100.0, 127.0, 5.0, -128.0, 64.0, 0.0, -1.0])
        inputs = (values.reshape(shape).cuda(),)
        trt_model = torch_tensorrt.dynamo.compile(
            torch.export.export(model, inputs), arg_inputs=inputs, min_block_size=1
        )
        torch.testing.assert_close(trt_model(*inputs), model(*inputs), rtol=0, atol=0)

    @parameterized.expand(
        [
            (
                "two_int8_tensors",
                lambda x: (x.to(torch.int8) + x.flip(-1).to(torch.int8)).sum(),
            ),
            (
                "0d_int64_tensor",
                lambda x: x.to(torch.int8) + torch.tensor(300, device=x.device),
            ),
        ]
    )
    def test_int8_arithmetic_on_int8_tensors_wraps(self, _, compute):
        # TensorRT would run these in INT8, as both operands are int8 tensors there.
        class Model(nn.Module):
            def forward(self, x):
                return compute(x).float()

        model = Model().cuda().eval()
        values = torch.tensor([100.0, -100.0, 127.0, 5.0, -128.0, 64.0, 0.0, -1.0])
        inputs = (values.reshape(2, 4).cuda(),)
        trt_model = torch_tensorrt.dynamo.compile(
            torch.export.export(model, inputs), arg_inputs=inputs, min_block_size=1
        )
        torch.testing.assert_close(trt_model(*inputs), model(*inputs), rtol=0, atol=0)


if __name__ == "__main__":
    run_tests()
