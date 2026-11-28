# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase, skip_if_trt_rtx_turing


class TestMatMulConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (
                "1_1",
                (1,),
                (1,),
            ),
            (
                "1_1",
                (2,),
                (2,),
            ),
            (
                "1_1",
                (3,),
                (3,),
            ),
        ]
    )
    def test_matmul_dot(self, _, input_shape, other_shape):
        skip_if_trt_rtx_turing(self, "FP32 GEMM")

        class MatMul(nn.Module):
            def __init__(self):
                super().__init__()
                self.other = nn.Parameter(torch.randn(*other_shape))

            def forward(self, input):
                return torch.ops.aten.dot.default(input, self.other)

        inputs = [torch.randn(*input_shape)]

        self.run_test(
            MatMul(),
            inputs,
        )

    @parameterized.expand(
        [
            (
                "2_2",
                (2, 3),
                (3, 2),
            ),
            (
                "4_6",
                (4, 5),
                (5, 6),
            ),
            (
                "2_1",
                (2, 3),
                (3, 1),
            ),
            (
                "4_1",
                (4, 1),
                (1, 1),
            ),
            (
                "1_2",
                (1, 3),
                (3, 2),
            ),
            (
                "1_3",
                (1, 2),
                (2, 3),
            ),
            # Following cases use torch.ops.aten.bmm.default
            # ("4_3", (3,1,3,2), (2,2,3)),
            # ("3_4", (3,1,3,2), (2,2,3)),
            # ("3_4", (2, 2, 3), (3, 1, 3, 3)),
            # ("4_2", (1, 2, 2, 3), (3, 2)),
        ]
    )
    def test_matmul_mm(self, _, input_shape, other_shape):
        skip_if_trt_rtx_turing(self, "FP32 GEMM")

        class MatMul(nn.Module):
            def __init__(self):
                super().__init__()
                self.other = nn.Parameter(torch.randn(*other_shape))

            def forward(self, input):
                return torch.ops.aten.mm.default(input, self.other)

        inputs = [torch.randn(*input_shape)]

        self.run_test(
            MatMul(),
            inputs,
        )

    @parameterized.expand(
        [
            (
                "1_1",
                (1, 1),
                (1,),
            ),
            (
                "1_1",
                (1, 2),
                (2,),
            ),
            (
                "2_1",
                (2, 1),
                (1,),
            ),
            (
                "3_1",
                (3, 4),
                (4,),
            ),
        ]
    )
    def test_matmul_mv(self, _, input_shape, other_shape):
        skip_if_trt_rtx_turing(self, "FP32 GEMM")

        class MatMul(nn.Module):
            def __init__(self):
                super().__init__()
                self.other = nn.Parameter(torch.randn(*other_shape))

            def forward(self, input):
                return torch.ops.aten.mv.default(input, self.other)

        inputs = [torch.randn(*input_shape)]

        self.run_test(
            MatMul(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2_2", (2, 3), (3, 2)),
            ("2_3", (2, 3), (2, 3, 4)),
            ("4_4", (2, 2, 2, 3), (2, 1, 3, 2)),
            ("4_2", (2, 1, 2, 3), (3, 2)),
            ("2_1", (2, 3), (3,)),
            ("1_2", (3,), (3, 2)),
            ("1_1", (3,), (3,)),
        ]
    )
    def test_matmul_matmul(self, _, input_shape, other_shape):
        skip_if_trt_rtx_turing(self, "FP32 GEMM")

        class MatMul(nn.Module):
            def forward(self, input, other):
                return torch.ops.aten.matmul(input, other)

        inputs = [torch.randn(*input_shape), torch.randn(*other_shape)]

        self.run_test(
            MatMul(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2_2", (2, 3), (3, 3), (4, 3), (3, 4), (3, 5), (3, 7)),
            ("3_2", (3, 2, 3), (5, 3, 3), (7, 4, 3), (3, 4), (3, 5), (3, 7)),
            ("3_2", (3, 2, 3), (5, 3, 5), (7, 4, 7), (3, 4), (5, 6), (7, 8)),
            (
                "3_4",
                (2, 4, 5),
                (3, 6, 7),
                (4, 7, 9),
                (2, 2, 5, 3),
                (4, 3, 7, 3),
                (5, 4, 9, 3),
            ),
            ("2_1", (2, 3), (3, 4), (5, 7), (3,), (4,), (7,)),
            ("1_2", (3,), (5,), (7,), (3, 2), (5, 4), (7, 6)),
            ("1_1", (3,), (5,), (7,), (3,), (5,), (7,)),
        ]
    )
    def test_matmul_matmul_dynamic_shape(self, *args):
        skip_if_trt_rtx_turing(self, "FP32 GEMM")

        class MatMul(nn.Module):
            def forward(self, input, other):
                return torch.ops.aten.matmul(input, other)

        input_specs = [
            Input(
                min_shape=args[1],
                opt_shape=args[2],
                max_shape=args[3],
            ),
            Input(
                min_shape=args[4],
                opt_shape=args[5],
                max_shape=args[6],
            ),
        ]
        self.run_test_with_dynamic_shape(
            MatMul(),
            input_specs,
        )

    @parameterized.expand(
        [
            (
                "int32_prod",
                lambda x: (lambda y: y @ y.T)(x.prod(-1)),
                (
                    torch.tensor(
                        [[[1, 2], [3, 4]], [[5, 1], [2, 3]]], dtype=torch.int32
                    ),
                ),
            ),
            (
                "int32_past_2_11",
                lambda x, y: x @ y,
                (
                    torch.arange(4096, dtype=torch.int32).view(64, 64) % 2046 + 2049,
                    torch.eye(64, dtype=torch.int32),
                ),
            ),
            (
                "int32_wraps",
                lambda x, y: x @ y,
                (
                    torch.full((1, 3), 2**30, dtype=torch.int32),
                    torch.ones(3, 1, dtype=torch.int32),
                ),
            ),
            (
                "int64_past_2_24",
                torch.mm,
                (torch.tensor([[2**33, 3], [5, -7]]), torch.tensor([[1, 2], [3, 4]])),
            ),
            (
                "int32_bmm",
                torch.bmm,
                (
                    torch.arange(12, dtype=torch.int32).view(2, 2, 3) * 1001 + 4097,
                    torch.arange(12, dtype=torch.int32).view(2, 3, 2) + 4092,
                ),
            ),
            (
                "int32_mv",
                torch.mv,
                (
                    torch.tensor([[4097, 4097, 1], [3, 5, 7]], dtype=torch.int32),
                    torch.tensor([4097, 1, 3], dtype=torch.int32),
                ),
            ),
            (
                "int32_vector_times_matrix",
                lambda x, y: x @ y,
                (
                    torch.tensor([3, 2049, -5], dtype=torch.int32),
                    torch.arange(6, dtype=torch.int32).view(3, 2) * 4097,
                ),
            ),
            (
                "int64_dot",
                torch.dot,
                (torch.tensor([2**31, 3, 5]), torch.tensor([2, -1, 4097])),
            ),
            (
                "argmax_times_int64_sum",
                lambda s, v: s.argmax(1) @ v.sum(1),
                (
                    torch.tensor([[5.0, 1.0], [0.0, 9.0]]),
                    torch.tensor([[0] * 4, [2**30] * 4], dtype=torch.int32),
                ),
            ),
            (
                "argmax_times_argmax",
                lambda x: (lambda y: (y @ y) * 300)(x.argmax(-1)),
                (torch.eye(2, 2001).flip(-1),),
            ),
        ]
    )
    def test_matmul_integer(self, _, matmul, inputs):
        # CUDA has no integer matmul, so the graph is exported and checked on the CPU.
        # TF32 rounds values past 2**11 and float32 rounds past 2**24, an int32 result
        # wraps, and an argmax is int64 although the engine holds it in int32.
        class MatMul(nn.Module):
            def forward(self, *args):
                return matmul(*args)

        model = MatMul().eval()
        compiled = torch_tensorrt.dynamo.compile(
            torch.export.export(model, inputs),
            arg_inputs=[x.cuda() for x in inputs],
            min_block_size=1,
            require_full_compilation=True,
        )
        torch.testing.assert_close(
            compiled(*[x.cuda() for x in inputs]).cpu(), model(*inputs), rtol=0, atol=0
        )


if __name__ == "__main__":
    run_tests()
