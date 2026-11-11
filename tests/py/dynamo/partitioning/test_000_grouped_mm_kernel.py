# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import torch
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests


class Experts(torch.nn.Module):
    """Two grouped matmuls around an activation, as in a mixture-of-experts layer."""

    def __init__(self, experts, k, n, dtype):
        super().__init__()
        self.up = torch.nn.Parameter(torch.randn(experts, k, n, dtype=dtype) * 0.05)
        self.down = torch.nn.Parameter(torch.randn(experts, n, k, dtype=dtype) * 0.05)

    def forward(self, x, offs):
        h = torch.ops.aten._grouped_mm.default(torch.relu(x), self.up, offs)
        return torch.ops.aten._grouped_mm.default(torch.relu(h), self.down, offs)


def _offsets(counts):
    return torch.cumsum(torch.tensor(counts), 0).to(torch.int32).cuda()


@unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    "The grouped GEMM kernel needs a GPU with compute capability 8.0 or newer",
)
@unittest.skipIf(
    not torch_tensorrt.ENABLED_FEATURES.qdp_plugin,
    "The grouped GEMM kernel is a TensorRT plugin",
)
class TestGroupedMMKernel(TestCase):
    @parameterized.expand([("bfloat16", torch.bfloat16), ("float16", torch.float16)])
    def test_dynamic_rows_stay_in_one_engine(self, _, dtype):
        experts, k, n = 16, 128, 256
        model = Experts(experts, k, n, dtype).eval().cuda()
        offs = _offsets([50] * 14 + [0, 0])
        x = torch.randn(700, k, dtype=dtype).cuda()
        rows = torch.export.Dim("rows", min=1, max=4096)
        ep = torch.export.export(model, (x, offs), dynamic_shapes=({0: rows}, None))
        compiled = torch_tensorrt.dynamo.compile(
            ep,
            arg_inputs=[
                torch_tensorrt.Input(
                    min_shape=(1, k),
                    opt_shape=(700, k),
                    max_shape=(4096, k),
                    dtype=dtype,
                ),
                torch_tensorrt.Input(shape=(experts,), dtype=torch.int32),
            ],
            min_block_size=1,
        )
        submodules = [name for name, _ in compiled.named_children()]
        self.assertEqual(submodules, ["_run_on_acc_0"])

        # Other row counts and routings, with empty groups first, last and in between.
        for counts in (
            [0, 3, 0, 1000, 0, 1, 129, 0, 2, 0, 0, 7, 0, 0, 900, 0],
            [1] + [0] * 15,
        ):
            offs = _offsets(counts)
            x = torch.randn(int(offs[-1]), k, dtype=dtype).cuda()
            with torch.no_grad():
                torch.testing.assert_close(
                    compiled(x, offs), model(x, offs), rtol=1e-2, atol=1e-2
                )

    def test_rows_past_the_last_offset_are_zero(self):
        # No group covers these rows, so PyTorch leaves them undefined. The kernel
        # zeroes them, as the dense lowering does, instead of leaving stale memory.
        experts, k, n = 4, 64, 128
        model = Experts(experts, k, n, torch.bfloat16).eval().cuda()
        offs = _offsets([10, 0, 118, 0])
        x = torch.randn(300, k, dtype=torch.bfloat16).cuda()
        rows = torch.export.Dim("rows", min=1, max=1024)
        ep = torch.export.export(model, (x, offs), dynamic_shapes=({0: rows}, None))
        compiled = torch_tensorrt.dynamo.compile(
            ep,
            arg_inputs=[
                torch_tensorrt.Input(
                    min_shape=(1, k),
                    opt_shape=(300, k),
                    max_shape=(1024, k),
                    dtype=torch.bfloat16,
                ),
                torch_tensorrt.Input(shape=(experts,), dtype=torch.int32),
            ],
            min_block_size=1,
        )
        for total in (300, 1000):
            x = torch.randn(total, k, dtype=torch.bfloat16).cuda()
            # Freed NaNs, so an unwritten output row would most likely show them.
            torch.full((total, k), float("nan"), dtype=torch.bfloat16, device="cuda")
            with torch.no_grad():
                out = compiled(x, offs)
                torch.testing.assert_close(
                    out[:128], model(x, offs)[:128], rtol=1e-2, atol=1e-2
                )
            self.assertEqual(torch.count_nonzero(out[128:]).item(), 0)

    def test_offsets_computed_in_pytorch(self):
        # The validator runs on the whole graph and again on the TensorRT submodule,
        # where the offsets arrive as an engine input. Both must keep the kernel.
        class FromCounts(Experts):
            def forward(self, x, counts):
                return super().forward(x, torch.cumsum(counts, 0).to(torch.int32))

        model = FromCounts(8, 64, 128, torch.bfloat16).eval().cuda()
        counts = torch.tensor([30, 0, 200, 1, 0, 0, 69, 0]).cuda()
        x = torch.randn(300, 64, dtype=torch.bfloat16).cuda()
        compiled = torch_tensorrt.compile(
            model,
            ir="dynamo",
            inputs=[x, counts],
            min_block_size=1,
            torch_executed_ops={"torch.ops.aten.cumsum.default"},
        )
        in_pytorch = [
            node.target
            for name, sub in compiled.named_children()
            if name.startswith("_run_on_gpu")
            for node in sub.graph.nodes
            if node.op == "call_function"
        ]
        self.assertNotIn(torch.ops.aten._grouped_mm.default, in_pytorch)
        self.assertIn(torch.ops.aten.cumsum.default, in_pytorch)
        with torch.no_grad():
            torch.testing.assert_close(
                compiled(x, counts), model(x, counts), rtol=1e-2, atol=1e-2
            )

    def test_require_full_compilation(self):
        model = Experts(4, 64, 64, torch.float16).eval().cuda()
        offs = _offsets([10, 0, 90, 28])
        x = torch.randn(128, 64, dtype=torch.float16).cuda()
        compiled = torch_tensorrt.compile(
            model,
            ir="dynamo",
            inputs=[x, offs],
            min_block_size=1,
            require_full_compilation=True,
        )
        with torch.no_grad():
            torch.testing.assert_close(
                compiled(x, offs), model(x, offs), rtol=1e-2, atol=1e-2
            )


if __name__ == "__main__":
    run_tests()
