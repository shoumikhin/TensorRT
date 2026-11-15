# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import operator
import unittest

import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
    sort_stable_validator,
    sort_validator,
)
from torch_tensorrt.dynamo.lowering import get_decompositions, post_lowering
from torch_tensorrt.dynamo.lowering.passes.annotate_sort_num_keys import (
    _num_index_values,
)

from .harness import DispatchTestCase


class TestSortConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((3, 2, 4), 0, True),
            ((2, 3, 4, 5), 1, True),
            ((2, 3, 4, 5), 2, False),
            ((6, 7, 5, 4, 5), 4, False),
            ((1, 5, 2, 1), -1, True),
            ((1, 2, 5, 3), -2, False),
            ((6, 2, 1, 3), -4, True),
            # ITopKLayer requires rank >= 2; converter broadcasts rank-1
            ((64,), 0, False),
            ((64,), 0, True),
            ((16,), -1, True),
        ]
    )
    def test_sort(self, input_shape, dim, descending):
        class Sort(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sort.default(x, dim, descending)

        inputs = [torch.randn(*input_shape)]
        self.run_test(
            Sort(),
            inputs,
            enable_passes=True,
            use_dynamo_tracer=True,
        )


class TestSortConverterDynamic(DispatchTestCase):
    @parameterized.expand(
        [
            (
                "3d_dynamic_descending",
                (2, 1, 4),
                (3, 2, 4),
                (3, 3, 4),
                2,
                True,
            ),
            (
                "4d_dynamic_ascending",
                (2, 2, 1, 4),
                (2, 2, 2, 4),
                (3, 3, 2, 4),
                3,
                False,
            ),
            (
                "4d_dynamic_descending_neg_dim",
                (1, 3, 1, 1),
                (2, 3, 2, 2),
                (3, 3, 2, 4),
                -3,
                True,
            ),
        ]
    )
    def test_sort_dynamic(self, _, min_shape, opt_shape, max_shape, dim, descending):
        class Sort(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sort.default(x, dim, descending)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=torch.float,
            ),
        ]
        self.run_test_with_dynamic_shape(
            Sort(),
            input_specs,
            output_dtypes=[torch.float, torch.int64],
            use_dynamo_tracer=True,
        )


@unittest.skipIf(not torch.cuda.is_available(), "Skip because CUDA is not available")
class TestSortValidatorDefaultDim(TestCase):
    """Regression test for https://github.com/pytorch/TensorRT/issues/3777.

    sort_validator is only consulted by the partitioner, which
    DispatchTestCase bypasses (it interprets the traced graph directly), so
    the cases above don't exercise it. torch.sort(x) with no explicit dim
    omits dim from the FX node's args entirely (relying on the schema
    default of -1), so this goes through the full compile() pipeline, where
    sort_validator used to crash with `IndexError: tuple index out of
    range` from indexing node.args[1] unconditionally.
    """

    def test_sort_default_dim_compiles(self):
        class Sort(nn.Module):
            def forward(self, x):
                values, _ = torch.sort(x)
                return values

        mod = Sort().eval().cuda()
        inputs = [torch.randn(4, 8).cuda()]
        trt_mod = torch_tensorrt.compile(
            mod, ir="dynamo", inputs=inputs, min_block_size=1
        )
        acc_count = sum(
            1 for name, _ in trt_mod.named_children() if "_run_on_acc" in name
        )
        self.assertEqual(acc_count, 1)
        torch.testing.assert_close(trt_mod(*inputs), mod(*inputs))

    def test_sort_scalar_input_falls_back(self):
        """A 0-D (scalar) input has no dim to normalize: get_positive_dim
        computes dim % rank, and rank=0 for a scalar raises ZeroDivisionError.
        sort_validator must reject scalar inputs before reaching that call so
        the node falls back to PyTorch instead of crashing the partitioner."""

        class Sort(nn.Module):
            def forward(self, x):
                values, _ = torch.sort(x)
                return values

        mod = Sort().eval().cuda()
        inputs = [torch.tensor(5.0).cuda()]
        trt_mod = torch_tensorrt.compile(
            mod, ir="dynamo", inputs=inputs, min_block_size=1
        )
        acc_count = sum(
            1 for name, _ in trt_mod.named_children() if "_run_on_acc" in name
        )
        self.assertEqual(acc_count, 0)
        torch.testing.assert_close(trt_mod(*inputs), mod(*inputs))


class TestSortValidator(unittest.TestCase):
    def _sort_node(self, fn):
        class Mod(nn.Module):
            def forward(self, x):
                return fn(x)

        gm = torch.export.export(Mod(), (torch.randn(4, 8),)).module()
        return next(
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.sort.default
        )

    def test_defaulted_dim_does_not_raise(self):
        # export drops dim=-1, so args is (x,) only; indexing args[1] used to IndexError
        node = self._sort_node(lambda x: torch.sort(x))
        self.assertEqual(len(node.args), 1)
        self.assertTrue(sort_validator(node))

    def test_explicit_dim_still_works(self):
        node = self._sort_node(lambda x: torch.sort(x, 0))
        self.assertGreaterEqual(len(node.args), 2)
        self.assertTrue(sort_validator(node))


class TestSortStableConverter(DispatchTestCase):
    """Stable sort of the indices that topk returns, as mixture-of-experts routing does."""

    @parameterized.expand(
        [
            # 8000 keys, above the 3840 elements a TopK layer can sort
            ("flat", (1000, 16), 8, None, False, False),
            ("flat_descending", (1000, 16), 8, None, True, False),
            ("flat_int32", (700, 128), 8, None, False, True),
            ("whole_tiles", (512, 16), 8, None, False, False),
            ("not_whole_tiles", (37, 5), 3, None, False, False),
            ("column", (600, 32), 1, 0, False, False),
            ("column_descending", (600, 32), 1, 0, True, False),
            ("row", (1, 120), 100, 1, True, False),
            ("first_of_three", (300, 1, 32), 1, 0, False, False),
            ("second_of_four", (1, 300, 1, 16), 1, 1, True, False),
        ]
    )
    def test_sort_stable(self, _, shape, k, dim, descending, int32):
        class Sort(nn.Module):
            def forward(self, x):
                ids = torch.topk(x, k, dim=-1)[1]
                if dim is None:
                    ids = ids.reshape(-1)
                if int32:
                    ids = ids.to(torch.int32)
                return torch.ops.aten.sort.stable(
                    ids,
                    stable=True,
                    dim=-1 if dim is None else dim,
                    descending=descending,
                )

        self.run_test(
            Sort(),
            [torch.randn(shape)],
            rtol=0,
            atol=0,
            enable_passes=True,
            use_dynamo_tracer=True,
        )

    def test_sort_stable_dynamic(self):
        class Sort(nn.Module):
            def forward(self, x):
                ids = torch.topk(x, 8, dim=-1)[1].reshape(-1)
                return torch.ops.aten.sort.stable(ids, stable=True)

        input_specs = [
            Input(
                min_shape=(1, 128),
                opt_shape=(500, 128),
                max_shape=(1000, 128),
                dtype=torch.float,
            ),
        ]
        self.run_test_with_dynamic_shape(
            Sort(),
            input_specs,
            rtol=0,
            atol=0,
            enable_passes=True,
            use_dynamo_tracer=True,
        )


_CONVERTERS_LOGGER = "torch_tensorrt.dynamo.conversion.aten_ops_converters"


class TestSortStableValidator(unittest.TestCase):
    def _sort_node(self, fn, x, dynamic_shapes=None):
        class Mod(nn.Module):
            def forward(self, x):
                return fn(x)

        # Partitioning sees the graph after decompositions and lowering passes.
        ep = torch.export.export(Mod(), (x,), dynamic_shapes=dynamic_shapes)
        gm = post_lowering(
            ep.run_decompositions(get_decompositions(False)).module(),
            CompilationSettings(),
        )
        return next(
            n
            for n in reversed(gm.graph.nodes)
            if n.op == "call_function" and n.target is torch.ops.aten.sort.stable
        )

    @parameterized.expand(
        [
            ("topk", lambda x: torch.topk(x, 2)[1].reshape(-1), 8),
            ("argmax_int32", lambda x: x.argmax(-1).to(torch.int32), 8),
            ("argmin", lambda x: x.argmin(0), 4),
            ("argmax_flat", lambda x: x.argmax().reshape(1), 32),
            ("sort", lambda x: torch.sort(x)[1].reshape(-1), 8),
            ("sort_stable", lambda x: torch.sort(x, stable=True)[1].reshape(-1), 8),
        ]
    )
    def test_known_range(self, _, keys, num_keys):
        node = self._sort_node(
            lambda x: torch.sort(keys(x), stable=True), torch.randn(4, 8)
        )
        self.assertEqual(node.meta["_num_sort_keys"], num_keys)
        self.assertTrue(sort_stable_validator(node))

    def test_unknown_range(self):
        node = self._sort_node(
            lambda x: torch.sort(x, stable=True), torch.randint(0, 8, (16,))
        )
        with self.assertLogs(_CONVERTERS_LOGGER, level="WARNING") as logs:
            self.assertFalse(sort_stable_validator(node))
        self.assertIn("Cannot tell the range", logs.output[0])

    def test_dynamic_key_range(self):
        node = self._sort_node(
            lambda x: torch.sort(torch.topk(x, 2)[1].reshape(-1), stable=True),
            torch.randn(4, 8),
            dynamic_shapes={"x": {1: torch.export.Dim("keys", min=2, max=64)}},
        )
        self.assertNotIn("_num_sort_keys", node.meta)
        self.assertFalse(sort_stable_validator(node))

    def test_float_keys(self):
        node = self._sort_node(
            lambda x: torch.sort(torch.topk(x, 2)[1].float(), stable=True),
            torch.randn(4, 8),
        )
        with self.assertLogs(_CONVERTERS_LOGGER, level="DEBUG") as logs:
            self.assertFalse(sort_stable_validator(node))
        self.assertIn("int32 or int64", logs.output[0])

    def test_unbounded_length(self):
        node = self._sort_node(
            lambda x: torch.sort(torch.topk(x, 2)[1].reshape(-1), stable=True),
            torch.randn(4, 8),
            dynamic_shapes={"x": {0: torch.export.Dim("tokens", min=2)}},
        )
        with self.assertLogs(_CONVERTERS_LOGGER, level="DEBUG") as logs:
            self.assertFalse(sort_stable_validator(node))
        self.assertIn("no upper bound", logs.output[0])

    def test_topk_dim_keyword(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.empty(4, 8)
        topk = graph.call_function(torch.ops.aten.topk.default, (x, 2), {"dim": 0})
        ids = graph.call_function(operator.getitem, (topk, 1))
        self.assertEqual(_num_index_values(ids), 4)

    def test_key_range_in_engine_cache_key(self):
        node = self._sort_node(
            lambda x: torch.sort(torch.topk(x, 2)[1].reshape(-1), stable=True),
            torch.randn(4, 8),
        )
        gm, inputs = node.graph.owning_module, [Input(shape=(4, 8))]
        before = BaseEngineCache.get_hash(gm, inputs, CompilationSettings())
        node.meta["_num_sort_keys"] = 16
        after = BaseEngineCache.get_hash(gm, inputs, CompilationSettings())
        self.assertNotEqual(before, after)

    def test_many_rows(self):
        node = self._sort_node(
            lambda x: torch.sort(torch.topk(x, 8)[1], stable=True),
            torch.randn(4, 300),
        )
        with self.assertLogs(_CONVERTERS_LOGGER, level="DEBUG") as logs:
            self.assertFalse(sort_stable_validator(node))
        self.assertIn("more than one row", logs.output[0])

    def test_data_dependent_rows(self):
        node = self._sort_node(
            lambda x: torch.sort(torch.topk(x[x[:, 0] > 0], 2)[1], stable=True),
            torch.randn(4, 8),
        )
        self.assertFalse(sort_stable_validator(node))

    def test_too_many_counts(self):
        # 27,784 keys with 1024 values each.
        node = self._sort_node(
            lambda x: torch.sort(torch.topk(x, 8)[1].reshape(-1), stable=True),
            torch.randn(3473, 1024),
        )
        with self.assertLogs(_CONVERTERS_LOGGER, level="DEBUG") as logs:
            self.assertFalse(sort_stable_validator(node))
        self.assertIn("counts", logs.output[0])


@unittest.skipIf(not torch.cuda.is_available(), "Skip because CUDA is not available")
class TestSortStableCompile(TestCase):
    def _compile(self, mod, inputs, **kwargs):
        trt_mod = torch_tensorrt.compile(
            mod, ir="dynamo", inputs=inputs, min_block_size=1, **kwargs
        )
        in_pytorch = [
            str(node.target)
            for name, sub in trt_mod.named_children()
            if "_run_on_gpu" in name
            for node in sub.graph.nodes
            if node.op == "call_function"
        ]
        return trt_mod, in_pytorch

    def test_expert_routing_stays_in_one_engine(self):
        class Routing(nn.Module):
            def forward(self, scores):
                experts = torch.topk(scores, 8, dim=-1)[1].reshape(-1)
                return experts.argsort(stable=True)

        mod = Routing().eval().cuda()
        inputs = [torch.randn(1000, 128).cuda()]
        trt_mod, in_pytorch = self._compile(mod, inputs)
        self.assertEqual(in_pytorch, [])
        torch.testing.assert_close(trt_mod(*inputs), mod(*inputs), rtol=0, atol=0)

    def test_keys_made_in_pytorch(self):
        # The sort lands in an engine whose input is the placeholder for the topk result.
        class Routing(nn.Module):
            def forward(self, scores):
                experts = torch.topk(scores, 8, dim=-1)[1].reshape(-1)
                return experts.argsort(stable=True) + 1

        mod = Routing().eval().cuda()
        inputs = [torch.randn(1000, 128).cuda()]
        trt_mod, in_pytorch = self._compile(
            mod, inputs, torch_executed_ops={"torch.ops.aten.topk.default"}
        )
        self.assertNotIn("aten.sort.stable", " ".join(in_pytorch))
        torch.testing.assert_close(trt_mod(*inputs), mod(*inputs), rtol=0, atol=0)

    def test_rank_of_rank(self):
        # The first sort has float keys and runs in PyTorch; the second sorts its indices.
        class Rank(nn.Module):
            def forward(self, x):
                return x.argsort(stable=True).argsort(stable=True) + 1

        mod = Rank().eval().cuda()
        inputs = [torch.randn(100).cuda()]
        trt_mod, in_pytorch = self._compile(mod, inputs)
        self.assertEqual(" ".join(in_pytorch).count("aten.sort.stable"), 1)
        torch.testing.assert_close(trt_mod(*inputs), mod(*inputs), rtol=0, atol=0)

    def test_dynamic_length_at_every_size(self):
        class Routing(nn.Module):
            def forward(self, scores):
                experts = torch.topk(scores, 8, dim=-1)[1].reshape(-1)
                return experts.argsort(stable=True)

        mod = Routing().eval().cuda()
        inputs = [
            torch_tensorrt.Input(
                min_shape=(1, 128), opt_shape=(500, 128), max_shape=(1000, 128)
            )
        ]
        trt_mod, in_pytorch = self._compile(mod, inputs)
        self.assertEqual(in_pytorch, [])
        # One partial tile, whole tiles (4096 keys) and the largest size.
        for tokens in (1, 512, 1000):
            scores = torch.randn(tokens, 128).cuda()
            torch.testing.assert_close(trt_mod(scores), mod(scores), rtol=0, atol=0)


if __name__ == "__main__":
    run_tests()
