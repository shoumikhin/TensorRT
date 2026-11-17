# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# type: ignore
"""End-to-end tests for ``aten.index_copy`` KV-cache aliasing.

Two converters are registered for ``aten.index_copy.default``:

* ``aten_ops_index_copy_kv`` — HIGH priority, validator-gated. Fires for
  the narrow KV-eligible case (4-D static cache, dim=2, batch=1, source seq
  dim size 1, or a module buffer written at ``start + arange(n)``) and emits
  ``IKVCacheUpdateLayer`` with aliased I/O.

* ``aten_ops_index_copy_fallback`` — STANDARD priority. Fires for
  everything else; produces correct results via the scatter path. No
  graph break.

These tests verify both paths end-to-end via the C++ runtime: the
fast path mutates in place, the fallback produces correct numerical
results without aliasing.
"""

import contextlib
import copy
import os
import tempfile
from unittest import mock

import torch
import torch_tensorrt
from torch.export import Dim, export
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo import refit_module_weights


def _compile(model, args, **kwargs):
    ep = export(model, tuple(args))
    return torch_tensorrt.compile(
        ep,
        ir="dynamo",
        inputs=list(args),
        enabled_precisions={torch.float32},
        min_block_size=1,
        use_python_runtime=False,
        **kwargs,
    )


def _aliased_io(compiled):
    for _name, mod in compiled.named_modules():
        if hasattr(mod, "aliased_io") and mod.aliased_io:
            return dict(mod.aliased_io)
    return {}


class TestIndexCopyKVFastPath(TestCase):
    """KV-eligible: 4-D static cache, dim=2, batch=1, single-position
    write. The validator passes and the fast path emits
    ``IKVCacheUpdateLayer`` with aliased output."""

    def test_a_buffer_written_at_one_position_is_updated_in_place(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(1, 4, 16, 8, device="cuda"))

            def forward(self, update, index):
                self.cache.index_copy_(2, index, update)
                return self.cache.sum(2)

        def step(i):
            g = torch.Generator().manual_seed(i)
            update = torch.randn(1, 4, 1, 8, generator=g).cuda()
            return update, torch.tensor([3 + i], device="cuda")

        eager = M()
        compiled = _compile(M(), step(0))
        self.assertEqual(
            list(_aliased_io(compiled).values()), [("buf_cache", "kv_cache_update")]
        )
        self.assertFalse(compiled.meta.get("_copyback_mutation_buffers"))
        for i in range(3):
            out = compiled(*step(i))
            out = out[0] if isinstance(out, (tuple, list)) else out
            torch.testing.assert_close(out, eager(*step(i)))
            torch.testing.assert_close(compiled.cache, eager.cache)

    def test_functional_write_of_an_input_leaves_it_alone(self):
        """The caller's cache is an input the model does not write, so the engine
        must not update it in place."""

        class M(torch.nn.Module):
            def forward(self, cache, index, update):
                return torch.ops.aten.index_copy.default(cache, 2, index, update)

        cache = torch.zeros(1, 4, 16, 8, device="cuda")
        index = torch.tensor([3], dtype=torch.int64, device="cuda")
        update = torch.ones(1, 4, 1, 8, device="cuda") * 7.0

        compiled = _compile(M().cuda(), (cache.clone(), index, update.clone()))
        self.assertEqual(_aliased_io(compiled), {})

        cache_run = cache.clone()
        out = compiled(cache_run, index, update)
        out_val = out[0] if isinstance(out, tuple) else out
        eager_out = torch.ops.aten.index_copy.default(cache, 2, index, update)
        self.assertTrue(torch.allclose(out_val, eager_out))
        self.assertTrue(torch.equal(cache_run, cache))


class _ChunkCache(torch.nn.Module):
    """A module-held cache written at ``input_pos + arange(T)``, as a prefill chunk
    or a decode step writes it, then read back."""

    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(1, 4, 16, 8))

    def forward(self, x, input_pos):
        idx = input_pos + torch.arange(x.shape[2], device=x.device)
        self.cache.index_copy_(2, idx, x)
        return self.cache.sum(2)


class _DoubledChunkCache(_ChunkCache):
    """Writes every token twice, so the number of positions is computed: ``2 * T``."""

    def forward(self, x, input_pos):
        return super().forward(torch.cat((x, x), dim=2), input_pos)


def _engines(compiled):
    return [n.target for n in compiled.graph.nodes if n.op == "call_module"]


def _check_calls(compiled, eager, calls):
    """Run each argument tuple through both modules in turn and compare every result
    and the whole cache, so a write that lands in the wrong place or never reaches
    the buffer shows on the call that reads it back."""
    for args in calls:
        want = eager(*args)
        got = compiled(*args)
        want = want if isinstance(want, tuple) else (want,)
        got = tuple(got) if isinstance(got, (tuple, list)) else (got,)
        assert len(got) == len(want), (len(got), len(want))
        for g, w in zip(got, want):
            torch.testing.assert_close(g, w)
        torch.testing.assert_close(compiled.cache, eager.cache)


def _chunk(t, pos):
    return (torch.randn(1, 4, t, 8, device="cuda"), torch.tensor([pos], device="cuda"))


_POSITIONS_IN_PYTORCH = {
    "torch.ops.aten.arange.default",
    "torch.ops.aten.arange.start",
    "torch.ops.aten.arange.start_step",
    "torch.ops.aten.add.Tensor",
    "torch.ops.aten.sym_size.int",
}


@torch.library.custom_op("torchtrt_kv_test::positive_rows", mutates_args=())
def _positive_rows(x: torch.Tensor) -> torch.Tensor:
    return x[x[:, 0] > 0].clone()


@_positive_rows.register_fake
def _positive_rows_fake(x: torch.Tensor) -> torch.Tensor:
    rows = torch.library.get_ctx().new_dynamic_size(min=0, max=x.shape[0])
    return x.new_empty(rows, x.shape[1])


class TestIndexCopyKVBufferWrites(TestCase):
    """A module buffer written at consecutive positions is updated in place, for any
    number of tokens, so one engine serves both prefill and decode."""

    def test_consecutive_positions_write_in_place(self):
        model = _ChunkCache().cuda()
        eager = copy.deepcopy(model)
        compiled = _compile(model, _chunk(4, 0))

        aliased = _aliased_io(compiled)
        self.assertEqual(len(aliased), 1)
        _, kind = next(iter(aliased.values()))
        self.assertEqual(kind, "kv_cache_update")
        self.assertEqual(compiled.meta.get("_copyback_mutation_buffers", []), [])
        # Overlapping positions, so each call reads back what the one before wrote.
        _check_calls(compiled, eager, [_chunk(4, p) for p in (0, 2, 9, 12)])

    def test_small_graph_writes_in_place_at_the_default_min_block_size(self):
        """The write's subgraph is smaller than the default ``min_block_size``, but
        its ``copy_`` is gone, so the partitioner keeps it in an engine."""
        model = _ChunkCache().cuda()
        eager = copy.deepcopy(model)
        ep = export(model, _chunk(4, 0))
        compiled = torch_tensorrt.compile(
            ep, ir="dynamo", inputs=list(_chunk(4, 0)), use_python_runtime=False
        )
        self.assertEqual(len(_aliased_io(compiled)), 1)
        _check_calls(compiled, eager, [_chunk(4, p) for p in (0, 2, 9)])

    def test_reader_the_write_depends_on_is_copied_back(self):
        """A position or a source computed from the cache reads the old cache. The
        engine can run that read after the write, so the write is copied back and
        every call matches eager PyTorch."""

        class FromCache(_ChunkCache):
            def __init__(self, source):
                super().__init__()
                self.source = source

            def forward(self, x, input_pos):
                if self.source == "position":
                    start = (self.cache[0, 0, :, 0] != 0).sum().reshape(1)
                    return super().forward(x, start)
                return super().forward(x + self.cache.mean(2, keepdim=True), input_pos)

        for source in ("position", "source"):
            for t in (1, 4):
                with self.subTest(source=source, tokens=t):
                    model = FromCache(source).cuda()
                    eager = copy.deepcopy(model)
                    compiled = _compile(model, _chunk(t, 0))
                    self.assertEqual(_aliased_io(compiled), {})
                    _check_calls(
                        compiled, eager, [_chunk(t, p) for p in (0, t, 1, 3 * t)]
                    )

    def test_one_position_write_with_a_reader_of_its_slot_is_copied_back(self):
        """Another op reads the slot the write fills, and nothing orders it first,
        so even a one-position write is copied back."""

        class M(_ChunkCache):
            def forward(self, x, input_pos):
                before = self.cache[:, :, 2:3].sum(2)
                return before, super().forward(x, input_pos)

        model = M().cuda()
        eager = copy.deepcopy(model)
        compiled = _compile(model, _chunk(1, 2))
        self.assertEqual(_aliased_io(compiled), {})
        _check_calls(compiled, eager, [_chunk(1, p) for p in (2, 9, 2)])

    def test_dynamic_length_write_in_place_in_one_engine(self):
        model = _ChunkCache().cuda()
        eager = copy.deepcopy(model)
        x = torch.randn(1, 4, 4, 8, device="cuda")
        input_pos = torch.tensor([0], device="cuda")
        ep = export(
            model, (x, input_pos), dynamic_shapes=({2: Dim("T", min=1, max=8)}, None)
        )
        compiled = torch_tensorrt.compile(
            ep,
            ir="dynamo",
            inputs=[
                torch_tensorrt.Input(
                    min_shape=(1, 4, 1, 8),
                    opt_shape=(1, 4, 4, 8),
                    max_shape=(1, 4, 8, 8),
                ),
                torch_tensorrt.Input(shape=(1,), dtype=torch.int64),
            ],
            min_block_size=1,
            use_python_runtime=False,
        )
        self.assertEqual(len(_engines(compiled)), 1, _engines(compiled))
        self.assertEqual(len(_aliased_io(compiled)), 1)

        _check_calls(
            compiled, eager, [_chunk(t, p) for t, p in ((8, 0), (1, 8), (3, 9), (4, 7))]
        )

    def _compile_chunk(self, model, x, dim, **kwargs):
        ep = export(
            model,
            (x, torch.tensor([0], device="cuda")),
            dynamic_shapes=({2: dim}, None),
        )
        return torch_tensorrt.dynamo.compile(
            ep, inputs=[x, torch.tensor([0], device="cuda")], min_block_size=1, **kwargs
        )

    def test_a_length_with_the_default_minimum_writes_in_place(self):
        """``Dim`` starts at 0 by default. The input enters the engine anyway, for the
        size the positions are computed from, so the write stays in it, in place."""
        model = _ChunkCache().cuda()
        eager = copy.deepcopy(model)
        compiled = self._compile_chunk(model, _chunk(3, 0)[0], Dim("T", max=8))
        self.assertEqual(len(_engines(compiled)), 1, _engines(compiled))
        self.assertEqual(len(_aliased_io(compiled)), 1)
        _check_calls(
            compiled, eager, [_chunk(t, p) for t, p in ((3, 0), (1, 3), (8, 4))]
        )

    def test_a_length_computed_in_the_engine_writes_in_place(self):
        """The engine computes ``2 * T`` itself, so its profile never has to hold it."""
        model = _DoubledChunkCache().cuda()
        eager = copy.deepcopy(model)
        compiled = self._compile_chunk(model, _chunk(2, 0)[0], Dim("T", min=1, max=4))
        self.assertEqual(len(_engines(compiled)), 1, _engines(compiled))
        self.assertEqual(len(_aliased_io(compiled)), 1)
        _check_calls(
            compiled, eager, [_chunk(t, p) for t, p in ((2, 0), (1, 4), (4, 6))]
        )

    def test_a_length_computed_in_pytorch_writes_in_place(self):
        """``2 * T`` comes into the engine from PyTorch. Its profile starts at 2, where
        ``T`` does, so a call with ``T = 1`` runs and the write stays in place."""
        model = _DoubledChunkCache().cuda()
        eager = copy.deepcopy(model)
        compiled = self._compile_chunk(
            model,
            _chunk(2, 0)[0],
            Dim("T", min=1, max=4),
            torch_executed_ops={"torch.ops.aten.cat.default"},
        )
        self.assertEqual(len(_aliased_io(compiled)), 1)
        _check_calls(
            compiled, eager, [_chunk(t, p) for t, p in ((2, 0), (1, 4), (4, 6))]
        )

    def test_a_length_exported_as_t_plus_1_takes_its_smallest_value(self):
        """Export records the chunk's size as ``T + 1`` with its own range ``[1, 9]``,
        from ``T``'s ``[0, 8]``, while the shape system starts ``T`` itself at 2."""
        model = _ChunkCache().cuda()
        eager = copy.deepcopy(model)
        T = Dim("T", min=0, max=8)
        ep = export(model, _chunk(5, 0), dynamic_shapes=({2: T + 1}, None))
        compiled = torch_tensorrt.dynamo.compile(
            ep, inputs=list(_chunk(5, 0)), min_block_size=1
        )
        _check_calls(
            compiled, eager, [_chunk(t, p) for t, p in ((5, 0), (1, 5), (9, 6))]
        )

    def test_a_length_that_can_be_zero_stays_in_pytorch(self):
        """The KV layer writes one position or more, and an engine profile for a
        ``Dim`` starts at 1. Here the sizes and positions are computed in PyTorch, so
        only the write would bring the source into an engine. It stays in PyTorch and
        is copied back, and a call that writes nothing works. When an engine reads the
        source for anything else, it fails on an empty input whatever the write does."""
        model = _ChunkCache().cuda()
        eager = copy.deepcopy(model)
        x, input_pos = _chunk(3, 0)
        ep = export(
            model, (x, input_pos), dynamic_shapes=({2: Dim("T", min=0, max=8)}, None)
        )
        compiled = torch_tensorrt.dynamo.compile(
            ep,
            inputs=[x, input_pos],
            min_block_size=1,
            torch_executed_ops={
                "torch.ops.aten.arange.default",
                "torch.ops.aten.arange.start",
                "torch.ops.aten.arange.start_step",
                "torch.ops.aten.add.Tensor",
                "torch.ops.aten.sym_size.int",
            },
        )
        self.assertEqual(_aliased_io(compiled), {})
        _check_calls(
            compiled, eager, [_chunk(t, p) for t, p in ((3, 0), (0, 3), (1, 3), (4, 5))]
        )

    def test_a_length_an_engine_takes_first_writes_in_place(self):
        """The engine that computes the source takes the tokens itself, so a call
        with ``T = 0`` fails there whatever the write does, and the write stays in
        that engine, in place."""
        sources = {
            "relu": (lambda model, h: torch.relu(h), (1, 4, 3, 8), 2),
            "projection": (
                lambda model, h: model.k(h).view(1, h.shape[1], 4, 8).transpose(1, 2),
                (1, 3, 32),
                1,
            ),
        }
        for name, (source, shape, dim) in sources.items():
            with self.subTest(source=name):

                class M(_ChunkCache):
                    def __init__(self):
                        super().__init__()
                        self.k = torch.nn.Linear(32, 32)

                    def forward(self, h, input_pos):
                        return super().forward(source(self, h), input_pos)

                def call(t, pos):
                    size = list(shape)
                    size[dim] = t
                    return (
                        torch.randn(*size, device="cuda"),
                        torch.tensor([pos], device="cuda"),
                    )

                model = M().cuda()
                eager = copy.deepcopy(model)
                ep = export(
                    model, call(3, 0), dynamic_shapes=({dim: Dim("T", max=8)}, None)
                )
                compiled = torch_tensorrt.dynamo.compile(
                    ep, inputs=list(call(3, 0)), min_block_size=1, disable_tf32=True
                )
                self.assertEqual(len(_engines(compiled)), 1, _engines(compiled))
                self.assertEqual(len(_aliased_io(compiled)), 1)
                _check_calls(
                    compiled, eager, [call(t, p) for t, p in ((3, 0), (1, 3), (8, 4))]
                )

    def test_an_engine_input_of_another_length_leaves_the_write_in_pytorch(self):
        """The engine takes ``U``, which a call with ``T = 0`` leaves at 3, so the
        write of ``T`` tokens stays in PyTorch and that call works."""

        class M(_ChunkCache):
            def forward(self, x, input_pos, y):
                return super().forward(x.clone(), input_pos), torch.relu(y)

        def call(t, pos):
            return (*_chunk(t, pos), torch.randn(3, 4, device="cuda"))

        model = M().cuda()
        eager = copy.deepcopy(model)
        ep = export(
            model,
            call(3, 0),
            dynamic_shapes=({2: Dim("T", max=8)}, None, {0: Dim("U", max=8)}),
        )
        compiled = torch_tensorrt.dynamo.compile(
            ep,
            inputs=list(call(3, 0)),
            min_block_size=1,
            torch_executed_ops=_POSITIONS_IN_PYTORCH | {"torch.ops.aten.clone.default"},
        )
        self.assertEqual(_aliased_io(compiled), {})
        _check_calls(compiled, eager, [call(t, p) for t, p in ((3, 0), (0, 3), (1, 3))])

    def test_a_data_dependent_length_stays_in_pytorch(self):
        """A data-dependent size gets a profile from 0, so the engine that takes it
        runs a call of size 0, while the write's converter cannot. The write stays in
        PyTorch and that call works."""

        class M(_ChunkCache):
            def forward(self, x, input_pos):
                rows = torch.ops.torchtrt_kv_test.positive_rows(x)
                source = (rows * 2).view(1, 1, rows.shape[0], 8).expand(1, 4, -1, 8)
                return super().forward(source, input_pos), (rows.abs() * 3 + 1).sum(0)

        def call(rows, pos):
            x = -torch.ones(6, 8, device="cuda")
            x[:rows] = torch.rand(rows, 8, device="cuda") + 1
            return x, torch.tensor([pos], device="cuda")

        model = M().cuda()
        eager = copy.deepcopy(model)
        ep = export(model, call(3, 0))
        compiled = torch_tensorrt.dynamo.compile(
            ep,
            inputs=list(call(3, 0)),
            min_block_size=1,
            torch_executed_ops=_POSITIONS_IN_PYTORCH,
        )
        self.assertEqual(_aliased_io(compiled), {})
        _check_calls(
            compiled, eager, [call(r, p) for r, p in ((3, 0), (0, 3), (1, 3), (2, 4))]
        )

    def test_a_source_folding_leaves_too_small_for_an_engine(self):
        """Before constant folding ``x * w.sin().cos()`` is an engine of its own that
        takes the tokens. Folding leaves ``x * c``, smaller than ``min_block_size``,
        so it runs in PyTorch and the write of its empty result does too."""

        class M(_ChunkCache):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.ones(1, 4, 1, 8))

            def forward(self, x, input_pos):
                return super().forward(x * self.w.sin().cos(), input_pos)

        model = M().cuda()
        eager = copy.deepcopy(model)
        x, input_pos = _chunk(3, 0)
        ep = export(model, (x, input_pos), dynamic_shapes=({2: Dim("T", max=8)}, None))
        compiled = torch_tensorrt.dynamo.compile(
            ep, inputs=[x, input_pos], min_block_size=5
        )
        _check_calls(
            compiled, eager, [_chunk(t, p) for t, p in ((3, 0), (0, 3), (1, 3))]
        )

    def test_operands_that_only_pytorch_can_compute(self):
        """No converter takes ``sort`` of a dynamic length or ``bessel_j0``, so their
        results enter the write's engine, if it has one, from PyTorch."""

        class M(torch.nn.Module):
            def forward(self, cache, index, values):
                index = torch.sort(index).values
                return torch.index_copy(
                    cache, 2, index, torch.special.bessel_j0(values)
                )

        def call(t):
            return (
                torch.zeros(1, 4, 16, 8, device="cuda"),
                torch.arange(2, 2 + t, device="cuda"),
                torch.randn(1, 4, t, 8, device="cuda"),
            )

        T = Dim("T", min=0, max=4)
        ep = export(M(), call(3), dynamic_shapes=(None, {0: T}, {2: T}))
        compiled = torch_tensorrt.dynamo.compile(
            ep, inputs=list(call(3)), min_block_size=1
        )
        for t in (3, 1, 0):
            args = call(t)
            torch.testing.assert_close(compiled(*args), M()(*args))

    def test_a_reader_no_converter_takes(self):
        """``sort`` has a converter, but not for a dynamic length, and a cast to
        float64 has one only with ``truncate_double``. Both run in PyTorch, so they
        do not take the empty source into an engine, and the write stays in PyTorch
        too."""
        readers = {
            "sort": lambda source: torch.sort(source, dim=2).values,
            "float64": lambda source: source.double(),
        }
        for name, reader in readers.items():
            with self.subTest(reader=name):

                class M(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.register_buffer("cache", torch.zeros(1, 4, 16, 8))

                    def forward(self, x, input_pos):
                        idx = input_pos + torch.arange(x.shape[2], device=x.device)
                        self.cache.index_copy_(2, idx, x)
                        return self.cache.sum(2), reader(x)

                model = M().cuda()
                eager = copy.deepcopy(model)
                compiled = self._compile_chunk(
                    model,
                    _chunk(3, 0)[0],
                    Dim("T", min=0, max=8),
                    torch_executed_ops={
                        "torch.ops.aten.arange.default",
                        "torch.ops.aten.arange.start",
                        "torch.ops.aten.arange.start_step",
                        "torch.ops.aten.add.Tensor",
                        "torch.ops.aten.sym_size.int",
                    },
                )
                _check_calls(
                    compiled, eager, [_chunk(t, p) for t, p in ((3, 0), (0, 3), (1, 3))]
                )

    def test_a_source_too_small_for_an_engine_of_its_own(self):
        """``x * 2`` and ``x.clone()`` have converters, but alone they are smaller
        than the default engine, so they run in PyTorch, and the write of their empty
        result does too."""
        sources = {"mul": lambda x: x * 2, "clone": lambda x: x.clone()}
        for name, source in sources.items():
            with self.subTest(source=name):

                class M(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.register_buffer("cache", torch.zeros(1, 4, 16, 8))

                    def forward(self, x, input_pos):
                        idx = input_pos + torch.arange(x.shape[2], device=x.device)
                        self.cache.index_copy_(2, idx, source(x))
                        return self.cache.sum(2)

                model = M().cuda()
                eager = copy.deepcopy(model)
                x, input_pos = _chunk(3, 0)
                ep = export(
                    model,
                    (x, input_pos),
                    dynamic_shapes=({2: Dim("T", min=0, max=8)}, None),
                )
                compiled = torch_tensorrt.dynamo.compile(ep, inputs=[x, input_pos])
                _check_calls(
                    compiled, eager, [_chunk(t, p) for t, p in ((3, 0), (0, 3), (1, 3))]
                )

    def test_a_reader_the_partitioner_keeps_out_of_the_engines(self):
        """``nonzero`` has a converter, but with ``fallback_data_dependent_ops`` the
        partitioner leaves it to PyTorch, so it takes no empty input into an engine."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(1, 4, 16, 8))

            def forward(self, x, input_pos):
                idx = input_pos + torch.arange(x.shape[2], device=x.device)
                self.cache.index_copy_(2, idx, x)
                return self.cache.sum(2), torch.nonzero(x)

        model = M().cuda()
        eager = copy.deepcopy(model)
        compiled = self._compile_chunk(
            model,
            _chunk(3, 0)[0],
            Dim("T", min=0, max=8),
            fallback_data_dependent_ops=True,
            torch_executed_ops={
                "torch.ops.aten.arange.default",
                "torch.ops.aten.arange.start",
                "torch.ops.aten.arange.start_step",
                "torch.ops.aten.add.Tensor",
                "torch.ops.aten.sym_size.int",
            },
        )
        _check_calls(
            compiled, eager, [_chunk(t, p) for t, p in ((3, 0), (0, 3), (1, 3))]
        )

    def test_writes_that_share_a_length_that_can_be_zero_stay_in_pytorch(self):
        """Both writes read the same source, so both stay in PyTorch and a call that
        writes nothing works."""

        class Shared(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("k", torch.zeros(1, 4, 16, 8))
                self.register_buffer("v", torch.zeros(1, 4, 16, 8))

            def forward(self, x, input_pos):
                idx = input_pos + torch.arange(x.shape[2], device=x.device)
                self.k.index_copy_(2, idx, x)
                self.v.index_copy_(2, idx, x)
                return self.k.sum(2) + self.v.sum(2)

        model = Shared().cuda()
        eager = copy.deepcopy(model)
        compiled = self._compile_chunk(
            model,
            _chunk(3, 0)[0],
            Dim("T", min=0, max=8),
            torch_executed_ops={
                "torch.ops.aten.arange.default",
                "torch.ops.aten.arange.start",
                "torch.ops.aten.arange.start_step",
                "torch.ops.aten.add.Tensor",
                "torch.ops.aten.sym_size.int",
            },
        )
        for t, p in ((3, 0), (0, 3), (1, 3), (4, 4)):
            args = _chunk(t, p)
            got = compiled(*args)
            got = got[0] if isinstance(got, (tuple, list)) else got
            torch.testing.assert_close(got, eager(*args))
            torch.testing.assert_close(compiled.k, eager.k)
            torch.testing.assert_close(compiled.v, eager.v)

    def test_an_input_spec_does_not_decide_where_the_write_runs(self):
        """The program allows ``T = 0`` and the ``Input`` spec starts at 1. The
        program decides, so the write stays in PyTorch, where refit puts it too."""
        model = _ChunkCache().cuda()
        eager = copy.deepcopy(model)
        x, input_pos = _chunk(3, 0)
        ep = export(
            model, (x, input_pos), dynamic_shapes=({2: Dim("T", min=0, max=8)}, None)
        )
        compiled = torch_tensorrt.dynamo.compile(
            ep,
            inputs=[
                torch_tensorrt.Input(
                    min_shape=(1, 4, 1, 8),
                    opt_shape=(1, 4, 3, 8),
                    max_shape=(1, 4, 8, 8),
                ),
                input_pos,
            ],
            min_block_size=1,
            torch_executed_ops={
                "torch.ops.aten.arange.default",
                "torch.ops.aten.arange.start",
                "torch.ops.aten.arange.start_step",
                "torch.ops.aten.add.Tensor",
                "torch.ops.aten.sym_size.int",
            },
        )
        self.assertEqual(_aliased_io(compiled), {})
        _check_calls(
            compiled, eager, [_chunk(t, p) for t, p in ((3, 0), (1, 3), (8, 4))]
        )

    def test_refit_splits_the_model_as_compile_did_under_an_input_spec(self):
        """Refit sees only the new program, not the ``Input`` specs, so compile must
        not let them move a write into an engine."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.k = torch.nn.Linear(8, 8)
                self.out = torch.nn.Linear(8, 8)

            def forward(self, cache, index, x):
                return self.out(torch.index_copy(cache, 2, index, self.k(x)))

        torch.manual_seed(0)
        old, new = M().cuda().eval(), M().cuda().eval()
        cache = torch.zeros(1, 4, 16, 8, device="cuda")
        args = (
            cache,
            torch.tensor([3, 9, 1], device="cuda"),
            torch.randn(1, 4, 3, 8, device="cuda"),
        )
        t = Dim("T", max=8)
        shapes = (None, {0: t}, {2: t})
        compiled = torch_tensorrt.dynamo.compile(
            export(old, args, dynamic_shapes=shapes),
            inputs=[
                cache,
                torch_tensorrt.Input(
                    min_shape=(1,), opt_shape=(3,), max_shape=(8,), dtype=torch.int64
                ),
                torch_tensorrt.Input(
                    min_shape=(1, 4, 1, 8),
                    opt_shape=(1, 4, 3, 8),
                    max_shape=(1, 4, 8, 8),
                ),
            ],
            min_block_size=1,
            immutable_weights=False,
        )
        refitted = refit_module_weights(
            compiled_module=compiled,
            new_weight_module=export(new, args, dynamic_shapes=shapes),
            arg_inputs=args,
            verify_output=True,
        )
        torch.testing.assert_close(refitted(*args), new(*args), atol=1e-3, rtol=1e-3)

    def test_dynamic_length_update_with_heads_unknown_at_build_time(self):
        """An update made from two inputs that share one dynamic length, then split
        into heads with ``view(1, T, -1, head_dim)``. TensorRT cannot prove the two
        lengths equal when it builds, so it does not know the head count, which the
        KV layer checks against the cache."""

        class M(_ChunkCache):
            def forward(self, x, y, input_pos):
                t = x.shape[1]
                update = torch.where(x > 0, x, y).view(1, t, -1, 8).transpose(1, 2)
                return super().forward(update, input_pos)

        model = M().cuda()
        eager = copy.deepcopy(model)
        x = torch.randn(1, 4, 32, device="cuda")
        input_pos = torch.tensor([0], device="cuda")
        T = Dim("T", min=1, max=8)
        ep = export(
            model, (x, x.clone(), input_pos), dynamic_shapes=({1: T}, {1: T}, None)
        )
        length = torch_tensorrt.Input(
            min_shape=(1, 1, 32), opt_shape=(1, 4, 32), max_shape=(1, 8, 32)
        )
        compiled = torch_tensorrt.compile(
            ep,
            ir="dynamo",
            inputs=[
                length,
                length,
                torch_tensorrt.Input(shape=(1,), dtype=torch.int64),
            ],
            min_block_size=1,
            use_python_runtime=False,
        )
        self.assertEqual(len(_aliased_io(compiled)), 1)

        calls = []
        for t, pos in ((5, 0), (1, 5), (8, 6), (3, 4)):
            x, y = torch.randn(2, 1, t, 32, device="cuda")
            calls.append((x, y, torch.tensor([pos], device="cuda")))
        _check_calls(compiled, eager, calls)


class TestProfilesOfComputedSizes(TestCase):
    """An engine input whose size is computed from a ``Dim``, as ``2 * n`` or
    ``n + 1``, starts where the program lets it start. Its largest size comes from
    an ``Input`` spec when the program has none, and without either it is as large
    as before this change."""

    class _Sum(torch.nn.Module):
        def forward(self, x, y):
            return x.sum() + y.sum()

    def test_an_input_spec_sets_the_largest_size(self):
        sizes = {"2 * n": lambda n: 2 * n, "n + 1": lambda n: n + 1}
        for name, size in sizes.items():
            with self.subTest(size=name):
                n = Dim("n", min=1)
                ep = export(
                    self._Sum(),
                    (torch.ones(4, 8).cuda(), torch.ones(size(4), 8).cuda()),
                    dynamic_shapes=({0: n}, {0: size(n)}),
                )
                compiled = torch_tensorrt.dynamo.compile(
                    ep,
                    inputs=[
                        torch_tensorrt.Input(
                            min_shape=(1, 8), opt_shape=(4, 8), max_shape=(20000, 8)
                        ),
                        torch_tensorrt.Input(
                            min_shape=(size(1), 8),
                            opt_shape=(size(4), 8),
                            max_shape=(size(20000), 8),
                        ),
                    ],
                    min_block_size=1,
                )
                # Both sizes reach past the largest one the fallback would give.
                for k in (1, 4, 20000):
                    args = (torch.ones(k, 8).cuda(), torch.ones(size(k), 8).cuda())
                    torch.testing.assert_close(compiled(*args), self._Sum()(*args))

    def test_a_size_made_in_the_graph_starts_at_the_recorded_minimum(self):
        """``T + 1`` made from a ``Dim(min=0)`` is 1 when ``T`` is 0, so the engine
        that updates the cache in place takes a single token."""

        class Cache(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(1, 4, 16, 8))

            def forward(self, x, pos):
                one = torch.ones((1, 4, 1, 8), dtype=x.dtype, device=x.device)
                update = torch.cat((x, one), 2)
                index = pos + torch.arange(update.shape[2], device=x.device)
                self.cache.index_copy_(2, index, update)
                return self.cache.sum(2)

        model, eager = Cache().cuda(), Cache().cuda()
        args = (torch.ones(1, 4, 3, 8).cuda(), torch.tensor([0]).cuda())
        ep = export(
            model, args, dynamic_shapes=({2: Dim("tokens", min=0, max=8)}, None)
        )
        compiled = torch_tensorrt.dynamo.compile(
            ep,
            inputs=list(args),
            min_block_size=1,
            torch_executed_ops={
                "torch.ops.aten.cat.default",
                "torch.ops.aten.arange.default",
                "torch.ops.aten.arange.start",
                "torch.ops.aten.arange.start_step",
                "torch.ops.aten.add.Tensor",
                "torch.ops.aten.sym_size.int",
            },
            cache_built_engines=False,
            reuse_cached_engines=False,
        )
        self.assertEqual(len(_aliased_io(compiled)), 1)
        for tokens, pos in ((0, 0), (1, 1), (3, 3)):
            x, p = torch.randn(1, 4, tokens, 8).cuda(), torch.tensor([pos]).cuda()
            out = compiled(x, p)
            out = out[0] if isinstance(out, (tuple, list)) else out
            torch.testing.assert_close(out, eager(x, p))
            torch.testing.assert_close(compiled.cache, eager.cache)

    def test_without_any_bound_the_largest_size_stays(self):
        """Before this change ``2 * n`` started at 4 and so reached 4 * 4096."""

        class Recorded(torch.nn.Module):
            def forward(self, x, y):
                return x, y.sum()

        class Computed(torch.nn.Module):
            def forward(self, x):
                return torch.cat((x, x)).sum()

        n = Dim("n", min=1)
        recorded = torch_tensorrt.dynamo.compile(
            export(
                Recorded(),
                (torch.ones(4, 8).cuda(), torch.ones(8, 8).cuda()),
                dynamic_shapes=({0: n}, {0: 2 * n}),
            ),
            inputs=[torch.ones(4, 8).cuda(), torch.ones(8, 8).cuda()],
            min_block_size=1,
        )
        computed = torch_tensorrt.dynamo.compile(
            export(Computed(), (torch.ones(4, 8).cuda(),), dynamic_shapes=({0: n},)),
            inputs=[torch.ones(4, 8).cuda()],
            min_block_size=1,
            torch_executed_ops={"torch.ops.aten.cat.default"},
        )
        for k in (1, 8000):
            x = torch.ones(k, 8).cuda()
            torch.testing.assert_close(
                recorded(x, torch.ones(2 * k, 8).cuda())[1], x.sum() * 2
            )
            torch.testing.assert_close(computed(x), x.sum() * 2)


class TestIndexCopyKVKeepsEagerSemantics(TestCase):
    """Writes that look like a KV update but must not be done in place."""

    def test_functional_write_leaves_the_input_alone(self):
        """A functional ``index_copy`` returns a new tensor; the caller's cache and
        any other reader of it keep the old values."""

        class M(torch.nn.Module):
            def forward(self, cache, input_pos, update):
                idx = input_pos + torch.arange(update.shape[2], device=update.device)
                return torch.index_copy(cache, 2, idx, update) - cache

        cache = torch.randn(1, 4, 16, 8, device="cuda")
        input_pos = torch.tensor([3], device="cuda")
        update = torch.randn(1, 4, 3, 8, device="cuda")
        compiled = _compile(M().cuda(), (cache.clone(), input_pos, update))
        self.assertEqual(_aliased_io(compiled), {})

        cache_run = cache.clone()
        out = compiled(cache_run, input_pos, update)
        out = out[0] if isinstance(out, tuple) else out
        torch.testing.assert_close(out, M()(cache, input_pos, update))
        torch.testing.assert_close(cache_run, cache)

    def test_buffer_read_by_another_node_is_copied_back(self):
        """The engine does not order an in-place write after a layer that reads the
        old cache without the write depending on it, so a multi-position write
        whose buffer has such a reader is copied back instead."""

        class M(_ChunkCache):
            def forward(self, x, input_pos):
                old = self.cache[:, :, 0:8].sum(2)
                return old, super().forward(x, input_pos)

        model = M().cuda()
        model.cache.fill_(1.0)
        eager = copy.deepcopy(model)
        args = _chunk(4, 2)
        compiled = _compile(model, args)
        self.assertEqual(_aliased_io(compiled), {})
        self.assertEqual(compiled.meta.get("_copyback_mutation_buffers"), ["cache"])
        for got, want in zip(compiled(*args), eager(*args)):
            torch.testing.assert_close(got, want)

    def test_positions_that_are_not_one_run_are_copied_back(self):
        """The layer writes one run of slots from a start, so an index with step 2,
        or a run added to several positions rather than one, is copied back."""

        class StepTwo(_ChunkCache):
            def forward(self, x, input_pos):
                idx = input_pos + torch.arange(0, 2 * x.shape[2], 2, device=x.device)
                self.cache.index_copy_(2, idx, x)
                return self.cache.sum(2)

        class PositionVector(_ChunkCache):
            def forward(self, x, positions):
                idx = positions + torch.arange(x.shape[2], device=x.device)
                self.cache.index_copy_(2, idx, x)
                return self.cache.sum(2)

        for model_type, make_pos in (
            (StepTwo, lambda p: torch.tensor([p], device="cuda")),
            (PositionVector, lambda p: torch.tensor([p, p + 3, p + 7], device="cuda")),
        ):
            with self.subTest(model=model_type.__name__):
                model = model_type().cuda()
                eager = copy.deepcopy(model)
                calls = [
                    (torch.randn(1, 4, 3, 8, device="cuda"), make_pos(p))
                    for p in (0, 2, 5)
                ]
                compiled = _compile(model, calls[0])
                self.assertEqual(_aliased_io(compiled), {})
                _check_calls(compiled, eager, calls)

    def test_index_built_in_pytorch_still_writes_in_place(self):
        """The index is proven consecutive before partitioning, so the write still
        happens in place when the ops that build the index run in PyTorch."""
        model = _ChunkCache().cuda()
        eager = copy.deepcopy(model)
        ep = export(model, _chunk(4, 2))
        compiled = torch_tensorrt.compile(
            ep,
            ir="dynamo",
            inputs=list(_chunk(4, 2)),
            min_block_size=1,
            torch_executed_ops={"torch.ops.aten.add.Tensor"},
            use_python_runtime=False,
        )
        self.assertEqual(len(_aliased_io(compiled)), 1)
        _check_calls(compiled, eager, [_chunk(4, p) for p in (2, 4, 11)])

    def test_positions_from_a_mutated_buffer_use_fallback(self):
        """Positions held in a buffer the model also updates are not known to be
        one consecutive run, so the cache is copied back."""

        class M(_ChunkCache):
            def __init__(self):
                super().__init__()
                self.register_buffer("positions", torch.tensor([3, 4, 5]))

            def forward(self, x, next_positions):
                self.cache.index_copy_(2, self.positions, x)
                self.positions.copy_(next_positions)
                return self.cache.sum(2)

        model = M().cuda()
        eager = copy.deepcopy(model)

        def call(positions):
            x = torch.randn(1, 4, 3, 8, device="cuda")
            return x, torch.tensor(positions, device="cuda")

        args = call([9, 2, 5])
        compiled = _compile(model, args)
        self.assertEqual(_aliased_io(compiled), {})
        self.assertIn("cache", compiled.meta.get("_copyback_mutation_buffers"))
        out = compiled(*args)
        out = out[0] if isinstance(out, tuple) else out
        torch.testing.assert_close(out, eager(*args))

    def test_engine_cache_keeps_in_place_and_functional_writes_apart(self):
        """A buffer write done in place and a functional write of a caller's tensor
        can be the same graph apart from the marks buffer lifting sets. An engine
        cached for one must not be reused for the other."""
        import tempfile

        class Functional(torch.nn.Module):
            def forward(self, x, input_pos, cache):
                idx = input_pos + torch.arange(x.shape[2], device=x.device)
                return torch.index_copy(cache, 2, idx, x).sum(2)

        x, input_pos = _chunk(4, 3)
        with tempfile.TemporaryDirectory() as directory:
            settings = dict(
                ir="dynamo",
                min_block_size=1,
                immutable_weights=False,
                cache_built_engines=True,
                reuse_cached_engines=True,
                engine_cache_dir=directory,
                use_python_runtime=False,
            )
            model = _ChunkCache().cuda()
            torch_tensorrt.compile(
                export(model, (x, input_pos)), inputs=[x, input_pos], **settings
            )

            cache = torch.randn(1, 4, 16, 8, device="cuda")
            before = cache.clone()
            compiled = torch_tensorrt.compile(
                export(Functional(), (x, input_pos, cache)),
                inputs=[x, input_pos, cache],
                **settings,
            )
            self.assertEqual(_aliased_io(compiled), {})
            out = compiled(x, input_pos, cache)
            out = out[0] if isinstance(out, tuple) else out
            torch.testing.assert_close(out, Functional()(x, input_pos, before))
            torch.testing.assert_close(cache, before)


class TestIndexCopyFallback(TestCase):
    """Cases where the validator denies the KV fast path. The fallback
    converter must produce correct results without aliasing."""

    def test_rank_3_input_uses_fallback(self):
        class M(torch.nn.Module):
            def forward(self, x, index, update):
                return torch.ops.aten.index_copy.default(x, 1, index, update)

        x = torch.zeros(2, 8, 16, device="cuda")
        index = torch.tensor([1, 3, 5], dtype=torch.int64, device="cuda")
        update = torch.randn(2, 3, 16, device="cuda")

        compiled = _compile(M().cuda(), (x.clone(), index, update.clone()))

        # No aliasing (validator rejected the KV path).
        self.assertEqual(_aliased_io(compiled), {})

        out = compiled(x.clone(), index, update)
        eager = torch.ops.aten.index_copy.default(x.clone(), 1, index, update)
        self.assertTrue(torch.allclose(out, eager))

    def test_dim_other_than_two_uses_fallback(self):
        class M(torch.nn.Module):
            def forward(self, cache, index, update):
                return torch.ops.aten.index_copy.default(cache, 1, index, update)

        cache = torch.zeros(1, 16, 4, 8, device="cuda")
        index = torch.tensor([3], dtype=torch.int64, device="cuda")
        update = torch.ones(1, 1, 4, 8, device="cuda") * 5.0

        compiled = _compile(M().cuda(), (cache.clone(), index, update.clone()))
        self.assertEqual(_aliased_io(compiled), {})

        cache_run = cache.clone()
        out = compiled(cache_run, index, update)
        eager = torch.ops.aten.index_copy.default(cache.clone(), 1, index, update)
        self.assertTrue(torch.allclose(out, eager))

    def test_batch_gt_one_uses_fallback(self):
        """Batch > 1 currently routes to fallback (broadcasting writeIndices
        is a Phase-2 extension)."""

        class M(torch.nn.Module):
            def forward(self, cache, index, update):
                return torch.ops.aten.index_copy.default(cache, 2, index, update)

        cache = torch.zeros(4, 4, 16, 8, device="cuda")
        index = torch.tensor([3], dtype=torch.int64, device="cuda")
        update = torch.ones(4, 4, 1, 8, device="cuda") * 7.0

        compiled = _compile(M().cuda(), (cache.clone(), index, update.clone()))
        self.assertEqual(_aliased_io(compiled), {})

        cache_run = cache.clone()
        out = compiled(cache_run, index, update)
        eager = torch.ops.aten.index_copy.default(cache.clone(), 2, index, update)
        self.assertTrue(torch.allclose(out, eager))

    def test_0d_input_uses_fallback(self):
        # x[0] * 2.0 and y[0] * 3.0 are 0-d, and index_copy puts one into the other.
        class M(torch.nn.Module):
            def forward(self, x, index, y):
                return torch.ops.aten.index_copy.default(
                    x[0] * 2.0, 0, index, y[0] * 3.0
                )

        x = torch.tensor([2.0, 5.0], device="cuda")
        index = torch.tensor([0], dtype=torch.int64, device="cuda")
        y = torch.tensor([7.0, 1.0], device="cuda")

        compiled = _compile(M().cuda(), (x, index, y))
        torch.testing.assert_close(compiled(x, index, y), M()(x, index, y))

    def test_0d_source_into_1d_input_uses_fallback(self):
        # y[0] * 3.0 is 0-d, and index_copy puts it into one element of x.
        class M(torch.nn.Module):
            def forward(self, x, index, y):
                return torch.ops.aten.index_copy.default(x * 2.0, 0, index, y[0] * 3.0)

        x = torch.tensor([2.0, 5.0, 4.0], device="cuda")
        index = torch.tensor([1], dtype=torch.int64, device="cuda")
        y = torch.tensor([7.0, 1.0], device="cuda")

        compiled = _compile(M().cuda(), (x, index, y))
        torch.testing.assert_close(compiled(x, index, y), M()(x, index, y))

    def test_negative_dim_uses_fallback(self):
        class M(torch.nn.Module):
            def forward(self, x, index, update):
                return torch.ops.aten.index_copy.default(x, -1, index, update)

        x = torch.zeros(2, 3, device="cuda")
        index = torch.tensor([2, 0], dtype=torch.int64, device="cuda")
        update = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")

        compiled = _compile(M().cuda(), (x.clone(), index, update))
        self.assertEqual(_aliased_io(compiled), {})

        out = compiled(x.clone(), index, update)
        eager = torch.ops.aten.index_copy.default(x.clone(), -1, index, update)
        torch.testing.assert_close(out, eager)

    def test_integer_cache_buffer_uses_fallback(self):
        """The KV layer takes floating-point caches only, so a one-token write into an
        integer cache buffer, by index_copy or by slice assignment, takes the
        fallback and is copied back."""

        class M(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.register_buffer("cache", torch.zeros(1, 4, 16, 8, dtype=dtype))
                self.register_buffer("other", torch.zeros(1, 4, 16, 8, dtype=dtype))

            def forward(self, x, input_pos):
                self.cache.index_copy_(2, input_pos, x)
                self.other[:, :, 5:6, :] = x
                return self.cache.sum(2) + self.other.sum(2)

        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):

                def step(i):
                    x = torch.full((1, 4, 1, 8), i + 1, device="cuda").to(dtype)
                    return x, torch.tensor([2 + i], device="cuda")

                eager = M(dtype).cuda()
                compiled = _compile(M(dtype).cuda(), step(0))
                self.assertEqual(_aliased_io(compiled), {})
                self.assertEqual(
                    sorted(compiled.meta.get("_copyback_mutation_buffers")),
                    ["cache", "other"],
                )
                for i in range(3):
                    out = compiled(*step(i))
                    out = out[0] if isinstance(out, tuple) else out
                    torch.testing.assert_close(out, eager(*step(i)))
                    torch.testing.assert_close(compiled.cache, eager.cache)
                    torch.testing.assert_close(compiled.other, eager.other)

    def test_integer_caller_cache_is_written_back(self):
        """The fallback is not in place, so its write-back stores the write in the
        caller's tensor."""

        class M(torch.nn.Module):
            def forward(self, cache, x, input_pos):
                cache.index_copy_(2, input_pos, x)
                return cache.sum(2)

        args = (
            torch.zeros(1, 4, 16, 8, dtype=torch.int32, device="cuda"),
            torch.ones(1, 4, 1, 8, dtype=torch.int32, device="cuda"),
            torch.tensor([2], device="cuda"),
        )
        compiled = _compile(M(), [a.clone() for a in args])
        given, expected = [a.clone() for a in args], [a.clone() for a in args]
        out = compiled(*given)
        out = out[0] if isinstance(out, (tuple, list)) else out
        torch.testing.assert_close(out, M()(*expected))
        torch.testing.assert_close(given[0], expected[0])


class _Scale(torch.nn.Module):
    """A submodule a test can name in ``torch_executed_modules``."""

    def forward(self, cache, x):
        return cache.mean(2, keepdim=True) * x


class TestOnlySafeWritesInPlace(TestCase):
    """A cache write is applied in place only when nothing but the write reads the
    old cache. Every call matches eager PyTorch: the output, the buffer, and the
    caller's tensors. A returned tensor also keeps its value after later calls, as
    eager's does, including one the engine would otherwise return in the cache's own
    memory."""

    def _check(self, model, make_args, in_place=False, **kwargs):
        eager = copy.deepcopy(model)
        compiled = _compile(model, make_args(0), **kwargs)
        self.assertEqual(bool(_aliased_io(compiled)), in_place)
        for step in range(3):
            args = make_args(step)
            run_args = [a.clone() for a in args]
            out, expected = compiled(*run_args), eager(*args)
            out = list(out) if isinstance(out, (tuple, list)) else [out]
            expected = (
                list(expected) if isinstance(expected, (tuple, list)) else [expected]
            )
            if step == 0:
                first, first_value = out[0], out[0].clone()
            torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)
            for got, given in zip(run_args, args):
                torch.testing.assert_close(got, given)
            # A model that only reads its buffer may not keep it as one.
            for name, buffer in eager.named_buffers():
                if hasattr(compiled, name):
                    torch.testing.assert_close(
                        getattr(compiled, name), buffer, rtol=1e-3, atol=1e-3
                    )
        torch.testing.assert_close(first, first_value)

    @staticmethod
    def _cache_model(forward, cache=None):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "cache",
                    torch.zeros(1, 4, 16, 8, device="cuda") if cache is None else cache,
                )
                self.scale = _Scale()

        M.forward = forward
        return M()

    @staticmethod
    def _step(step):
        g = torch.Generator().manual_seed(step)
        x = torch.randn(1, 4, 1, 8, generator=g).cuda()
        return x, torch.tensor([2 + step], device="cuda")

    def test_an_independent_reader_of_the_written_slot(self):
        def forward(self, x, pos):
            old = self.cache[:, :, 3:4, :] * 2
            self.cache.index_copy_(2, pos, x)
            return self.cache.sum(2) + old.sum()

        self._check(self._cache_model(forward), self._step)

    def test_a_reader_through_a_view(self):
        def forward(self, x, pos):
            view = self.cache.view(1, 4, 16, 8)
            old = view * 2
            src = x + view.mean(2, keepdim=True)
            self.cache.index_copy_(2, pos, src)
            return self.cache - old

        self._check(self._cache_model(forward), self._step)

    def test_a_returned_clone(self):
        def forward(self, x, pos):
            self.cache.index_copy_(2, pos, x)
            return self.cache.clone()

        self._check(self._cache_model(forward), self._step, in_place=True)

    def test_a_reader_in_an_excluded_module(self):
        def forward(self, x, pos):
            scale = self.scale(self.cache, x)
            self.cache.index_copy_(2, pos, x + scale)
            return self.cache.sum(2)

        self._check(
            self._cache_model(forward),
            self._step,
            torch_executed_modules={f"{_Scale.__module__}.{_Scale.__qualname__}"},
        )

    def test_a_reader_of_a_split_part(self):
        def forward(self, x, pos):
            whole = self.cache.split(16, dim=2)[0]
            old = whole * 2
            self.cache.index_copy_(2, pos, x + whole.mean(2, keepdim=True))
            return self.cache - old

        self._check(self._cache_model(forward), self._step)

    def test_a_reader_of_a_zero_pad(self):
        def forward(self, x, pos):
            padded = torch.nn.functional.pad(self.cache, (0, 0, 0, 0))
            self.cache.index_copy_(2, pos, x + padded.mean(2, keepdim=True))
            return self.cache.sum(2) + padded.sum(2)

        self._check(self._cache_model(forward), self._step)

    def test_a_value_computed_from_a_copy_into_a_new_tensor(self):
        """The converter of ``copy`` hands back the cache it copies, its second
        input, so the value computed from the copy reads the cache."""

        def forward(self, x, pos):
            old = torch.empty_like(self.cache).copy_(self.cache)
            self.cache.index_copy_(2, pos, x + old.mean(2, keepdim=True))
            return old * 2 + self.cache.sum()

        self._check(self._cache_model(forward), self._step)

    def test_a_returned_zero_pad_of_the_new_cache(self):
        """The pad converter hands back its input, here the updated cache."""

        def forward(self, x, pos):
            self.cache.index_copy_(2, pos, x)
            return torch.nn.functional.pad(self.cache, (0, 0, 0, 0))

        self._check(self._cache_model(forward), self._step, in_place=True)

    def test_a_returned_pool_to_the_cache_size(self):
        """So does the pool converter, for a pool to the input's own size."""

        def forward(self, x, pos):
            self.cache.index_copy_(2, pos, x)
            return torch.nn.functional.adaptive_avg_pool2d(self.cache, (16, 8))

        self._check(self._cache_model(forward), self._step, in_place=True)

    def test_a_returned_max_pool_to_the_cache_size(self):
        def forward(self, x, pos):
            self.cache.index_copy_(2, pos, x)
            return torch.nn.functional.adaptive_max_pool2d(self.cache, (16, 8))

        self._check(self._cache_model(forward), self._step, in_place=True)

    def test_an_old_value_reader_of_a_truncated_double(self):
        """With ``truncate_double`` the cast to float64 is the cache itself inside
        the engine."""

        def forward(self, x, pos):
            old = self.cache.double()
            self.cache.index_copy_(2, pos, x)
            return self.cache.sum(2) + old.sum(2).float()

        self._check(self._cache_model(forward), self._step, truncate_double=True)

    def test_a_buffer_that_stores_the_old_cache(self):
        """Another buffer's write-back reads the cache before the cache is written.
        Its ``copy_`` is gone by the time the cache is classified, so the value it
        stores still counts as a reader of the old cache."""
        for name, f in (("copy", lambda t: t), ("computed", lambda t: t * 2)):
            with self.subTest(name):

                def forward(self, x, pos):
                    self.snap.copy_(f(self.cache))
                    self.cache.index_copy_(2, pos, x)
                    return self.cache.sum(2)

                model = self._cache_model(forward, torch.randn(1, 4, 16, 8).cuda())
                model.register_buffer("snap", torch.zeros(1, 4, 16, 8).cuda())
                self._check(model, self._step)

    def test_a_copy_of_a_buffer_written_with_values_from_the_caller(self):
        """The clone runs in PyTorch, so the engine is handed the copy as an input,
        and the model returns the copy's old contents."""

        def forward(self, x, pos):
            cache = self.cache.clone()
            updated = torch.index_copy(cache, 2, pos, x)
            return cache, updated.sum()

        self._check(
            self._cache_model(forward),
            self._step,
            torch_executed_ops={"torch.ops.aten.clone.default"},
        )

    def test_a_channels_last_buffer(self):
        def forward(self, x, pos):
            self.cache.index_copy_(2, pos, x)
            return self.cache.sum(2)

        cache = torch.zeros(1, 4, 16, 8, device="cuda").contiguous(
            memory_format=torch.channels_last
        )
        self._check(self._cache_model(forward, cache), self._step)

    def test_a_write_of_a_computed_tensor(self):
        """The partitioner hands the sine to the engine as an input, which the
        write must not update in place, since the model reads it again."""

        class M(torch.nn.Module):
            def forward(self, x, pos, v):
                y = torch.sin(x)
                return y.index_copy(2, pos, v) + y

        def make_args(step):
            x, pos = self._step(step)
            return torch.randn(1, 4, 16, 8, device="cuda"), pos, x

        self._check(M(), make_args, torch_executed_ops={"torch.ops.aten.sin.default"})

    def test_a_cache_written_twice(self):
        def forward(self, x, pos):
            self.cache.index_copy_(2, pos, x)
            self.cache.index_copy_(2, pos, x + self.cache.sum(2, keepdim=True))
            return self.cache.sum(2)

        self._check(self._cache_model(forward), self._step)

    def test_two_writes_with_an_op_in_pytorch_between(self):
        """The op puts the writes in different engines, and the second engine would
        update the first one's result in place while PyTorch still reads it."""

        def forward(self, x, pos):
            self.cache.index_copy_(2, pos, x)
            scale = torch.sin(self.cache.sum())
            old = self.cache * (scale + 2)
            self.cache.index_copy_(2, pos, x * (scale + 4))
            return self.cache - old

        self._check(
            self._cache_model(forward),
            self._step,
            torch_executed_ops={"torch.ops.aten.sin.default"},
        )


class TestCallerTensorsKeepWhatEagerDoes(TestCase):
    """A compiled model changes a caller's tensor exactly when eager PyTorch does."""

    def test_a_slice_write_of_a_returned_input_is_stored(self):
        """The model returns the tensor it writes, so its write-back copy_ stays
        and runs after the engines. The write does not need to be done in place."""

        class M(torch.nn.Module):
            def forward(self, x, y):
                x[:, 2:4] = y
                return x

        args = (torch.randn(4, 8, device="cuda"), torch.randn(4, 2, device="cuda"))
        compiled = _compile(M(), [a.clone() for a in args])
        given = [a.clone() for a in args]
        expected = [a.clone() for a in args]
        out = compiled(*given)
        out = out[0] if isinstance(out, (tuple, list)) else out
        torch.testing.assert_close(out, M()(*expected))
        torch.testing.assert_close(given[0], expected[0])

    def test_a_computed_value_of_a_written_cache_is_returned(self):
        """The model writes the caller's cache and returns a value computed from
        it, which the engine gives its own memory."""

        for name, f in (("tanh", torch.tanh), ("times one", lambda t: t * 1.0)):
            with self.subTest(name):

                class M(torch.nn.Module):
                    def forward(self, cache, x, pos):
                        cache.index_copy_(2, pos, x)
                        return f(cache)

                args = (
                    torch.zeros(1, 4, 16, 8, device="cuda"),
                    torch.randn(1, 4, 1, 8, device="cuda"),
                    torch.tensor([3], device="cuda"),
                )
                compiled = _compile(M(), [a.clone() for a in args])
                given = [a.clone() for a in args]
                expected = [a.clone() for a in args]
                out = compiled(*given)
                out = out[0] if isinstance(out, (tuple, list)) else out
                first = out.clone()
                torch.testing.assert_close(out, M()(*expected))
                torch.testing.assert_close(given[0], expected[0])
                compiled(given[0], torch.randn_like(args[1]), args[2] + 1)
                torch.testing.assert_close(out, first)

    def test_torch_compile_leaves_a_functionally_written_cache_alone(self):
        """torch.compile hands the backend a functional index_copy, which must not
        change the caller's cache, whatever its layout."""

        class M(torch.nn.Module):
            def forward(self, cache, index, update):
                update = torch.relu(update + 1) * 2 - 1
                return torch.index_copy(cache, 2, index, update).sum(2)

        base = torch.arange(512, device="cuda", dtype=torch.float32)
        caches = {
            "contiguous": base.reshape(1, 4, 16, 8),
            "other strides": base.reshape(1, 16, 4, 8).transpose(1, 2),
        }
        index = torch.tensor([3], device="cuda")
        update = torch.full((1, 4, 1, 8), 7.0, device="cuda")
        for name, cache in caches.items():
            with self.subTest(cache=name):
                torch._dynamo.reset()
                compiled = torch.compile(M(), backend="torch_tensorrt", fullgraph=True)
                before = cache.clone()
                expected = M()(before.clone(), index, update)
                torch.testing.assert_close(compiled(cache, index, update), expected)
                torch.testing.assert_close(cache, before)
        torch._dynamo.reset()

    def test_torch_compile_leaves_the_cache_of_a_written_clone_alone(self):
        """The model writes its own clone of the cache, so the caller's cache stays
        as it was."""

        class M(torch.nn.Module):
            def forward(self, cache, x, pos):
                y = cache.clone()
                y.index_copy_(2, pos, x)
                return y.sum(2)

        cache = torch.randn(1, 4, 16, 8, device="cuda")
        x = torch.randn(1, 4, 1, 8, device="cuda")
        pos = torch.tensor([3], device="cuda")
        before = cache.clone()
        torch._dynamo.reset()
        try:
            compiled = torch.compile(
                M(),
                backend="torch_tensorrt",
                fullgraph=True,
                options={"min_block_size": 1, "pass_through_build_failures": True},
            )
            for _ in range(2):
                torch.testing.assert_close(compiled(cache, x, pos), M()(before, x, pos))
                torch.testing.assert_close(cache, before)
        finally:
            torch._dynamo.reset()


class TestEntryPointsWithoutLiftingKeepTheWrite(TestCase):
    """The torch.compile backend, the serialized engine and the Windows cross compile
    do not lift buffers, and they still store an in-place write of a caller's
    tensor: in place when the engine can, otherwise through its write-back."""

    class _Writer(torch.nn.Module):
        def forward(self, cache, x, pos):
            cache.index_copy_(2, pos, x)
            return cache.sum(2)

    @staticmethod
    def _step(n):
        g = torch.Generator().manual_seed(n)
        return torch.randn(1, 4, 1, 8, generator=g).cuda(), torch.tensor(
            [2 + n], device="cuda"
        )

    def _check_steps(self, compiled, eager, cache, expected):
        for n in range(3):
            x, pos = self._step(n)
            torch.testing.assert_close(compiled(cache, x, pos), eager(expected, x, pos))
            torch.testing.assert_close(cache, expected)

    def test_torch_compile_stores_the_write_of_a_caller_cache(self):
        caches = {
            "contiguous": (
                lambda: torch.zeros(1, 4, 16, 8, device="cuda"),
                {"arg0_1": "kv_cache_update"},
            ),
            "other strides": (
                lambda: torch.zeros(1, 16, 4, 8, device="cuda").transpose(1, 2),
                {},
            ),
        }
        from torch_tensorrt.dynamo.backend.backends import torch_tensorrt_backend

        for name, (make, aliased) in caches.items():
            with self.subTest(cache=name):
                torch._dynamo.reset()
                built = []

                def backend(gm, sample_inputs):
                    built.append(
                        torch_tensorrt_backend(
                            gm,
                            sample_inputs,
                            options={
                                "min_block_size": 1,
                                "pass_through_build_failures": True,
                            },
                        )
                    )
                    return built[-1]

                compiled = torch.compile(
                    self._Writer(), backend=backend, fullgraph=True
                )
                self._check_steps(compiled, self._Writer(), make(), make())
                self.assertEqual(dict(_aliased_io(built[0]).values()), aliased)
        torch._dynamo.reset()

    def test_torch_compile_stores_a_write_through_data(self):
        """A write through ``.data`` leaves the version counter alone."""

        class DataWriter(torch.nn.Module):
            def forward(self, cache, x, pos):
                cache.data.index_copy_(2, pos, x)
                return cache.sum(2)

        torch._dynamo.reset()
        compiled = torch.compile(
            DataWriter(),
            backend="torch_tensorrt",
            options={"min_block_size": 1, "pass_through_build_failures": True},
        )
        cache, expected = (
            torch.zeros(1, 4, 16, 8).cuda(),
            torch.zeros(1, 4, 16, 8).cuda(),
        )
        self._check_steps(compiled, DataWriter(), cache, expected)
        torch._dynamo.reset()

    def test_torch_compile_returns_a_reshape_of_an_input_it_does_not_write(self):
        """Only written inputs get a traced final value: AOT refuses one that is a
        view of another output, as the input and this reshape of it are."""

        class Reshape(torch.nn.Module):
            def forward(self, x, targets):
                return x.reshape(targets.size(0), -1)

        torch._dynamo.reset()
        x, targets = torch.randn(16, 64).cuda(), torch.zeros(16, 1).cuda()
        torch._dynamo.mark_dynamic(x, 0, min=1, max=2048)
        torch._dynamo.mark_dynamic(targets, 0, min=1, max=2048)
        compiled = torch.compile(
            Reshape(),
            backend="torch_tensorrt",
            options={"min_block_size": 1, "pass_through_build_failures": True},
        )
        torch.testing.assert_close(compiled(x, targets), x.reshape(16, -1))
        torch._dynamo.reset()

    def test_torch_compile_stores_the_write_of_a_buffer(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(1, 4, 16, 8, device="cuda"))

            def forward(self, x, pos):
                self.cache.index_copy_(2, pos, x)
                return self.cache.sum(2)

        torch._dynamo.reset()
        model, eager = M(), M()
        compiled = torch.compile(
            model,
            backend="torch_tensorrt",
            options={"min_block_size": 1, "pass_through_build_failures": True},
        )
        for n in range(3):
            x, pos = self._step(n)
            torch.testing.assert_close(compiled(x, pos), eager(x, pos))
            torch.testing.assert_close(model.cache, eager.cache)
        torch._dynamo.reset()

    def _example(self):
        return (torch.zeros(1, 4, 16, 8, device="cuda"),) + self._step(0)

    def test_the_serialized_engine_updates_the_cache_in_place(self):
        import tensorrt as trt

        example = self._example()
        engine = (
            torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
                export(self._Writer(), example),
                arg_inputs=list(example),
                min_block_size=1,
            )
        )
        engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(
            bytes(engine)
        )
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.assertEqual(
            [engine.get_aliased_input_tensor(n) for n in names[3:]], [None, "cache"]
        )

    def test_full_compilation_keeps_the_write_in_place(self):
        """With full support compile returns the engine module itself, whose
        ``aliased_io`` the check must read too."""
        example = self._example()
        compiled = torch_tensorrt.dynamo.compile(
            export(self._Writer(), example),
            arg_inputs=list(example),
            min_block_size=1,
            require_full_compilation=True,
        )
        cache, expected = example[0].clone(), example[0].clone()
        self._check_steps(compiled, self._Writer(), cache, expected)

    def test_the_serialized_engine_with_binding_names(self):
        import tensorrt as trt

        example = self._example()
        engine = (
            torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
                export(self._Writer(), example),
                arg_inputs=list(example),
                arg_input_binding_names=("cache_in", "x_in", "pos_in"),
                min_block_size=1,
            )
        )
        engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(
            bytes(engine)
        )
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.assertIn("cache_in", [engine.get_aliased_input_tensor(n) for n in names])

    def test_the_serialized_engine_from_host_examples(self):
        """The caller binds the engine's memory, so a host example still gives a
        cache updated in place."""
        import tensorrt as trt

        example = tuple(t.cpu() for t in self._example())
        engine = (
            torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
                export(self._Writer(), example),
                arg_inputs=list(example),
                min_block_size=1,
            )
        )
        engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(
            bytes(engine)
        )
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.assertIn("cache", [engine.get_aliased_input_tensor(n) for n in names])

    def test_the_serialized_engine_refuses_a_write_it_cannot_do_in_place(self):
        class M(torch.nn.Module):
            def forward(self, cache, x):
                cache[:, :, 0:8:2, :] = x
                return cache.sum(2)

        example = (
            torch.zeros(1, 4, 16, 8, device="cuda"),
            torch.ones(1, 4, 4, 8, device="cuda"),
        )
        with self.assertRaisesRegex(
            RuntimeError, r"cannot update input\(s\) \['cache'\]"
        ):
            torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
                export(M(), example), arg_inputs=list(example), min_block_size=1
            )

    def test_the_windows_cross_compile_marks_the_write(self):
        """Building for Windows needs a Windows-enabled build, so this checks the
        graph the build would be handed."""
        from torch_tensorrt import _features
        from torch_tensorrt.dynamo import _compiler

        handed = []

        def compile_module(gm, *args, **kwargs):
            handed.append(gm)
            return gm

        example = self._example()
        enabled = _features.ENABLED_FEATURES._replace(windows_cross_compile=True)
        with (
            mock.patch.object(_features, "ENABLED_FEATURES", enabled),
            mock.patch.object(_compiler, "compile_module", compile_module),
        ):
            torch_tensorrt.dynamo.cross_compile_for_windows(
                export(self._Writer(), example),
                arg_inputs=list(example),
                min_block_size=1,
                dryrun=True,
            )
        (write,) = [
            n
            for n in handed[0].graph.nodes
            if n.target is torch.ops.aten.index_copy.default
        ]
        self.assertTrue(write.meta.get("_trt_kv_alias"))


def _python_runtime():
    """The Python runtime runs the engine when the C++ runtime library is not
    installed, which is the only way to select it."""
    from torch_tensorrt.dynamo.runtime import _TorchTensorRTModule as runtime

    without_cpp = runtime.ENABLED_FEATURES._replace(torch_tensorrt_runtime=False)
    return mock.patch.object(runtime, "ENABLED_FEATURES", without_cpp)


def _python_engine_op():
    """The C++ op is loaded here, so the Python engine is called as its own op
    calls it."""

    def python_op(input_tensors, engine):
        outputs = engine.execute(input_tensors)
        return [outputs] if isinstance(outputs, torch.Tensor) else list(outputs)

    return mock.patch.object(torch.ops.tensorrt, "execute_engine", python_op)


class TestCallerCacheTheEngineGetsACopyOf(TestCase):
    """A caller's cache with other strides, or on the host, reaches the engine as a
    copy. The engine updates the copy, and the write-back of the returned cache
    stores it, also in a saved and loaded program and in both engine runtimes."""

    class _Writer(torch.nn.Module):
        def forward(self, cache, x, pos):
            cache.index_copy_(2, pos, x)
            return cache, cache.sum(2)

    @staticmethod
    def _step(n):
        g = torch.Generator().manual_seed(n)
        x = torch.randn(1, 4, 1, 8, generator=g).cuda()
        return x, torch.tensor([2 + n], device="cuda")

    def _check(self, modules):
        caches = {
            "contiguous on the GPU": lambda: torch.randn(1, 4, 16, 8, device="cuda"),
            "other strides": lambda: torch.randn(1, 16, 4, 8, device="cuda").transpose(
                1, 2
            ),
            "host": lambda: torch.randn(1, 4, 16, 8),
        }
        for kind, module in modules.items():
            for name, make in caches.items():
                with self.subTest(module=kind, cache=name):
                    cache = make()
                    expected = cache.clone()
                    for n in range(3):
                        x, pos = self._step(n)
                        out = module(cache, x, pos)
                        want = self._Writer()(
                            expected, x.to(expected.device), pos.to(expected.device)
                        )
                        torch.testing.assert_close(out[1].cpu(), want[1].cpu())
                        torch.testing.assert_close(cache, expected)

    def test_a_cache_with_other_strides_or_on_the_host(self):
        example = (torch.zeros(1, 4, 16, 8, device="cuda"),) + self._step(0)
        compiled = _compile(self._Writer(), example)
        self.assertEqual(len(_aliased_io(compiled)), 1)
        modules = {"compiled": compiled}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.ep")
            torch_tensorrt.save(
                compiled, path, output_format="exported_program", arg_inputs=example
            )
            modules["exported_program"] = torch.export.load(path).module()
            # TorchScript holds the engine as a C++ object.
            if torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime:
                path = os.path.join(tmp, "m.ts")
                torch_tensorrt.save(
                    compiled, path, output_format="torchscript", arg_inputs=example
                )
                modules["torchscript"] = torch.jit.load(path)
        self._check(modules)

    def test_the_python_runtime(self):
        from torch_tensorrt.dynamo.runtime import _TorchTensorRTModule as runtime

        example = (torch.zeros(1, 4, 16, 8, device="cuda"),) + self._step(0)
        with _python_runtime():
            compiled = _compile(self._Writer(), example)
            engines = [
                type(m.engine).__name__
                for m in compiled.modules()
                if isinstance(m, runtime.TorchTensorRTModule)
            ]
            self.assertEqual(engines, ["TRTEngine"])
            self.assertEqual(len(_aliased_io(compiled)), 1)
            with _python_engine_op():
                self._check({"compiled": compiled})


class TestInputsInTheMemoryOfAnUpdatedCache(TestCase):
    """The engine updates a KV cache in place while it reads its other inputs. An
    input in the cache's memory still reads the old contents, in both engine
    runtimes, because it is bound to a copy taken before the call. Two caches that
    share memory cannot both be updated, so that call is refused."""

    class _Reader(torch.nn.Module):
        def forward(self, cache, reader, x, pos):
            old = reader.clone()
            cache.index_copy_(2, pos, x)
            return cache + old

    class _SlotReader(torch.nn.Module):
        def forward(self, cache, reader, x, pos):
            old = reader.clone()
            cache.index_copy_(2, pos, x)
            return cache.sum(2) + old.sum(2)

    class _TwoCaches(torch.nn.Module):
        def forward(self, k, v, x, pos):
            k.index_copy_(2, pos, x)
            v.index_copy_(2, pos, x * 2)
            return k.sum(2) + v.sum(2)

    def _call(self, model, compiled, make, x, pos):
        given, expected = make(), make()
        out = compiled(*given, x, pos)
        out = out[0] if isinstance(out, (tuple, list)) else out
        torch.testing.assert_close(out, model(*expected, x, pos))
        torch.testing.assert_close(given[0], expected[0])

    def _check(self, cudagraphs=False, calls=contextlib.nullcontext):
        # Each call makes the same values, for the compiled and the eager model.
        def values(*shape, seed=0):
            g = torch.Generator().manual_seed(seed)
            return torch.randn(*shape, generator=g).cuda()

        def separate():
            return values(1, 4, 16, 8), values(1, 4, 16, 8, seed=1)

        def disjoint_halves():
            both = values(2, 4, 16, 8)
            return both[:1], both[1:]

        def a_view_of_the_cache():
            cache = values(1, 4, 16, 8)
            return cache, cache.view_as(cache)

        def a_strided_view_of_the_cache():
            cache = values(1, 4, 16, 8)
            return cache, cache.view(1, 4, 8, 16).transpose(2, 3)

        def a_slot_the_engine_does_not_write():
            cache = values(1, 1, 16, 8)
            return cache, cache[:, :, :1]

        x4, x1 = torch.ones(1, 4, 1, 8).cuda(), torch.ones(1, 1, 1, 8).cuda()
        pos = torch.tensor([3], device="cuda")
        reader = _compile(self._Reader(), [*separate(), x4, pos])
        slot = _compile(
            self._SlotReader(), [values(1, 1, 16, 8), values(1, 1, 1, 8), x1, pos]
        )
        two = _compile(self._TwoCaches(), [*separate(), x4, pos])
        self.assertEqual(len(_aliased_io(reader)), 1)
        self.assertEqual(len(_aliased_io(slot)), 1)
        self.assertEqual(len(_aliased_io(two)), 2)
        if cudagraphs:
            # One call each, since a recorded graph keeps the cache's address.
            accepted = [a_view_of_the_cache]
        else:
            accepted = [
                separate,
                disjoint_halves,
                a_view_of_the_cache,
                a_strided_view_of_the_cache,
            ]
        with contextlib.ExitStack() as stack:
            stack.enter_context(calls())
            if cudagraphs:
                reader, slot, two = (
                    stack.enter_context(torch_tensorrt.runtime.enable_cudagraphs(m))
                    for m in (reader, slot, two)
                )
            for make in accepted:
                with self.subTest(make.__name__):
                    self._call(self._Reader(), reader, make, x4, pos)
            with self.subTest("a_slot_the_engine_does_not_write"):
                self._call(
                    self._SlotReader(), slot, a_slot_the_engine_does_not_write, x1, pos
                )
            with self.subTest("two caches that share memory"):
                cache = torch.randn(1, 4, 16, 8, device="cuda")
                with self.assertRaisesRegex(RuntimeError, "share memory"):
                    two(cache, cache, x4, pos)

    def test_the_cpp_runtime(self):
        self._check()

    def test_cuda_graphs(self):
        self._check(cudagraphs=True)

    def test_the_python_runtime(self):
        with _python_runtime():
            self._check(calls=_python_engine_op)

    def test_an_input_a_plugin_updates_may_be_another_input(self):
        """An alias declared for a plugin is not a KV cache: ``x.add_(y)`` with the
        same tensor for ``x`` and ``y`` reads both before it writes."""
        import tensorrt as trt
        from torch_tensorrt.dynamo._settings import CompilationSettings
        from torch_tensorrt.dynamo.runtime import TorchTensorRTModule
        from torch_tensorrt.dynamo.runtime._TRTEngine import TRTEngine

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        added = network.add_elementwise(
            network.add_input("x", trt.float32, (16,)),
            network.add_input("y", trt.float32, (16,)),
            trt.ElementWiseOperation.SUM,
        ).get_output(0)
        added.name = "updated_x"
        network.mark_output(added)
        plan = bytes(
            builder.build_serialized_network(network, builder.create_builder_config())
        )
        module = TorchTensorRTModule(
            plan,
            ["x", "y"],
            ["updated_x"],
            name="user_alias_add",
            settings=CompilationSettings(device=torch_tensorrt.Device("cuda:0")),
            aliased_io={"updated_x": ("x", "user")},
            num_user_outputs=1,
        )
        engine = TRTEngine(module._pack_engine_info())
        for name, run in (
            ("cpp", module),
            ("python", lambda x, y: engine.execute((x, y))),
        ):
            with self.subTest(name):
                x = torch.arange(1.0, 17.0, device="cuda")
                out = run(x, x)
                out = out[0] if isinstance(out, (tuple, list)) else out
                torch.testing.assert_close(out, torch.arange(1.0, 17.0).cuda() * 2)

    def test_an_input_a_plugin_updates_may_not_share_a_cache(self):
        """The engine updates both in place, so a private copy of the plugin's input
        would take its update: the call is refused. Separate tensors still work."""
        import numpy as np
        import tensorrt as trt
        from torch_tensorrt.dynamo._settings import CompilationSettings
        from torch_tensorrt.dynamo.runtime import TorchTensorRTModule
        from torch_tensorrt.dynamo.runtime._TRTEngine import TRTEngine

        builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
        network = builder.create_network(0)
        cache = network.add_input("cache", trt.float32, (1, 4, 16, 8))
        reader = network.add_input("reader", trt.float32, (1, 4, 16, 8))
        update = network.add_input("update", trt.float32, (1, 4, 1, 8))
        pos = network.add_input("pos", trt.int32, (1,))
        written = network.add_kv_cache_update(
            cache, update, pos, trt.KVCacheMode.LINEAR
        ).get_output(0)
        written.name = "cache_out"
        two = network.add_constant(
            (1, 1, 1, 1), np.array([2], dtype=np.float32)
        ).get_output(0)
        doubled = network.add_elementwise(
            reader, two, trt.ElementWiseOperation.PROD
        ).get_output(0)
        doubled.name = "reader_out"
        total = network.add_elementwise(
            network.add_reduce(written, trt.ReduceOperation.SUM, 15, True).get_output(
                0
            ),
            network.add_reduce(doubled, trt.ReduceOperation.SUM, 15, True).get_output(
                0
            ),
            trt.ElementWiseOperation.SUM,
        ).get_output(0)
        total.name = "result"
        for tensor in (total, written, doubled):
            network.mark_output(tensor)
        plan = bytes(
            builder.build_serialized_network(network, builder.create_builder_config())
        )
        module = TorchTensorRTModule(
            plan,
            ["cache", "reader", "update", "pos"],
            ["result", "cache_out", "reader_out"],
            name="kv_and_user_alias",
            settings=CompilationSettings(device=torch_tensorrt.Device("cuda:0")),
            aliased_io={
                "cache_out": ("cache", "kv_cache_update"),
                "reader_out": ("reader", "user"),
            },
            num_user_outputs=1,
        )
        engine = TRTEngine(module._pack_engine_info())
        for name, run in (
            ("cpp", module),
            ("python", lambda *args: engine.execute(args)),
        ):
            for shared in (False, True):
                with self.subTest(name, shared=shared):
                    c = torch.full((1, 4, 16, 8), 3.0, device="cuda")
                    r = c if shared else c.clone()
                    args = (
                        c,
                        r,
                        torch.full((1, 4, 1, 8), 7.0, device="cuda"),
                        torch.tensor([2], dtype=torch.int32, device="cuda"),
                    )
                    if shared:
                        with self.assertRaisesRegex(RuntimeError, "share memory"):
                            run(*args)
                        continue
                    out = run(*args)
                    out = out[0] if isinstance(out, (tuple, list)) else out
                    torch.testing.assert_close(out, torch.full_like(out, 4736.0))
                    torch.testing.assert_close(r, torch.full_like(r, 6.0))


class _Decoder(torch.nn.Module):
    """Two attention layers that write their key and value caches at one position
    per call, as a decoder does for each new token."""

    def __init__(self, dim=64, heads=4, length=16, layers=2):
        super().__init__()
        self.heads, self.head_dim, self.length = heads, dim // heads, length
        for i in range(layers):
            for name in ("k", "v"):
                self.register_buffer(
                    f"{name}{i}", torch.zeros(1, heads, length, dim // heads)
                )
        self.qkv = torch.nn.ModuleList(
            torch.nn.Linear(dim, 3 * dim) for _ in range(layers)
        )
        self.out = torch.nn.ModuleList(torch.nn.Linear(dim, dim) for _ in range(layers))

    def forward(self, h, pos):
        mask = torch.arange(self.length, device=h.device)[None, :] <= pos[:, None]
        for i, (qkv, out) in enumerate(zip(self.qkv, self.out)):
            q, k, v = (
                t.view(1, 1, self.heads, self.head_dim).transpose(1, 2)
                for t in qkv(h).chunk(3, dim=-1)
            )
            k_cache, v_cache = getattr(self, f"k{i}"), getattr(self, f"v{i}")
            k_cache.index_copy_(2, pos, k)
            v_cache.index_copy_(2, pos, v)
            a = torch.nn.functional.scaled_dot_product_attention(
                q, k_cache, v_cache, attn_mask=mask[None, None]
            )
            h = h + out(a.transpose(1, 2).reshape(1, 1, -1))
        return h


class TestDecoderCachesStayInPlace(TestCase):
    """Every cache of a decoder is updated in place, also when attention, which
    reads only the updated caches, is sent to PyTorch. The engine then hands the
    updated caches to attention in their own memory, without a copy."""

    def _check(self, shared=(), **kwargs):
        torch.manual_seed(0)
        model = _Decoder().cuda().eval()
        eager = copy.deepcopy(model)

        def step(i):
            g = torch.Generator().manual_seed(i)
            return torch.randn(1, 1, 64, generator=g).cuda(), torch.tensor(
                [i], device="cuda"
            )

        compiled = _compile(model, step(0), **kwargs)
        aliased = [
            buffer
            for _, mod in compiled.named_modules()
            for buffer, _ in (getattr(mod, "aliased_io", None) or {}).values()
        ]
        self.assertEqual(sorted(aliased), ["buf_k0", "buf_k1", "buf_v0", "buf_v1"])
        self.assertFalse(compiled.meta.get("_copyback_mutation_buffers"))
        engine_outputs = []
        for mod in compiled.modules():
            if hasattr(mod, "aliased_io"):
                mod.register_forward_hook(
                    lambda _m, _i, out: engine_outputs.extend(
                        o.data_ptr()
                        for o in (out if isinstance(out, (tuple, list)) else [out])
                    )
                )
        for i in range(4):
            with torch.no_grad():
                want = eager(*step(i))
            out = compiled(*step(i))
            out = out[0] if isinstance(out, (tuple, list)) else out
            torch.testing.assert_close(out, want, rtol=1e-3, atol=1e-3)
            for name in ("k0", "v0", "k1", "v1"):
                torch.testing.assert_close(
                    getattr(compiled, name), getattr(eager, name), rtol=1e-3, atol=1e-3
                )
        for name in shared:
            self.assertIn(getattr(compiled, name).data_ptr(), engine_outputs)

    def test_a_decoder_built_into_one_engine(self):
        self._check()

    def test_a_decoder_with_attention_in_pytorch(self):
        self._check(
            shared=("k0", "v0", "k1", "v1"),
            torch_executed_ops={
                "torch.ops.aten.scaled_dot_product_attention.default",
                "torch.ops.aten._scaled_dot_product_efficient_attention.default",
                "torch.ops.aten._scaled_dot_product_flash_attention.default",
            },
        )


class TestFullCachePrefillStaysInPlace(TestCase):
    """A prefill that fills the whole cache, then attends over it, is updated in
    place: attention reads only the updated cache, even though its output has the
    cache's shape."""

    def test_a_full_capacity_prefill(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("k", torch.zeros(1, 4, 16, 8))
                self.register_buffer("v", torch.zeros(1, 4, 16, 8))

            def forward(self, q, k, v, pos):
                index = pos + torch.arange(k.shape[2], device=k.device)
                self.k.index_copy_(2, index, k)
                self.v.index_copy_(2, index, v)
                return torch.softmax(q @ self.k.transpose(-1, -2), -1) @ self.v

        torch.manual_seed(0)
        model = M().cuda()
        eager = copy.deepcopy(model)
        args = [torch.randn(1, 4, 16, 8, device="cuda") for _ in range(3)]
        args.append(torch.tensor([0], device="cuda"))
        compiled = _compile(model, args, disable_tf32=True)
        aliased = sorted(
            buffer
            for _, mod in compiled.named_modules()
            for buffer, _ in (getattr(mod, "aliased_io", None) or {}).values()
        )
        self.assertEqual(aliased, ["buf_k", "buf_v"])
        out = compiled(*args)
        out = out[0] if isinstance(out, (tuple, list)) else out
        torch.testing.assert_close(out, eager(*args), rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(compiled.k, eager.k)


class TestHostTracedBuffer(TestCase):
    """ExecuTorch exports a model traced on the host, so its cache is a host tensor
    at compile time. The write is still applied in place, and the compiled module
    holds the cache on the GPU, so calls from PyTorch update it too."""

    def test_a_host_traced_cache_is_updated_in_place(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(1, 4, 16, 8))

            def forward(self, x, pos):
                self.cache.index_copy_(2, pos, x)
                return self.cache.sum(2)

        def step(i):
            g = torch.Generator().manual_seed(i)
            return torch.randn(1, 4, 1, 8, generator=g), torch.tensor([2 + i])

        eager = M()
        compiled = _compile(M(), step(0))
        self.assertEqual(len(_aliased_io(compiled)), 1)
        self.assertTrue(compiled.cache.is_cuda)
        for i in range(3):
            out = compiled(*step(i))
            out = out[0] if isinstance(out, (tuple, list)) else out
            want = eager(*step(i))
            torch.testing.assert_close(out.cpu(), want)
            torch.testing.assert_close(compiled.cache.cpu(), eager.cache)


class TestCompileKeepsOneCopyOfTheCache(TestCase):
    """The compiled module holds its own copy of a cache it updates in place, and
    compiling makes no other copy of it on the GPU."""

    _SHAPE = (1, 8, 1024, 512)
    _BYTES = 8 * 1024 * 512 * 4

    @classmethod
    def _model(cls):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(cls._SHAPE))

            def forward(self, x, pos):
                self.cache.index_copy_(2, pos, x)
                return self.cache[:, :, :4].sum(2)

        return M()

    def _peak_while_compiling(self, model, args):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.memory_allocated()
        compiled = _compile(model, args)
        torch.cuda.synchronize()
        self.assertEqual(len(_aliased_io(compiled)), 1)
        self.assertTrue(compiled.cache.is_cuda)
        return torch.cuda.max_memory_allocated() - start

    def test_a_host_traced_cache(self):
        args = (torch.ones(1, 8, 1, 512), torch.tensor([3]))
        peak = self._peak_while_compiling(self._model(), args)
        self.assertLess(peak, 1.5 * self._BYTES)

    def test_a_cache_traced_on_the_gpu(self):
        args = (
            torch.ones(1, 8, 1, 512, device="cuda"),
            torch.tensor([3], device="cuda"),
        )
        peak = self._peak_while_compiling(self._model().cuda(), args)
        self.assertLess(peak, 1.5 * self._BYTES)


class TestEngineCacheNamesTheAliasedInput(TestCase):
    """A reused engine updates the input it was built for, by name, so a model whose
    buffer has another name gets its own engine."""

    def test_a_renamed_buffer_builds_its_own_engine(self):
        from torch_tensorrt.dynamo._engine_cache import BaseEngineCache

        class MemoryCache(BaseEngineCache):
            def __init__(self):
                self.data = {}

            def save(self, key, blob, *args, **kwargs):
                self.data[key] = blob

            def load(self, key, *args, **kwargs):
                return self.data.get(key)

        class M(torch.nn.Module):
            def __init__(self, name):
                super().__init__()
                self.name = name
                self.register_buffer(name, torch.zeros(1, 4, 16, 8, device="cuda"))

            def forward(self, x, pos):
                cache = getattr(self, self.name)
                cache.index_copy_(2, pos, x)
                return cache.sum(2)

        engines = MemoryCache()
        args = (torch.ones(1, 4, 1, 8, device="cuda"), torch.tensor([2], device="cuda"))
        for name in ("cache", "state"):
            compiled = _compile(
                M(name),
                args,
                cache_built_engines=True,
                reuse_cached_engines=True,
                immutable_weights=False,
                custom_engine_cache=engines,
            )
            out = compiled(*args)
            out = out[0] if isinstance(out, (tuple, list)) else out
            torch.testing.assert_close(out, M(name)(*args))
        self.assertEqual(len(engines.data), 2)

    def test_binding_names_a_caller_sets_build_their_own_engine(self):
        """A reused engine keeps the names it was built with, and the check that the
        cache is updated in place looks for the new ones."""
        import tensorrt as trt
        from torch_tensorrt.dynamo._engine_cache import BaseEngineCache

        class MemoryCache(BaseEngineCache):
            def __init__(self):
                self.data = {}

            def save(self, key, blob, *args, **kwargs):
                self.data[key] = blob

            def load(self, key, *args, **kwargs):
                return self.data.get(key)

        class Writer(torch.nn.Module):
            def forward(self, cache, x, pos):
                cache.index_copy_(2, pos, x)
                return cache.sum(2)

        engines = MemoryCache()
        args = (
            torch.zeros(1, 4, 16, 8, device="cuda"),
            torch.ones(1, 4, 1, 8, device="cuda"),
            torch.tensor([3], device="cuda"),
        )
        for prefix in ("a", "b"):
            plan = (
                torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
                    export(Writer(), args),
                    arg_inputs=list(args),
                    min_block_size=1,
                    immutable_weights=False,
                    cache_built_engines=True,
                    reuse_cached_engines=True,
                    custom_engine_cache=engines,
                    arg_input_binding_names=(
                        f"{prefix}_cache",
                        f"{prefix}_x",
                        f"{prefix}_pos",
                    ),
                )
            )
            engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(
                bytes(plan)
            )
            self.assertEqual(engine.get_tensor_name(0), f"{prefix}_cache")
        self.assertEqual(len(engines.data), 2)

    def test_a_cache_the_caller_receives_has_its_own_key(self):
        """Such a cache output gets its own copy in the engine, one the caller does
        not receive keeps the cache's memory."""
        from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
        from torch_tensorrt.dynamo._settings import CompilationSettings

        def key(received):
            graph = torch.fx.Graph()
            cache, x, pos = (graph.placeholder(n) for n in ("cache", "x", "pos"))
            write = graph.call_function(
                torch.ops.aten.index_copy.default, (cache, 2, pos, x)
            )
            write.meta["_trt_kv_alias"] = True
            if received is not None:
                write.meta["_trt_reaches_caller"] = received
            graph.output((write,))
            gm = torch.fx.GraphModule(torch.nn.Module(), graph)
            return BaseEngineCache.get_hash(gm, [], CompilationSettings())

        self.assertNotEqual(key(True), key(False))
        self.assertEqual(key(True), key(None))


class TestWithoutTheKVCacheLayer(TestCase):
    """TensorRT 10.3, used for JetPack, has no IKVCacheUpdateLayer, so a cache write
    is copied back there instead of updated in place."""

    def test_cache_writes_are_copied_back(self):
        from unittest import mock

        from torch_tensorrt.dynamo.conversion.impl import slice_scatter

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(1, 4, 16, 8, device="cuda"))
                self.register_buffer("other", torch.zeros(1, 4, 16, 8, device="cuda"))

            def forward(self, x, pos):
                self.cache.index_copy_(2, pos, x)
                self.other[:, :, 5:6, :] = x
                return self.cache.sum(2) + self.other.sum(2)

        model = M().cuda()
        eager = copy.deepcopy(model)
        args = (torch.ones(1, 4, 1, 8, device="cuda"), torch.tensor([2], device="cuda"))
        with mock.patch.object(
            slice_scatter, "kv_cache_update_supported", return_value=False
        ):
            compiled = _compile(model, args)
        self.assertEqual(_aliased_io(compiled), {})
        self.assertEqual(
            sorted(compiled.meta.get("_copyback_mutation_buffers")), ["cache", "other"]
        )
        for step in range(3):
            x = torch.full((1, 4, 1, 8), float(step + 1), device="cuda")
            pos = torch.tensor([step], device="cuda")
            out = compiled(x, pos)
            out = out[0] if isinstance(out, (tuple, list)) else out
            torch.testing.assert_close(out, eager(x, pos))
            torch.testing.assert_close(compiled.cache, eager.cache)

    def test_a_caller_cache_write_is_written_back(self):
        """The write is not done in place, so its write-back stores it in the
        caller's tensor."""
        from unittest import mock

        from torch_tensorrt.dynamo.conversion.impl import slice_scatter

        class M(torch.nn.Module):
            def forward(self, cache, x, pos):
                cache.index_copy_(2, pos, x)
                return cache.sum(2)

        args = (
            torch.zeros(1, 4, 16, 8, device="cuda"),
            torch.ones(1, 4, 1, 8, device="cuda"),
            torch.tensor([2], device="cuda"),
        )
        with mock.patch.object(
            slice_scatter, "kv_cache_update_supported", return_value=False
        ):
            compiled = _compile(M(), [a.clone() for a in args])
        self.assertEqual(_aliased_io(compiled), {})
        given, expected = [a.clone() for a in args], [a.clone() for a in args]
        out = compiled(*given)
        out = out[0] if isinstance(out, (tuple, list)) else out
        torch.testing.assert_close(out, M()(*expected))
        torch.testing.assert_close(given[0], expected[0])


if __name__ == "__main__":
    run_tests()
