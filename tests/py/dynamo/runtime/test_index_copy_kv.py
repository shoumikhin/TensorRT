# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# type: ignore
"""End-to-end tests for ``aten.index_copy`` KV-cache aliasing.

Two converters are registered for ``aten.index_copy.default``:

* ``aten_ops_index_copy_kv`` — HIGH priority, validator-gated. Fires for
  the narrow KV-eligible case (4-D static cache, dim=2, source seq dim
  size 1, batch=1) and emits ``IKVCacheUpdateLayer`` with aliased I/O.

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
from torch.export import export
from torch.testing._internal.common_utils import TestCase, run_tests


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


if __name__ == "__main__":
    run_tests()
