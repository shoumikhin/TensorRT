# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# type: ignore
"""Unit tests for ``lift_mutated_buffers`` and ``inline_lifted_buffers_into_gm``.

``lift_mutated_buffers`` is a pre-compile rewrite that detects mutated
buffers (the trailing ``aten.copy_(get_attr, _)`` pattern that
``ExportedProgram.module()`` generates for each ``BUFFER_MUTATION``) and
lifts each one from a ``get_attr`` to a ``placeholder``. The rebuilt
GraphModule's ``forward`` signature reflects the new placeholder set —
which requires resetting the graph's ``_codegen`` from the
``_PyTreeCodeGen`` baked in by ``ep.module()`` to the default ``CodeGen``.

These tests verify:

* Buffers ARE lifted when mutated.
* Buffers are NOT lifted when only read.
* The rebuilt GraphModule's ``forward`` accepts the new placeholders.
* The rebuilt GraphModule produces the same outputs as the original
  pre-lift gm when both are given the same inputs (buffers + user inputs).
* The original buffer tensors are returned alongside the placeholder
  names for downstream wiring.
* ``inline_lifted_buffers_into_gm`` rewrites the lifted-buffer
  placeholders into ``get_attr`` reads and registers the buffers as
  module state. The result is a plain ``fx.GraphModule`` that
  serializes via ``torch_tensorrt.save`` without an external wrapper.
"""

import inspect
import io
import sys
import unittest
from unittest import mock

import torch
from torch.export import export
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering._buffer_lifting import (
    aliased_input_bindings,
    assert_predicted_kv_aliased,
    erase_export_guards,
    hide_copyback_outputs,
    inline_lifted_buffers_into_gm,
    lift_mutated_buffers,
)


def _ep_module_decomposed(model, args):
    """Run the prefix of the compile pipeline up through ``ep.module()``."""
    ep = export(model, tuple(args))
    ep = ep.run_decompositions({})
    return ep.module()


def _ep_module_on_cuda(model, args):
    """``_ep_module_decomposed`` for a model whose buffers live on the GPU, as an
    engine needs them to update one in place."""
    return _ep_module_decomposed(model.cuda(), tuple(a.cuda() for a in args))


_needs_cuda = unittest.skipIf(not torch.cuda.is_available(), "CUDA required")


class TestLiftMutatedBuffers(TestCase):
    def test_no_mutation_no_lift(self):
        """A module that reads buffers but doesn't mutate them returns
        ``(gm, [])`` — no rewrite happens."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("table", torch.arange(16, dtype=torch.float32))

            def forward(self, x):
                return x + self.table.sum()

        gm = _ep_module_decomposed(M(), (torch.zeros(4),))
        new_gm, lifted = lift_mutated_buffers(gm)
        self.assertEqual(lifted, [])
        # The same gm is returned when nothing is lifted.
        self.assertIs(new_gm, gm)

    def test_single_buffer_lifted(self):
        """A buffer that's mutated should be lifted to a placeholder, the
        trailing copy_ removed, and the rebuilt forward should accept it
        as an argument."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(2, 4, 16, 8))

            def forward(self, x):
                self.cache[:, :, 3:4, :] = x
                return self.cache.sum()

        gm = _ep_module_decomposed(M(), (torch.ones(2, 4, 1, 8),))
        new_gm, lifted = lift_mutated_buffers(gm)

        # Exactly one buffer was lifted.
        self.assertEqual(len(lifted), 1)
        ph_name, buf_name, tensor = lifted[0]
        self.assertEqual(buf_name, "cache")
        self.assertEqual(tuple(tensor.shape), (2, 4, 16, 8))
        self.assertEqual(ph_name, "buf_cache")

        # The rebuilt forward should now accept (x, buf_cache).
        sig = inspect.signature(new_gm.forward)
        param_names = list(sig.parameters.keys())
        self.assertEqual(param_names, ["x", "buf_cache"])

        # No get_attr nodes for `cache` remain in the graph.
        for node in new_gm.graph.nodes:
            if node.op == "get_attr":
                self.assertNotEqual(node.target, "cache")
        # No trailing aten.copy_ to the (now removed) cache get_attr.
        for node in new_gm.graph.nodes:
            self.assertNotEqual(node.target, torch.ops.aten.copy_.default)

    def test_nested_buffer_lifted(self):
        """A buffer owned by a submodule should be lifted too.

        ``get_attr`` targets are fully qualified, so this one arrives as
        ``inner.cache``. Resolving it with ``getattr`` reports it as missing and
        skips the rewrite, which leaves the mutation in place."""

        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(2, 4, 16, 8))

            def forward(self, x):
                self.cache[:, :, 3:4, :] = x
                return self.cache.sum()

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Inner()

            def forward(self, x):
                return self.inner(x)

        gm = _ep_module_decomposed(M(), (torch.ones(2, 4, 1, 8),))
        new_gm, lifted = lift_mutated_buffers(gm)

        self.assertEqual(len(lifted), 1)
        ph_name, buf_name, tensor = lifted[0]
        self.assertEqual(buf_name, "inner.cache")
        self.assertEqual(tuple(tensor.shape), (2, 4, 16, 8))
        self.assertEqual(ph_name, "buf_inner_cache")

        sig = inspect.signature(new_gm.forward)
        self.assertEqual(list(sig.parameters.keys()), ["x", "buf_inner_cache"])

        for node in new_gm.graph.nodes:
            self.assertNotEqual(node.target, torch.ops.aten.copy_.default)

    def test_paired_buffers_lifted(self):
        """Two mutated buffers are both lifted; placeholders appear in a
        stable order so callers can match them positionally."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache_k", torch.zeros(2, 4, 16, 8))
                self.register_buffer("cache_v", torch.zeros(2, 4, 16, 8))

            def forward(self, x_k, x_v):
                self.cache_k[:, :, 3:4, :] = x_k
                self.cache_v[:, :, 3:4, :] = x_v
                return self.cache_k.sum() + self.cache_v.sum()

        gm = _ep_module_decomposed(
            M(), (torch.ones(2, 4, 1, 8), torch.ones(2, 4, 1, 8))
        )
        new_gm, lifted = lift_mutated_buffers(gm)
        self.assertEqual(len(lifted), 2)
        buf_names = {b for _, b, _ in lifted}
        self.assertEqual(buf_names, {"cache_k", "cache_v"})

        # forward signature should have all 4 params (2 user + 2 buffer).
        sig = inspect.signature(new_gm.forward)
        self.assertEqual(len(sig.parameters), 4)

    def test_rebuilt_forward_matches_original(self):
        """The rebuilt GraphModule, when given (user_args..., buffers...),
        should produce the same outputs as the original ep.module() when
        given the same user_args (with buffers used from internal state)."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(2, 4, 16, 8))

            def forward(self, x):
                self.cache[:, :, 3:4, :] = x
                return self.cache.sum() * 2.0

        x = torch.randn(2, 4, 1, 8)
        gm_original = _ep_module_decomposed(M(), (x.clone(),))
        original_out = gm_original(x.clone())
        # ep.module() returns its outputs as a tuple. Take the first element
        # to compare against the rebuilt gm (whose default CodeGen returns
        # tuples too, but possibly with a different surrounding shape).
        if isinstance(original_out, tuple):
            original_out = original_out[0]

        # Re-create gm for the lift (in-place mutation of the first gm's
        # graph would change its forward behavior).
        gm_for_lift = _ep_module_decomposed(M(), (x.clone(),))
        new_gm, lifted = lift_mutated_buffers(gm_for_lift)
        _, _, buf_tensor = lifted[0]
        # Call rebuilt gm with the original buffer state.
        new_out = new_gm(x.clone(), buf_tensor.clone())
        if isinstance(new_out, tuple):
            new_out = new_out[0]
        self.assertTrue(torch.allclose(new_out, original_out))


class TestInlineLiftedBuffers(TestCase):
    """``inline_lifted_buffers_into_gm`` should register each lifted
    buffer as module state on the gm and rewrite the corresponding
    placeholder node into a ``get_attr`` read. After inlining, the gm's
    forward should accept only the user inputs."""

    def _build_simple_gm(self):
        """Construct an fx GraphModule with two placeholders (x, buf) and
        a body that sums them, matching what ``lift_mutated_buffers``
        would produce."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        buf = graph.placeholder("buf_cache")
        out = graph.call_function(torch.add, args=(x, buf))
        graph.output(out)
        gm = torch.fx.GraphModule({}, graph)
        gm.recompile()
        return gm

    def test_inline_registers_buffer_and_rewrites_placeholder(self):
        gm = self._build_simple_gm()
        buf_tensor = torch.tensor([1.0, 2.0, 3.0])

        new_gm = inline_lifted_buffers_into_gm(
            gm, lifted_buffers=[("buf_cache", "cache", buf_tensor)]
        )

        # Buffer registered as module state.
        self.assertTrue(hasattr(new_gm, "cache"))
        self.assertTrue(torch.allclose(new_gm.cache, buf_tensor))

        # Placeholder count is now 1 (only `x`); buffer is a get_attr.
        placeholders = [n for n in new_gm.graph.nodes if n.op == "placeholder"]
        self.assertEqual(len(placeholders), 1)
        self.assertEqual(placeholders[0].name, "x")
        get_attrs = [n for n in new_gm.graph.nodes if n.op == "get_attr"]
        self.assertEqual(len(get_attrs), 1)
        self.assertEqual(get_attrs[0].target, "cache")

        # forward(x) computes x + cache via the inlined get_attr.
        x = torch.tensor([10.0, 20.0, 30.0])
        out = new_gm(x)
        if isinstance(out, tuple):
            out = out[0]
        self.assertTrue(torch.allclose(out, x + buf_tensor))

    def test_inline_is_noop_for_empty_lifted(self):
        gm = self._build_simple_gm()
        ph_before = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
        result = inline_lifted_buffers_into_gm(gm, lifted_buffers=[])
        self.assertIs(result, gm)
        ph_after = [n.name for n in result.graph.nodes if n.op == "placeholder"]
        self.assertEqual(ph_before, ph_after)

    def test_inline_preserves_user_input_order(self):
        """When multiple buffers are inlined, the user inputs come first
        and are unchanged; the buffers become get_attr reads."""
        graph = torch.fx.Graph()
        u1 = graph.placeholder("u1")
        u2 = graph.placeholder("u2")
        b1 = graph.placeholder("buf_a")
        b2 = graph.placeholder("buf_b")
        s1 = graph.call_function(torch.add, args=(u1, b1))
        s2 = graph.call_function(torch.add, args=(u2, b2))
        out = graph.call_function(torch.add, args=(s1, s2))
        graph.output(out)
        gm = torch.fx.GraphModule({}, graph)
        gm.recompile()

        new_gm = inline_lifted_buffers_into_gm(
            gm,
            lifted_buffers=[
                ("buf_a", "a", torch.tensor(1.0)),
                ("buf_b", "b", torch.tensor(2.0)),
            ],
        )
        placeholders = [n.name for n in new_gm.graph.nodes if n.op == "placeholder"]
        self.assertEqual(placeholders, ["u1", "u2"])
        # Numerical: (10 + 1) + (20 + 2) = 33
        out = new_gm(torch.tensor(10.0), torch.tensor(20.0))
        if isinstance(out, tuple):
            out = out[0]
        self.assertEqual(out.item(), 33.0)

    def test_inline_remaps_copyback_targets_for_nested_buffers(self):
        """A nested (dotted) buffer is registered under a flattened
        ``lifted_buf_*`` name (``register_buffer`` rejects "."), so the recorded
        copy-back targets -- which still use the original dotted name -- must be
        remapped through the same mapping. Otherwise the ``BUFFER_MUTATION``
        OutputSpec names a buffer that no longer exists and the ExportedProgram
        verifier rejects the program. A flat name maps to itself and is unchanged.
        """
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        b_state = graph.placeholder("buf_state")
        b_nested = graph.placeholder("buf_nested")
        s = graph.call_function(torch.add, args=(x, b_state))
        out = graph.call_function(torch.add, args=(s, b_nested))
        graph.output(out)
        gm = torch.fx.GraphModule({}, graph)
        gm.recompile()
        # lift_mutated_buffers records copy-back targets under the buffers'
        # original names (dotted for a submodule-owned cache).
        gm.meta["_copyback_mutation_buffers"] = ["state", "layers.0.attn.k_cache"]

        new_gm = inline_lifted_buffers_into_gm(
            gm,
            lifted_buffers=[
                ("buf_state", "state", torch.zeros(4)),
                ("buf_nested", "layers.0.attn.k_cache", torch.zeros(4)),
            ],
        )

        registered = dict(new_gm.named_buffers())
        # Flat name kept; nested name flattened to a valid attribute.
        self.assertIn("state", registered)
        self.assertIn("lifted_buf_layers_0_attn_k_cache", registered)
        # Copy-back targets remapped: flat unchanged, nested -> flattened name.
        self.assertEqual(
            new_gm.meta["_copyback_mutation_buffers"],
            ["state", "lifted_buf_layers_0_attn_k_cache"],
        )
        # Every recorded target now names a buffer that actually exists.
        for name in new_gm.meta["_copyback_mutation_buffers"]:
            self.assertIn(name, registered)


class TestCopyBackClassification(TestCase):
    """``lift_mutated_buffers`` splits mutated buffers into two kinds.

    An *eligible* ``slice_scatter`` / ``index_copy`` KV write (one the converter
    lowers to an ``IKVCacheUpdateLayer`` with in-place aliased I/O) relies on that
    engine aliasing and is NOT recorded for copy-back. Every other mutation --
    including a ``slice_scatter`` / ``index_copy`` that fails the converter's
    eligibility (wrong rank/dim/shape), whichever of the converter's ineligible
    paths it then takes -- a non-aliasing scatter, an outright return of the source,
    or a raise -- has no engine aliasing, so its new value is re-appended as a trailing graph
    output and its buffer name recorded in
    ``gm.meta['_copyback_mutation_buffers']`` for the exporters to reclassify as
    a BUFFER_MUTATION.
    """

    def test_kv_slice_scatter_write_no_copyback(self):
        """A slice-assignment KV write is aliased downstream, not copied back."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(2, 4, 16, 8))

            def forward(self, x):
                self.cache[:, :, 3:4, :] = x
                return self.cache.sum()

        gm = _ep_module_decomposed(M(), (torch.ones(2, 4, 1, 8),))
        new_gm, lifted = lift_mutated_buffers(gm)
        self.assertEqual(len(lifted), 1)
        self.assertEqual(new_gm.meta["_copyback_mutation_buffers"], [])
        # Predicted-KV binding recorded so compile() can assert it actually aliases.
        self.assertEqual(new_gm.meta["_predicted_kv_bindings"], ["buf_cache"])

    def test_kv_index_copy_write_no_copyback(self):
        """An eligible ``index_copy`` KV write (4-D static cache, dim=2, batch 1,
        single-position source) is aliased by the KV converter, so it is not
        copied back."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(1, 4, 16, 8))

            def forward(self, x):
                self.cache.index_copy_(2, torch.tensor([3]), x)
                return self.cache.sum()

        gm = _ep_module_decomposed(M(), (torch.ones(1, 4, 1, 8),))
        new_gm, lifted = lift_mutated_buffers(gm)
        self.assertEqual(len(lifted), 1)
        self.assertEqual(new_gm.meta["_copyback_mutation_buffers"], [])

    def test_non_kv_mutation_recorded_for_copyback(self):
        """A non-KV in-place mutation is recorded for copy-back by buffer name."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(4))

            def forward(self, x):
                self.state.add_(x)
                return self.state.sum()

        gm = _ep_module_decomposed(M(), (torch.ones(4),))
        new_gm, lifted = lift_mutated_buffers(gm)
        self.assertEqual(len(lifted), 1)
        self.assertEqual(new_gm.meta["_copyback_mutation_buffers"], ["state"])
        # A non-KV mutation is not predicted to alias, so nothing to assert later.
        self.assertEqual(new_gm.meta["_predicted_kv_bindings"], [])

    def test_copyback_value_appended_as_last_output(self):
        """The non-KV new value is re-attached as a trailing graph output so it
        survives DCE; at the lift stage it is the LAST output and equals the
        updated buffer."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(4))

            def forward(self, x):
                self.state.add_(x)
                return self.state.sum()

        x = torch.arange(4, dtype=torch.float32)
        gm = _ep_module_decomposed(M(), (x.clone(),))
        new_gm, lifted = lift_mutated_buffers(gm)
        _, buf_name, buf_tensor = lifted[0]
        self.assertEqual(buf_name, "state")

        out = new_gm(x.clone(), buf_tensor.clone())
        self.assertIsInstance(out, tuple)
        # Last output is the copy-back value == state + x.
        self.assertTrue(torch.allclose(out[-1], buf_tensor + x))

    def test_index_put_is_copyback_not_kv(self):
        """Regression: ``index_put`` has no aliasing converter, so it must fall
        into copy-back rather than being dropped in expectation of aliasing."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(4, 8))

            def forward(self, x):
                self.state[torch.tensor([1, 3])] = x
                return self.state.sum()

        gm = _ep_module_decomposed(M(), (torch.ones(2, 8),))
        new_gm, lifted = lift_mutated_buffers(gm)
        self.assertEqual(len(lifted), 1)
        self.assertEqual(new_gm.meta["_copyback_mutation_buffers"], ["state"])

    def test_mixed_kv_and_copyback(self):
        """One KV buffer + one non-KV buffer: only the non-KV one is recorded."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(2, 4, 16, 8))
                self.register_buffer("state", torch.zeros(4))

            def forward(self, x_kv, x_state):
                self.cache[:, :, 3:4, :] = x_kv
                self.state.add_(x_state)
                return self.cache.sum() + self.state.sum()

        gm = _ep_module_decomposed(M(), (torch.ones(2, 4, 1, 8), torch.ones(4)))
        new_gm, lifted = lift_mutated_buffers(gm)
        self.assertEqual(len(lifted), 2)
        self.assertEqual(new_gm.meta["_copyback_mutation_buffers"], ["state"])

    def test_ineligible_index_copy_is_copyback(self):
        """Regression: an ``index_copy`` the KV converter cannot alias (here a 2-D
        cache / dim 0, not the 4-D dim=2 layout ``IKVCacheUpdateLayer`` requires)
        is lowered to a non-aliasing scatter, so its write-back must be preserved
        as copy-back rather than dropped."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(4, 8))

            def forward(self, x):
                self.cache.index_copy_(0, torch.tensor([2]), x)
                return self.cache.sum()

        gm = _ep_module_decomposed(M(), (torch.ones(1, 8),))
        new_gm, lifted = lift_mutated_buffers(gm)
        self.assertEqual(len(lifted), 1)
        self.assertEqual(new_gm.meta["_copyback_mutation_buffers"], ["cache"])

    def test_ineligible_slice_scatter_is_copyback(self):
        """Regression: a ``slice_scatter`` on a KV-shaped 4-D cache but the wrong
        axis (dim 1, not dim 2) is not IKVCacheUpdateLayer-eligible, so it is
        lowered to a non-aliasing scatter and must fall to copy-back rather than
        being dropped."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(2, 16, 4, 8))

            def forward(self, x):
                self.cache[:, 3:4, :, :] = x  # write on dim 1, not the seq dim 2
                return self.cache.sum()

        gm = _ep_module_decomposed(M(), (torch.ones(2, 1, 4, 8),))
        new_gm, lifted = lift_mutated_buffers(gm)
        self.assertEqual(len(lifted), 1)
        self.assertEqual(new_gm.meta["_copyback_mutation_buffers"], ["cache"])

    def test_strided_slice_scatter_is_copyback(self):
        """A strided write cannot use the contiguous KV-cache layer, so buffer
        lifting must preserve it through copy-back instead of predicting an alias."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(2, 4, 16, 8))

            def forward(self, x):
                self.cache[:, :, 0:16:2, :] = x
                return self.cache.sum()

        gm = _ep_module_decomposed(M(), (torch.ones(2, 4, 8, 8),))
        new_gm, lifted = lift_mutated_buffers(gm)
        self.assertEqual(len(lifted), 1)
        self.assertEqual(new_gm.meta["_copyback_mutation_buffers"], ["cache"])
        self.assertEqual(new_gm.meta["_predicted_kv_bindings"], [])

    def test_torch_executed_write_is_copyback_not_kv(self):
        """A KV-shaped write whose op the caller excluded from TensorRT cannot reach a
        converter, so no IKVCacheUpdateLayer can alias it.

        It must still be lifted and copied back. Dropping the buffer instead would
        leave the copy_ with no users, and post_lowering's dead-node pass would erase
        it, silently losing the write."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(2, 4, 16, 8))

            def forward(self, x):
                self.cache[:, :, 3:4, :] = x
                return self.cache.sum()

        args = (torch.ones(2, 4, 1, 8),)
        # Without the exclusion the same write is predicted to alias inside the engine.
        gm = _ep_module_decomposed(M(), args)
        baseline_gm, baseline_lifted = lift_mutated_buffers(gm)
        self.assertEqual(len(baseline_lifted), 1)
        self.assertEqual(baseline_gm.meta["_copyback_mutation_buffers"], [])
        self.assertEqual(len(baseline_gm.meta["_predicted_kv_bindings"]), 1)

        gm = _ep_module_decomposed(M(), args)
        settings = CompilationSettings(
            torch_executed_ops={"torch.ops.aten.slice_scatter.default"}
        )
        new_gm, lifted = lift_mutated_buffers(gm, settings)
        self.assertEqual(len(lifted), 1)
        self.assertEqual(new_gm.meta["_copyback_mutation_buffers"], ["cache"])
        self.assertEqual(new_gm.meta["_predicted_kv_bindings"], [])

        # The write survives as a real graph output, so it cannot be dead-code
        # eliminated the way an orphaned copy_ would be.
        out_node = next(n for n in new_gm.graph.nodes if n.op == "output")
        self.assertEqual(len(out_node.args[0]), 2)
        erasable = [
            n
            for n in new_gm.graph.nodes
            if n is not out_node and not n.users and n.all_input_nodes
        ]
        self.assertEqual(erasable, [])


class TestSliceScatterDerivationIsShared(TestCase):
    """The predictor and the converter must derive ``_kv_eligible``'s arguments
    the same way.

    ``_kv_write_will_alias`` reuses the converter's eligibility predicate, but a
    predicate is only as good as what it is handed: computing ``start`` /
    ``update_len`` differently on the two sides mispredicts just as effectively
    as a different predicate would. ``resolve_slice_scatter_write`` is the single
    derivation both sides call, and these pin the corners where an independent
    derivation would part company with the converter.
    """

    @staticmethod
    def _classify(cache_shape, src_shape, *slice_args):
        """Route a hand-built ``slice_scatter`` through the predictor."""
        from torch_tensorrt.dynamo.lowering._buffer_lifting import _kv_write_will_alias

        graph = torch.fx.Graph()
        cache = graph.placeholder("cache")
        cache.meta["val"] = torch.empty(cache_shape, device="meta")
        src = graph.placeholder("src")
        src.meta["val"] = torch.empty(src_shape, device="meta")
        node = graph.call_function(
            torch.ops.aten.slice_scatter.default, args=(cache, src, *slice_args)
        )
        return _kv_write_will_alias(node, tuple(cache_shape))

    def test_full_overwrite_is_not_kv(self):
        """The converter short-circuits a full overwrite by returning the source,
        so it emits no KV layer and nothing aliases the cache."""
        self.assertFalse(self._classify((2, 4, 16, 8), (2, 4, 16, 8), 2, 0, 16))
        # Same slice written implicitly (no start/end args).
        self.assertFalse(self._classify((2, 4, 16, 8), (2, 4, 16, 8), 2))

    def test_open_ended_slice_is_clamped_to_the_dim(self):
        """``cache[:, :, 3:, :]`` lowers with ``end == INT64_MAX``, clamped to the dim
        as aten does, leaving a KV-eligible 13-slot write. Both sides have to clamp, or
        the predictor files copy-back for a write the converter goes on to alias."""
        for open_end in (sys.maxsize, 9223372036854775807):
            self.assertTrue(
                self._classify((2, 4, 16, 8), (2, 4, 13, 8), 2, 3, open_end)
            )

    def test_negative_start_is_normalised(self):
        """A negative ``start`` counts from the end for the converter, so the predictor
        has to normalise it the same way. No slice can fail the ``start + update_len <=
        s_max`` bound now that ``end`` is clamped; it still guards ``index_copy``."""
        # -4 normalises to 12; the write is the 4 slots from 12 to s_max.
        self.assertTrue(self._classify((2, 4, 16, 8), (2, 4, 4, 8), 2, -4, 16))
        # An end past the dim is clamped back to it, leaving the same 4-slot write.
        self.assertTrue(self._classify((2, 4, 16, 8), (2, 4, 4, 8), 2, -4, 20))

    def test_non_int_start_is_not_kv(self):
        """A non-constant bound makes the converter raise rather than emit a KV
        layer, so it cannot be predicted to alias. Reading it as ``start = 0``
        would predict aliasing for a write that never gets converted at all."""
        graph = torch.fx.Graph()
        pos = graph.placeholder("pos")
        self.assertFalse(self._classify((2, 4, 16, 8), (2, 4, 1, 8), 2, pos, 4))

    def test_step_is_returned_unchanged(self):
        """``step`` passes through the shared derivation untouched, which is what
        both sides need: the derivation reads it for the full-overwrite shortcut, the
        shared eligibility predicate rejects non-unit steps, and the scatter fallback
        uses the stride to build its indices."""
        from torch_tensorrt.dynamo.conversion.impl.slice_scatter import (
            KVWriteStatus,
            resolve_slice_scatter_write,
        )

        shape = (2, 4, 16, 8)
        for step in (1, 2, 3):
            self.assertEqual(
                resolve_slice_scatter_write(shape, 2, 0, 8, step),
                (0, 8, step, KVWriteStatus.OK),
            )

    def test_strided_write_is_not_kv_eligible(self):
        """Only the contiguous write is classified for KV aliasing; strided writes
        must use the scatter fallback, which writes the requested non-adjacent slots."""
        self.assertTrue(self._classify((2, 4, 16, 8), (2, 4, 8, 8), 2, 0, 8, 1))
        self.assertFalse(self._classify((2, 4, 16, 8), (2, 4, 4, 8), 2, 0, 8, 2))
        self.assertFalse(self._classify((2, 4, 16, 8), (2, 4, 3, 8), 2, 0, 8, 3))

    def test_an_integer_cache_is_not_kv_eligible(self):
        """IKVCacheUpdateLayer takes floating-point caches only."""
        from torch_tensorrt.dynamo.lowering._buffer_lifting import _kv_write_will_alias

        graph = torch.fx.Graph()
        cache = graph.placeholder("cache")
        cache.meta["val"] = torch.empty((2, 4, 16, 8), dtype=torch.int32, device="meta")
        src = graph.placeholder("src")
        src.meta["val"] = torch.empty((2, 4, 1, 8), dtype=torch.int32, device="meta")
        node = graph.call_function(
            torch.ops.aten.slice_scatter.default, args=(cache, src, 2, 3, 4)
        )
        self.assertFalse(_kv_write_will_alias(node, (2, 4, 16, 8)))

    def test_resolve_reports_the_converter_early_exits(self):
        """Each status comes back with the bounds its contract promises, since a
        caller reads the bounds on the strength of the status alone."""
        from torch_tensorrt.dynamo.conversion.impl.slice_scatter import (
            KVWriteStatus,
            resolve_slice_scatter_write,
        )

        shape = (2, 4, 16, 8)
        self.assertEqual(
            resolve_slice_scatter_write(shape, 2, 3, 4, 1), (3, 4, 1, KVWriteStatus.OK)
        )
        # Defaults filled in and negative indices counted from the end.
        self.assertEqual(
            resolve_slice_scatter_write(shape, 2, -4, None, None),
            (12, 16, 1, KVWriteStatus.OK),
        )
        # An end past the dim is clamped to it, as aten does.
        self.assertEqual(
            resolve_slice_scatter_write(shape, 2, -4, 20, 1),
            (12, 16, 1, KVWriteStatus.OK),
        )
        self.assertEqual(
            resolve_slice_scatter_write(shape, 2, None, None, None),
            (0, 16, 1, KVWriteStatus.FULL_OVERWRITE),
        )

        class _StepEqualToOne:
            """A step that is not an ``int`` but compares equal to 1, which is what a
            symbolic step out of a dynamic-shape trace can be."""

            def __eq__(self, other):
                return other == 1

        self.assertEqual(
            resolve_slice_scatter_write(shape, 2, 0, 16, _StepEqualToOne()),
            (0, 16, 1, KVWriteStatus.FULL_OVERWRITE),
        )
        # Nothing resolved, so nothing is reported as resolved.
        self.assertEqual(
            resolve_slice_scatter_write(shape, 2, "sym", 4, 1),
            (None, None, None, KVWriteStatus.DYNAMIC_BOUNDS),
        )
        self.assertEqual(
            resolve_slice_scatter_write(shape, 9, 0, 4, 1),
            (None, None, None, KVWriteStatus.BAD_DIM),
        )

    def test_a_dynamic_dim_leaves_every_bound_unresolved(self):
        """Reading TensorRT's -1 as a size is how ``cache[:, :, 3:]`` used to resolve to
        ``arange(3, -2)``, an empty write. Both -1 and a ``SymInt`` stand-in are
        checked, since the two callers have to read them the same way."""
        from torch_tensorrt.dynamo.conversion.impl.slice_scatter import (
            KVWriteStatus,
            resolve_slice_scatter_write,
        )

        unresolved = (None, None, None, KVWriteStatus.DYNAMIC_DIM_SIZE)
        for dynamic_size in (-1, "sym"):
            shape = (2, 4, dynamic_size, 8)
            for start, end in (
                (3, sys.maxsize),
                (3, 9223372036854775807),
                (3, None),
                (None, None),
                (-4, None),
                (-4, 12),
                (0, 40),
                (1, 5),
            ):
                self.assertEqual(
                    resolve_slice_scatter_write(shape, 2, start, end, 1), unresolved
                )


class TestCacheMustReachTheConverterAsANetworkInput(TestCase):
    """``emit_kv_cache_update_layer`` aliases the cache only when it is handed a
    network input, so the classifier has to predict what the converter will be handed
    -- after ``post_lowering``, not what the graph holds at classification time.

    Both directions are damaging. Reading the cache through an op that survives means
    no KV layer, so a write called KV loses its write-back. A ``clone`` that
    ``remove_input_alias_fixing_clones`` then erases means the KV layer *is* emitted,
    so a write called copy-back adds a trailing output the runtime truncates as an
    engine side effect while the outer graph still reads it.
    """

    CACHE = (2, 4, 16, 8)
    SRC = (2, 4, 1, 8)

    @classmethod
    def _classify(cls, build_cache_arg, extra_reader=False):
        """Route a KV-eligible ``slice_scatter`` whose cache argument is whatever
        ``build_cache_arg`` returns through the predictor."""
        from torch_tensorrt.dynamo.lowering._buffer_lifting import _kv_write_will_alias

        graph = torch.fx.Graph()
        buffer = graph.placeholder("buf_cache")
        buffer.meta["val"] = torch.empty(cls.CACHE, device="meta")
        cache_arg = build_cache_arg(graph, buffer)
        if extra_reader:
            graph.call_function(torch.ops.aten.sum.default, args=(buffer,))
        src = graph.placeholder("src")
        src.meta["val"] = torch.empty(cls.SRC, device="meta")
        node = graph.call_function(
            torch.ops.aten.slice_scatter.default, args=(cache_arg, src, 2, 3, 4)
        )
        return _kv_write_will_alias(node, cls.CACHE)

    @staticmethod
    def _classify_index_copy(through_clone, extra_reader=False):
        """Same, for ``index_copy``, whose converter applies its own placeholder
        check to ``args[0]`` rather than relying on the binding name."""
        from torch_tensorrt.dynamo.lowering._buffer_lifting import _kv_write_will_alias

        cache_shape, src_shape = (1, 4, 16, 8), (1, 4, 1, 8)
        graph = torch.fx.Graph()
        buffer = graph.placeholder("buf_cache")
        buffer.meta["val"] = torch.empty(cache_shape, device="meta")
        cache_arg = buffer
        if through_clone:
            cache_arg = graph.call_function(
                torch.ops.aten.clone.default, args=(buffer,)
            )
        if extra_reader:
            graph.call_function(torch.ops.aten.sum.default, args=(buffer,))
        index = graph.placeholder("index")
        index.meta["val"] = torch.empty((1,), dtype=torch.int64, device="meta")
        src = graph.placeholder("src")
        src.meta["val"] = torch.empty(src_shape, device="meta")
        node = graph.call_function(
            torch.ops.aten.index_copy.default, args=(cache_arg, 2, index, src)
        )
        return _kv_write_will_alias(node, cache_shape)

    def test_direct_placeholder_is_kv(self):
        self.assertTrue(self._classify(lambda _graph, buffer: buffer))

    def test_index_copy_through_a_sole_use_clone_is_kv(self):
        """``_index_copy_kv_eligible`` checks ``args[0]`` itself, so peeling the
        clone for the binding-name question is not enough -- the validator has to be
        pointed at the same node, or an ``index_copy`` decode write through a clone
        is filed copy-back and the engine aliases it anyway."""
        self.assertTrue(self._classify_index_copy(through_clone=False))
        self.assertTrue(self._classify_index_copy(through_clone=True))

    def test_index_copy_through_a_shared_clone_is_not_kv(self):
        self.assertFalse(
            self._classify_index_copy(through_clone=True, extra_reader=True)
        )

    def test_index_copy_validator_still_reads_args0_by_default(self):
        """The partitioner passes no override and must go on seeing the node's own
        input: by the time it runs, lowering has settled what that is."""
        from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
            _index_copy_kv_eligible,
        )

        graph = torch.fx.Graph()
        buffer = graph.placeholder("buf_cache")
        buffer.meta["val"] = torch.empty((1, 4, 16, 8), device="meta")
        clone = graph.call_function(torch.ops.aten.clone.default, args=(buffer,))
        clone.meta["val"] = torch.empty((1, 4, 16, 8), device="meta")
        index = graph.placeholder("index")
        src = graph.placeholder("src")
        src.meta["val"] = torch.empty((1, 4, 1, 8), device="meta")
        node = graph.call_function(
            torch.ops.aten.index_copy.default, args=(clone, 2, index, src)
        )
        self.assertFalse(_index_copy_kv_eligible(node))
        self.assertTrue(_index_copy_kv_eligible(node, input_node=buffer))

    def test_clone_of_a_sole_use_placeholder_is_kv(self):
        """``clone -> scatter -> copy_`` is the ordinary shape when the post-write
        cache is read in the same forward. Lifting erases the ``copy_``, leaving the
        clone as the placeholder's only user, which is exactly the condition
        ``remove_input_alias_fixing_clones`` erases it under -- so the converter does
        see a network input and does alias."""
        self.assertTrue(
            self._classify(
                lambda graph, buffer: graph.call_function(
                    torch.ops.aten.clone.default, args=(buffer,)
                )
            )
        )

    def test_clone_of_a_shared_placeholder_is_not_kv(self):
        """``remove_input_alias_fixing_clones`` only erases a clone that is its
        placeholder's sole user, so a cache something else also reads keeps its clone
        and the converter never sees a network input."""
        self.assertFalse(
            self._classify(
                lambda graph, buffer: graph.call_function(
                    torch.ops.aten.clone.default, args=(buffer,)
                ),
                extra_reader=True,
            )
        )

    def test_clone_of_a_non_placeholder_is_not_kv(self):
        self.assertFalse(
            self._classify(
                lambda graph, buffer: graph.call_function(
                    torch.ops.aten.clone.default,
                    args=(
                        graph.call_function(
                            torch.ops.aten.mul.Tensor, args=(buffer, 1.0)
                        ),
                    ),
                )
            )
        )

    def test_cache_read_through_another_op_is_not_kv(self):
        """Nothing erases an arbitrary op between the placeholder and the write, so
        the converter is handed a ``call_function`` with no input binding name, falls
        back to a plain scatter, and aliases nothing. Without this the shape and dim
        alone say KV-eligible and the write-back is dropped for an aliasing that never
        happens."""
        self.assertFalse(
            self._classify(
                lambda graph, buffer: graph.call_function(
                    torch.ops.aten.mul.Tensor, args=(buffer, 1.0)
                )
            )
        )

    def test_cache_read_from_a_get_attr_is_not_kv(self):
        """An unlifted buffer is a ``get_attr``, which constant-folds into the engine
        rather than becoming an input binding."""
        self.assertFalse(self._classify(lambda graph, _buffer: graph.get_attr("cache")))


class _FakeEngine:
    """Stand-in for a compiled TRT submodule exposing an ``aliased_io`` map."""

    def __init__(self, aliased_io):
        self.aliased_io = aliased_io


class _FakeGM:
    """Stand-in for a compiled GraphModule whose children are TRT engines."""

    def __init__(self, children):
        self._children = children

    def named_children(self):
        return list(self._children.items())


class TestPredictedKvAssertion(TestCase):
    """`assert_predicted_kv_aliased` is the ground-truth backstop for the
    pre-conversion KV prediction: every write predicted to alias must actually
    appear in a compiled engine's `aliased_io`, else its write-back would be
    silently dropped."""

    @staticmethod
    def _aliased_in(gm):
        """Ground truth the way ``compile()`` assembles it, off the compiled
        submodules."""
        return aliased_input_bindings(
            getattr(sub, "aliased_io", None) for _name, sub in gm.named_children()
        )

    def test_passes_when_predicted_kv_is_aliased(self):
        gm = _FakeGM(
            {
                "_run_on_acc_0": _FakeEngine(
                    {"out_k": ("buf_k_cache", "kv_cache_update")}
                )
            }
        )
        # buf_k_cache is aliased -> no error.
        assert_predicted_kv_aliased(self._aliased_in(gm), ["buf_k_cache"])

    def test_raises_when_predicted_kv_not_aliased(self):
        # Predicted KV for buf_conv_state, but the engine aliased only buf_k_cache
        # (the converter emitted no IKVCacheUpdateLayer for conv_state) -> must
        # raise rather than silently drop the write-back.
        gm = _FakeGM(
            {
                "_run_on_acc_0": _FakeEngine(
                    {"out_k": ("buf_k_cache", "kv_cache_update")}
                )
            }
        )
        with self.assertRaises(RuntimeError):
            assert_predicted_kv_aliased(self._aliased_in(gm), ["buf_conv_state"])

    def test_aggregates_aliased_io_across_engines(self):
        gm = _FakeGM(
            {
                "_run_on_acc_0": _FakeEngine(
                    {"out_k": ("buf_k_cache", "kv_cache_update")}
                ),
                "_run_on_acc_1": _FakeEngine(
                    {"out_v": ("buf_v_cache", "kv_cache_update")}
                ),
            }
        )
        # Both predicted-KV bindings are aliased across the two engines -> no error.
        assert_predicted_kv_aliased(
            self._aliased_in(gm), ["buf_k_cache", "buf_v_cache"]
        )

    def test_noop_when_no_prediction(self):
        assert_predicted_kv_aliased(self._aliased_in(_FakeGM({})), [])

    def test_no_engines_built_does_not_raise(self):
        """With no engine built, aliased_io is empty for every prediction whatever
        the predictions were worth, so there is nothing to check and raising would
        fail a run on the strength of its own absence of evidence."""
        assert_predicted_kv_aliased(
            self._aliased_in(_FakeGM({})),
            ["buf_k_cache"],
            engines_built=False,
        )

    def test_dryrun_alone_does_not_skip(self):
        """``dryrun`` is not what the skip keys on. The engine converter accepts it,
        never acts on it and builds an engine regardless, so treating it as "no
        engines" there would turn a check the converter deliberately has no opt-out
        for into one with an opt-out kwarg."""
        with self.assertRaises(RuntimeError):
            assert_predicted_kv_aliased(
                self._aliased_in(_FakeGM({})),
                ["buf_k_cache"],
                CompilationSettings(dryrun=True),
            )

    def test_raises_when_a_copyback_write_is_aliased(self):
        """The other direction. A copy-back write carries its new value as a trailing
        graph output; if the engine aliases that buffer anyway the runtime truncates
        the output as a side effect while the outer graph still reads it, and the
        module raises on every call. ``predicted_kv_bindings`` is empty in that case,
        so the KV direction returns at its early exit and sees nothing."""
        gm = _FakeGM(
            {"_run_on_acc_0": _FakeEngine({"output1": ("buf_k", "kv_cache_update")})}
        )
        with self.assertRaisesRegex(RuntimeError, "aliased them in place anyway"):
            assert_predicted_kv_aliased(
                self._aliased_in(gm), [], copyback_bindings=["buf_k"]
            )

    def test_passes_when_a_copyback_write_is_not_aliased(self):
        gm = _FakeGM(
            {"_run_on_acc_0": _FakeEngine({"output1": ("buf_k", "kv_cache_update")})}
        )
        assert_predicted_kv_aliased(
            self._aliased_in(gm), ["buf_k"], copyback_bindings=["buf_state"]
        )

    def test_no_engines_built_does_not_raise_for_copyback_either(self):
        assert_predicted_kv_aliased(
            self._aliased_in(_FakeGM({})),
            [],
            copyback_bindings=["buf_k"],
            engines_built=False,
        )

    def test_message_names_min_block_size(self):
        """The classification runs before partitioning, so the usual cause is the
        partitioner rejecting the subgraph the write landed in. Without the value in
        force and the remedy, there is nothing in the error to act on."""
        with self.assertRaises(RuntimeError) as caught:
            assert_predicted_kv_aliased(
                self._aliased_in(_FakeGM({})),
                ["buf_k_cache"],
                CompilationSettings(min_block_size=5),
            )
        message = str(caught.exception)
        self.assertIn("min_block_size (5)", message)
        self.assertIn("min_block_size=1", message)

    def test_message_does_not_offer_min_block_size_1_when_it_is_already_1(self):
        """The remedy has to be one the reader can act on, and at ``min_block_size=1``
        blaming the value and recommending it are the same sentence. So this branch
        names the cause and mentions ``min_block_size`` nowhere."""
        with self.assertRaises(RuntimeError) as caught:
            assert_predicted_kv_aliased(
                self._aliased_in(_FakeGM({})),
                ["buf_k_cache"],
                CompilationSettings(min_block_size=1),
            )
        message = str(caught.exception)
        self.assertIn("a converter or a capability validator rejected the op", message)
        self.assertNotIn("min_block_size", message)


class _Block(torch.nn.Module):
    """A submodule a test can name in ``torch_executed_modules``."""

    def forward(self, cache, x):
        return cache.mean(2, keepdim=True) * x


class _Writer(torch.nn.Module):
    """A submodule that writes the cache, for ``torch_executed_modules``."""

    def forward(self, cache, x, pos):
        cache.index_copy_(2, pos, x)


@_needs_cuda
class TestInPlaceRule(TestCase):
    """A buffer write is left to the engine to apply in place only when nothing else
    can see the cache change too early or too late. Everything else is copied back,
    and only a write applied in place is marked for the converters to alias."""

    @staticmethod
    def _lift(model, args, **settings):
        gm = _ep_module_on_cuda(model, args)
        return lift_mutated_buffers(
            gm, CompilationSettings(**settings) if settings else None
        )[0]

    def _assert_copied_back(self, gm):
        self.assertEqual(gm.meta["_copyback_mutation_buffers"], ["cache"])
        self.assertEqual(gm.meta["_predicted_kv_bindings"], [])
        writes = [
            n
            for n in gm.graph.nodes
            if n.target
            in (torch.ops.aten.index_copy.default, torch.ops.aten.slice_scatter.default)
        ]
        self.assertTrue(writes)
        self.assertFalse(any(n.meta.get("_trt_kv_alias") for n in writes))

    @staticmethod
    def _cache_model(forward):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(1, 4, 16, 8))
                self.block = _Block()
                self.writer = _Writer()

        M.forward = forward
        return M()

    _args = (torch.ones(1, 4, 1, 8), torch.tensor([3]))

    def test_a_reader_the_write_depends_on_is_copied_back(self):
        """TensorRT can run a layer that reads the cache after the KV layer that
        updates it, even when that layer feeds the update."""

        def forward(self, x, pos):
            src = x + self.cache.mean(2, keepdim=True)
            self.cache.index_copy_(2, pos, src)
            return self.cache.sum()

        self._assert_copied_back(self._lift(self._cache_model(forward), self._args))

    def test_an_independent_reader_is_copied_back(self):
        """The reader reads the slot the write fills, and nothing orders it first."""

        def forward(self, x, pos):
            old = self.cache[:, :, 3:4, :] * 2
            self.cache.index_copy_(2, pos, x)
            return self.cache.sum() + old.sum()

        self._assert_copied_back(self._lift(self._cache_model(forward), self._args))

    def test_a_reader_through_a_view_is_copied_back(self):
        def forward(self, x, pos):
            view = self.cache.view(1, 4, 16, 8)
            old = view * 2
            src = x + view.mean(2, keepdim=True)
            self.cache.index_copy_(2, pos, src)
            return self.cache - old

        self._assert_copied_back(self._lift(self._cache_model(forward), self._args))

    def test_a_reader_through_a_reshape_is_copied_back(self):
        """The lowering decompositions turn ``view`` into ``_reshape_copy``, which
        TensorRT can turn into a no-op that reads the cache itself."""

        def forward(self, x, pos):
            view = torch.ops.aten._reshape_copy.default(self.cache, [1, 4, 16, 8])
            old = view * 2
            src = x + view.mean(2, keepdim=True)
            self.cache.index_copy_(2, pos, src)
            return self.cache - old

        self._assert_copied_back(self._lift(self._cache_model(forward), self._args))

    def test_a_reader_of_a_clone_is_copied_back(self):
        """A converter returns a clone as its input, so the clone is the cache."""

        def forward(self, x, pos):
            old = self.cache.clone()
            self.cache.index_copy_(2, pos, x)
            return self.cache - old

        self._assert_copied_back(self._lift(self._cache_model(forward), self._args))

    def test_a_returned_clone_keeps_it_in_place(self):
        """The clone reads only the new cache. The engine gives the returned value
        its own memory (see ``test_index_copy_kv``)."""

        def forward(self, x, pos):
            self.cache.index_copy_(2, pos, x)
            return self.cache.clone()

        gm = self._lift(self._cache_model(forward), self._args)
        self.assertEqual(gm.meta["_predicted_kv_bindings"], ["buf_cache"])
        self.assertEqual(gm.meta["_copyback_mutation_buffers"], [])

    def test_a_write_after_another_write_is_copied_back(self):
        """The KV layer updates only a network input, so the second write cannot
        be applied in place, and the first must not be either."""

        def forward(self, x, pos):
            self.cache.index_copy_(2, pos, x)
            self.cache.index_copy_(2, pos, x + self.cache.sum(2, keepdim=True))
            return self.cache.sum()

        self._assert_copied_back(self._lift(self._cache_model(forward), self._args))

    def test_a_reader_in_an_excluded_module_is_copied_back(self):
        """The reader runs before the write, but in PyTorch, and the engine that
        writes in place cannot order itself against it."""

        def forward(self, x, pos):
            scale = self.block(self.cache, x)
            self.cache.index_copy_(2, pos, x + scale)
            return self.cache.sum()

        self._assert_copied_back(
            self._lift(
                self._cache_model(forward),
                self._args,
                torch_executed_modules={f"{_Block.__module__}.{_Block.__qualname__}"},
            )
        )

    def test_a_write_in_an_excluded_module_is_copied_back(self):
        def forward(self, x, pos):
            self.writer(self.cache, x, pos)
            return self.cache.sum()

        self._assert_copied_back(
            self._lift(
                self._cache_model(forward),
                self._args,
                torch_executed_modules={f"{_Writer.__module__}.{_Writer.__qualname__}"},
            )
        )

    def test_an_old_value_reader_in_pytorch_is_copied_back(self):
        def forward(self, x, pos):
            src = x + torch.sigmoid(self.cache).mean(2, keepdim=True)
            self.cache.index_copy_(2, pos, src)
            return self.cache.sum()

        self._assert_copied_back(
            self._lift(
                self._cache_model(forward),
                self._args,
                torch_executed_ops={"torch.ops.aten.sigmoid.default"},
            )
        )

    def test_a_new_value_reader_in_pytorch_keeps_it_in_place(self):
        """An op that reads only the updated cache runs after the engine, wherever
        it runs, as attention does when it is sent to PyTorch."""

        def forward(self, x, pos):
            self.cache.index_copy_(2, pos, x)
            return torch.sigmoid(self.cache).sum()

        gm = self._lift(
            self._cache_model(forward),
            self._args,
            torch_executed_ops={"torch.ops.aten.sigmoid.default"},
        )
        self.assertEqual(gm.meta["_predicted_kv_bindings"], ["buf_cache"])
        self.assertEqual(gm.meta["_copyback_mutation_buffers"], [])

    def test_a_reader_of_a_split_part_is_copied_back(self):
        """``split`` hands back a tuple, and one part is read through it."""

        def forward(self, x, pos):
            whole = self.cache.split(16, dim=2)[0]
            old = whole * 2
            src = x + whole.mean(2, keepdim=True)
            self.cache.index_copy_(2, pos, src)
            return self.cache - old

        self._assert_copied_back(self._lift(self._cache_model(forward), self._args))

    def test_a_reader_of_a_zero_pad_is_copied_back(self):
        """TensorRT turns a pad of zero into its input, so the pad is the cache."""

        def forward(self, x, pos):
            padded = torch.nn.functional.pad(self.cache, (0, 0, 0, 0))
            src = x + padded.mean(2, keepdim=True)
            self.cache.index_copy_(2, pos, src)
            return self.cache.sum() + padded.sum()

        self._assert_copied_back(self._lift(self._cache_model(forward), self._args))

    def test_a_reader_of_a_copy_into_a_new_tensor_is_copied_back(self):
        def forward(self, x, pos):
            old = torch.empty_like(self.cache).copy_(self.cache)
            src = x + old.mean(2, keepdim=True)
            self.cache.index_copy_(2, pos, src)
            return self.cache.sum() + old.sum()

        self._assert_copied_back(self._lift(self._cache_model(forward), self._args))

    def test_a_metadata_check_keeps_it_in_place(self):
        """``.to()`` to the cache's own type leaves only a check of its metadata in
        the graph, which never reads the data, so it does not need ordering."""

        def forward(self, x, pos):
            same = self.cache.to(torch.float32)
            self.cache.index_copy_(2, pos, x)
            return self.cache.sum() + same.shape[2]

        gm = self._lift(self._cache_model(forward), self._args)
        self.assertEqual(gm.meta["_predicted_kv_bindings"], ["buf_cache"])

    def test_a_float64_buffer_is_copied_back(self):
        """With ``truncate_double`` the engine is handed a float32 copy, which
        would take the write instead of the buffer."""

        def forward(self, x, pos):
            self.cache[:, :, 3:4, :] = x
            return self.cache.sum()

        model = self._cache_model(forward).double()
        self._assert_copied_back(
            self._lift(
                model, (self._args[0].double(), self._args[1]), truncate_double=True
            )
        )

    def test_a_buffer_that_is_not_contiguous_is_copied_back(self):
        """The runtime hands the engine a contiguous copy, which would take the
        write instead of the buffer."""

        def forward(self, x, pos):
            self.cache.index_copy_(2, pos, x)
            return self.cache.sum()

        model = self._cache_model(forward).cuda()
        model.cache = model.cache.contiguous(memory_format=torch.channels_last)
        self._assert_copied_back(self._lift(model, self._args))

    def test_a_host_buffer_is_written_in_place_on_the_engine_device(self):
        """A model traced on the host, as ExecuTorch exports it, keeps its in-place
        write. The buffer the compiled module holds moves to the engine's device,
        so the engine is handed the buffer and not a per-call copy of it."""

        def forward(self, x, pos):
            self.cache.index_copy_(2, pos, x)
            return self.cache.sum()

        gm, lifted = lift_mutated_buffers(
            _ep_module_decomposed(self._cache_model(forward), self._args)
        )
        self.assertEqual(gm.meta["_predicted_kv_bindings"], ["buf_cache"])
        self.assertEqual(gm.meta["_copyback_mutation_buffers"], [])
        gm = inline_lifted_buffers_into_gm(
            gm, lifted, in_place=["buf_cache"], device=torch.device("cuda")
        )
        self.assertTrue(gm.cache.is_cuda)

    def test_a_functional_write_of_an_input_is_not_in_place(self):
        class M(torch.nn.Module):
            def forward(self, cache, x, pos):
                return cache.index_copy(2, pos, x) * 1.0 + cache

        gm = self._lift(M(), (torch.zeros(1, 4, 16, 8),) + self._args)
        (write,) = [
            n for n in gm.graph.nodes if n.target is torch.ops.aten.index_copy.default
        ]
        self.assertFalse(write.meta.get("_trt_kv_alias"))
        self.assertFalse(gm.meta.get("_predicted_kv_bindings"))

    def _assert_written_back(self, gm):
        """The write is not done in place, so its ``copy_`` stays to store it."""
        self.assertEqual(gm.meta["_predicted_kv_bindings"], [])
        (copy,) = [
            n for n in gm.graph.nodes if n.target is torch.ops.aten.copy_.default
        ]
        self.assertTrue(copy.meta.get("_trt_write_back"))

    def test_an_input_written_in_place_with_another_reader_is_written_back(self):
        class M(torch.nn.Module):
            def forward(self, cache, x):
                old = cache[:, :, 3:4, :] * 2
                cache[:, :, 3:4, :] = x
                return cache.sum() + old.sum()

        self._assert_written_back(
            self._lift(M(), (torch.zeros(1, 4, 16, 8), torch.ones(1, 4, 1, 8)))
        )

    class _InputCache(torch.nn.Module):
        def forward(self, cache, x, pos):
            cache.index_copy_(2, pos, x)
            return cache.sum()

    def test_an_input_on_the_host_is_written_back(self):
        """The engine is handed a per-call copy of a host tensor."""
        args = (torch.zeros(1, 4, 16, 8),) + self._args
        gm, _ = lift_mutated_buffers(_ep_module_decomposed(self._InputCache(), args))
        self._assert_written_back(gm)

    def test_an_input_that_is_not_contiguous_is_written_back(self):
        cache = torch.zeros(1, 16, 4, 8).transpose(1, 2)
        self._assert_written_back(self._lift(self._InputCache(), (cache,) + self._args))

    def test_an_input_written_and_not_read_again_is_written_back(self):
        """The engine updates the cache only as part of a value it computes, and
        nothing here reads the updated value."""

        class M(torch.nn.Module):
            def forward(self, cache, x, pos):
                cache.index_copy_(2, pos, x)
                return x + 1

        self._assert_written_back(
            self._lift(M(), (torch.zeros(1, 4, 16, 8),) + self._args)
        )

    def test_a_strided_write_of_an_input_is_written_back(self):
        class M(torch.nn.Module):
            def forward(self, cache, x):
                cache[:, :, 0:8:2, :] = x
                return cache.sum()

        self._assert_written_back(
            self._lift(M(), (torch.zeros(1, 4, 16, 8), torch.ones(1, 4, 4, 8)))
        )

    def test_an_input_written_and_returned_keeps_its_write_back(self):
        """The write is not done in place, but the model returns the tensor, so the
        ``copy_`` that stores the write is read and stays."""

        class M(torch.nn.Module):
            def forward(self, cache, x):
                cache[:, :, 0:8:2, :] = x
                return cache

        gm = self._lift(M(), (torch.zeros(1, 4, 16, 8), torch.ones(1, 4, 4, 8)))
        self.assertEqual(gm.meta["_predicted_kv_bindings"], [])
        self.assertTrue(
            any(n.target is torch.ops.aten.copy_.default for n in gm.graph.nodes)
        )

    def test_an_input_written_in_place_stays_in_place(self):
        gm = self._lift(self._InputCache(), (torch.zeros(1, 4, 16, 8),) + self._args)
        self.assertEqual(gm.meta["_predicted_kv_bindings"], ["cache"])


@_needs_cuda
class TestInPlaceRuleKeepsSymbolicSizes(TestCase):
    """Deciding a write adds no guard: a dynamic size stays dynamic, and a size known
    only at run time is not asked for its value."""

    def test_a_full_capacity_example_keeps_the_size_dynamic(self):
        """The example fills the whole cache, so the attention output has the cache's
        shape for that example only."""

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

        args = tuple(torch.randn(1, 4, 16, 8, device="cuda") for _ in range(3))
        args += (torch.tensor([0], device="cuda"),)
        T = torch.export.Dim("T", min=1, max=16)
        ep = export(M().cuda(), args, dynamic_shapes=({2: T}, {2: T}, {2: T}, None))
        gm = ep.run_decompositions({}).module()
        (q,) = [n for n in gm.graph.nodes if n.op == "placeholder" and n.name == "q"]
        shape_env = q.meta["val"].shape[2].node.shape_env
        guards = [str(g.expr) for g in shape_env.guards]
        gm, _ = lift_mutated_buffers(gm)
        self.assertEqual([str(g.expr) for g in shape_env.guards], guards)

    def test_a_reader_of_data_dependent_size(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(1, 2, 8, 4))

            def forward(self, k, pos, mask):
                old = self.cache.index_select(2, mask.nonzero().squeeze(1))
                self.cache.index_copy_(2, pos, k)
                return old.sum(2) + self.cache.sum(2)

        args = (torch.ones(1, 2, 1, 4), torch.tensor([3]), torch.tensor([True] * 8))
        gm, _ = lift_mutated_buffers(_ep_module_on_cuda(M(), args))
        self.assertEqual(gm.meta["_copyback_mutation_buffers"], ["cache"])


class TestDeadKvWriteRouting(TestCase):
    """A KV write nothing else consumes is routed to copy-back instead of aliased.

    Erasing its ``copy_`` leaves such a write dead, so it is eliminated before any
    engine can alias it and the prediction can never be fulfilled. Copy-back makes it
    live again by re-attaching its value as a graph output, which is only safe because
    the converter aliases only writes marked ``_trt_kv_alias``: a buffer that is both
    raises on every call.
    """

    @staticmethod
    def _dead_write_gm():
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(2, 4, 16, 8))

            def forward(self, x):
                self.cache[:, :, 3:4, :] = x
                return x.sum() * 2.0

        return _ep_module_decomposed(M(), (torch.ones(2, 4, 1, 8),))

    @staticmethod
    def _live_index_copy_gm():
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(1, 4, 16, 8))

            def forward(self, x):
                self.cache.index_copy_(2, torch.tensor([3]), x)
                return self.cache.sum()

        return _ep_module_decomposed(M(), (torch.ones(1, 4, 1, 8),))

    def test_dead_write_is_copyback_rather_than_predicted_kv(self):
        """The same write with a reader is predicted KV (see
        ``test_kv_slice_scatter_write_no_copyback``); without one it goes to
        copy-back, which is what stops the cross-check failing a compile the merge
        base ran."""
        gm, lifted = lift_mutated_buffers(self._dead_write_gm())
        self.assertEqual(len(lifted), 1)
        self.assertEqual(gm.meta["_copyback_mutation_buffers"], ["cache"])
        self.assertEqual(gm.meta["_predicted_kv_bindings"], [])

    def test_the_rerouted_write_is_not_marked_in_place(self):
        gm, _lifted = lift_mutated_buffers(self._dead_write_gm())
        (node,) = [
            n
            for n in gm.graph.nodes
            if n.target is torch.ops.aten.slice_scatter.default
        ]
        self.assertFalse(node.meta.get("_trt_kv_alias"))

    def test_a_live_kv_write_is_marked_in_place(self):
        gm, _lifted = lift_mutated_buffers(self._live_index_copy_gm())
        (node,) = [
            n for n in gm.graph.nodes if n.target is torch.ops.aten.index_copy.default
        ]
        self.assertTrue(node.meta.get("_trt_kv_alias"))
        self.assertEqual(gm.meta["_predicted_kv_bindings"], ["buf_cache"])

    def test_a_non_kv_write_is_copied_back(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(4))

            def forward(self, x):
                self.state.add_(x)
                return x.sum()

        gm, _lifted = lift_mutated_buffers(_ep_module_decomposed(M(), (torch.ones(4),)))
        self.assertEqual(gm.meta["_copyback_mutation_buffers"], ["state"])

    def test_index_copy_takes_only_a_marked_write(self):
        """``index_copy`` applies its own eligibility check rather than
        ``slice_scatter``'s, so both have to read the mark, or a write left to
        copy-back aliases while its value is also a copy-back output."""
        from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
            _index_copy_kv_eligible,
        )

        gm, _lifted = lift_mutated_buffers(self._live_index_copy_gm())
        node = next(
            n for n in gm.graph.nodes if n.target is torch.ops.aten.index_copy.default
        )
        self.assertTrue(_index_copy_kv_eligible(node))
        del node.meta["_trt_kv_alias"]
        self.assertFalse(_index_copy_kv_eligible(node))


class TestHiddenCopybackOutputs(TestCase):
    """``hide_copyback_outputs`` splits what the graph returns from what ``forward``
    does: the trailing copy-back values stay on the output node so the exporters
    keep finding them and dead-code elimination cannot reach them, while ``forward``
    writes them into their buffers and returns the arity of the model that was
    compiled."""

    @staticmethod
    def _two_output_gm():
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        doubled = graph.call_function(torch.mul, args=(x, 2))
        graph.output((x, doubled))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        gm.register_buffer("state", torch.zeros(3))
        return gm

    def test_hidden_output_is_written_into_its_buffer(self):
        gm = self._two_output_gm()
        hide_copyback_outputs(gm, ["state"])

        x = torch.arange(3, dtype=torch.float32)
        self.assertEqual(gm(x), (x,))
        self.assertEqual(gm.state, x * 2)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        self.assertEqual(len(output_node.args[0]), 2)
        # Nothing is dead, so eliminate_dead_code cannot take the hidden value.
        gm.graph.eliminate_dead_code()
        self.assertEqual(len(output_node.args[0]), 2)

    def test_hiding_nothing_is_a_no_op(self):
        gm = self._two_output_gm()
        hide_copyback_outputs(gm, [])
        x = torch.arange(3, dtype=torch.float32)
        self.assertEqual(len(gm(x)), 2)
        self.assertEqual(gm.state, torch.zeros(3))

    def test_a_value_read_from_another_written_buffer_is_read_first(self):
        """``b`` is written before ``a``, and ``a``'s new value is ``b``'s old one."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        old_b = graph.get_attr("b")
        doubled = graph.call_function(torch.mul, args=(x, 2))
        graph.output((x, doubled, old_b))
        root = torch.nn.Module()
        root.register_buffer("b", torch.ones(3))
        gm = torch.fx.GraphModule(root, graph)
        gm.register_buffer("a", torch.zeros(3))
        hide_copyback_outputs(gm, ["b", "a"])

        x = torch.arange(3, dtype=torch.float32)
        self.assertEqual(gm(x), (x,))
        self.assertEqual(gm.b, x * 2)
        self.assertEqual(gm.a, torch.ones(3))

    def test_a_value_that_is_a_view_of_its_own_buffer_is_read_first(self):
        """``self.state = self.state.t()`` left to PyTorch: the new value is a view of
        the buffer it is written into."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        old_state = graph.get_attr("state")
        transposed = graph.call_function(
            torch.ops.aten.permute.default, args=(old_state, [1, 0])
        )
        graph.output((x, transposed))
        root = torch.nn.Module()
        root.register_buffer("state", torch.arange(9.0).reshape(3, 3))
        gm = torch.fx.GraphModule(root, graph)
        hide_copyback_outputs(gm, ["state"])

        x = torch.zeros(3)
        self.assertEqual(gm(x), (x,))
        self.assertEqual(gm.state, torch.arange(9.0).reshape(3, 3).t())

    def test_an_input_named_like_a_builtin(self):
        """The generated ``forward`` takes the inputs under the model's own names,
        so it must not name a builtin such as ``getattr`` itself."""

        class M(torch.nn.Module):
            def forward(self, getattr):
                return getattr, getattr * 2

        gm = torch.export.export(M(), (torch.zeros(3),)).module()
        self.assertIn(
            "getattr", [n.target for n in gm.graph.nodes if n.op == "placeholder"]
        )
        gm.register_buffer("state", torch.zeros(3))
        hide_copyback_outputs(gm, ["state"])

        x = torch.arange(3, dtype=torch.float32)
        self.assertEqual(gm(x), (x,))
        self.assertEqual(gm.state, x * 2)

    def test_graph_consumers_still_see_the_hidden_value(self):
        """Only ``forward`` changes. A non-strict ``torch.export`` runs a GraphModule
        through ``fx.Interpreter``, which reads the output node and never runs the
        write, so a retrace carries the copy-back value into the program as an output
        for the exporter downstream to reclassify, and does not record the write a
        second time."""
        gm = self._two_output_gm()
        hide_copyback_outputs(gm, ["state"])

        x = torch.arange(3, dtype=torch.float32)
        self.assertEqual(len(torch.fx.Interpreter(gm).run(x)), 2)
        self.assertEqual(gm.state, torch.zeros(3))


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestCompileSeam(TestCase):
    """End-to-end coverage of the wire between the two halves of the design.

    ``lift_mutated_buffers`` classifies each write before partitioning and
    ``assert_predicted_kv_aliased`` re-checks that classification after conversion.
    Every other test in this file drives one half directly, so the two could be
    disconnected in ``_compiler.py`` -- the copy-back list never reaching
    ``trt_gm.meta``, or the cross-check never being called -- without a single
    failure. These go through ``compile()`` so that wiring is observed.
    """

    @staticmethod
    def _compile(model, args, **kwargs):
        import torch_tensorrt

        ep = torch.export.export(model, args, strict=False)
        return torch_tensorrt.dynamo.compile(
            ep,
            inputs=list(args),
            cache_built_engines=False,
            reuse_cached_engines=False,
            **kwargs,
        )

    @staticmethod
    def _kv_model(batch):
        """A model and inputs whose single-position cache write lift classifies as
        engine-aliased, so it drops the ``copy_`` and the cross-check has something
        to be right or wrong about."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(batch, 4, 16, 8).cuda())

            def forward(self, x):
                self.cache[:, :, 3:4, :] = x
                return self.cache.sum()

        return M().cuda().eval(), (torch.ones(batch, 4, 1, 8).cuda(),)

    def test_compile_threads_copyback_buffers_into_trt_gm_meta(self):
        """The copy-back list ``lift_mutated_buffers`` records has to survive
        ``compile_module`` and land on the compiled module's meta -- that is the only
        channel by which ``save()`` learns which trailing outputs to reclassify as
        BUFFER_MUTATIONs."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(8).cuda())

            def forward(self, x):
                self.state.add_(x)
                return self.state.sum()

        x = torch.arange(8, dtype=torch.float32).cuda()
        trt_gm = self._compile(M().cuda().eval(), (x,), min_block_size=1)

        self.assertEqual(trt_gm.meta.get("_copyback_mutation_buffers"), ["state"])
        # The recorded name must resolve to a buffer that still exists, else the
        # ExportedProgram verifier rejects the BUFFER_MUTATION downstream.
        self.assertIn("state", dict(trt_gm.named_buffers()))

        # The copy-back value is a real trailing graph output carrying the post-write
        # buffer, not a stale read. It is hidden from the module's return, so read it
        # the way anything that walks the graph does.
        outs = torch.fx.Interpreter(trt_gm).run(x)
        self.assertIsInstance(outs, (tuple, list))
        self.assertEqual(len(outs), 2)
        self.assertTrue(torch.allclose(outs[-1].float().cpu(), x.cpu()))

    def test_compile_runs_the_predicted_kv_cross_check(self):
        """``compile()`` must actually call ``assert_predicted_kv_aliased`` with the
        predictions lift made; without that call a misclassified write silently
        loses its write-back."""
        from torch_tensorrt.dynamo import _compiler as C

        model, args = self._kv_model(1)
        calls = []
        original = C.assert_predicted_kv_aliased

        def _spy(aliased_in, bindings, settings=None, **kwargs):
            calls.append(list(bindings))
            return original(aliased_in, bindings, settings, **kwargs)

        with mock.patch.object(C, "assert_predicted_kv_aliased", _spy):
            self._compile(model, args, min_block_size=1)

        self.assertEqual(calls, [["buf_cache"]])

    def test_compile_raises_when_a_predicted_kv_write_is_not_aliased(self):
        """The cross-check is load-bearing, not decorative: a write predicted to alias
        whose node the partitioner then leaves out of every engine (here via the
        default ``min_block_size``) fails the compile instead of returning a module
        whose buffer never updates."""
        model, args = self._kv_model(2)

        with self.assertRaisesRegex(RuntimeError, "did not alias them"):
            self._compile(model, args, min_block_size=5)

    def test_compile_writes_the_copyback_value_into_the_buffer(self):
        """A copy-back value is carried on the graph output for the exporters. Called
        from PyTorch, the compiled module writes it into the buffer itself, so the
        buffer and the result follow eager PyTorch call after call, and it keeps the
        arity of the model it came from while the graph keeps the value."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(8).cuda())

            def forward(self, x):
                self.state.add_(x)
                return self.state.sum()

        x = torch.arange(8, dtype=torch.float32).cuda()
        model = M().cuda().eval()
        trt_gm = self._compile(model, (x,), min_block_size=1)

        self.assertEqual(trt_gm.meta.get("_copyback_mutation_buffers"), ["state"])
        for step in range(3):
            xi = x * (step + 1)
            eager = model(xi)
            out = trt_gm(xi)
            self.assertEqual(len(out), 1)
            torch.testing.assert_close(out[0], eager)
            torch.testing.assert_close(trt_gm.state, model.state)

        # Still on the output node, where the exporters read it and dead-code
        # elimination cannot reach it.
        output_node = next(n for n in trt_gm.graph.nodes if n.op == "output")
        self.assertEqual(len(output_node.args[0]), 2)

    @staticmethod
    def _accumulator():
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(8).cuda())

            def forward(self, x):
                self.state.add_(x)
                return self.state * 2

        return M().cuda().eval(), (torch.arange(8, dtype=torch.float32).cuda(),)

    def test_compile_keeps_a_clone_of_a_copyback_buffer(self):
        """The clone holds the old value, and forward writes the new one into the
        buffer before it returns. Lowering used to erase the clone, so the module
        returned the buffer itself, holding the new value."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.ones(8).cuda())

            def forward(self, x):
                old = self.state.clone()
                self.state.copy_(x * 2)
                return old

        model = M().cuda().eval()
        x = torch.arange(8, dtype=torch.float32).cuda()
        trt_gm = self._compile(M().cuda().eval(), (x,), min_block_size=1)
        for step in range(3):
            xi = x + step
            out = trt_gm(xi)
            torch.testing.assert_close(out[0], model(xi))
            torch.testing.assert_close(trt_gm.state, model.state)

    def test_a_new_value_computed_in_pytorch_is_not_copied(self):
        """``add`` returns new memory, so its result needs no copy before it is
        written into the buffer it was computed from."""
        model, args = self._accumulator()
        trt_gm = self._compile(
            *self._accumulator(),
            min_block_size=1,
            torch_executed_ops={"torch.ops.aten.add.Tensor"},
        )
        self.assertNotIn(".clone()", trt_gm.code)
        for _ in range(3):
            torch.testing.assert_close(trt_gm(*args)[0], model(*args))
            torch.testing.assert_close(trt_gm.state, model.state)

    def test_full_compilation_of_a_model_with_written_buffers(self):
        """With full support compile used to return the engine module itself, which
        has no buffers to write: a copied-back buffer and a cache updated in place
        both failed."""
        for make in (self._accumulator, lambda: self._kv_model(1)):
            model, args = make()
            trt_gm = self._compile(
                *make(), min_block_size=1, require_full_compilation=True
            )
            for _ in range(3):
                out = trt_gm(*args)
                out = out[0] if isinstance(out, (tuple, list)) else out
                torch.testing.assert_close(out, model(*args))
            name = "state" if hasattr(model, "state") else "cache"
            torch.testing.assert_close(getattr(trt_gm, name), getattr(model, name))

    def test_compile_writes_a_copyback_buffer_owned_by_a_submodule(self):
        """A nested buffer is registered on the compiled module under a flat name,
        and forward has to write it under that name."""

        class Counter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(8).cuda())

            def forward(self, x):
                self.state.add_(x)
                return self.state * 2

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.counter = Counter()

            def forward(self, x):
                return self.counter(x) + 1

        model = M().cuda().eval()
        x = torch.arange(8, dtype=torch.float32).cuda()
        trt_gm = self._compile(M().cuda().eval(), (x,), min_block_size=1)
        for step in range(3):
            xi = x * (step + 1)
            torch.testing.assert_close(trt_gm(xi)[0], model(xi))

    def test_a_nested_buffer_flattens_past_a_model_buffer_of_that_name(self):
        """``a.b`` flattens to ``lifted_buf_a_b``, which the model already uses for a
        buffer it only reads, so the compiled module holds ``a.b`` under another
        name. Refit's output check has to find both again."""
        from torch_tensorrt.dynamo import refit_module_weights

        class Holder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("b", torch.zeros(8).cuda())

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.arange(1.0, 9.0).cuda())
                self.a = Holder()
                self.register_buffer("lifted_buf_a_b", torch.full((8,), 5.0).cuda())

            def forward(self, x):
                self.a.b.add_(x * self.weight)
                return (self.a.b + self.lifted_buf_a_b,)

        model = M()
        args = (torch.ones(8).cuda(),)
        ep = torch.export.export(M(), args, strict=False)
        # The global partitioner keeps the model's buffer on the compiled module.
        trt_gm = self._compile(
            M(),
            args,
            min_block_size=1,
            immutable_weights=False,
            use_fast_partitioner=False,
        )
        for _ in range(3):
            torch.testing.assert_close(trt_gm(*args)[0], model(*args)[0])
        torch.testing.assert_close(trt_gm.lifted_buf_a_b, model.lifted_buf_a_b)
        torch.testing.assert_close(trt_gm.lifted_buf_a_b_1, model.a.b)
        with self.assertLogs("torch_tensorrt.dynamo._refit", level="INFO") as logs:
            refit_module_weights(
                trt_gm, ep, arg_inputs=list(args), in_place=True, verify_output=True
            )
        self.assertTrue(any("Refitting Succeed" in line for line in logs.output))

    def test_torchscript_save_leaves_the_buffers_alone(self):
        """Tracing calls the module, and used to save it a call ahead."""
        import tempfile

        import torch_tensorrt

        model, args = self._accumulator()
        trt_gm = self._compile(*self._accumulator(), min_block_size=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/model.ts"
            torch_tensorrt.save(
                trt_gm, path, output_format="torchscript", arg_inputs=list(args)
            )
            self.assertEqual(trt_gm.state, torch.zeros(8).cuda())
            loaded = torch.jit.load(path)
        out = loaded(*args)
        out = out[0] if isinstance(out, (tuple, list)) else out
        torch.testing.assert_close(out, model(*args))

    def test_torchscript_save_leaves_a_cache_written_in_place_alone(self):
        """The engine updates this cache itself, so tracing used to advance it."""
        import tempfile

        import torch_tensorrt

        model, args = self._kv_model(1)
        trt_gm = self._compile(model, args, min_block_size=1)
        self.assertFalse(trt_gm.meta.get("_copyback_mutation_buffers"))
        with tempfile.TemporaryDirectory() as tmp:
            torch_tensorrt.save(
                trt_gm,
                f"{tmp}/model.ts",
                output_format="torchscript",
                arg_inputs=list(args),
            )
        self.assertEqual(trt_gm.cache, torch.zeros(1, 4, 16, 8).cuda())

    def test_torchscript_save_keeps_a_read_only_expanded_buffer(self):
        """Saving used to write every buffer back, and an expanded buffer, whose
        elements share memory, cannot be written."""
        import tempfile

        import torch_tensorrt

        class M(torch.nn.Module):
            def __init__(self, expanded):
                super().__init__()
                bias = torch.tensor([2.0]).cuda().expand(8)
                self.register_buffer("bias", bias if expanded else bias.clone())

            def forward(self, x):
                return x + self.bias

        x = torch.arange(8.0).cuda()
        for expanded in (True, False):
            graph = torch.fx.Graph()
            out = graph.call_function(
                torch.ops.aten.add.Tensor,
                (graph.placeholder("x"), graph.get_attr("bias")),
            )
            graph.output(out)
            gm = torch.fx.GraphModule(M(expanded), graph)
            with tempfile.TemporaryDirectory() as tmp:
                path = f"{tmp}/model.ts"
                torch_tensorrt.save(
                    gm, path, output_format="torchscript", arg_inputs=[x]
                )
                loaded = torch.jit.load(path)
            torch.testing.assert_close(loaded(x), x + 2)

    def test_refit_verification_leaves_the_buffers_alone(self):
        """The output check calls the refitted module, and used to leave it a call
        ahead. It also has to compare the two modules from the same buffers, or a
        module that has run once fails the check."""
        from torch_tensorrt.dynamo import refit_module_weights

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.arange(1.0, 9.0).cuda())
                self.register_buffer("state", torch.zeros(8).cuda())

            def forward(self, x):
                self.state.add_(x * self.weight)
                return (self.state * 3,)

        args = (torch.ones(8).cuda(),)
        ep = torch.export.export(M(), args, strict=False)
        trt_gm = self._compile(M(), args, min_block_size=1, immutable_weights=False)
        # Two calls, so the one call the check makes from the new model's zeros
        # cannot end where these did.
        trt_gm(*args)
        trt_gm(*args)
        before = trt_gm.state.clone()
        with self.assertLogs("torch_tensorrt.dynamo._refit", level="INFO") as logs:
            refitted = refit_module_weights(
                trt_gm, ep, arg_inputs=list(args), in_place=True, verify_output=True
            )
        self.assertTrue(any("Refitting Succeed" in line for line in logs.output))
        torch.testing.assert_close(refitted.state, before)

    def test_refit_verification_seeds_nested_buffers_that_flatten_alike(self):
        """``a.b_c`` and ``a_b.c`` both flatten to ``lifted_buf_a_b_c``, so the
        compiled module holds the second as ``lifted_buf_a_b_c_1``. The check has to
        start each from the new model's value, not write both into the first, also
        on a module that was saved and loaded."""
        from torch_tensorrt.dynamo import refit_module_weights

        class Holder(torch.nn.Module):
            def __init__(self, name, value):
                super().__init__()
                self.register_buffer(name, torch.full((8,), value).cuda())

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.arange(1.0, 9.0).cuda())
                self.a = Holder("b_c", 1.0)
                self.a_b = Holder("c", 2.0)

            def forward(self, x):
                self.a.b_c.add_(x * self.weight)
                self.a_b.c.add_(x)
                return (self.a.b_c + self.a_b.c,)

        args = (torch.ones(8).cuda(),)
        ep = torch.export.export(M(), args, strict=False)
        trt_gm = self._compile(M(), args, min_block_size=1, immutable_weights=False)
        data = io.BytesIO()
        torch.save(trt_gm, data)
        data.seek(0)
        modules = {"compiled": trt_gm, "pickled": torch.load(data, weights_only=False)}
        for kind, module in modules.items():
            with self.subTest(module=kind):
                module(*args)
                module(*args)
                with self.assertLogs(
                    "torch_tensorrt.dynamo._refit", level="INFO"
                ) as logs:
                    refit_module_weights(
                        module,
                        ep,
                        arg_inputs=list(args),
                        in_place=True,
                        verify_output=True,
                    )
                self.assertTrue(
                    any("Refitting Succeed" in line for line in logs.output)
                )

    def test_an_input_named_like_the_write_back_helper(self):
        """The generated ``forward`` calls the helper by a name no input has."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(8).cuda())

            def forward(self, write_copyback_buffers):
                self.state.add_(write_copyback_buffers)
                return (self.state * 2,)

        args = (torch.ones(8).cuda(),)
        trt_gm = self._compile(M(), args, min_block_size=1)
        eager = M()
        for _ in range(2):
            torch.testing.assert_close(trt_gm(*args)[0], eager(*args)[0])
        torch.testing.assert_close(trt_gm.state, eager.state)

    def test_a_module_compiled_in_inference_mode_runs_outside_it(self):
        """A buffer cloned in inference mode refused the write forward makes."""
        model, args = self._accumulator()
        with torch.inference_mode():
            trt_gm = self._compile(*self._accumulator(), min_block_size=1)
        with torch.no_grad():
            for _ in range(2):
                out = trt_gm(*args)
                torch.testing.assert_close(out[0], model(*args))

    def test_compile_hides_copyback_beside_an_engine_aliased_kv_buffer(self):
        """One module holding both kinds of mutated buffer at once.

        ``hide_copyback_outputs`` truncates the last *N* values of the return, so it
        is right only while the copy-back values really are the trailing ones. Two
        mechanisms append outputs: lift appends the copy-back values last, and the
        interpreter appends an aliased output per KV write inside each submodule.
        The second is truncated at the submodule boundary and must never reach the
        outer graph, else the *N* counted here would take a KV output and leave a
        copy-back value in the return. Each mechanism is covered alone; this is the
        intersection."""

        def _model():
            class M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.register_buffer("k", torch.zeros(1, 4, 16, 8).cuda())
                    self.register_buffer("state", torch.zeros(8).cuda())

                def forward(self, x_kv, x_state):
                    self.k[:, :, 3:4, :] = x_kv
                    self.state.add_(x_state)
                    return self.k.sum() + self.state.sum()

            return M().cuda().eval()

        args = (
            torch.full((1, 4, 1, 8), 2.0).cuda(),
            torch.arange(8, dtype=torch.float32).cuda(),
        )
        eager = _model()
        with torch.no_grad():
            expected = eager(*args).clone()
            expected_k = eager.k.clone()
            expected_state = eager.state.clone()

        trt_gm = self._compile(_model(), args, min_block_size=1)

        # The KV write is aliased in the engine; only the other buffer is copy-back.
        self.assertEqual(trt_gm.meta.get("_copyback_mutation_buffers"), ["state"])
        aliased = aliased_input_bindings(
            getattr(sub, "aliased_io", None) for _name, sub in trt_gm.named_children()
        )
        self.assertEqual(aliased, {"buf_k"})

        # The graph keeps exactly one more value than the return, and it is the
        # copy-back buffer's new contents rather than a KV output. The interpreter
        # reads the graph without writing the copy-back buffer, so run it first.
        output_node = next(n for n in trt_gm.graph.nodes if n.op == "output")
        self.assertEqual(len(output_node.args[0]), 2)
        outs = torch.fx.Interpreter(trt_gm).run(*args)
        self.assertEqual(len(outs), 2)
        self.assertTrue(
            torch.allclose(outs[-1].float().cpu(), expected_state.cpu(), atol=1e-3)
        )

        out = trt_gm(*args)
        self.assertEqual(len(out), 1)
        self.assertTrue(torch.allclose(out[0].cpu(), expected.cpu(), atol=1e-3))
        # Aliased, so the engine wrote the cache in place; the other buffer is
        # written back by forward.
        self.assertTrue(torch.allclose(trt_gm.k.cpu(), expected_k.cpu(), atol=1e-3))
        self.assertTrue(
            torch.allclose(trt_gm.state.cpu(), expected_state.cpu(), atol=1e-3)
        )

    def test_saved_program_declares_the_hidden_copyback_mutation(self):
        """Hiding the value from the return must not hide it from the exporter: the
        saved program still has to carry the BUFFER_MUTATION that tells the
        ExecuTorch runtime to write the buffer back."""
        import tempfile

        import torch_tensorrt

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(8).cuda())

            def forward(self, x):
                self.state.add_(x)
                return self.state.sum()

        x = torch.arange(8, dtype=torch.float32).cuda()
        trt_gm = self._compile(M().cuda().eval(), (x,), min_block_size=1)

        with tempfile.TemporaryDirectory() as directory:
            # Both exporters have to see the value the module stopped returning.
            for retrace in (False, True):
                path = f"{directory}/program_{retrace}.ep"
                torch_tensorrt.save(
                    trt_gm,
                    path,
                    output_format="exported_program",
                    retrace=retrace,
                    arg_inputs=(x,) if retrace else None,
                )
                loaded = torch.export.load(path)
                # The user output has to survive alongside the declaration. An
                # exporter that never saw the hidden value declares the buffer
                # mutation out of the user output instead, which passes a check for
                # the mutation alone and leaves the program returning nothing.
                self.assertEqual(
                    [
                        (spec.kind.name, spec.target)
                        for spec in loaded.graph_signature.output_specs
                    ],
                    [("BUFFER_MUTATION", "state"), ("USER_OUTPUT", None)],
                    f"retrace={retrace}",
                )
                # And the module the caller still holds returns what it did before.
                self.assertEqual(len(trt_gm(x)), 1, f"retrace={retrace}")

    def test_strict_export_records_the_copyback_write(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(8).cuda())

            def forward(self, x):
                self.state.add_(x)
                return self.state.sum()

        x = torch.arange(8, dtype=torch.float32).cuda()
        eager = M().cuda().eval()
        trt_gm = self._compile(M().cuda().eval(), (x,), min_block_size=1)
        program = torch.export.export(trt_gm, (x,), strict=True)
        specs = program.run_decompositions({}).graph_signature.output_specs
        self.assertEqual(
            [(s.kind.name, s.target) for s in specs if s.kind.name != "TOKEN"],
            [("BUFFER_MUTATION", "state"), ("USER_OUTPUT", None)],
        )
        exported = program.module()
        for _ in range(2):
            out = exported(x)
            torch.testing.assert_close(out[0].reshape(()), eager(x))
            torch.testing.assert_close(exported.state, eager.state)

    @staticmethod
    def _swaps():
        class Swaps(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("left", torch.ones(4).cuda())
                self.register_buffer("right", torch.full((4,), 3.0).cuda())

            def forward(self, x):
                left, right = self.left.clone(), self.right.clone()
                self.left.copy_(right * x)
                self.right.copy_(left)
                return x + self.left

        return Swaps().cuda().eval(), (torch.full((4,), 2.0).cuda(),)

    def test_a_pickled_module_keeps_writing_its_buffers(self):
        """``torch.load`` rebuilds the graph by tracing the saved ``forward``. That
        must keep the writes in it and must not run them."""
        import copy
        import io

        for min_block_size in (1, 100):
            model, args = self._swaps()
            eager = copy.deepcopy(model)
            trt_gm = self._compile(model, args, min_block_size=min_block_size)
            data = io.BytesIO()
            torch.save(trt_gm, data)
            data.seek(0)
            loaded = torch.load(data, weights_only=False)
            self.assertEqual(loaded.left, eager.left)
            self.assertEqual(loaded.right, eager.right)
            for _ in range(3):
                out = loaded(*args)
                torch.testing.assert_close(out[0], eager(*args))
                self.assertEqual(loaded.left, eager.left)
                self.assertEqual(loaded.right, eager.right)

    def test_retracing_exporter_sees_the_hidden_copyback_value(self):
        """``dynamo._exporter.export`` without the legacy exporter re-traces, and it
        is the retrace, not the module's return, that has to carry the hidden value.
        It does: the export is non-strict, so the GraphModule is interpreted off its
        output node and the value enters the program with nothing exposing it first.
        The declaration pass downstream then reclassifies it as the mutation."""
        from torch_tensorrt.dynamo._exporter import export as exporter_export

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(8).cuda())

            def forward(self, x):
                self.state.add_(x)
                return self.state.sum()

        x = torch.arange(8, dtype=torch.float32).cuda()
        trt_gm = self._compile(M().cuda().eval(), (x,), min_block_size=1)

        exported = exporter_export(trt_gm, arg_inputs=(x,), use_legacy_exporter=False)
        # Undeclared at this layer -- the copy-back value is still a user output --
        # but it must be there for the declaration pass to reclassify.
        self.assertEqual(len(exported.graph_signature.output_specs), 2)

    @staticmethod
    def _cloned_cache_model():
        """Clone, scatter, write back -- the ordinary shape when the post-write cache
        is also read in the same forward."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("k", torch.zeros(1, 2, 16, 4).cuda())

            def forward(self, x):
                k = self.k.clone()
                k[:, :, 3:4, :] = x
                self.k.copy_(k)
                return k.sum(dim=-1)

        return M().cuda().eval(), (torch.full((1, 2, 1, 4), 3.0).cuda(),)

    def test_compile_runs_a_cloned_cache_write_and_writes_it_back(self):
        """``remove_input_alias_fixing_clones`` erases the clone after
        classification, so the converter emits the KV layer whatever the classifier
        decided. Classified copy-back, the module raised ``IndexError`` on every call
        -- the trailing output the runtime truncates as an engine side effect while
        the outer graph still reads it."""
        model, args = self._cloned_cache_model()
        with torch.no_grad():
            expected = model(*args).clone()
            expected_cache = model.k.clone()

        trt_gm = self._compile(*self._cloned_cache_model(), min_block_size=1)

        self.assertEqual(trt_gm.meta.get("_copyback_mutation_buffers", []), [])
        aliased = aliased_input_bindings(
            getattr(sub, "aliased_io", None) for _name, sub in trt_gm.named_children()
        )
        self.assertEqual(aliased, {"buf_k"})

        out = trt_gm(*args)
        self.assertEqual(len(out), 1)
        self.assertTrue(torch.allclose(out[0].cpu(), expected.cpu(), atol=1e-3))
        self.assertTrue(torch.allclose(trt_gm.k.cpu(), expected_cache.cpu(), atol=1e-3))

    def test_compile_runs_a_cloned_index_copy_write_and_writes_it_back(self):
        """The `index_copy` decode shape of the same case: a per-step cache-position
        write through a clone. Its validator checks `args[0]` on its own account, so
        this fails for a different reason than the `slice_scatter` one even though
        the model shape and the symptom are the same."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("k", torch.zeros(1, 2, 16, 4).cuda())

            def forward(self, x, pos):
                k = self.k.clone()
                k = k.index_copy(2, pos, x)
                self.k.copy_(k)
                return k.sum(dim=-1)

        args = (
            torch.full((1, 2, 1, 4), 3.0).cuda(),
            torch.tensor([3], dtype=torch.int64).cuda(),
        )
        model = M().cuda().eval()
        with torch.no_grad():
            expected = model(*args).clone()
            expected_cache = model.k.clone()

        trt_gm = self._compile(M().cuda().eval(), args, min_block_size=1)

        self.assertEqual(trt_gm.meta.get("_copyback_mutation_buffers", []), [])
        out = trt_gm(*args)
        self.assertEqual(len(out), 1)
        self.assertTrue(torch.allclose(out[0].cpu(), expected.cpu(), atol=1e-3))
        self.assertTrue(torch.allclose(trt_gm.k.cpu(), expected_cache.cpu(), atol=1e-3))

    def test_compile_raises_when_a_copyback_write_is_aliased(self):
        """The backstop for the next divergence between what the classifier predicts
        and what the converter emits, whatever that turns out to be. Standing in for
        it by making the classifier file everything copy-back while the write keeps
        its in-place mark: the engine aliases the cache anyway, and the compile has
        to say so rather than hand back a module that raises ``IndexError`` on every
        call."""
        from torch_tensorrt.dynamo.lowering import _buffer_lifting

        def copy_back_but_keep_the_mark(value, cache, *args, **kwargs):
            _buffer_lifting._cache_write_of(value, cache).meta["_trt_kv_alias"] = True

        model, args = self._kv_model(batch=1)
        with mock.patch.object(
            _buffer_lifting, "_in_place_write", copy_back_but_keep_the_mark
        ):
            with self.assertRaisesRegex(RuntimeError, "aliased them in place anyway"):
                self._compile(model, args, min_block_size=1)

    @staticmethod
    def _dead_slice_scatter_model():
        """A KV-shaped write whose result nothing else consumes."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 4, bias=False).cuda()
                self.register_buffer("cache", torch.zeros(2, 4, 16, 4).cuda())

            def forward(self, x):
                self.cache[:, :, 3:4, :] = self.lin(x)
                return x.sum() * 2.0

        return M().cuda().eval(), (torch.ones(2, 4, 1, 4).cuda(),)

    def _assert_copyback_and_unaliased(self, trt_gm, model, args, buffers):
        """The compile produced a module that runs, and the buffer it could not alias
        is declared copy-back rather than both."""
        self.assertEqual(trt_gm.meta.get("_copyback_mutation_buffers"), buffers)
        for _name, sub in trt_gm.named_children():
            self.assertFalse(getattr(sub, "aliased_io", None))
        out = trt_gm(*args)
        self.assertEqual(len(out), 1)
        eager = model(*args)
        # These models return a 0-d sum, which comes back from the engine as rank-1.
        # Only the value is under test here.
        torch.testing.assert_close(
            out[0].reshape(eager.shape), eager, rtol=1e-3, atol=1e-3
        )

    def test_compile_routes_a_dead_slice_scatter_write_to_copyback(self):
        """Erasing the ``copy_`` leaves the write dead, so no engine could ever alias
        it and the predicted-KV cross-check would fail the compile. It is filed
        copy-back instead, and the marker on the write is what keeps the engine from
        aliasing it once the copy-back output makes it live again."""
        model, args = self._dead_slice_scatter_model()
        for min_block_size in (1, 5):
            trt_gm = self._compile(model, args, min_block_size=min_block_size)
            self._assert_copyback_and_unaliased(trt_gm, model, args, ["cache"])

    def test_compile_routes_a_dead_cloned_cache_write_to_copyback(self):
        """The two mechanisms meet here: the write reads its cache through a clone
        *and* nothing consumes its result. The clone peel is what classifies it KV in
        the first place, and the dead-write routing is what stops that classification
        failing the cross-check -- so a write that is both is the case each mechanism
        can only handle with the other. Without the peel it is filed copy-back
        unmarked, ``remove_input_alias_fixing_clones`` erases the clone, and the
        engine aliases a buffer that is also copy-back; without the routing it is
        predicted KV, dies with its ``copy_``, and no engine claims it."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(1, 2, 16, 4).cuda())

            def forward(self, x):
                k = self.cache.clone()
                k[:, :, 3:4, :] = x
                self.cache.copy_(k)
                return x.sum() * 2.0

        model, args = M().cuda().eval(), (torch.full((1, 2, 1, 4), 3.0).cuda(),)
        for min_block_size in (1, 5):
            trt_gm = self._compile(model, args, min_block_size=min_block_size)
            self._assert_copyback_and_unaliased(trt_gm, model, args, ["cache"])

    def test_compile_routes_a_dead_write_whose_buffer_has_another_reader(self):
        """The buffer is read before the write, so the *placeholder* still has a
        reader in the lowered graph while the *write* is dead. Keyed on the
        placeholder this looked like a write the partitioner had rejected."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 4, bias=False).cuda()
                self.register_buffer("cache", torch.zeros(2, 4, 16, 4).cuda())

            def forward(self, x):
                prev = self.cache.sum()
                self.cache[:, :, 3:4, :] = self.lin(x)
                return prev * 2.0 + x.sum()

        model, args = M().cuda().eval(), (torch.ones(2, 4, 1, 4).cuda(),)
        trt_gm = self._compile(model, args, min_block_size=1)
        self._assert_copyback_and_unaliased(trt_gm, model, args, ["cache"])

    def test_compile_routes_a_dead_index_copy_write_to_copyback(self):
        """``index_copy``'s eligibility check is a different function from
        ``slice_scatter``'s, so honouring the marker in one does not honour it in the
        other -- and the decode write is the ``index_copy`` one."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("k", torch.zeros(1, 2, 16, 4).cuda())

            def forward(self, x, pos):
                self.k.index_copy_(2, pos, x)
                return x.sum() * 2.0

        model = M().cuda().eval()
        args = (torch.ones(1, 2, 1, 4).cuda(), torch.tensor([3]).cuda())
        trt_gm = self._compile(model, args, min_block_size=1)
        self._assert_copyback_and_unaliased(trt_gm, model, args, ["k"])

    def test_saved_program_declares_the_dead_write_mutation(self):
        """Compiling is not the point -- the write has to reach the runtime. A
        compile that succeeded without declaring the mutation would leave the buffer
        at its compile-time value; the saved program declares it for the runtime to
        apply."""
        import tempfile

        import torch_tensorrt

        model, args = self._dead_slice_scatter_model()
        trt_gm = self._compile(model, args, min_block_size=1)
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/program.ep"
            torch_tensorrt.save(
                trt_gm, path, output_format="exported_program", arg_inputs=args
            )
            loaded = torch.export.load(path)
        self.assertIn(
            ("BUFFER_MUTATION", "cache"),
            [
                (spec.kind.name, spec.target)
                for spec in loaded.graph_signature.output_specs
            ],
        )

    def test_compile_folds_a_write_of_constants(self):
        """Constant folding turns these writes into constants before conversion,
        with the default settings and no buffer the model writes."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("w", torch.ones(1, 4, 16, 8).cuda())
                self.register_buffer("u", torch.full((1, 4, 2, 8), 2.0).cuda())
                self.register_buffer("pos", torch.tensor([7]).cuda())

            def forward(self, x):
                w = self.w.clone()
                w[:, :, 4:6, :] = self.u
                v = self.w.index_copy(2, self.pos, self.u[:, :, :1, :])
                return x * w + v

        model, args = M().cuda().eval(), (torch.randn(1, 4, 16, 8).cuda(),)
        out = self._compile(model, args)(*args)
        out = out[0] if isinstance(out, (tuple, list)) else out
        torch.testing.assert_close(out, model(*args))

    def test_compile_does_not_cross_check_a_dryrun(self):
        """A dryrun returns before any engine is built, so every prediction is absent
        from ``aliased_io`` for a reason that says nothing about the prediction.

        ``compile()`` is the only caller that knows this, so it is the only one that
        tells the check -- the check itself cannot read it off ``dryrun``, which the
        engine converter accepts while building anyway. Nothing else observes that
        ``compile()`` says so, and a run whose only job is to report would otherwise
        fail outright."""
        model, args = self._kv_model(1)

        self._compile(model, args, dryrun=True, min_block_size=1)


class TestEraseExportGuards(TestCase):
    def test_erase_guards_fn_from_exported_module(self):
        class Add(torch.nn.Module):
            def forward(self, x):
                return x + x

        gm = export(Add(), (torch.ones(2, 2),)).module()
        had_guard = any(
            n.op == "call_module" and n.target == "_guards_fn" for n in gm.graph.nodes
        )
        if not had_guard:
            self.skipTest("this PyTorch export did not emit _guards_fn")
        gm = erase_export_guards(gm)
        self.assertFalse(
            any(
                n.op == "call_module" and n.target == "_guards_fn"
                for n in gm.graph.nodes
            )
        )
        self.assertNotIn("_guards_fn", dict(gm.named_children()))


if __name__ == "__main__":
    run_tests()
