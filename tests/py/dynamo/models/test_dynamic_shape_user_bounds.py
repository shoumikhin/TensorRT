# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# type: ignore
"""Tests for ``user_symbol_bounds`` plumbed from ``compile_module`` through
the partitioner into ``extract_var_range_info``."""

import os
import sys
import unittest

import pytest
import sympy
import torch
import torch_tensorrt as torchtrt
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo._compiler import _build_user_symbol_bounds
from torch_tensorrt.dynamo.partitioning.common import construct_dynamic_input
from torch_tensorrt.dynamo.utils import extract_var_range_info

assertions = unittest.TestCase()


def _first_sym_placeholder(ep: torch.export.ExportedProgram, sym_dim: int = 0):
    """First placeholder with a ``SymInt`` at ``sym_dim``; skips lifted
    param/buffer placeholders that ``strict=False`` may prepend."""
    for node in ep.graph.nodes:
        if node.op != "placeholder":
            continue
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor):
            continue
        if sym_dim < len(val.size()) and isinstance(val.size()[sym_dim], torch.SymInt):
            return val, val.size()[sym_dim]
    return None, None


class _SmallLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.linear(x).relu()


@torch.library.custom_op("torchtrt_profile_test::dynamic_rows", mutates_args=())
def _profile_dynamic_rows(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_profile_dynamic_rows.register_fake
def _profile_dynamic_rows_fake(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty(torch.library.get_ctx().new_dynamic_size(), x.shape[1])


class _ConstrainedDynamicRows(torch.nn.Module):
    def __init__(self, lower: int, upper: int) -> None:
        super().__init__()
        self.lower = lower
        self.upper = upper

    def forward(self, x):
        rows = torch.ops.torchtrt_profile_test.dynamic_rows(x)
        torch._check(rows.shape[0] >= self.lower)
        torch._check(rows.shape[0] <= self.upper)
        return rows


def _constrained_dynamic_shape(lower: int, upper: int) -> torch.Size:
    exported = torch.export.export(
        _ConstrainedDynamicRows(lower, upper), (torch.ones(1100, 4),)
    )
    dynamic_rows = next(
        node
        for node in exported.graph.nodes
        if node.target == torch.ops.torchtrt_profile_test.dynamic_rows.default
    )
    return dynamic_rows.meta["val"].shape


@pytest.mark.unit
def test_extract_var_range_info_fills_unbounded_max_from_user():
    """User bounds must fill the gap when the exporter's upper is ``int_oo``,
    and successive calls must not leak state across invocations."""

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(2, 8, device="cuda")

    dyn_batch = torch.export.Dim.DYNAMIC
    ep = torch.export.export(
        model, (sample,), dynamic_shapes=({0: dyn_batch},), strict=False
    )

    fake_val, sym_dim = _first_sym_placeholder(ep, sym_dim=0)
    assert sym_dim is not None
    assert isinstance(sym_dim, torch.SymInt)
    expr = sym_dim.node.expr
    assert isinstance(expr, sympy.Symbol)

    assert extract_var_range_info(sym_dim)["max"] is None

    info_with_user = extract_var_range_info(sym_dim, user_symbol_bounds={expr: (1, 8)})
    assert info_with_user["max"] == 8
    assert info_with_user["min"] == 1

    # Different map on the same SymInt must return the new bounds (no caching).
    info_wider = extract_var_range_info(sym_dim, user_symbol_bounds={expr: (1, 16)})
    assert info_wider["max"] == 16
    assert info_wider["min"] == 1

    # Dropping the map must revert (no ShapeEnv mutation).
    assert extract_var_range_info(sym_dim)["max"] is None


@pytest.mark.unit
def test_extract_var_range_info_does_not_widen_lower_bound():
    """Lower bound is intersected so the 0/1 specialization survives even
    when the user passes ``min_shape=0``."""

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(2, 8, device="cuda")
    ep = torch.export.export(
        model,
        (sample,),
        dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
        strict=False,
    )
    _, sym_dim = _first_sym_placeholder(ep, sym_dim=0)
    assert sym_dim is not None
    expr = sym_dim.node.expr

    info = extract_var_range_info(sym_dim, user_symbol_bounds={expr: (0, 8)})
    assert info["min"] == 1, "exporter lower must not be widened to user's 0"
    assert info["max"] == 8


@pytest.mark.unit
def test_extract_var_range_info_does_not_override_finite_max():
    """A finite exporter max must win over ``user_symbol_bounds``."""

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(2, 8, device="cuda")
    ep = torch.export.export(
        model,
        (sample,),
        dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=4)},),
        strict=False,
    )
    _, sym_dim = _first_sym_placeholder(ep, sym_dim=0)
    assert sym_dim is not None
    expr = sym_dim.node.expr

    info = extract_var_range_info(sym_dim, user_symbol_bounds={expr: (1, 16)})
    assert info["max"] == 4


@pytest.mark.unit
def test_extract_var_range_info_handles_sympy_oo_from_bound_sympy():
    """Composite exprs (e.g. ``s0 + s1``) hit ``bound_sympy``, which returns
    ``sympy.oo`` (not ``int_oo``) for unbounded operands. A naive
    ``!= int_oo`` guard misses it and crashes on ``int(sympy.oo)``."""

    class _ConcatBatch(torch.nn.Module):
        def forward(self, a, b):
            return torch.cat([a, b], dim=0)

    model = _ConcatBatch().eval().cuda()
    a = torch.randn(2, 8, device="cuda")
    b = torch.randn(3, 8, device="cuda")
    ep = torch.export.export(
        model,
        (a, b),
        dynamic_shapes=(
            {0: torch.export.Dim.DYNAMIC},
            {0: torch.export.Dim.DYNAMIC},
        ),
        strict=False,
    )

    cat_node = next(
        (
            n
            for n in ep.graph.nodes
            if n.op == "call_function" and "cat" in str(n.target)
        ),
        None,
    )
    if cat_node is None or "val" not in cat_node.meta:
        pytest.skip("no cat node with composite SymInt on this PyTorch")

    composite_dim = cat_node.meta["val"].size()[0]
    if not isinstance(composite_dim, torch.SymInt):
        pytest.skip("composite dim specialized to a concrete int")
    if isinstance(composite_dim.node.expr, sympy.Symbol):
        pytest.skip("expr collapsed to a plain symbol on this PyTorch")

    # Pre-fix this raised ``AttributeError`` from ``int(sympy.oo)``.
    info = extract_var_range_info(composite_dim)
    assert info["max"] is None
    assert info["min"] is not None


@pytest.mark.unit
def test_build_user_symbol_bounds_uses_dynamic_inputs_only():
    """Static ``Input``s contribute nothing to the user-bounds map."""

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(2, 8, device="cuda")
    ep = torch.export.export(
        model,
        (sample,),
        dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
        strict=False,
    )

    dynamic_input = Input(
        min_shape=(1, 8),
        opt_shape=(4, 8),
        max_shape=(8, 8),
        dtype=torch.float32,
    )
    bounds = _build_user_symbol_bounds(
        ep.module(), [dynamic_input], {}, ep.graph_signature
    )
    assert len(bounds) == 1
    ((sym, (lo, hi)),) = bounds.items()
    assert isinstance(sym, sympy.Symbol)
    assert (lo, hi) == (1, 8)

    # Static input -> empty map.
    static_input = Input(shape=(2, 8), dtype=torch.float32)
    assert (
        _build_user_symbol_bounds(ep.module(), [static_input], {}, ep.graph_signature)
        == {}
    )


@pytest.mark.unit
@pytest.mark.parametrize("exported_min, warns", [(1, False), (2, True)])
def test_compile_warns_only_when_input_min_1_is_below_the_exported_min(
    caplog, exported_min, warns
):
    """``Dim(min=1)`` matches ``Input.min_shape=1`` even though the ShapeEnv
    reports its lower bound as 2. A program exported with ``Dim(min=2)`` still
    warns, because the engine accepts a size of 1 it was not exported for."""

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(4, 8, device="cuda")
    ep = torch.export.export(
        model,
        (sample,),
        dynamic_shapes=({0: torch.export.Dim("batch", min=exported_min, max=8)},),
        strict=False,
    )
    input_min1 = Input(
        min_shape=(1, 8),
        opt_shape=(4, 8),
        max_shape=(8, 8),
        dtype=torch.float32,
    )

    import logging

    with caplog.at_level(logging.WARNING, logger="torch_tensorrt.dynamo._compiler"):
        trt_module = torchtrt.dynamo.compile(ep, inputs=[input_min1], min_block_size=1)

    warned = any(
        "exported program's min is 2" in rec.message
        for rec in caplog.records
        if rec.levelno >= logging.WARNING
    )
    assert warned == warns
    one_row = torch.randn(1, 8, device="cuda")
    torch.testing.assert_close(
        trt_module(one_row), model(one_row), rtol=1e-3, atol=1e-3
    )


def _profile_warnings(caplog, model, example, dynamic_shapes, inputs, **kwargs):
    import logging

    ep = torch.export.export(
        model, example, dynamic_shapes=dynamic_shapes, strict=False
    )
    with caplog.at_level(logging.WARNING, logger="torch_tensorrt.dynamo._compiler"):
        compiled = torchtrt.dynamo.compile(
            ep, inputs=inputs, min_block_size=1, **kwargs
        )
    warnings = [
        rec.message
        for rec in caplog.records
        if rec.levelno >= logging.WARNING
        and ("Its profile starts at" in rec.message or "min is 2" in rec.message)
    ]
    return warnings, compiled


def _assert_runs_or_warns(warnings, compiled, model, smallest):
    """A call at the smallest size the program allows either runs or was warned
    about at compile time."""
    try:
        torch.testing.assert_close(
            compiled(*smallest), model(*smallest), rtol=1e-3, atol=1e-3
        )
    except RuntimeError:
        assert any("Its profile starts at" in w for w in warnings), warnings


class _Repeat(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x, z):
        return self.linear(x).repeat(2, 1) + z


_b = torch.export.Dim("b", min=1, max=8)
_derived_inputs = [
    Input(min_shape=(1, 8), opt_shape=(4, 8), max_shape=(8, 8)),
    Input(min_shape=(2, 8), opt_shape=(8, 8), max_shape=(16, 8)),
]


@pytest.mark.unit
def test_compile_warns_when_an_engine_input_size_is_computed_from_a_symbol(caplog):
    """``z`` is ``2 * b``. A profile that came from the ShapeEnv bounds of ``b``,
    which start at 2 for ``Dim(min=1)``, would leave a call with ``b = 1`` outside."""
    model = _Repeat().eval().cuda()
    warnings, compiled = _profile_warnings(
        caplog,
        model,
        (torch.randn(4, 8, device="cuda"), torch.randn(8, 8, device="cuda")),
        ({0: _b}, {0: 2 * _b}),
        _derived_inputs,
    )
    smallest = (torch.randn(1, 8, device="cuda"), torch.randn(2, 8, device="cuda"))
    _assert_runs_or_warns(warnings, compiled, model, smallest)


@pytest.mark.unit
def test_compile_warns_for_a_computed_size_made_inside_the_graph(caplog):
    """No input of the program has the size ``2 * b``, but the engine after the
    op sent to PyTorch has one."""

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(8, 8)
            self.second = torch.nn.Linear(8, 8)

        def forward(self, x):
            return self.second(torch.relu(self.first(x).repeat(2, 1)))

    model = Inner().eval().cuda()
    warnings, compiled = _profile_warnings(
        caplog,
        model,
        (torch.randn(4, 8, device="cuda"),),
        ({0: _b},),
        [Input(min_shape=(1, 8), opt_shape=(4, 8), max_shape=(8, 8))],
        torch_executed_ops={"torch.ops.aten.relu.default"},
    )
    _assert_runs_or_warns(
        warnings, compiled, model, (torch.randn(1, 8, device="cuda"),)
    )


@pytest.mark.unit
def test_compile_warns_when_every_size_of_an_input_is_computed(caplog):
    """No dimension of ``x`` is ``b`` itself, so no ``Input`` spec bounds ``b``.
    The program's own bounds still give the smallest size."""
    half = torch.export.Dim("half", min=1, max=8)
    model = _SmallLinear().eval().cuda()
    warnings, compiled = _profile_warnings(
        caplog,
        model,
        (torch.randn(8, 8, device="cuda"),),
        ({0: 2 * half},),
        [Input(min_shape=(2, 8), opt_shape=(8, 8), max_shape=(16, 8))],
    )
    _assert_runs_or_warns(
        warnings, compiled, model, (torch.randn(2, 8, device="cuda"),)
    )


@pytest.mark.unit
def test_compile_warns_when_compiled_from_example_tensors(caplog):
    example = (torch.randn(4, 8, device="cuda"), torch.randn(8, 8, device="cuda"))
    model = _Repeat().eval().cuda()
    warnings, compiled = _profile_warnings(
        caplog, model, example, ({0: _b}, {0: 2 * _b}), list(example)
    )
    smallest = (torch.randn(1, 8, device="cuda"), torch.randn(2, 8, device="cuda"))
    _assert_runs_or_warns(warnings, compiled, model, smallest)


@pytest.mark.unit
def test_compile_does_not_warn_for_a_computed_size_only_pytorch_reads(caplog):
    class TorchOnly(torch.nn.Module):
        def forward(self, x, z):
            return x.relu(), z.sin()

    model = TorchOnly()
    warnings, _ = _profile_warnings(
        caplog,
        model,
        (torch.randn(4, 8, device="cuda"), torch.randn(8, 8, device="cuda")),
        ({0: _b}, {0: 2 * _b}),
        _derived_inputs,
        torch_executed_ops={"torch.ops.aten.sin.default"},
    )
    assert warnings == []


@pytest.mark.unit
def test_compile_does_not_warn_about_sizes_the_input_spec_rules_out(caplog):
    """``x`` is ``2 * b``, and its ``Input`` spec starts at 8, so ``b`` starts at 4."""
    b = torch.export.Dim("b", min=1, max=8)
    model = _SmallLinear().eval().cuda()
    warnings, compiled = _profile_warnings(
        caplog,
        model,
        (torch.randn(8, 8, device="cuda"),),
        ({0: 2 * b},),
        [Input(min_shape=(8, 8), opt_shape=(8, 8), max_shape=(16, 8))],
    )
    assert warnings == []
    x = torch.randn(8, 8, device="cuda")
    torch.testing.assert_close(compiled(x), model(x), rtol=1e-3, atol=1e-3)


@pytest.mark.unit
def test_compile_warns_only_about_a_size_the_program_allows(caplog):
    """``2 * b`` with ``b`` from 0 is never 1. A size of 0 never enters an engine, so
    the smallest size that can is 2."""
    b = torch.export.Dim("b", min=0, max=8)
    model = _SmallLinear().eval().cuda()
    example = (torch.randn(8, 8, device="cuda"),)
    warnings, compiled = _profile_warnings(
        caplog, model, example, ({0: 2 * b},), list(example)
    )
    assert not any("that size is 1," in w for w in warnings), warnings
    _assert_runs_or_warns(
        warnings, compiled, model, (torch.randn(2, 8, device="cuda"),)
    )


@torch.library.custom_op("torchtrt_profile_test::positive_rows", mutates_args=())
def _positive_rows(x: torch.Tensor) -> torch.Tensor:
    return x[x[:, 0] > 0].clone()


@_positive_rows.register_fake
def _positive_rows_fake(x: torch.Tensor) -> torch.Tensor:
    rows = torch.library.get_ctx().new_dynamic_size(min=0, max=8)
    return x.new_empty((rows, x.shape[1]))


@pytest.mark.unit
def test_compile_warns_for_a_size_a_symbol_of_0_makes(caplog):
    """The linear layer reads ``2 * b + 2 * rows``, where ``rows`` can be 0, so its
    smallest size is 2, with ``b = 1``."""

    class Joined(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 8)

        def forward(self, x):
            rows = torch.ops.torchtrt_profile_test.positive_rows(x)
            return self.linear(torch.cat((x.repeat(2, 1), rows.repeat(2, 1))))

    model = Joined().eval().cuda()
    warnings, compiled = _profile_warnings(
        caplog,
        model,
        (torch.ones(4, 8, device="cuda"),),
        ({0: _b},),
        [Input(min_shape=(1, 8), opt_shape=(4, 8), max_shape=(8, 8))],
        torch_executed_ops={
            "torch.ops.aten.repeat.default",
            "torch.ops.aten.cat.default",
        },
    )
    _assert_runs_or_warns(
        warnings, compiled, model, (-torch.ones(1, 8, device="cuda"),)
    )


@pytest.mark.unit
def test_compile_names_the_smallest_size_a_symbol_of_0_makes(caplog):
    """``b + 1`` with ``b`` from 0 starts at 1, not at the 2 that ``b = 1`` gives."""
    b = torch.export.Dim("b", min=0, max=8)
    model = _SmallLinear().eval().cuda()
    warnings, compiled = _profile_warnings(
        caplog,
        model,
        (torch.randn(5, 8, device="cuda"),),
        ({0: b + 1},),
        [Input(min_shape=(1, 8), opt_shape=(5, 8), max_shape=(9, 8))],
    )
    assert not any("that size is 2," in w for w in warnings), warnings
    _assert_runs_or_warns(
        warnings, compiled, model, (torch.randn(1, 8, device="cuda"),)
    )


@pytest.mark.unit
def test_compile_does_not_warn_about_a_size_below_its_program_range(caplog):
    """``a`` is ``2 * b`` and the program starts ``b`` at 2, so ``a`` starts at 4,
    however far the ``Input`` spec of ``b + 1`` lets ``b`` go down."""

    class Two(torch.nn.Module):
        def forward(self, a, c):
            return a.sum() + c.sum()

    b = torch.export.Dim("b", min=2, max=8)
    example = (torch.randn(8, 8, device="cuda"), torch.randn(5, 8, device="cuda"))
    warnings, _ = _profile_warnings(
        caplog,
        Two(),
        example,
        ({0: 2 * b}, {0: b + 1}),
        [example[0], Input(min_shape=(1, 8), opt_shape=(5, 8), max_shape=(9, 8))],
    )
    assert not any("Input a " in w for w in warnings), warnings


@pytest.mark.unit
def test_compile_names_an_even_size_of_two_symbols_from_0(caplog):
    """``2 * T + 2 * U`` with both from 0 is never 1: its smallest size above 0 is 2."""

    class Joined(torch.nn.Module):
        def forward(self, x, y):
            return torch.cat((x.repeat(2, 1), y.repeat(2, 1))).abs() * 2 + 1

    T, U = torch.export.Dim("T", min=0, max=8), torch.export.Dim("U", min=0, max=8)
    example = (torch.randn(3, 8, device="cuda"), torch.randn(2, 8, device="cuda"))
    warnings, _ = _profile_warnings(
        caplog,
        Joined(),
        example,
        ({0: T}, {0: U}),
        list(example),
        torch_executed_ops={
            "torch.ops.aten.repeat.default",
            "torch.ops.aten.cat.default",
        },
    )
    assert not any("that size is 1," in w for w in warnings), warnings


@pytest.mark.unit
def test_compile_names_the_smallest_size_of_a_product(caplog):
    """``8 * b * r`` is 0 or at least 8, whichever symbol is raised above 0."""

    class Product(torch.nn.Module):
        def forward(self, x):
            flat = x.reshape(-1)
            return torch.cat((flat, flat)).abs() + 1

    b, r = torch.export.Dim("b", min=0, max=8), torch.export.Dim("r", min=0, max=8)
    example = (torch.randn(2, 3, 4, device="cuda"),)
    warnings, _ = _profile_warnings(
        caplog,
        Product(),
        example,
        ({0: b, 1: r},),
        list(example),
        torch_executed_ops={
            "torch.ops.aten.cat.default",
            "torch.ops.aten.view.default",
            "torch.ops.aten.reshape.default",
        },
    )
    assert not any(
        "that size is 1," in w or "that size is 2," in w for w in warnings
    ), warnings


@pytest.mark.unit
def test_compile_keeps_a_symbol_in_its_program_range(caplog):
    """``2 * b + 2`` with ``b`` from 2 starts at 6, though the ``Input`` spec of
    ``b + 1`` lets ``b`` go to 0."""

    class Repeated(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 8)

        def forward(self, x):
            return self.linear(x.repeat(2, 1))

    b = torch.export.Dim("b", min=2, max=8)
    warnings, _ = _profile_warnings(
        caplog,
        Repeated().eval().cuda(),
        (torch.randn(5, 8, device="cuda"),),
        ({0: b + 1},),
        [Input(min_shape=(1, 8), opt_shape=(5, 8), max_shape=(9, 8))],
        torch_executed_ops={"torch.ops.aten.repeat.default"},
    )
    assert not any(
        "that size is 2," in w or "that size is 4," in w for w in warnings
    ), warnings


class _Plain(torch.nn.Module):
    """``x`` has size ``b`` and ``z`` size ``2 * b``, in either order."""

    def __init__(self, z_first):
        super().__init__()
        self.z_first = z_first

    def forward(self, first, second):
        x, z = (second, first) if self.z_first else (first, second)
        return x.sum() + z.sum()


@pytest.mark.unit
def test_an_input_for_a_computed_size_does_not_hide_the_plain_one():
    """An ``Input`` for ``x`` past the exported maximum is refused whichever input
    comes first. The ``Input`` for ``z``, of size ``2 * b``, bounds ``b`` for the
    profile warning only, and changes the bounds that build profiles not at all."""
    b = torch.export.Dim("b", min=1, max=8)
    for z_first in (False, True):

        def specs(x_max):
            x = Input(min_shape=(1, 8), opt_shape=(4, 8), max_shape=(x_max, 8))
            z = Input(min_shape=(2, 8), opt_shape=(8, 8), max_shape=(15, 8))
            return [z, x] if z_first else [x, z]

        example = (torch.randn(8, 8), torch.randn(4, 8))
        shapes = ({0: 2 * b}, {0: b})
        if not z_first:
            example, shapes = example[::-1], shapes[::-1]
        ep = torch.export.export(
            _Plain(z_first), example, dynamic_shapes=shapes, strict=False
        )
        with pytest.raises(ValueError, match="Input.max_shape"):
            _build_user_symbol_bounds(ep.module(), specs(32), {}, ep.graph_signature)
        computed = {}
        bounds = _build_user_symbol_bounds(
            ep.module(),
            specs(8),
            {},
            ep.graph_signature,
            computed_size_bounds=computed,
        )
        assert list(bounds.values()) == [(1, 8)], bounds
        assert list(computed.values()) == [(1, 7)], computed


@pytest.mark.unit
def test_compile_rejects_input_min_0_for_a_dim_exported_from_0():
    """``range_constraints`` record ``Dim(min=0)`` as 0, but Torch-TensorRT starts
    every engine profile at 1 or more, so ``Input.min_shape=0`` is rejected at
    compile time rather than accepted and then refused by the engine at run time."""

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(4, 8, device="cuda")
    ep = torch.export.export(
        model,
        (sample,),
        dynamic_shapes=({0: torch.export.Dim("batch", min=0, max=8)},),
        strict=False,
    )
    input_min0 = Input(
        min_shape=(0, 8), opt_shape=(4, 8), max_shape=(8, 8), dtype=torch.float32
    )
    with pytest.raises(ValueError, match="starts every engine profile at 1"):
        torchtrt.dynamo.compile(ep, inputs=[input_min0], min_block_size=1)


@pytest.mark.unit
def test_build_user_symbol_bounds_raises_when_input_min_genuinely_below_export():
    """``user_min < exp_min`` (not the 1->2 case) must raise."""

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(10, 8, device="cuda")
    ep = torch.export.export(
        model,
        (sample,),
        dynamic_shapes=({0: torch.export.Dim("batch", min=10, max=20)},),
        strict=False,
    )
    input_too_low = Input(
        min_shape=(2, 8),
        opt_shape=(5, 8),
        max_shape=(10, 8),
        dtype=torch.float32,
    )
    with pytest.raises(ValueError) as exc_info:
        _build_user_symbol_bounds(ep.module(), [input_too_low], {}, ep.graph_signature)
    msg = str(exc_info.value)
    assert "Input.min_shape" in msg and "exported program" in msg
    assert "re-export" in msg.lower() or "Input.min_shape >=" in msg


@pytest.mark.unit
def test_build_user_symbol_bounds_raises_when_input_max_above_export():
    """``user_max > exp_max`` must raise (TRT would reject at runtime)."""

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(10, 8, device="cuda")
    ep = torch.export.export(
        model,
        (sample,),
        dynamic_shapes=({0: torch.export.Dim("batch", min=10, max=20)},),
        strict=False,
    )
    too_wide_input = Input(
        min_shape=(10, 8),
        opt_shape=(15, 8),
        max_shape=(25, 8),
        dtype=torch.float32,
    )
    with pytest.raises(ValueError) as exc_info:
        _build_user_symbol_bounds(ep.module(), [too_wide_input], {}, ep.graph_signature)
    msg = str(exc_info.value)
    assert "Input.max_shape" in msg and "exported program" in msg
    assert "re-export" in msg.lower() or "Input.max_shape <=" in msg


@pytest.mark.unit
def test_build_user_symbol_bounds_no_warning_on_matching_bounds(caplog):
    """Matching bounds must not warn or raise."""

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(10, 8, device="cuda")
    ep = torch.export.export(
        model,
        (sample,),
        dynamic_shapes=({0: torch.export.Dim("batch", min=10, max=20)},),
        strict=False,
    )
    matching_input = Input(
        min_shape=(10, 8),
        opt_shape=(15, 8),
        max_shape=(20, 8),
        dtype=torch.float32,
    )

    import logging

    with caplog.at_level(logging.WARNING, logger="torch_tensorrt.dynamo._compiler"):
        _build_user_symbol_bounds(ep.module(), [matching_input], {}, ep.graph_signature)

    warning_messages = [
        rec.message for rec in caplog.records if rec.levelno >= logging.WARNING
    ]
    assert not warning_messages, f"unexpected warnings: {warning_messages}"


@pytest.mark.unit
def test_build_user_symbol_bounds_narrows_to_user_on_subset(caplog):
    """Strict-subset Input narrows the engine profile to the user's bounds.
    No warning -- the user got exactly what they asked for; just an info log."""

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(10, 8, device="cuda")
    ep = torch.export.export(
        model,
        (sample,),
        dynamic_shapes=({0: torch.export.Dim("batch", min=10, max=20)},),
        strict=False,
    )
    subset_input = Input(
        min_shape=(12, 8),
        opt_shape=(15, 8),
        max_shape=(18, 8),
        dtype=torch.float32,
    )

    import logging

    with caplog.at_level(logging.INFO, logger="torch_tensorrt.dynamo._compiler"):
        bounds = _build_user_symbol_bounds(
            ep.module(), [subset_input], {}, ep.graph_signature
        )

    # No warning -- subset is a valid, intended narrowing.
    warning_messages = [
        rec.message for rec in caplog.records if rec.levelno >= logging.WARNING
    ]
    assert not warning_messages, f"unexpected warnings: {warning_messages}"

    # Narrowing must surface as an info log so users can see what envelope
    # their engine actually has.
    info_messages = [
        rec.message for rec in caplog.records if rec.levelno == logging.INFO
    ]
    assert any("Narrowing engine profile" in m for m in info_messages), info_messages

    # And the engine-side resolution must actually narrow to (12, 18).
    _, sym_dim = _first_sym_placeholder(ep, sym_dim=0)
    assert sym_dim is not None
    info = extract_var_range_info(sym_dim, user_symbol_bounds=bounds)
    assert info["min"] == 12, info
    assert info["max"] == 18, info


@pytest.mark.unit
def test_build_user_symbol_bounds_no_warning_on_dim_dynamic(caplog):
    """Dim.DYNAMIC + Input is the intended fill case -- no mismatch warning."""

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(2, 8, device="cuda")
    ep = torch.export.export(
        model,
        (sample,),
        dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
        strict=False,
    )
    fill_input = Input(
        min_shape=(1, 8),
        opt_shape=(4, 8),
        max_shape=(8, 8),
        dtype=torch.float32,
    )

    import logging

    with caplog.at_level(logging.WARNING, logger="torch_tensorrt.dynamo._compiler"):
        _build_user_symbol_bounds(ep.module(), [fill_input], {}, ep.graph_signature)

    mismatch_warnings = [
        rec.message
        for rec in caplog.records
        if rec.levelno >= logging.WARNING and "differ from the exporter" in rec.message
    ]
    assert not mismatch_warnings, f"unexpected mismatch warnings: {mismatch_warnings}"


@pytest.mark.unit
def test_build_user_symbol_bounds_mixed_tensor_and_symint():
    """Tensor + SymInt placeholders: only the tensor contributes to the map
    (SymInt is filtered by the ``isinstance(fake_val, torch.Tensor)`` guard)."""

    class _TensorAndScalar(torch.nn.Module):
        def forward(self, x, k):
            return x.relu() + k

    model = _TensorAndScalar().eval().cuda()
    x_sample = torch.randn(2, 8, device="cuda")
    k_sample = 4  # >1 to avoid 0/1 specialization

    dyn = torch.export.Dim.DYNAMIC
    ep = torch.export.export(
        model,
        (x_sample, k_sample),
        dynamic_shapes=({0: dyn}, dyn),
        strict=False,
    )

    placeholders = [n for n in ep.graph.nodes if n.op == "placeholder"]
    metas = [p.meta.get("val") for p in placeholders]
    if not any(isinstance(m, torch.SymInt) for m in metas):
        pytest.skip("export collapsed the scalar SymInt on this PyTorch")
    assert any(isinstance(m, torch.Tensor) for m in metas)

    tensor_input = Input(
        min_shape=(1, 8),
        opt_shape=(4, 8),
        max_shape=(8, 8),
        dtype=torch.float32,
    )
    # Index-aligned placeholder; never inspected (SymInt path skips it).
    scalar_input = Input(
        min_shape=(1,),
        opt_shape=(1,),
        max_shape=(1,),
        dtype=torch.int64,
    )

    bounds = _build_user_symbol_bounds(
        ep.module(), [tensor_input, scalar_input], {}, ep.graph_signature
    )
    assert len(bounds) == 1, bounds
    ((sym, (lo, hi)),) = bounds.items()
    assert isinstance(sym, sympy.Symbol)
    assert (lo, hi) == (1, 8)


@pytest.mark.unit
def test_build_user_symbol_bounds_mixed_arg_and_kwarg():
    """Positional args (matched by index) and kwargs (matched by name) both
    contribute to the map."""

    class _TwoTensors(torch.nn.Module):
        def forward(self, x, y):
            # Independent paths so the exporter doesn't unify x[0] and y[0].
            return x.relu(), y.relu()

    model = _TwoTensors().eval().cuda()
    # Distinct sizes discourage symbol unification.
    x_sample = torch.randn(2, 8, device="cuda")
    y_sample = torch.randn(3, 8, device="cuda")

    dyn = torch.export.Dim.DYNAMIC
    ep = torch.export.export(
        model,
        args=(x_sample,),
        kwargs={"y": y_sample},
        dynamic_shapes={"x": {0: dyn}, "y": {0: dyn}},
        strict=False,
    )

    placeholder_names = [n.target for n in ep.graph.nodes if n.op == "placeholder"]
    assert "x" in placeholder_names, placeholder_names
    assert "y" in placeholder_names, placeholder_names

    # Distinct bounds for x and y so we can verify the kwarg path by value.
    x_input = Input(
        min_shape=(1, 8),
        opt_shape=(2, 8),
        max_shape=(8, 8),
        dtype=torch.float32,
    )
    y_input = Input(
        min_shape=(2, 8),
        opt_shape=(3, 8),
        max_shape=(6, 8),
        dtype=torch.float32,
    )

    bounds = _build_user_symbol_bounds(
        ep.module(),
        sample_arg_inputs=[x_input],
        sample_kwarg_inputs={"y": y_input},
        graph_signature=ep.graph_signature,
    )

    bounds_values = set(bounds.values())
    assert (1, 8) in bounds_values, bounds  # positional-arg branch
    assert (2, 6) in bounds_values, bounds  # kwarg branch
    assert len(bounds) == 2, bounds


@pytest.mark.unit
@pytest.mark.critical
def test_dim_dynamic_save_preserves_range_constraints(tmpdir):
    """End-to-end: Dim.DYNAMIC + Input(max_shape) round-trips through
    ``torchtrt.save`` without mutating the exporter's range_constraints."""

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(2, 8, device="cuda")

    dyn_batch = torch.export.Dim.DYNAMIC
    exp_program = torch.export.export(
        model, (sample,), dynamic_shapes=({0: dyn_batch},), strict=False
    )

    # Snapshot before Torch-TRT touches anything.
    expected_constraints = {
        str(k): (v.lower, v.upper) for k, v in exp_program.range_constraints.items()
    }

    compile_spec = {
        "inputs": [
            Input(
                min_shape=(1, 8),
                opt_shape=(4, 8),
                max_shape=(8, 8),
                dtype=torch.float32,
            )
        ],
        "ir": "dynamo",
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)

    after_constraints = {
        str(k): (v.lower, v.upper) for k, v in exp_program.range_constraints.items()
    }
    assertions.assertEqual(expected_constraints, after_constraints)

    trt_ep_path = os.path.join(tmpdir, "trt_dim_dynamic.ep")
    torchtrt.save(
        trt_module,
        trt_ep_path,
        output_format="exported_program",
        arg_inputs=[sample],
        dynamic_shapes=({0: dyn_batch},),
        retrace=True,
    )

    reloaded = torch.export.load(trt_ep_path)
    reloaded_constraints = {
        str(k): (v.lower, v.upper) for k, v in reloaded.range_constraints.items()
    }
    assertions.assertEqual(expected_constraints, reloaded_constraints)

    # Engine accepts shapes across [min_shape, max_shape].
    trt_module(torch.randn(1, 8, device="cuda"))
    trt_module(torch.randn(4, 8, device="cuda"))
    trt_module(torch.randn(8, 8, device="cuda"))

    # And rejects shapes beyond ``Input.max_shape`` even though the eager
    # graph is unbounded -- ``Input(max_shape=8)`` is a strict cap. Pins
    # the contract against regressions that re-widen to the heuristic.
    too_big = torch.randn(16, 8, device="cuda")
    with assertions.assertRaises(Exception):
        trt_module(too_big)


@pytest.mark.unit
def test_construct_dynamic_input_uses_profile_midpoint():
    symbolic_shape = _constrained_dynamic_shape(1000, 1200)
    input_spec = construct_dynamic_input(
        symbolic_shape, torch.float32, name="dynamic_rows"
    )

    assert input_spec.shape["min_shape"] == (1000, 4)
    assert input_spec.shape["opt_shape"] == (1100, 4)
    assert input_spec.shape["max_shape"] == (1200, 4)


@pytest.mark.unit
def test_construct_dynamic_input_unbounded_zero_min_uses_nonzero_max():
    """An unbounded data-dependent extent must not collapse to [0, 0, 0]."""

    class UnboundedDynamicRows(torch.nn.Module):
        def forward(self, x):
            return torch.ops.torchtrt_profile_test.dynamic_rows(x)

    exported = torch.export.export(UnboundedDynamicRows(), (torch.ones(4, 4),))
    dynamic_rows = next(
        node
        for node in exported.graph.nodes
        if node.target == torch.ops.torchtrt_profile_test.dynamic_rows.default
    )
    symbolic_shape = dynamic_rows.meta["val"].shape
    range_info = extract_var_range_info(symbolic_shape[0])

    assert range_info["min"] == 0
    assert range_info["max"] is None

    input_spec = construct_dynamic_input(
        symbolic_shape, torch.float32, name="dynamic_rows"
    )
    assert input_spec.shape["min_shape"] == (0, 4)
    assert input_spec.shape["opt_shape"] == (2048, 4)
    assert input_spec.shape["max_shape"] == (4096, 4)


@pytest.mark.unit
def test_construct_dynamic_input_rejects_invalid_profile(monkeypatch):
    """Generated profiles are validated before reaching the TensorRT builder."""

    symbolic_shape = _constrained_dynamic_shape(1, 8)
    monkeypatch.setattr(
        "torch_tensorrt.dynamo.partitioning.common.extract_var_range_info",
        lambda *_args, **_kwargs: {"min": 1, "opt": 9, "max": 8},
    )

    with pytest.raises(
        ValueError,
        match=(
            "Invalid optimization profile for input 'dynamic_rows' at dimension 0: "
            "expected min <= opt <= max, but got min=1, opt=9, max=8"
        ),
    ):
        construct_dynamic_input(symbolic_shape, torch.float32, name="dynamic_rows")


@pytest.mark.unit
def test_largest_size_symbol_bound_is_treated_as_unbounded():
    symbolic_shape = _constrained_dynamic_shape(1, sys.maxsize - 1)
    range_info = extract_var_range_info(symbolic_shape[0])
    input_spec = construct_dynamic_input(
        symbolic_shape, torch.float32, name="dynamic_rows"
    )

    assert range_info["max"] is None
    assert input_spec.shape["min_shape"] == (1, 4)
    assert input_spec.shape["opt_shape"] == (2048, 4)
    assert input_spec.shape["max_shape"] == (4096, 4)


@pytest.mark.unit
def test_construct_dynamic_input_preserves_zero_minimum():
    """Data-dependent extents retain their legal zero profile minimum."""

    class Nonzero(torch.nn.Module):
        def forward(self, x):
            return torch.nonzero(x)

    exported = torch.export.export(
        Nonzero(), (torch.tensor([True, False, True, False]),)
    )
    nonzero = next(
        node
        for node in exported.graph.nodes
        if node.target == torch.ops.aten.nonzero.default
    )
    fake_value = nonzero.meta["val"]
    assert isinstance(fake_value.shape[0], torch.SymInt)

    input_spec = construct_dynamic_input(
        fake_value.shape, fake_value.dtype, name="nonzero_output"
    )
    assert input_spec.shape["min_shape"] == (0, 1)
    assert input_spec.shape["opt_shape"] == (2, 1)
    assert input_spec.shape["max_shape"] == (4, 1)


@pytest.mark.unit
def test_an_input_spec_that_rules_out_the_traced_size():
    """The program is traced with 4 rows, and the ``Input`` spec starts at 8."""

    class M(torch.nn.Module):
        def forward(self, x):
            return x * 2 + 1

    b = torch.export.Dim("b", min=1, max=8)
    for name, rows in (("plain", b), ("computed", 2 * b)):
        x = torch.randn(4, 4, device="cuda")
        ep = torch.export.export(M().cuda(), (x,), dynamic_shapes={"x": {0: rows}})
        trt = torchtrt.dynamo.compile(
            ep,
            inputs=[Input(min_shape=(8, 4), opt_shape=(8, 4), max_shape=(8, 4))],
            min_block_size=1,
        )
        y = torch.randn(8, 4, device="cuda")
        torch.testing.assert_close(trt(y), M()(y), msg=name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
