# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Lift mutated module buffers back to engine input bindings.

PyTorch's ``ExportedProgram.module()`` converts BUFFER placeholders into
``get_attr`` nodes that read the buffer from the GraphModule's state, plus a
trailing ``aten.copy_(get_attr_buffer, new_value)`` per BUFFER_MUTATION
output. From Torch-TensorRT's point of view those ``get_attr`` nodes are
parameters that get constant-folded; the buffer becomes baked into the engine
and the trailing ``copy_`` is dropped. Per-call buffer state is lost and the
KV-cache aliasing path cannot fire (the cache isn't a network input).

This module provides:

* :func:`lift_mutated_buffers` — pre-compile rewrite that turns each mutated
  buffer's ``get_attr`` into a ``placeholder`` and removes the trailing
  ``copy_``. The buffer becomes an engine input binding; downstream the
  slice_scatter converter's KV-cache fast path can recognize the cache as a
  network input and emit ``IKVCacheUpdateLayer`` with aliased I/O.

* :func:`inline_lifted_buffers_into_gm` — post-compile transform that
  registers each lifted buffer as state on the compiled GraphModule and
  rewrites the corresponding placeholder nodes to ``get_attr`` reads. The
  resulting module's ``forward`` takes only user inputs (buffers are
  threaded internally via the fx graph). Because everything is fx +
  module state, the result serializes naturally through
  ``torch_tensorrt.save`` / ``torch.export``.

What copy-back costs: only an engine-aliased (KV) write is free. A write routed to
copy-back re-attaches the *whole post-write buffer* as a graph output, because a
BUFFER_MUTATION output is defined as the buffer's new contents rather than the slice
that changed. Each call therefore pays, per copy-back buffer:

* the graph materializing the full buffer as that output — whatever applied the
  update produces the whole tensor rather than the changed slice, whether that is a
  scatter inside an engine or an op the partitioner left in PyTorch — and
* ExecuTorch's write-back pass copying that output into the caller-owned buffer
  after the delegate returns.

For a decode step writing one position into a multi-megabyte KV cache, that is a
full-cache-sized materialization plus a full-cache-sized copy to record a
single-slot update, per buffer, per call, and easily dominates the step. It is the
right correctness tradeoff — the alternative is silently losing the write — but it
is why the classifier routes a cache write to engine aliasing instead, and why a
model that unexpectedly lands in copy-back comes out slow rather than wrong.
``gm.meta['_copyback_mutation_buffers']`` lists which buffers are paying it.

Where the copy happens depends on who runs the module. A serialized program declares
the re-attached value a BUFFER_MUTATION, which the ExecuTorch runtime applies after
the delegate returns. A compiled module called directly from PyTorch applies it itself
at the end of ``forward`` (:func:`hide_copyback_outputs`), and does not return the
value. It returns the model's own outputs as a flat tuple, so a model with one
output gives a 1-tuple. A compiled module without copy-back returns them in the
model's own structure.
"""

from __future__ import annotations

import contextlib
import logging
import operator
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq
from torch_tensorrt.dynamo._settings import CompilationSettings

logger = logging.getLogger(__name__)


def _is_torch_executed(
    node: object,
    settings: Optional[CompilationSettings] = None,
) -> bool:
    """Whether the caller sent this node to PyTorch, by its op
    (``settings.torch_executed_ops``) or by a module it was traced through
    (``settings.torch_executed_modules``).

    Matched the way the partitioners match both, so the two agree. These are only
    some of the reasons a node can end up in PyTorch (a missing converter, a failed
    capability validator and others do the same), so this is a sufficient reason to
    rule engine aliasing out, never a proof that aliasing happens.
    ``assert_predicted_kv_aliased`` remains the ground-truth check for the rest.
    """
    if settings is None:
        return False
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"):
        return False
    from torch_tensorrt.dynamo.conversion._ConverterRegistry import ConverterRegistry
    from torch_tensorrt.dynamo.partitioning.common import (
        node_in_torch_executed_module,
    )

    excluded = set(settings.torch_executed_ops)
    target = node.target
    return (
        ConverterRegistry.qualified_name_or_str(target) in excluded
        or target in excluded
        or node_in_torch_executed_module(node, settings.torch_executed_modules)
    )


def _effective_cache_input(cache_node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """The node the converter will see as this write's cache argument, if it is a
    network input, else ``None``.

    A lifted buffer is a placeholder, and ``emit_kv_cache_update_layer`` aliases the
    cache only when the argument it is handed is one -- anything else has no input
    binding name and the converter falls back to a non-aliasing concatenation or
    scatter.

    One rewrite stands between this classification and the converter:
    ``remove_input_alias_fixing_clones``, an ATEN lowering pass inside
    ``post_lowering``, erases a ``clone`` whose placeholder has no other user, which
    is the shape ``clone -> scatter -> copy_`` leaves behind once lifting has erased
    the ``copy_``. The converter then does see a direct network input and does emit
    the KV layer, so this has to predict what the converter will be handed rather
    than what is in the graph now. Getting it wrong in that direction is worse than
    a lost write-back: the write is filed copy-back, its value is re-attached as a
    trailing graph output, and the engine aliases the cache anyway -- so the runtime
    truncates that output as an engine side effect while the outer graph still reads
    it, and every call raises ``IndexError``.

    The pass's own condition is reproduced exactly, sole user and all: a placeholder
    with another reader keeps its clone, the converter sees a ``call_function``, and
    nothing aliases.

    Predicting the peel is not free. A cloned cache write is classified KV, so if no
    engine goes on to claim it -- at the *default* ``min_block_size=5`` a small
    model's subgraph often does not -- ``assert_predicted_kv_aliased`` fails the
    compile rather than leaving the buffer at its compile-time value. A loud failure
    beats a silently stale buffer, and this failure is reachable at the default
    ``min_block_size``, not only at ``min_block_size=1``.
    """
    if cache_node.op == "placeholder":
        return cache_node
    if not (
        cache_node.op == "call_function"
        and cache_node.target is torch.ops.aten.clone.default
        and cache_node.args
        and isinstance(cache_node.args[0], torch.fx.Node)
        and cache_node.args[0].op == "placeholder"
    ):
        return None
    placeholder = cache_node.args[0]
    users = list(placeholder.users)
    if len(users) == 1 and users[0] is cache_node:
        return placeholder
    return None


def _kv_write_will_alias(
    value_node: object,
    cache_shape: Tuple[int, ...],
    settings: Optional[CompilationSettings] = None,
    *,
    cache_input: Optional[torch.fx.Node] = None,
) -> bool:
    """Whether the converter will emit an ``IKVCacheUpdateLayer`` (with in-place
    aliased I/O) for this mutated buffer's new-value node.

    Eligibility depends on the write dim, static shapes, and the cache being a
    direct network input, so this reuses the converters' own eligibility
    predicates -- and, for ``slice_scatter``, the converter's own derivation of
    the arguments those predicates take (:func:`resolve_slice_scatter_write`), since
    a divergence there mispredicts just as effectively as a divergence in the
    predicate. Returning ``False`` routes the write to copy-back, which is right
    whatever the converter goes on to do with it: it lowers most ineligible writes
    to a non-aliasing concatenation or scatter, returns the source in the input's
    dtype for a full overwrite, and raises for the rest (a ``slice_scatter`` with
    dynamic bounds or a bad ``dim``, an ``index_copy`` its fallback cannot express).
    The first two both have a write-back to preserve -- for a full overwrite the
    source *is* the buffer's new contents -- and the ones that raise never get as far
    as needing one. One more never reaches a converter: a ``slice_scatter`` written
    on a dynamic dim, which its validator runs in PyTorch, and which needs the
    copy-back like any other.
    ``cache_input`` names the placeholder the converter will see as the cache, when
    the caller has already worked it out.
    Imports are local to avoid a lowering<->conversion import cycle.
    """
    if not (isinstance(value_node, torch.fx.Node) and value_node.op == "call_function"):
        return False
    args = value_node.args
    # An op or module the caller excluded from TensorRT never reaches a converter, so it
    # cannot emit an IKVCacheUpdateLayer and the engine will not alias it. Without
    # this the write is classified as engine-aliased, its copy_ is dropped, and
    # compile() later fails its own aliased_io cross-check. It still gets lifted and
    # copied back like any other non-aliasing write.
    if _is_torch_executed(value_node, settings):
        return False
    # The KV layer aliases the cache only if it is a direct network input; after
    # lifting, the mutated buffer is a placeholder the write op reads from.
    if not (args and isinstance(args[0], torch.fx.Node)):
        return False
    if cache_input is None:
        cache_input = _effective_cache_input(args[0])
    if cache_input is None:
        return False

    if value_node.target is torch.ops.aten.index_copy.default:
        from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
            _index_copy_kv_eligible,
        )

        # The validator applies its own placeholder check to args[0], and here that
        # can still be the clone lowering is about to erase. Point it at what the
        # converter will be handed instead; the partitioner, running after lowering,
        # passes nothing and reads args[0] itself.
        return bool(_index_copy_kv_eligible(value_node, input_node=cache_input))

    if value_node.target is torch.ops.aten.slice_scatter.default:
        from torch_tensorrt.dynamo.conversion.impl.slice_scatter import (
            KVWriteStatus,
            _kv_eligible,
            resolve_slice_scatter_write,
        )

        # Any status other than OK is a case in which the converter returns or
        # raises before it reaches _kv_eligible, so nothing aliases the cache.
        # Returning False routes the write to copy-back -- the copy_ is erased
        # either way, and the new value is re-attached as a graph output instead. A
        # full overwrite needs that: it returns the source, emits no KV layer, and
        # its write still has to be recorded. Two of the rest raise out of the
        # converter, so the compile aborts and the output is never reached; the
        # dynamic-dim one is validated away to PyTorch and keeps its copy-back.
        dim = args[2] if len(args) > 2 else 0
        start, end, step, status = resolve_slice_scatter_write(
            tuple(cache_shape),
            dim,
            args[3] if len(args) > 3 else None,
            args[4] if len(args) > 4 else None,
            args[5] if len(args) > 5 else None,
        )
        if status is not KVWriteStatus.OK:
            return False
        # OK is the only status that resolves all three bounds to Python ints.
        assert start is not None and end is not None and step is not None
        eligible, _reason = _kv_eligible(
            tuple(cache_shape), dim, start, end - start, step
        )
        return bool(eligible)

    return False


_WRITE_OPS = (
    torch.ops.aten.index_copy.default,
    torch.ops.aten.slice_scatter.default,
)


def _is_cache_write(node: object) -> bool:
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target in _WRITE_OPS
    )


def _in_place_write(
    value: torch.fx.Node,
    cache: torch.fx.Node,
    settings: Optional[CompilationSettings],
    *,
    write_back: Optional[torch.fx.Node] = None,
    stored: Iterable[torch.fx.Node] = (),
) -> Optional[torch.fx.Node]:
    """The write that produces ``value``, if an engine can apply it to ``cache``'s
    memory in place; otherwise ``None``.

    ``cache`` is a placeholder whose new contents are ``value``. ``write_back`` is the
    ``copy_`` that stores ``value`` into ``cache``, when it is still in the graph.
    ``stored`` are the new values of the other written buffers, whose ``copy_`` is
    already gone: each is read, to be stored, as if the graph returned it.

    The KV layer updates only a network input, so the write has to be a single
    eligible write of ``cache`` itself, in one engine; a buffer written twice is
    copied back. Nothing else may read the cache's old contents. TensorRT orders the
    write only after the layers it depends on, and may compute an op that only moves
    data, such as a copy, a cast to the same type or a reshape, from the cache again
    after the write, even when the write depends on it. So any other reader of the
    old contents could see the new ones. Ops that read the new contents run after
    the write, and a returned value that is the cache's memory gets its own copy in
    the engine (``TRTInterpreter.output``).
    """
    write = _cache_write_of(value, cache)
    if not (
        write is not None
        and isinstance(write.args[0], torch.fx.Node)
        and _effective_cache_input(write.args[0]) is cache
        and _kv_write_will_alias(
            write, tuple(cache.meta["val"].shape), settings, cache_input=cache
        )
    ):
        return None
    if _reads_old_contents(cache, write, {write, value, write_back}, set(stored)):
        return None
    if any(_is_torch_executed(node, settings) for node in (write, value)):
        return None
    return write


def _reads_old_contents(
    cache: torch.fx.Node,
    write: torch.fx.Node,
    stop: Set[Optional[torch.fx.Node]],
    stored: Set[torch.fx.Node],
) -> bool:
    """Whether a node other than those in ``stop`` reads the data of ``cache``
    before ``write`` updates it: directly, or through the clone lowering peels off
    (see :func:`_effective_cache_input`). A node whose result is used for its shape
    alone, as a slice assignment uses the slice it copies into, reads nothing. A
    value in ``stored``, which another buffer stores, counts as read.
    """
    sources = {cache, write.args[0]}
    if sources & stored:
        return True
    for source in sources:
        for user in source.users:
            if user in sources or user in stop or not _uses_data_of(user, source):
                continue
            if (
                user.op == "output"
                or user in stored
                or (
                    user.op == "call_function"
                    and any(_uses_data_of(u, user) for u in user.users)
                )
            ):
                return True
    return False


def _uses_data_of(user: torch.fx.Node, node: torch.fx.Node) -> bool:
    """Whether ``user`` reads the values of ``node``, rather than only its shape:
    a metadata check does not, and neither does ``copy(dst, src)`` of its ``dst``,
    whose result holds the values of ``src``."""
    if user.target is torch.ops.aten._assert_tensor_metadata.default:
        return False
    return not (
        user.target is torch.ops.aten.copy.default
        and user.args[0] is node
        and user.args[1:2] != (node,)
    )


def _cache_write_of(
    value: torch.fx.Node, cache: torch.fx.Node
) -> Optional[torch.fx.Node]:
    """The ``index_copy`` or ``slice_scatter`` that produces ``value``, if any.

    ``cache.copy_(write)`` in the model reaches here as ``copy(cache, write)``,
    which converts to the write itself when the two match in shape and dtype.
    """
    if (
        value.target is torch.ops.aten.copy.default
        and value.args[0] is cache
        and isinstance(value.args[1], torch.fx.Node)
        and _same_shape_and_dtype(value, value.args[1])
    ):
        value = value.args[1]
    return value if _is_cache_write(value) else None


def _engine_takes_its_memory(tensor: torch.Tensor) -> bool:
    """Whether an engine is handed this tensor's own memory.

    A non-contiguous tensor reaches it as a contiguous copy, and a float64 one as a
    float32 copy (TensorRT has no float64), so the write would land in the copy.
    """
    return bool(tensor.is_contiguous()) and tensor.dtype != torch.float64


def _same_shape_and_dtype(a: torch.fx.Node, b: torch.fx.Node) -> bool:
    """Whether the two values have one shape and dtype for every input. A symbolic
    comparison that is not known to hold is False rather than a guard, which would
    specialize the engine to the traced sizes or fail on a data-dependent size."""
    va, vb = a.meta.get("val"), b.meta.get("val")
    return (
        isinstance(va, torch.Tensor)
        and isinstance(vb, torch.Tensor)
        and va.dtype == vb.dtype
        and statically_known_true(sym_eq(va.shape, vb.shape))
    )


def aliased_input_bindings(aliased_io_maps: Iterable[Any]) -> Set[str]:
    """Collect the input-binding side of one or more ``aliased_io`` maps.

    Each map is ``output binding -> (input binding, kind)``, and a bare input binding
    is tolerated in place of the pair.
    """
    bindings: Set[str] = set()
    for amap in aliased_io_maps:
        if not amap:
            continue
        for v in amap.values():
            bindings.add(v[0] if isinstance(v, (tuple, list)) else v)
    return bindings


def assert_predicted_kv_aliased(
    aliased_in: Set[str],
    predicted_kv_bindings: List[str],
    settings: Optional[CompilationSettings] = None,
    *,
    engines_built: bool = True,
    copyback_bindings: Iterable[str] = (),
) -> None:
    """Ground-truth check on both directions of the classification.

    Each write ``lift_mutated_buffers`` classified as KV-aliased had its ``copy_``
    dropped in the expectation that the engine would alias it in place. If the
    converter did not actually emit an ``IKVCacheUpdateLayer`` for it, that
    write-back is silently lost. So assert every predicted-KV input binding
    appears in a compiled engine's ``aliased_io``, and raise loudly otherwise.
    Keyed on the ``buf_*`` binding name, which is stable across the later buffer
    rename and is what ``aliased_io`` records on the input side.

    The other direction fails just as badly and is not the same check. A write filed
    copy-back has its new value re-attached as a trailing graph output; if the engine
    aliases that buffer anyway, the runtime truncates the aliased output as an engine
    side effect while the outer graph still reads it, and the module raises on every
    call. So ``copyback_bindings`` -- the input bindings of the writes filed
    copy-back -- must be absent from ``aliased_io``.

    ``aliased_in`` is the ground truth, supplied by the caller because the two entry
    points hold it in different shapes: :func:`compile` reads it off the compiled
    submodules, while the engine converter has it on the interpreter result.

    ``engines_built=False`` says the caller produced no engine, so ``aliased_in`` is
    empty for reasons that say nothing about the predictions and every one of them
    would look unfulfilled. The check is skipped there instead. :func:`compile` passes
    ``not settings.dryrun``, because a dryrun returns before conversion; the engine
    converter takes ``dryrun`` too but never acts on it and builds either way, so it
    keeps the default.
    """
    copyback_bindings = list(copyback_bindings)
    if not predicted_kv_bindings and not copyback_bindings:
        return
    if not engines_built:
        logger.debug(
            "no engines were built, so the aliasing cross-check for predicted-KV "
            "%s and copy-back %s has nothing to verify against and is skipped",
            predicted_kv_bindings,
            copyback_bindings,
        )
        return
    missing = [b for b in predicted_kv_bindings if b not in aliased_in]
    if missing:
        cause = "the write did not end up inside a TensorRT engine"
        if settings is not None:
            if settings.min_block_size > 1:
                cause += (
                    f", most often because min_block_size "
                    f"({settings.min_block_size}) rejected the subgraph it landed "
                    "in; min_block_size=1 rules that out"
                )
            else:
                # The sibling branch's remedy is the value already in force here, so
                # naming min_block_size at all would read as "1 rejected the
                # subgraph; set it to 1" -- the value blamed and the value
                # recommended being the same is no remedy the reader can act on.
                # Name the cause instead.
                cause += (
                    ", most often because a converter or a capability validator "
                    "rejected the op"
                )
        raise RuntimeError(
            "lift_mutated_buffers classified these buffer writes as KV-cache "
            "(engine-aliased) and dropped their copy_, but the compiled engine "
            f"did not alias them (absent from aliased_io): {missing}. Their "
            "write-back would be silently dropped. The classification runs before "
            f"lowering and partitioning, so: {cause}."
        )

    unexpected = [b for b in copyback_bindings if b in aliased_in]
    if unexpected:
        # What raising here prevents: the runtime treats an aliased output as an
        # engine side effect and does not return it, so the surrounding graph reads
        # past the end of the results and the module raises on every call.
        raise RuntimeError(
            "lift_mutated_buffers classified these buffer writes as copy-back (not "
            "engine-aliased) and re-attached their new values as trailing graph "
            "outputs, but the compiled engine aliased them in place anyway (present "
            f"in aliased_io): {unexpected}. The classification runs before lowering, "
            "so a lowering pass rewrote the write into a form the converter could "
            "alias after all -- remove_input_alias_fixing_clones erasing a clone of "
            "the cache is the known case."
        )


def erase_export_guards(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Remove ``torch.export`` ``_guards_fn`` call_module nodes.

    The partitioner does not delete these nodes: AccNodesFinder marks them
    unsupported, ``split()`` parks them on ``_run_on_gpu_*``, and
    ``compile_module`` skips convert for non-``_run_on_acc`` children. Direct
    ``convert_module(gm)`` would otherwise try to compile ``GuardsFn``.
    """
    for node in list(gm.graph.nodes):
        if (
            node.op == "call_module"
            and isinstance(node.target, str)
            and node.target == "_guards_fn"
        ):
            name = node.target
            gm.graph.erase_node(node)
            if name in dict(gm.named_children()):
                gm.delete_submodule(name)
            break
    gm.graph.lint()
    return gm


def lift_mutated_buffers(
    gm: torch.fx.GraphModule,
    settings: Optional[CompilationSettings] = None,
    *,
    lift_buffers: bool = True,
    host_inputs_copied: bool = True,
) -> Tuple[torch.fx.GraphModule, List[Tuple[str, str, torch.Tensor]]]:
    """Lift each mutated buffer from a ``get_attr`` to a ``placeholder``.

    A mutated buffer is identified by a trailing
    ``aten.copy_(get_attr_buffer, new_value)`` pattern, which is how
    ``ExportedProgram.module()`` represents a BUFFER_MUTATION.

    Returns ``(new_gm, lifted)`` where:

    * ``new_gm`` is a plain ``torch.fx.GraphModule`` whose ``forward``
      signature reflects the updated placeholder set. Necessary because
      ``ExportedProgram.module()`` produces a module whose forward is
      fixed by a pytree spec — recompiling alone doesn't pick up new
      placeholders.
    * ``lifted`` is a list of ``(placeholder_name, buffer_name, buffer_tensor)``
      tuples, in the order placeholders were appended (which matches the
      order they appear in the new gm's forward signature, after the
      original user inputs).

    Side effects: the trailing ``copy_`` of each mutated buffer is erased. A buffer
    whose write an engine can apply to its memory in place (an eligible
    ``slice_scatter`` / ``index_copy``, per :func:`_in_place_write`) relies on that
    engine aliasing for its write-back, so nothing further is added, and the write
    gets ``node.meta['_trt_kv_alias']``. Every other mutation has no engine aliasing, so
    its new value is re-appended as a graph output ("copy-back") and its buffer name
    recorded, in output order, in ``gm.meta['_copyback_mutation_buffers']`` -- the
    downstream exporters (``create_trt_exp_program`` /
    ``_declare_aliased_kv_mutations_on_ep``) read that list to reclassify those
    outputs as BUFFER_MUTATIONs. Copy-back is correct but not free; see the module
    docstring for what each such buffer costs per call.

    A caller's tensor that the model writes in place, a ``copy_`` into a placeholder,
    is classified by the same rule. Its ``copy_`` stays: it is what stores the write
    when the engine does not, so for an ``index_copy`` or ``slice_scatter`` that is
    not done in place it gets ``node.meta['_trt_write_back']``, which keeps it when
    nothing reads it. Other in-place ops on a caller's tensor keep their old
    behaviour.

    With ``lift_buffers=False`` the buffers stay ``get_attr`` constants of the
    engine and only the caller's tensors are classified. That is for the entry
    points whose result has no buffer state to update.

    The PyTorch runtimes give an engine a CUDA copy of a host tensor, so a caller's
    tensor traced on the host is not written in place. With
    ``host_inputs_copied=False``, for a serialized engine whose caller binds the
    memory itself, the example's device does not matter.

    The markers go on the write *nodes* while the classification is per *buffer*, so
    one value node written back into two buffers would put the two in conflict.
    ``torch.export`` rejects that shape with ``SpecViolationError`` before the
    classifier sees it, so it is a constraint on what can arrive here rather than a
    case to guard against.
    """
    # Find all aten.copy_(get_attr_X, _) calls. The first arg's target is
    # the buffer name. Some EPs emit copy_.default, others copy_.
    mutation_pairs: List[Tuple[torch.fx.Node, torch.fx.Node]] = (
        []
    )  # (copy_node, get_attr_node)
    # (copy_node, placeholder) for a caller's tensor the model writes.
    input_writes: List[Tuple[torch.fx.Node, torch.fx.Node]] = []
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if node.target not in (torch.ops.aten.copy_.default, torch.ops.aten.copy_):
            continue
        if not node.args:
            continue
        target = node.args[0]
        if isinstance(target, torch.fx.Node) and target.op == "get_attr":
            if lift_buffers:
                mutation_pairs.append((node, target))
        elif (
            isinstance(target, torch.fx.Node)
            and target.op == "placeholder"
            and len(node.args) > 1
            and isinstance(node.args[1], torch.fx.Node)
        ):
            input_writes.append((node, target))

    if not mutation_pairs and not input_writes:
        return gm, []

    # Find the position to insert new placeholders (after the last existing placeholder).
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    insert_after = placeholders[-1] if placeholders else None

    lifted: List[Tuple[str, str, torch.Tensor]] = []
    seen_buffers: Dict[str, torch.fx.Node] = {}  # buffer name -> new placeholder node

    # Each mutated buffer's write-back is handled one of two ways downstream:
    #   - Engine-level aliasing (zero-copy): the slice_scatter / index_copy
    #     converters emit an IKVCacheUpdateLayer whose output is aliased in-place
    #     to the cache input. We drop the copy_ and rely on that aliasing.
    #   - Copy-back: any other mutation (a non-KV buffer such as a convolution-
    #     state ring-buffer, OR a slice_scatter / index_copy the converter cannot
    #     turn into an IKVCacheUpdateLayer) has no aliasing, so its new value is
    #     re-attached as an ordinary BUFFER_MUTATION output that ExecuTorch copies
    #     back after the delegate runs.
    # `_kv_write_will_alias` decides between them by reusing the converters' own
    # eligibility predicates (not the op target alone), so a non-aliasable
    # slice_scatter / index_copy falls to copy-back rather than being dropped.
    # index_put has no aliasing converter, so it always falls to copy-back too.
    # (new_value_node, buffer_name, input_binding_name)
    copyback: List[Tuple[torch.fx.Node, str, str]] = []
    # (new_value_node, buffer_name, placeholder, buffer_tensor), classified below.
    written: List[Tuple[torch.fx.Node, str, torch.fx.Node, torch.Tensor]] = []
    # Input-binding names of writes predicted to alias (KV). compile() asserts each
    # actually appears in the engine's aliased_io, and that no copy-back binding
    # does, turning either misprediction into a loud error instead of a silently
    # dropped write-back or a module that raises on every call. The binding name
    # (buf_*) is the stable key: it survives the buffer renaming inline does later,
    # and it is exactly what aliased_io records on the input side.
    predicted_kv_bindings: set[str] = set()

    for copy_node, get_attr_node in mutation_pairs:
        buffer_name = get_attr_node.target
        # A get_attr target is fully qualified, so a buffer owned by a submodule
        # arrives as "layers.0.self_attn.kv_cache.k_cache". getattr does not walk a
        # dotted path, so it reports every nested buffer as missing; get_buffer
        # resolves through the submodules.
        try:
            buffer_tensor = gm.get_buffer(buffer_name)
        except AttributeError:
            logger.warning(
                "lift_mutated_buffers: get_attr target %s not found on gm; skipping",
                buffer_name,
            )
            continue
        if not isinstance(buffer_tensor, torch.Tensor):
            logger.debug(
                "lift_mutated_buffers: attribute %s is not a Tensor; skipping",
                buffer_name,
            )
            continue

        if buffer_name in seen_buffers:
            # Same buffer mutated more than once — already lifted; just remove
            # this copy_ node and rely on the existing placeholder.
            replacement = seen_buffers[buffer_name]
        else:
            # Build a unique placeholder name from the buffer name.
            placeholder_name = "buf_" + buffer_name.replace(".", "_")
            base = placeholder_name
            suffix = 0
            existing = {n.name for n in gm.graph.nodes}
            while placeholder_name in existing:
                suffix += 1
                placeholder_name = f"{base}_{suffix}"

            if insert_after is not None:
                with gm.graph.inserting_after(insert_after):
                    new_ph = gm.graph.placeholder(placeholder_name)
            else:
                # No existing placeholders — insert at graph start.
                with gm.graph.inserting_before(next(iter(gm.graph.nodes))):
                    new_ph = gm.graph.placeholder(placeholder_name)
            new_ph.meta["val"] = get_attr_node.meta.get(
                "val",
                torch.empty_like(buffer_tensor, device="meta"),
            )
            new_ph.meta["_lifted_buffer"] = buffer_name
            insert_after = new_ph
            seen_buffers[buffer_name] = new_ph
            replacement = new_ph
            lifted.append((placeholder_name, buffer_name, buffer_tensor))

        # Re-route every use of the original get_attr (other than the copy_ itself)
        # to the new placeholder.
        get_attr_node.replace_all_uses_with(replacement)

        # KV writes rely on engine-level aliasing, so the trailing copy_ is
        # redundant and dropped. Other (non-KV) mutations have no aliasing:
        # record their new value so we can re-attach it as a copy-back
        # BUFFER_MUTATION output below.
        #
        # Drop the (now input-target) copy_ first, so the classification reads the
        # graph the lowering passes and the partitioner will read. The copy_ is the
        # last reader of the buffer placeholder in the ordinary write-only shape, and
        # whether it is still there decides whether a clone of the cache survives
        # into conversion.
        new_value = copy_node.args[1] if len(copy_node.args) > 1 else None
        gm.graph.erase_node(copy_node)
        if isinstance(new_value, torch.fx.Node):
            written.append((new_value, buffer_name, replacement, buffer_tensor))

        # Erase the now-unused get_attr.
        if not get_attr_node.users:
            gm.graph.erase_node(get_attr_node)

    # Classified once every buffer is lifted, so each decision sees the graph the
    # partitioner will see, plus the values the erased copy_ nodes store.
    stored = [new_value for new_value, _, _, _ in written]
    for new_value, buffer_name, replacement, buffer_tensor in written:
        write = (
            _in_place_write(
                new_value,
                replacement,
                settings,
                stored=[v for v in stored if v is not new_value],
            )
            if _engine_takes_its_memory(buffer_tensor)
            else None
        )
        # A value nothing else reads is dead once its copy_ is gone, and dead-code
        # elimination would take it before any engine saw it. Copy-back keeps it.
        if write is not None and new_value.users:
            predicted_kv_bindings.add(replacement.name)
            write.meta["_trt_kv_alias"] = True
        else:
            copyback.append((new_value, buffer_name, replacement.name))
            # A clone of this buffer holds its old value past the point where
            # forward writes the new one, so lowering must keep it.
            replacement.meta["_copyback_buffer"] = True

    # A caller's tensor that the model writes keeps its write-back copy_. While
    # something reads that copy_'s result, as when the model returns the tensor, it
    # runs in PyTorch after the engines and stores the write. Otherwise dead-code
    # elimination would drop it, so an index_copy or slice_scatter the engine does
    # not do in place marks it with _trt_write_back, which keeps it. Other in-place
    # ops on a caller's tensor keep their old behaviour.
    for copy_node, cache in input_writes:
        new_value = copy_node.args[1]
        val = cache.meta.get("val")
        write = (
            _in_place_write(
                new_value, cache, settings, write_back=copy_node, stored=stored
            )
            if isinstance(val, torch.Tensor)
            and (val.is_cuda or not host_inputs_copied)
            and _engine_takes_its_memory(val)
            and len(new_value.users) > 1
            else None
        )
        if write is not None:
            predicted_kv_bindings.add(cache.name)
            write.meta["_trt_kv_alias"] = True
        elif _cache_write_of(new_value, cache) is not None:
            copy_node.meta["_trt_write_back"] = True

    # Re-attach non-KV mutation new-values as graph outputs so they survive
    # compilation (otherwise, with the copy_ gone, they are dead and eliminated).
    # Appended in order; recorded so create_trt_exp_program / _declare can tag the
    # corresponding engine outputs as BUFFER_MUTATION (copy-back) downstream.
    copyback_buffers: List[str] = []
    copyback_bindings: List[str] = []
    if copyback:
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        out_args = list(output_node.args[0])
        out_args.extend(nv for nv, _, _ in copyback)
        output_node.args = (tuple(out_args),)
        copyback_buffers = [buf for _, buf, _ in copyback]
        copyback_bindings = sorted({binding for _, _, binding in copyback})

    gm.graph.lint()

    if not lifted:
        gm.meta["_copyback_mutation_buffers"] = copyback_buffers
        gm.meta["_copyback_bindings"] = copyback_bindings
        gm.meta["_predicted_kv_bindings"] = sorted(predicted_kv_bindings)
        return gm, []

    # ExportedProgram.module() produces a GraphModule whose forward is
    # generated by a ``_PyTreeCodeGen`` baked into the graph: the body
    # unpacks args through a stored pytree spec, ignoring any added
    # placeholders. Rebuild the gm with the default ``CodeGen`` so the
    # forward signature reflects the placeholder set as written.
    # First remove the call to ``_guards_fn`` (generated for the original
    # arity; would fail after lifting).
    gm = erase_export_guards(gm)

    # Reset codegen to the plain CodeGen so the forward args = placeholders.
    gm.graph.set_codegen(torch.fx.graph.CodeGen())
    gm.graph.lint()

    new_gm = torch.fx.GraphModule(gm, gm.graph)
    for attr in ("_in_spec", "_out_spec"):
        if hasattr(new_gm, attr):
            try:
                delattr(new_gm, attr)
            except AttributeError:
                pass
    new_gm.recompile()
    new_gm.meta["_copyback_mutation_buffers"] = copyback_buffers
    new_gm.meta["_copyback_bindings"] = copyback_bindings
    new_gm.meta["_predicted_kv_bindings"] = sorted(predicted_kv_bindings)

    logger.debug(
        "Lifted %d mutated buffer(s) to placeholders: %s; copy-back buffers: %s",
        len(lifted),
        [(p, b) for p, b, _ in lifted],
        copyback_buffers,
    )

    return new_gm, lifted


class _HiddenCopybackOutputs(torch.fx.graph.CodeGen):  # type: ignore[misc]
    """Generate a ``forward`` that writes the trailing copy-back values into their
    buffers instead of returning them.

    The values stay on the graph's output node; only the generated ``forward`` differs.
    Everything that reads the graph rather than calling the module still sees them as
    outputs -- the exporters, dead-code elimination, and the ``torch.fx.Interpreter``
    that a *non-strict* ``torch.export`` runs a GraphModule through, so a retrace
    keeps them too and declares each one a BUFFER_MUTATION. Non-strict is what both
    retrace paths get: ``_compile.save`` passes ``strict=False`` and
    ``dynamo._exporter.export`` takes the default, which is ``False``. A strict
    ``torch.export`` of the compiled module traces ``forward`` itself, where the
    writes are plain ``copy_`` calls into the buffers, so it records them as buffer
    mutations. ``_TorchTensorRTModule.forward`` hides the aliased outputs the
    interpreter appends the same way: the engine produces them, the caller never sees
    them.
    """

    def __init__(
        self, buffers: Sequence[str], copied: Sequence[bool], helper: str
    ) -> None:
        super().__init__()
        self.buffers = list(buffers)
        self.copied = list(copied)
        # The name forward calls write_copyback_buffers by. The generated code can
        # name nothing else from outside: an input may be called anything, even
        # getattr.
        self.helper = helper

    def additional_globals(self) -> List[Tuple[str, Any]]:
        return [(self.helper, write_copyback_buffers)]

    def generate_output(self, output_args: Any, *args: Any, **kwargs: Any) -> str:
        write_back = ""
        if self.buffers and isinstance(output_args, (list, tuple)):
            split = len(output_args) - len(self.buffers)
            repr_fn = kwargs.get("repr_fn") or repr
            names = "".join(f"{name!r}, " for name in self.buffers)
            values = "".join(
                f"{repr_fn(value)}{'.clone()' if copied else ''}, "
                for value, copied in zip(output_args[split:], self.copied)
            )
            write_back = f"{self.helper}(self, ({names}), ({values}))\n"
            output_args = output_args[:split]
            # descs, when the caller passes it, is one entry per untruncated output.
            kwargs.pop("descs", None)
        return write_back + str(super().generate_output(output_args, *args, **kwargs))


class _CopybackTracer(torch.fx.Tracer):  # type: ignore[misc]
    """Rebuilds a pickled compiled module from its ``forward``.

    ``torch.load`` traces the saved ``forward`` again to get the graph back. The
    default tracer reads buffers as plain tensors, so it would run the copy-back
    writes then, once, and drop them from the rebuilt ``forward``. Reading them as
    graph nodes keeps the writes in it instead.
    """

    proxy_buffer_attributes = True


def mark_values_the_caller_receives(gm: torch.fx.GraphModule) -> None:
    """Set ``node.meta['_trt_reaches_caller']`` on each call of ``gm``: whether the
    caller of the compiled module can receive its memory, as an output or through
    views of one.

    A KV cache update's engine output is the cache's own memory, which the next call
    changes, so the interpreter copies it only where the caller can receive it. One
    that only feeds the next graph piece keeps the cache's memory. A call this pass
    did not see, such as one a later pass adds, counts as received.
    """

    def sources_of_its_memory(node: torch.fx.Node) -> Iterable[torch.fx.Node]:
        if node.op != "call_function" or not isinstance(
            node.target, torch._ops.OpOverload
        ):
            return list(node.all_input_nodes)
        # A view without an alias annotation in its schema.
        if node.target is torch.ops.aten._unsafe_view.default:
            return list(node.all_input_nodes)[:1]
        sources: List[torch.fx.Node] = []
        for i, argument in enumerate(node.target._schema.arguments):
            if argument.alias_info is None:
                continue
            value = (
                node.args[i] if i < len(node.args) else node.kwargs.get(argument.name)
            )
            torch.fx.node.map_arg(value, sources.append)
        return sources

    output = next(n for n in reversed(gm.graph.nodes) if n.op == "output")
    received: Set[torch.fx.Node] = set()
    pending = list(output.all_input_nodes)
    while pending:
        node = pending.pop()
        if node not in received:
            received.add(node)
            pending.extend(sources_of_its_memory(node))
    for node in gm.graph.nodes:
        if node.op == "call_function":
            node.meta["_trt_reaches_caller"] = node in received


def write_copyback_buffers(
    module: torch.nn.Module, names: Sequence[str], values: Sequence[torch.Tensor]
) -> None:
    """Copy each value into ``module``'s buffer of the same position in ``names``.

    The buffers are read here rather than in the generated ``forward``, under the
    tracer too, which hands back a graph node for each.
    """
    for name, value in zip(names, values):
        getattr(module, name).copy_(value)


def _may_share_a_buffer(
    gm: torch.fx.GraphModule, value: torch.fx.Node, buffers: Set[str]
) -> bool:
    """Whether ``value`` can be one of ``buffers``, or a view of one, when it is read.

    The walk follows only what can hand back its input's memory: an op whose schema
    returns an alias, an item of a tuple, and the matching output of a submodule left
    to PyTorch, into whose graph it goes. An engine and every other op return new
    memory, so a value computed from a buffer is not copied for nothing.
    """
    from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

    buffer_storages = {
        gm.get_buffer(name).untyped_storage().data_ptr()
        for name in buffers
        if name in dict(gm.named_buffers())
    }
    # Each entry is a node, its module, and the calls that entered that module.
    pending: List[Tuple[Any, Any, Tuple[Any, ...]]] = [(value, gm, ())]
    seen = set()

    def enter(sub: Any, outputs: Any, frames: Tuple[Any, ...]) -> None:
        torch.fx.node.map_arg(outputs, lambda n: pending.append((n, sub, frames)))

    while pending:
        node, module, frames = pending.pop()
        if not isinstance(node, torch.fx.Node) or (node, module, frames) in seen:
            continue
        seen.add((node, module, frames))
        if node.op == "placeholder":
            if frames:
                outer, call = frames[-1]
                inputs = [n for n in module.graph.nodes if n.op == "placeholder"]
                index = inputs.index(node)
                if index < len(call.args):
                    pending.append((call.args[index], outer, frames[:-1]))
        elif node.op == "get_attr":
            if not frames:
                if node.target in buffers:
                    return True
            else:
                attr = getattr(module, node.target, None)
                if (
                    isinstance(attr, torch.Tensor)
                    and attr.untyped_storage().data_ptr() in buffer_storages
                ):
                    return True
        elif node.op == "call_module":
            sub = module.get_submodule(node.target)
            if isinstance(sub, TorchTensorRTModule):
                continue
            if isinstance(sub, torch.fx.GraphModule):
                out = next(iter(sub.graph.find_nodes(op="output")))
                enter(sub, out.args[0], frames + ((module, node),))
            else:
                pending.extend((n, module, frames) for n in node.all_input_nodes)
        elif node.op == "call_function" and node.target is operator.getitem:
            base, index = node.args[0], node.args[1]
            sub = (
                module.get_submodule(base.target)
                if isinstance(base, torch.fx.Node) and base.op == "call_module"
                else None
            )
            if isinstance(sub, torch.fx.GraphModule) and not isinstance(
                sub, TorchTensorRTModule
            ):
                out = next(iter(sub.graph.find_nodes(op="output"))).args[0]
                if isinstance(out, (tuple, list)) and isinstance(index, int):
                    enter(sub, out[index], frames + ((module, base),))
                    continue
            pending.append((base, module, frames))
        elif node.op == "call_function" and isinstance(
            node.target, torch._ops.OpOverload
        ):
            if any(r.alias_info is not None for r in node.target._schema.returns):
                pending.extend((n, module, frames) for n in node.all_input_nodes)
        else:
            pending.extend((n, module, frames) for n in node.all_input_nodes)
    return False


def lifted_buffer_attrs(
    names: Iterable[str], model_names: Iterable[str] = ()
) -> Dict[str, str]:
    """The attribute each lifted buffer, by its name in the model, is registered
    under on the compiled module.

    ``register_buffer`` takes no ``.``, so a nested name is flattened, and a
    flattened name that another lifted buffer or a top-level name of the model
    (``model_names``, the model's state names) already uses gets a suffix, in sorted
    order. A flat name stays as it is, so ``state_dict`` keys do not change. The
    attributes follow from the model's names alone, so they are found again on a
    module that was saved and loaded.
    """
    names = set(names)
    attrs = {name: name for name in names if "." not in name}
    taken = set(attrs.values()) | {name.split(".")[0] for name in model_names}
    for name in sorted(names - set(attrs)):
        attr = base = "lifted_buf_" + name.replace(".", "_")
        suffix = 0
        while attr in taken:
            suffix += 1
            attr = f"{base}_{suffix}"
        taken.add(attr)
        attrs[name] = attr
    return attrs


def copy_into_lifted_buffers(
    module: torch.nn.Module, state: Dict[str, torch.Tensor], lifted: Iterable[str]
) -> None:
    """Copy each value in ``state``, the model's state keyed by its names, into the
    buffer of the compiled ``module`` that holds it. ``lifted`` names the buffers
    the model writes, the ones compile() registered on the module."""
    for name, attr in lifted_buffer_attrs(lifted, state).items():
        buffer = module._buffers.get(attr)
        value = state.get(name)
        if buffer is not None and value is not None and buffer.shape == value.shape:
            buffer.copy_(value)


@contextlib.contextmanager
def buffers_restored(module: torch.nn.Module) -> Iterator[None]:
    """Put ``module``'s buffers back as they were when the block exits.

    Calling a compiled module advances the buffers it writes, so code that runs it
    only to trace it or to check its output would otherwise leave it a call ahead.
    The copies are kept on the CPU, so a large cache does not need twice the device
    memory. Only a buffer whose bytes changed is written back: one the block never
    wrote can be read-only, as an expanded buffer is.
    """
    saved = {
        name: buffer.detach().to("cpu", copy=True)
        for name, buffer in module.named_buffers()
    }
    try:
        yield
    finally:
        # Inference mode accepts the write whether or not a buffer was made in it.
        with torch.inference_mode():
            for name, buffer in module.named_buffers():
                old = saved.get(name)
                if old is None:
                    continue
                now = buffer.detach().to("cpu").reshape(-1).view(torch.uint8)
                if not torch.equal(now, old.reshape(-1).view(torch.uint8)):
                    buffer.copy_(old)


def hide_copyback_outputs(gm: torch.fx.GraphModule, buffers: Sequence[str]) -> None:
    """Make ``gm``'s ``forward`` copy its last ``len(buffers)`` graph outputs into
    those buffers, in order, and return the outputs before them.

    They remain outputs of the graph, so the exporters still find them and declare
    them BUFFER_MUTATIONs. Every value is read before any buffer is written: a value
    that can share memory with one of the buffers, as when one buffer's new contents
    are another's old ones or a view of its own, is copied first.
    """
    if not buffers:
        return
    output_node = next(n for n in gm.graph.nodes if n.op == "output")
    values = output_node.args[0][-len(buffers) :]
    copied = [
        isinstance(value, torch.fx.Node)
        and _may_share_a_buffer(gm, value, set(buffers))
        for value in values
    ]
    used = {node.name for node in gm.graph.nodes}
    helper = "write_copyback_buffers"
    while helper in used:
        helper += "_"
    gm.graph.set_codegen(_HiddenCopybackOutputs(buffers, copied, helper))
    gm.graph._tracer_cls = gm._tracer_cls = _CopybackTracer
    gm.recompile()


def inline_lifted_buffers_into_gm(
    gm: torch.fx.GraphModule,
    lifted_buffers: List[Tuple[str, str, torch.Tensor]],
    model_names: Iterable[str] = (),
    in_place: Iterable[str] = (),
    device: Optional[torch.device] = None,
) -> torch.fx.GraphModule:
    """Inline lifted buffers as ``get_attr`` reads on the compiled GraphModule.

    After ``lift_mutated_buffers`` + ``compile_module``, ``gm`` is a
    ``torch.fx.GraphModule`` whose top-level ``forward`` takes the user's
    inputs *plus* the lifted buffers as placeholders. To make the result
    look like a normal ``nn.Module`` (and to make it serializable via
    ``torch_tensorrt.save`` / ``torch.export``) we:

    1. Register each lifted buffer as a ``register_buffer`` on ``gm``.
    2. Replace each buffer-placeholder node with a ``get_attr`` node that
       reads from ``gm.<buffer_name>``.
    3. Recompile.

    The resulting GraphModule's ``forward`` takes only the user's inputs;
    the buffers are threaded through internally via the get_attr nodes.
    The engine still sees the buffers as input bindings (and writes through
    them via aliased I/O); the buffer storage lives on ``gm`` so subsequent
    calls reuse the mutated state.

    The buffers whose placeholders are named in ``in_place`` are the ones the
    engine updates in place, and they are registered on ``device``. A buffer
    traced on the host would otherwise reach the engine as a per-call device
    copy, which would take the write.

    This transform is a no-op if ``lifted_buffers`` is empty (returns
    ``gm`` unchanged).
    """
    if not lifted_buffers:
        return gm

    placeholder_to_buf: Dict[str, str] = {
        ph_name: buf_name for ph_name, buf_name, _ in lifted_buffers
    }
    # Register buffers as module state, under the attribute names
    # lifted_buffer_attrs gives. Copy so the gm owns its own storage; for a buffer
    # moved to the engine's device, that move is the copy.
    in_place = set(in_place)
    first: Dict[str, Tuple[str, torch.Tensor]] = {}
    for ph_name, buf_name, tensor in lifted_buffers:
        first.setdefault(buf_name, (ph_name, tensor))
    buf_to_attr = lifted_buffer_attrs(first, model_names)
    for buf_name, (ph_name, tensor) in first.items():
        attr_name = buf_to_attr[buf_name]
        if hasattr(gm, attr_name):
            if "." in buf_name:
                raise RuntimeError(
                    f"Cannot register buffer '{buf_name}' as '{attr_name}': the "
                    "compiled module already has an attribute of that name."
                )
            continue
        # A clone made under inference mode is an inference tensor, and forward
        # writes copy-back buffers in place, which such a tensor refuses outside
        # that mode.
        with torch.inference_mode(False):
            gm.register_buffer(
                attr_name,
                (
                    tensor.to(device, copy=True)
                    if ph_name in in_place and device is not None
                    else tensor.clone()
                ),
            )

    # Copy-back mutation targets were recorded (in gm.meta) under the buffers'
    # original names; nested ones were just flattened to lifted_buf_* above. Remap
    # the recorded targets through the same mapping so the downstream exporter
    # names a buffer that still exists -- otherwise the ExportedProgram verifier
    # rejects the BUFFER_MUTATION ("output ... does not point to a buffer that
    # exists").
    copyback = gm.meta.get("_copyback_mutation_buffers")
    if copyback:
        gm.meta["_copyback_mutation_buffers"] = [
            buf_to_attr.get(name, name) for name in copyback
        ]

    # Find placeholders we need to replace. Insert get_attr nodes BEFORE
    # removing the placeholders so the graph remains valid throughout.
    placeholders_to_remove = []
    for node in list(gm.graph.nodes):
        if node.op != "placeholder":
            continue
        if node.name not in placeholder_to_buf:
            continue
        buf_name = placeholder_to_buf[node.name]
        with gm.graph.inserting_after(node):
            get_attr_node = gm.graph.get_attr(buf_to_attr[buf_name])
        # Carry over fake-tensor metadata so downstream passes see the right
        # shape/dtype.
        if "val" in node.meta:
            get_attr_node.meta["val"] = node.meta["val"]
        node.replace_all_uses_with(get_attr_node)
        placeholders_to_remove.append(node)

    for node in placeholders_to_remove:
        gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()
    logger.debug(
        "Inlined %d lifted buffer(s) into gm as get_attr reads: %s",
        len(lifted_buffers),
        [b for _, b, _ in lifted_buffers],
    )
    return gm
