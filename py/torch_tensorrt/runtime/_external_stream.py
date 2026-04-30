import logging
from typing import Any, Dict, List, Tuple, Union

import torch
import torch_tensorrt
from torch_tensorrt.dynamo.runtime import PythonTorchTensorRTModule, TorchTensorRTModule

logger = logging.getLogger(__name__)

StreamLike = Union[int, "torch.cuda.Stream"]


def _to_handle(stream: StreamLike) -> int:
    """Normalize a stream argument to an int handle (cudaStream_t value)."""
    if isinstance(stream, torch.cuda.Stream):
        return int(stream.cuda_stream)
    if isinstance(stream, int):
        return stream
    # Duck-type: anything int-able (e.g. cuda-python's CUstream).
    try:
        return int(stream)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"stream must be int, torch.cuda.Stream, or any int-able CUDA stream handle, "
            f"got {type(stream).__name__}"
        ) from exc


def _collect_rt_modules(
    module: Any,
) -> List[Tuple[str, Union[PythonTorchTensorRTModule, TorchTensorRTModule]]]:
    """Walk ``module`` and collect every TRT runtime submodule with its name."""
    rt_mods = []
    for name, rt_mod in getattr(module, "named_children", lambda: [])():
        if "_run_on_acc" in name and isinstance(
            rt_mod, (PythonTorchTensorRTModule, TorchTensorRTModule)
        ):
            rt_mods.append((name, rt_mod))
    return rt_mods


class _ExternalStreamContextManager(object):
    """Helper class used in conjunction with `set_external_stream(s)`.

    Restores prior bindings on `__exit__`.
    """

    def __init__(
        self,
        prior: List[Tuple[Union[PythonTorchTensorRTModule, TorchTensorRTModule], int]],
    ) -> None:
        self.prior = prior

    def __enter__(self) -> "_ExternalStreamContextManager":
        return self

    def __exit__(self, *args: Any) -> None:
        for rt_mod, prior_handle in self.prior:
            if prior_handle == 0:
                rt_mod.clear_external_stream()
            else:
                rt_mod.set_external_stream(prior_handle)


def set_external_stream(
    module: Any,
    stream: StreamLike,
) -> _ExternalStreamContextManager:
    """Bind every TRT engine in ``module`` to a single externally-managed CUDA stream.

    For modules with a single TRT submodule this is the natural API. For
    modules with multiple TRT submodules where each should bind to a
    DIFFERENT stream (e.g., per-engine SM partitioning via distinct CUDA
    Green Contexts), use :func:`set_external_streams` instead.

    See :func:`set_external_streams` for the lifetime contract, cudagraph
    interaction, and CUDA version requirements (those notes apply here too).

    Arguments:
        module: A compiled module returned by ``torch_tensorrt.compile``.
        stream: Either a ``torch.cuda.Stream``, an ``int`` (raw cudaStream_t
            cast to int), or any int-able CUDA stream handle (e.g.
            ``cuda.CUstream`` from cuda-python).

    Returns:
        A context manager that restores prior per-engine bindings on exit.
    """
    handle = _to_handle(stream)
    if handle == 0:
        raise ValueError(
            "stream must wrap a non-null CUDA stream. To revert to the default "
            "stream pool, call torch_tensorrt.runtime.clear_external_stream(module)."
        )

    rt_mods = _collect_rt_modules(module)
    if not rt_mods:
        raise ValueError(
            "No TRT runtime submodules found on the supplied module. "
            "Was the module compiled with torch_tensorrt.compile()?"
        )

    prior: List[Tuple[Union[PythonTorchTensorRTModule, TorchTensorRTModule], int]] = []
    for _, rt_mod in rt_mods:
        prior.append((rt_mod, rt_mod.get_external_stream()))
        rt_mod.set_external_stream(handle)

    logger.info(
        f"Bound external stream {handle:#x} to {len(rt_mods)} TRT runtime submodule(s)"
    )

    return _ExternalStreamContextManager(prior)


def set_external_streams(
    module: Any,
    streams: Dict[str, StreamLike],
) -> _ExternalStreamContextManager:
    """Bind each TRT engine in ``module`` to its own externally-managed CUDA stream.

    Primary use case: multi-engine SM partitioning via CUDA Green Contexts
    (cuda 12.4+) where a single compiled module contains several TRT
    submodules and each should run on a distinct SM partition. Pass a dict
    keyed by submodule name (e.g., ``"_run_on_acc_0"``) → stream.

    Caller owns each stream's lifetime: every underlying stream (and its
    parent green context, if any) MUST outlive the bound engines, OR the
    binding must be cleared before the stream is destroyed (use the returned
    context manager or call :func:`clear_external_stream`).

    Arguments:
        module: A compiled module returned by ``torch_tensorrt.compile``.
        streams: Mapping from submodule name to stream-like value. Use
            :func:`list_trt_submodules` (or inspect ``module.named_children()``)
            to discover the available names. All keys must resolve to a TRT
            runtime submodule on ``module``; unknown keys raise ``ValueError``.
            Submodules not mentioned in ``streams`` are left at their current
            binding (typically the default stream pool).

    Returns:
        A context manager that restores prior per-engine bindings on exit.

    Example:

        .. code-block:: py

            from cuda import cuda  # cuda-python

            # Create three green contexts, one stream each.
            streams = {}
            _, sm = cuda.cuDeviceGetDevResource(
                0, cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
            )
            _, parts, _ = cuda.cuDevSmResourceSplitByCount(3, sm, 0, 0)
            for i, name in enumerate(["_run_on_acc_0", "_run_on_acc_1", "_run_on_acc_2"]):
                _, desc = cuda.cuDevResourceGenerateDesc([parts[i]], 1)
                _, gctx = cuda.cuGreenCtxCreate(desc, 0, 0)
                _, s = cuda.cuGreenCtxStreamCreate(gctx, 0, 0)
                streams[name] = int(s)

            with torch_tensorrt.runtime.set_external_streams(trt_module, streams):
                outputs = trt_module(*inputs)

    Notes:
        - CUDA Graphs are mutually exclusive with an external stream; setting
          both will raise at execute time.
        - The green context's parent CUcontext must match the runtime primary
          context for the device.
        - Submodule names like ``_run_on_acc_*`` come from torch-tensorrt's
          partitioner and are stable for a given compilation but may change
          across compile settings or torch-tensorrt versions. Re-discover
          them with ``module.named_children()`` after recompiling.
    """
    if not isinstance(streams, dict):
        raise TypeError(
            f"streams must be a dict mapping submodule name to stream, "
            f"got {type(streams).__name__}. For a single stream applied to "
            f"all engines, use set_external_stream() instead."
        )

    rt_mods = _collect_rt_modules(module)
    if not rt_mods:
        raise ValueError(
            "No TRT runtime submodules found on the supplied module. "
            "Was the module compiled with torch_tensorrt.compile()?"
        )

    rt_mod_by_name = {name: rt_mod for name, rt_mod in rt_mods}
    unknown_keys = set(streams.keys()) - set(rt_mod_by_name.keys())
    if unknown_keys:
        raise ValueError(
            f"Unknown submodule keys: {sorted(unknown_keys)}. "
            f"Available TRT submodule names on this module: {sorted(rt_mod_by_name.keys())}."
        )

    # Validate all stream handles up front before mutating any engine state
    # (transactional: bind all or bind none).
    handles: Dict[str, int] = {}
    for name, stream in streams.items():
        handle = _to_handle(stream)
        if handle == 0:
            raise ValueError(
                f"streams[{name!r}] wraps a null CUDA stream. To revert to the "
                f"default stream pool, omit the key or call clear_external_stream(module)."
            )
        handles[name] = handle

    prior: List[Tuple[Union[PythonTorchTensorRTModule, TorchTensorRTModule], int]] = []
    for name, handle in handles.items():
        rt_mod = rt_mod_by_name[name]
        prior.append((rt_mod, rt_mod.get_external_stream()))
        rt_mod.set_external_stream(handle)

    logger.info(
        f"Bound {len(handles)} per-engine external stream(s) on TRT runtime submodules: "
        f"{sorted(handles.keys())}"
    )

    return _ExternalStreamContextManager(prior)


def clear_external_stream(module: Any) -> None:
    """Revert every TRT engine in ``module`` back to the default stream pool."""
    for _, rt_mod in _collect_rt_modules(module):
        rt_mod.clear_external_stream()


def list_trt_submodules(module: Any) -> List[str]:
    """Return the names of TRT runtime submodules in ``module``.

    Useful for discovering which keys to pass to :func:`set_external_streams`.
    """
    return [name for name, _ in _collect_rt_modules(module)]
