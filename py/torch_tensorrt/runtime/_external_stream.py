import logging
from typing import Any, Dict, List, Tuple, Union

import torch
from torch_tensorrt.dynamo.runtime import PythonTorchTensorRTModule, TorchTensorRTModule

logger = logging.getLogger(__name__)

StreamLike = Union[int, "torch.cuda.Stream"]
RTModule = Union[PythonTorchTensorRTModule, TorchTensorRTModule]


def _to_handle(stream: StreamLike) -> int:
    if isinstance(stream, torch.cuda.Stream):
        return int(stream.cuda_stream)
    if isinstance(stream, int):
        return stream
    raise TypeError(
        f"stream must be int or torch.cuda.Stream, got {type(stream).__name__}"
    )


def _iter_rt_modules(module: Any) -> List[Tuple[str, RTModule]]:
    if not hasattr(module, "named_modules"):
        raise TypeError(f"expected nn.Module, got {type(module).__name__}")
    # named_modules() recurses; needed because real-world compiled outputs
    # (e.g. HF blocks above min_block_size) nest TRT submodules under wrapper
    # GraphModules where named_children() would miss them. Returned names are
    # dotted paths and unique.
    return [
        (name, m)
        for name, m in module.named_modules()
        if "_run_on_acc" in name
        and isinstance(m, (PythonTorchTensorRTModule, TorchTensorRTModule))
    ]


class _ExternalStreamContextManager:
    def __init__(self, prior: List[Tuple[RTModule, int]]) -> None:
        self._prior = prior

    def __enter__(self) -> "_ExternalStreamContextManager":
        return self

    def __exit__(self, *args: Any) -> None:
        for rt_mod, prior_handle in self._prior:
            if prior_handle == 0:
                rt_mod.clear_external_stream()
            else:
                rt_mod.set_external_stream(prior_handle)


def set_external_stream(
    module: Any,
    stream: Union[StreamLike, Dict[str, StreamLike]],
) -> _ExternalStreamContextManager:
    """Bind TRT engine(s) in ``module`` to externally-managed CUDA stream(s).

    ``stream`` is either a single ``StreamLike`` bound to every TRT engine, or a
    ``Dict[submodule_name, StreamLike]`` for per-engine binding (the canonical
    multi-engine SM-partitioning case via CUDA Green Contexts; cuda 12.4+).
    Submodule names are dotted paths from ``module.named_modules()`` (recursive),
    so deeply-nested TRT submodules are reachable.

    Returns a context manager that restores prior bindings on exit.

    Caller owns each stream's lifetime: the underlying stream MUST outlive the
    bound engines, OR the binding must be cleared before the stream is destroyed.
    Mutually exclusive with CUDA Graphs (will raise at execute time).
    """
    rt_mods = dict(_iter_rt_modules(module))
    if not rt_mods:
        raise ValueError(
            "No TRT runtime submodules found on the supplied module. "
            "Was the module compiled with torch_tensorrt.compile()?"
        )

    if isinstance(stream, dict):
        unknown = set(stream) - set(rt_mods)
        if unknown:
            raise ValueError(
                f"Unknown submodule keys: {sorted(unknown)}. "
                f"Available: {sorted(rt_mods)}."
            )
        bindings = {n: _to_handle(s) for n, s in stream.items()}
    else:
        h = _to_handle(stream)
        bindings = {n: h for n in rt_mods}

    # Validate up front so a bad value doesn't leave a partially-bound module.
    for name, handle in bindings.items():
        if handle == 0:
            raise ValueError(
                f"streams[{name!r}] wraps a non-null CUDA stream is required; "
                "use clear_external_stream() to revert to the default stream pool."
            )

    prior = [(rt_mods[n], rt_mods[n].get_external_stream()) for n in bindings]
    for name, handle in bindings.items():
        rt_mods[name].set_external_stream(handle)

    logger.debug("Bound external stream(s) on %d TRT submodule(s)", len(bindings))
    return _ExternalStreamContextManager(prior)


def clear_external_stream(module: Any) -> None:
    for _, rt_mod in _iter_rt_modules(module):
        rt_mod.clear_external_stream()
