import logging
from typing import Any, Iterable, List, Tuple, Union

import torch
import torch_tensorrt

logger = logging.getLogger(__name__)

StreamHandle = Union[int, "torch.cuda.Stream"]


def _to_handle(stream: StreamHandle) -> int:
    """Normalize a stream argument to an int handle (cudaStream_t value)."""
    if isinstance(stream, int):
        return stream
    if isinstance(stream, torch.cuda.Stream):
        # torch.cuda.Stream stores the underlying cudaStream_t in cuda_stream
        return int(stream.cuda_stream)
    raise TypeError(
        f"stream must be int or torch.cuda.Stream, got {type(stream).__name__}"
    )


def _iter_engines(module: Any) -> Iterable[Any]:
    """Walk a TRT-compiled module and yield each TRTEngine custom-class instance."""
    if hasattr(module, "engine") and module.engine is not None:
        yield module.engine
        return
    for child in getattr(module, "children", lambda: [])():
        yield from _iter_engines(child)


class _ExternalStreamContextManager:
    """Helper class used in conjunction with `set_external_stream`.

    Restores prior stream bindings on `__exit__`.
    """

    def __init__(self, prior: List[Tuple[Any, int]]) -> None:
        self.prior = prior

    def __enter__(self) -> "_ExternalStreamContextManager":
        return self

    def __exit__(self, *args: Any) -> None:
        for engine, prior_handle in self.prior:
            if prior_handle == 0:
                engine.clear_external_stream()
            else:
                engine.set_external_stream(prior_handle)


def set_external_stream(
    module: Any,
    stream: StreamHandle,
) -> _ExternalStreamContextManager:
    """Bind every TRT engine in ``module`` to an externally-managed CUDA stream.

    When set, ``execute_engine`` calls into the engine use the supplied stream
    for ``enqueueV3`` instead of pulling one from torch's internal stream pool.
    The primary use case is binding engines to a CUDA Green Context (cuda
    12.4+) created via ``cuGreenCtxStreamCreate`` for SM partitioning across
    concurrent models.

    Caller owns the stream's lifetime: the underlying stream (and its parent
    green context, if any) MUST outlive the bound engines, OR the binding
    must be cleared before the stream is destroyed (use the returned context
    manager or call this function again with the prior stream).

    Arguments:
        module: A TorchScript / FX / dynamo module containing TRT engines, or
            a single TRT engine custom-class instance.
        stream: Either an ``int`` (raw cudaStream_t cast to int) or a
            ``torch.cuda.Stream`` whose underlying cudaStream_t will be used.

    Example:

        .. code-block:: py

            # Create a CUDA Green Context and a stream within it (cuda-python).
            from cuda import cuda
            _, sm = cuda.cuDeviceGetDevResource(
                0, cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
            )
            _, parts, _ = cuda.cuDevSmResourceSplitByCount(2, sm, 0, 0)
            _, desc = cuda.cuDevResourceGenerateDesc([parts[0]], 1)
            _, green_ctx = cuda.cuGreenCtxCreate(desc, 0, 0)
            _, gctx_stream = cuda.cuGreenCtxStreamCreate(green_ctx, 0, 0)

            # Use as a one-shot setter:
            torch_tensorrt.runtime.set_external_stream(trt_module, int(gctx_stream))
            outputs = trt_module(*inputs)

            # Or as a context manager (auto-restores prior binding on exit):
            with torch_tensorrt.runtime.set_external_stream(trt_module, int(gctx_stream)):
                outputs = trt_module(*inputs)

    Notes:
        - CUDA Graphs are mutually exclusive with an external stream; setting
          both will throw at execute time.
        - The green context's parent CUcontext must match the runtime primary
          context for the device (the standard case when using torch's CUDA
          tensors).
    """
    handle = _to_handle(stream)
    if handle == 0:
        raise ValueError(
            "stream must wrap a non-null CUDA stream. To revert to the default "
            "stream pool, call torch_tensorrt.runtime.clear_external_stream(module)."
        )

    prior: List[Tuple[Any, int]] = []
    for engine in _iter_engines(module):
        prior.append((engine, engine.get_external_stream()))
        engine.set_external_stream(handle)

    if not prior:
        logger.warning(
            "set_external_stream: no TRT engines found on the supplied module; "
            "binding had no effect."
        )

    return _ExternalStreamContextManager(prior)


def clear_external_stream(module: Any) -> None:
    """Revert every TRT engine in ``module`` back to the default stream pool."""
    for engine in _iter_engines(module):
        engine.clear_external_stream()
