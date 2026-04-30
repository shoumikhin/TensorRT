import logging
from typing import Any, List, Tuple, Union

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
) -> List[Union[PythonTorchTensorRTModule, TorchTensorRTModule]]:
    """Walk ``module`` and collect every TRT runtime submodule."""
    rt_mods = []
    for name, rt_mod in getattr(module, "named_children", lambda: [])():
        if "_run_on_acc" in name and isinstance(
            rt_mod, (PythonTorchTensorRTModule, TorchTensorRTModule)
        ):
            rt_mods.append(rt_mod)
    return rt_mods


class _ExternalStreamContextManager(object):
    """Helper class used in conjunction with `set_external_stream`.

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
    """Bind every TRT engine in ``module`` to an externally-managed CUDA stream.

    When set, the runtime uses the supplied stream for engine execution
    (``enqueueV3``) instead of pulling one from torch's internal stream pool.
    The primary use case is binding engines to a CUDA Green Context (cuda
    12.4+) created via ``cuGreenCtxStreamCreate`` for SM partitioning across
    concurrent models.

    Caller owns the stream's lifetime: the underlying stream (and its parent
    green context, if any) MUST outlive the bound engines, OR the binding
    must be cleared before the stream is destroyed (use the returned context
    manager or call ``clear_external_stream``).

    Arguments:
        module: A compiled module returned by ``torch_tensorrt.compile``.
        stream: Either a ``torch.cuda.Stream``, an ``int`` (raw cudaStream_t
            cast to int), or any int-able CUDA stream handle (e.g.
            ``cuda.CUstream`` from cuda-python).

    Example:

        .. code-block:: py

            from cuda import cuda  # cuda-python
            _, sm = cuda.cuDeviceGetDevResource(
                0, cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
            )
            _, parts, _ = cuda.cuDevSmResourceSplitByCount(2, sm, 0, 0)
            _, desc = cuda.cuDevResourceGenerateDesc([parts[0]], 1)
            _, green_ctx = cuda.cuGreenCtxCreate(desc, 0, 0)
            _, gctx_stream = cuda.cuGreenCtxStreamCreate(green_ctx, 0, 0)

            with torch_tensorrt.runtime.set_external_stream(trt_module, gctx_stream):
                outputs = trt_module(*inputs)

    Notes:
        - CUDA Graphs are mutually exclusive with an external stream; setting
          both will raise at execute time.
        - The green context's parent CUcontext must match the runtime primary
          context for the device (the standard case when using torch's CUDA
          tensors and ``torch.cuda.set_device``).
        - CUDA Green Contexts require CUDA 12.4+. Binding any pre-created
          CUstream is supported on any CUDA version.
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
    for rt_mod in rt_mods:
        prior.append((rt_mod, rt_mod.get_external_stream()))
        rt_mod.set_external_stream(handle)

    logger.info(
        f"Bound external stream {handle:#x} to {len(rt_mods)} TRT runtime submodule(s)"
    )

    return _ExternalStreamContextManager(prior)


def clear_external_stream(module: Any) -> None:
    """Revert every TRT engine in ``module`` back to the default stream pool."""
    for rt_mod in _collect_rt_modules(module):
        rt_mod.clear_external_stream()
