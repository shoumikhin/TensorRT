import importlib

if importlib.util.find_spec("executorch") is None:

    def __getattr__(name: str):
        raise ImportError(
            f"Cannot access torch_tensorrt.executorch.{name}: "
            "ExecuTorch is required. Install with: pip install executorch"
        )

else:
    from typing import Any, Dict, Optional, Sequence, Set, Union

    import logging

    import torch
    from torch.export import ExportedProgram

    from torch_tensorrt.dynamo import _defaults
    from torch_tensorrt.executorch._backend import TensorRTBackend
    from torch_tensorrt.executorch._partitioner import TensorRTPartitioner

    import torch_tensorrt.executorch.register_et_ops  # noqa: F401

    logger = logging.getLogger(__name__)

    def _move_to_cpu(inputs: Sequence[Any]) -> tuple:
        """Recursively move tensors to CPU; returns nested tuples."""
        result = []
        for x in inputs:
            if isinstance(x, torch.Tensor):
                result.append(x.cpu())
            elif isinstance(x, (list, tuple)):
                result.append(_move_to_cpu(x))
            else:
                result.append(x)
        return tuple(result)

    def to_trt(
        exported_program: ExportedProgram,
        inputs: Sequence[torch.Tensor],
        kwarg_inputs: Optional[Dict[str, Any]] = None,
        *,
        dynamic_shapes: Any = None,
        enabled_precisions: Set[torch.dtype] = _defaults.ENABLED_PRECISIONS,
        device: Optional[Union[torch.device, str]] = _defaults.DEVICE,
        min_block_size: int = 1,
        require_full_compilation: bool = True,
        truncate_double: bool = _defaults.TRUNCATE_DOUBLE,
        optimization_level: Optional[int] = _defaults.OPTIMIZATION_LEVEL,
        workspace_size: int = _defaults.WORKSPACE_SIZE,
        debug: bool = False,
        **kwargs,
    ) -> ExportedProgram:
        """Compile an ExportedProgram with TensorRT and return an ET-ready ExportedProgram.

        This is the low-level entry point: call to_trt() to compile TRT engines,
        then pass the result to executorch's
        to_edge_transform_and_lower(TensorRTPartitioner()).
        """
        import torch_tensorrt.dynamo
        from torch_tensorrt.dynamo._exporter import transform
        from torch_tensorrt.executorch._converter import (
            convert_engines,
            export_trt_module,
        )

        if "use_python_runtime" in kwargs:
            logger.warning(
                "use_python_runtime is not supported for ExecuTorch export "
                "and will be ignored. The C++ TRT runtime is always used."
            )
        # Keys consumed by to_trt() (popped so they don't reach dynamo.compile
        # a second time): use_python_runtime, enabled_precisions, device,
        # min_block_size, require_full_compilation, truncate_double,
        # optimization_level, workspace_size, debug.
        # Everything remaining in kwargs is forwarded to dynamo.compile().
        kwargs.pop("use_python_runtime", None)
        for key in (
            "enabled_precisions",
            "device",
            "min_block_size",
            "require_full_compilation",
            "truncate_double",
            "optimization_level",
            "workspace_size",
            "debug",
        ):
            kwargs.pop(key, None)

        gm = torch_tensorrt.dynamo.compile(
            exported_program,
            inputs=inputs,
            kwarg_inputs=kwarg_inputs,
            use_python_runtime=False,
            enabled_precisions=set(enabled_precisions),
            device=device,
            min_block_size=min_block_size,
            require_full_compilation=require_full_compilation,
            truncate_double=truncate_double,
            optimization_level=optimization_level,
            workspace_size=workspace_size,
            debug=debug,
            **kwargs,
        )
        gm = transform(gm)
        convert_engines(gm)

        engine_count = sum(
            1 for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops.tensorrt.execute_engine_et.default
        )
        if engine_count > 1:
            logger.warning(
                "%d TRT engines detected. Multi-engine .pte incurs "
                "H2D/D2H transfer overhead between delegates.",
                engine_count,
            )

        # ExecuTorch's emitter requires CPU tensors for non-delegated ops.
        # TRT engine blobs are already CPU ByteTensors from convert_engines().
        gm = gm.cpu()
        cpu_inputs = _move_to_cpu(inputs)
        cpu_kwarg_inputs = (
            {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in kwarg_inputs.items()}
            if kwarg_inputs is not None
            else None
        )

        trt_ep = export_trt_module(
            gm,
            arg_inputs=cpu_inputs,
            kwarg_inputs=cpu_kwarg_inputs,
            dynamic_shapes=dynamic_shapes,
        )

        # Decompose any remaining attention ops that survived as fallback ops.
        # These ops lack out-variants required by ExecuTorch's ToOutVarPass.
        from torch_tensorrt.dynamo.lowering._decompositions import get_decompositions

        trt_ep = trt_ep.run_decompositions(
            get_decompositions(decompose_attention=True)
        )

        return trt_ep

    def get_edge_compile_config() -> "EdgeCompileConfig":
        """Return the EdgeCompileConfig used for all TRT ExecuTorch exports.

        TRT engines are pre-compiled and don't need Edge IR validation or Edge ops.
        """
        from executorch.exir import EdgeCompileConfig

        return EdgeCompileConfig(_check_ir_validity=False, _use_edge_ops=False)

    __all__ = [
        "to_trt",
        "get_edge_compile_config",
        "TensorRTPartitioner",
        "TensorRTBackend",
    ]
