import base64
import json
import logging
from itertools import product
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.export import ExportedProgram

from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
    DEVICE_IDX,
    ENGINE_IDX,
    HW_COMPATIBLE_IDX,
    INPUT_BINDING_NAMES_IDX,
    OUTPUT_BINDING_NAMES_IDX,
    REQUIRES_OUTPUT_ALLOCATOR_IDX,
    SERIALIZED_METADATA_IDX,
    TARGET_PLATFORM_IDX,
)
from torch_tensorrt.executorch._serialization import (
    TensorRTBlobMetadata,
    TensorRTIOBinding,
    serialize_engine,
)

import torch_tensorrt.executorch.register_et_ops  # noqa: F401

logger = logging.getLogger(__name__)

_BINDING_DELIM = "%"


def _is_trt_module(mod: torch.nn.Module) -> bool:
    """Check whether a module is a TensorRT runtime module.

    Args:
        mod: The module to check.

    Returns:
        True if ``mod`` is a ``PythonTorchTensorRTModule`` or
        ``TorchTensorRTModule`` instance, False otherwise.
    """
    from torch_tensorrt.dynamo.runtime._PythonTorchTensorRTModule import (
        PythonTorchTensorRTModule,
    )

    if isinstance(mod, PythonTorchTensorRTModule):
        return True
    try:
        from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
            TorchTensorRTModule,
        )

        if isinstance(mod, TorchTensorRTModule):
            return True
    except ImportError:
        pass
    return False


def _build_output_spec(val: Any) -> str:
    """Build a JSON output specification from traced tensor metadata.

    Determines the upper-bound shape for every output tensor so that
    ExecuTorch can pre-allocate output buffers.  Symbolic integer
    dimensions are resolved by evaluating their expression at every
    corner of the known symbol ranges.

    Args:
        val: A ``torch.Tensor`` or a list/tuple of tensors carrying
            shape metadata (possibly with ``SymInt`` dimensions).

    Returns:
        A JSON string encoding a list of ``{"shape": [...], "dtype": "..."}``
        dictionaries, one per output tensor.

    Raises:
        RuntimeError: If ``val`` is not a tensor or sequence of tensors,
            if a ``SymInt`` is found without a ``ShapeEnv``, or if a
            symbol cannot be resolved.
    """
    from torch._guards import detect_fake_mode

    if isinstance(val, torch.Tensor):
        tensors = [val]
    elif isinstance(val, (list, tuple)):
        tensors = list(val)
    else:
        raise RuntimeError(f"Unexpected output type: {type(val)}")

    fake_mode = detect_fake_mode(tensors)
    shape_env = fake_mode.shape_env if fake_mode else None

    spec = []
    for t in tensors:
        shape = []
        for d in t.shape:
            if isinstance(d, torch.SymInt):
                if shape_env is None:
                    raise RuntimeError(
                        f"SymInt dimension {d} found but no ShapeEnv available. "
                        "Cannot determine upper bound for ET memory planning. "
                        "Ensure the model was exported with dynamic_shapes."
                    )
                expr = d.node.expr
                free_syms = expr.free_symbols
                missing = [s for s in free_syms if s not in shape_env.var_to_range]
                if missing:
                    raise RuntimeError(
                        f"Cannot resolve upper bound for {expr}: "
                        f"symbols {missing} not in var_to_range"
                    )
                corners = list(
                    product(
                        *(
                            (
                                shape_env.var_to_range[sym].lower,
                                shape_env.var_to_range[sym].upper,
                            )
                            for sym in free_syms
                        )
                    )
                )
                upper = max(
                    int(expr.subs(dict(zip(free_syms, combo))))
                    for combo in corners
                )
                shape.append(upper)
            else:
                shape.append(int(d))
        spec.append(
            {
                "shape": shape,
                "dtype": str(t.dtype).replace("torch.", ""),
            }
        )
    return json.dumps(spec)


def _parse_device_id(device_str: str) -> int:
    """Extract the GPU device ID from a serialized device string.

    Args:
        device_str: Serialized device info in the format
            ``"gpu_id%major%minor%device_type%device_name"``.

    Returns:
        The integer GPU device ID, or ``0`` if the string is empty or
        cannot be parsed.
    """
    if not device_str:
        return 0
    parts = device_str.split(_BINDING_DELIM)
    try:
        return int(parts[0])
    except (ValueError, IndexError):
        return 0


def convert_engines(gm: torch.fx.GraphModule) -> None:
    """Replace execute_engine ScriptObject nodes with execute_engine_et + ByteTensor buffers.

    Must be called after transform() which inlines TRT/torch modules and produces
    execute_engine call_function nodes with ScriptObject engine references.

    Mutates the GraphModule in-place.
    """
    if ENGINE_IDX < 0:
        raise RuntimeError(
            "TensorRT C++ runtime is required for ExecuTorch export but is not available. "
            "Install torch_tensorrt with C++ runtime support."
        )

    engine_count = 0
    old_engine_attrs: List[str] = []

    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        if node.target != torch.ops.tensorrt.execute_engine.default:
            continue

        input_args = node.args[0]
        engine_attr_node = node.args[1]

        engine = getattr(gm, engine_attr_node.target)
        if not hasattr(engine, "__getstate__"):
            raise TypeError(
                f"Expected ScriptObject for {engine_attr_node.target}, "
                f"got {type(engine)} (no __getstate__ method)"
            )
        state = engine.__getstate__()
        serialized = state[0] if isinstance(state, tuple) else state

        try:
            engine_bytes = base64.b64decode(serialized[ENGINE_IDX])
        except Exception as e:
            raise RuntimeError(
                f"Failed to base64-decode engine {engine_count}: {e}. "
                "The serialized engine data may be corrupt or not properly encoded."
            ) from e
        if not engine_bytes:
            raise RuntimeError(
                f"Engine {engine_count} has empty serialized data. "
                "TRT compilation may have failed silently."
            )

        input_names = [
            n for n in serialized[INPUT_BINDING_NAMES_IDX].split(_BINDING_DELIM) if n
        ]
        output_names = [
            n for n in serialized[OUTPUT_BINDING_NAMES_IDX].split(_BINDING_DELIM) if n
        ]

        device_id = _parse_device_id(serialized[DEVICE_IDX])

        if (
            len(serialized) > REQUIRES_OUTPUT_ALLOCATOR_IDX
            and serialized[REQUIRES_OUTPUT_ALLOCATOR_IDX] == "1"
        ):
            raise RuntimeError(
                "Engines requiring output allocator (data-dependent output shapes) "
                "are not supported in the ExecuTorch export path."
            )

        io_bindings = []
        for name in input_names:
            io_bindings.append(TensorRTIOBinding(name, "", [], True))
        for name in output_names:
            io_bindings.append(TensorRTIOBinding(name, "", [], False))

        hw_compat = (
            serialized[HW_COMPATIBLE_IDX] == "1"
            if len(serialized) > HW_COMPATIBLE_IDX
            else False
        )
        ser_meta = (
            serialized[SERIALIZED_METADATA_IDX]
            if len(serialized) > SERIALIZED_METADATA_IDX
            else ""
        )
        target_plat = (
            serialized[TARGET_PLATFORM_IDX]
            if len(serialized) > TARGET_PLATFORM_IDX
            else ""
        )

        if not target_plat:
            try:
                import tensorrt as trt

                target_plat = f"trt:{trt.__version__}"
            except ImportError:
                pass

        metadata = TensorRTBlobMetadata(
            io_bindings=io_bindings,
            device_id=device_id,
            hardware_compatible=hw_compat,
            serialized_metadata=ser_meta,
            target_platform=target_plat,
        )

        blob = serialize_engine(engine_bytes, metadata)
        blob_tensor = torch.frombuffer(bytearray(blob), dtype=torch.uint8)

        buffer_name = f"__trt_engine_blob_{engine_count}"
        gm.register_buffer(buffer_name, blob_tensor)

        val = node.meta.get("val")
        if val is None:
            raise RuntimeError(
                f"Node {node.name} has no 'val' in meta. "
                "Cannot determine output shapes for execute_engine_et."
            )

        output_spec = _build_output_spec(val)

        with gm.graph.inserting_before(node):
            buffer_node = gm.graph.get_attr(buffer_name)
            et_node = gm.graph.call_function(
                torch.ops.tensorrt.execute_engine_et.default,
                (list(input_args), buffer_node, output_spec),
            )
            et_node.meta["val"] = node.meta["val"]

        node.replace_all_uses_with(et_node)
        gm.graph.erase_node(node)

        old_engine_attrs.append(engine_attr_node.target)

        engine_count += 1
        logger.debug(
            "Converted engine %d: %d inputs, %d outputs, %d bytes",
            engine_count - 1,
            len(input_names),
            len(output_names),
            len(engine_bytes),
        )

    gm.graph.eliminate_dead_code()
    for attr_name in old_engine_attrs:
        if hasattr(gm, attr_name):
            delattr(gm, attr_name)
    gm.graph.lint()
    gm.recompile()

    if engine_count == 0:
        logger.warning(
            "No TRT engines found in GraphModule. "
            "The exported .pte will have zero TRT delegation."
        )


def export_trt_module(
    gm: torch.fx.GraphModule,
    arg_inputs: Optional[Sequence[torch.Tensor]] = None,
    kwarg_inputs: Optional[Dict[str, Any]] = None,
    dynamic_shapes: Optional[Any] = None,
) -> ExportedProgram:
    """Re-export the GraphModule (with execute_engine_et ops) as an ExportedProgram."""
    args = tuple(arg_inputs) if arg_inputs is not None else ()
    kwargs = kwarg_inputs if kwarg_inputs is not None else {}
    return torch.export.export(
        gm, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes, strict=False
    )
