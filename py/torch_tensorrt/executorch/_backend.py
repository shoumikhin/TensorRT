from typing import Any, List, final

try:
    from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
except ImportError as e:
    raise ImportError(
        "ExecuTorch is required for TensorRTBackend but could not be imported. "
        "Please install it following https://pytorch.org/executorch/."
    ) from e

from torch_tensorrt.executorch._operator_support import is_trt_engine_op


@final
class TensorRTBackend(BackendDetails):
    """ExecuTorch backend delegate that extracts serialized TensorRT engine blobs."""

    @staticmethod
    def preprocess(
        edge_program: "ExportedProgram",
        compile_specs: List[Any],
    ) -> PreprocessResult:
        """Extract the serialized TensorRT engine blob from a partitioned program.

        Arguments:
            edge_program (ExportedProgram): The partitioned ExecuTorch program
                containing a single ``execute_engine_et`` node.
            compile_specs (List[Any]): Compile specifications forwarded by the
                ExecuTorch runtime (currently unused).

        Returns:
            PreprocessResult containing the raw engine bytes for the runtime.
        """
        gm = edge_program.graph_module
        engine_nodes = [n for n in gm.graph.nodes if is_trt_engine_op(n)]
        if len(engine_nodes) != 1:
            raise RuntimeError(
                f"Expected exactly 1 execute_engine_et node per partition, "
                f"found {len(engine_nodes)}"
            )
        node = engine_nodes[0]
        blob_node = node.args[1]
        for input_spec in edge_program.graph_signature.input_specs:
            if (
                hasattr(input_spec.arg, "name")
                and input_spec.arg.name == blob_node.name
            ):
                blob_tensor = edge_program.state_dict.get(input_spec.target)
                if blob_tensor is None:
                    blob_tensor = edge_program.constants.get(input_spec.target)
                if blob_tensor is None:
                    raise RuntimeError(
                        f"Engine blob buffer '{input_spec.target}' not found "
                        f"in state_dict or constants"
                    )
                return PreprocessResult(
                    processed_bytes=blob_tensor.cpu().contiguous().numpy().tobytes()
                )
        raise RuntimeError(
            f"Engine blob buffer '{blob_node.name}' not found in graph_signature"
        )
