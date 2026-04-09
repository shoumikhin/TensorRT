import operator
from typing import Any, Dict, List, Optional

from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data

from torch_tensorrt.executorch._backend import TensorRTBackend
from torch_tensorrt.executorch._operator_support import is_trt_engine_op


class TensorRTPartitioner(Partitioner):
    """Partitioner that tags TensorRT engine nodes for ExecuTorch delegation."""

    def __init__(self, compile_specs: Optional[List[Any]] = None) -> None:
        super().__init__()
        compile_specs = compile_specs or []
        self.delegation_spec = DelegationSpec(
            TensorRTBackend.__name__, compile_specs
        )

    def partition(
        self, exported_program: "ExportedProgram"
    ) -> "PartitionResult":
        """Partition the exported program by tagging TensorRT engine nodes.

        Arguments:
            exported_program (ExportedProgram): The exported program whose graph
                contains ``execute_engine_et`` call-function nodes.

        Returns:
            PartitionResult with the tagged program and per-partition delegation specs.
        """
        partition_tags: Dict[str, DelegationSpec] = {}
        engine_count = 0
        for node in exported_program.graph.nodes:
            if is_trt_engine_op(node):
                tag = f"trt_engine_{engine_count}"
                node.meta["delegation_tag"] = tag
                partition_tags[tag] = self.delegation_spec
                for user in node.users:
                    if user.op == "call_function" and user.target is operator.getitem:
                        user.meta["delegation_tag"] = tag
                engine_count += 1
        tag_constant_data(exported_program)
        return PartitionResult(
            tagged_exported_program=exported_program,
            partition_tags=partition_tags,
        )
