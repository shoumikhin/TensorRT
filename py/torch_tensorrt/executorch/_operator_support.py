"""Operator-support helpers for TensorRT ExecuTorch integration."""

import torch


def is_trt_engine_op(node: torch.fx.Node) -> bool:
    """Return ``True`` if *node* is a ``tensorrt::execute_engine_et`` call."""
    return (
        node.op == "call_function"
        and node.target == torch.ops.tensorrt.execute_engine_et.default
    )
