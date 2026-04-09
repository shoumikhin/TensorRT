"""Custom operators for TensorRT engine execution in ExecuTorch.

Registers the ``tensorrt::execute_engine_et`` custom op and its fake-tensor
implementation so that the op can be traced, exported, and lowered by the
ExecuTorch toolchain.
"""

import json
from typing import List

import torch


@torch.library.custom_op("tensorrt::execute_engine_et", mutates_args=())
def execute_engine_et(
    inputs: List[torch.Tensor],
    engine_blob: torch.Tensor,
    output_spec: str,
) -> List[torch.Tensor]:
    raise RuntimeError(
        "execute_engine_et is a placeholder op for ExecuTorch export. "
        "It should never be executed directly in Python."
    )


@torch.library.register_fake("tensorrt::execute_engine_et")
def fake_execute_engine_et(
    inputs: List[torch.Tensor],
    engine_blob: torch.Tensor,
    output_spec: str,
) -> List[torch.Tensor]:
    spec = json.loads(output_spec)
    outputs = []
    for entry in spec:
        shape = entry["shape"]
        dtype = getattr(torch, entry["dtype"], None)
        if dtype is None:
            raise ValueError(f"Unknown dtype: {entry['dtype']}")
        device = inputs[0].device if inputs else torch.device("cpu")
        outputs.append(torch.empty(shape, dtype=dtype, device=device))
    return outputs
