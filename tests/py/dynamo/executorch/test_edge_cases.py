import importlib
import os
import unittest
from unittest import mock

import pytest
import torch

import torch_tensorrt

from .models import AddModel, SimpleConvRelu

assertions = unittest.TestCase()


def _compile(model, inputs, **extra):
    ep = torch.export.export(model, tuple(inputs))
    compile_spec = {
        "inputs": inputs,
        "enabled_precisions": {torch.float},
        "device": torch_tensorrt.Device("cuda:0"),
        "use_python_runtime": False,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }
    compile_spec.update(extra)
    return torch_tensorrt.dynamo.compile(ep, **compile_spec)


def test_no_trt_ops(tmp_path):
    model = SimpleConvRelu().eval().cuda()
    inputs = [torch.randn(1, 3, 32, 32).cuda()]

    trt_gm = _compile(model, inputs, min_block_size=100)

    pte_path = os.path.join(str(tmp_path), "model.pte")
    torch_tensorrt.save(
        trt_gm,
        pte_path,
        output_format="executorch",
        inputs=inputs,
    )
    assertions.assertTrue(os.path.exists(pte_path))


def test_all_trt(tmp_path):
    model = SimpleConvRelu().eval().cuda()
    inputs = [torch.randn(1, 3, 32, 32).cuda()]

    trt_gm = _compile(
        model, inputs,
        min_block_size=1,
        require_full_compilation=True,
    )

    pte_path = os.path.join(str(tmp_path), "model.pte")
    torch_tensorrt.save(
        trt_gm,
        pte_path,
        output_format="executorch",
        inputs=inputs,
    )
    assertions.assertTrue(os.path.exists(pte_path))


def test_multi_engine(tmp_path):
    model = SimpleConvRelu().eval().cuda()
    inputs = [torch.randn(1, 3, 32, 32).cuda()]

    trt_gm = _compile(
        model, inputs,
        min_block_size=1,
        torch_executed_ops={"torch.ops.aten.relu.default"},
    )

    pte_path = os.path.join(str(tmp_path), "model.pte")
    torch_tensorrt.save(
        trt_gm,
        pte_path,
        output_format="executorch",
        inputs=inputs,
    )
    assertions.assertTrue(os.path.exists(pte_path))


@pytest.mark.unit
def test_missing_executorch():
    import torch_tensorrt.executorch as et_mod
    try:
        with mock.patch.dict("sys.modules", {"executorch": None, "executorch.exir": None}):
            importlib.reload(et_mod)
        # The lazy __getattr__ guard rejects attribute access with an informative error
        with pytest.raises(ImportError, match="ExecuTorch is required"):
            et_mod.__getattr__("to_trt")
    finally:
        importlib.reload(et_mod)
        et_mod.__dict__.pop("__getattr__", None)


@pytest.mark.unit
def test_missing_inputs(tmp_path):
    from torch_tensorrt._compile import _save_as_executorch

    model = SimpleConvRelu()
    with pytest.raises(ValueError, match="inputs are required"):
        _save_as_executorch(model, str(tmp_path / "test.pte"), arg_inputs=(), kwarg_inputs={})


@pytest.mark.unit
def test_exported_program_with_executorch_raises(tmp_path):
    model = SimpleConvRelu()
    ep = torch.export.export(model, (torch.randn(1, 3, 32, 32),))
    with pytest.raises(ValueError, match="ExportedProgram cannot be saved directly"):
        torch_tensorrt.save(ep, str(tmp_path / "test.pte"), output_format="executorch")


@pytest.mark.unit
def test_empty_inputs_with_executorch_raises(tmp_path):
    from torch_tensorrt._compile import _save_as_executorch

    model = SimpleConvRelu()
    with pytest.raises(ValueError, match="inputs are required"):
        _save_as_executorch(model, str(tmp_path / "test.pte"), arg_inputs=[], kwarg_inputs={})


def test_multi_input_model(tmp_path):
    model = AddModel().eval().cuda()
    inputs = [torch.randn(1, 10).cuda(), torch.randn(1, 10).cuda()]

    trt_gm = _compile(model, inputs, min_block_size=1)

    pte_path = os.path.join(str(tmp_path), "model.pte")
    torch_tensorrt.save(
        trt_gm,
        pte_path,
        output_format="executorch",
        inputs=inputs,
    )
    assertions.assertTrue(os.path.exists(pte_path))


def test_retrace_with_executorch_succeeds(tmp_path):
    model = SimpleConvRelu().eval().cuda()
    inputs = [torch.randn(1, 3, 32, 32).cuda()]

    trt_gm = _compile(model, inputs, min_block_size=1)

    pte_path = os.path.join(str(tmp_path), "model.pte")
    torch_tensorrt.save(
        trt_gm,
        pte_path,
        output_format="executorch",
        retrace=True,
        inputs=inputs,
    )
    assertions.assertTrue(os.path.exists(pte_path))
