import json
import unittest

import pytest
import torch

import torch_tensorrt.executorch.register_et_ops  # noqa: F401

assertions = unittest.TestCase()


@pytest.mark.unit
def test_op_exists():
    assert hasattr(torch.ops.tensorrt, "execute_engine_et")
    assert callable(torch.ops.tensorrt.execute_engine_et.default)


@pytest.mark.unit
def test_fake_kernel_shapes():
    from torch._subclasses.fake_tensor import FakeTensorMode

    spec = json.dumps(
        [
            {"shape": [1, 16, 30, 30], "dtype": "float32"},
            {"shape": [1], "dtype": "float64"},
        ]
    )
    with FakeTensorMode() as fake_mode:
        fake_input = fake_mode.from_tensor(torch.randn(1, 3, 32, 32))
        fake_blob = fake_mode.from_tensor(torch.zeros(100, dtype=torch.uint8))
        result = torch.ops.tensorrt.execute_engine_et.default(
            [fake_input], fake_blob, spec
        )
    assertions.assertEqual(len(result), 2)
    assertions.assertEqual(result[0].shape, torch.Size([1, 16, 30, 30]))
    assertions.assertEqual(result[0].dtype, torch.float32)
    assertions.assertEqual(result[1].shape, torch.Size([1]))
    assertions.assertEqual(result[1].dtype, torch.float64)


@pytest.mark.unit
def test_fake_kernel_dynamic_shapes():
    from torch._subclasses.fake_tensor import FakeTensorMode

    spec = json.dumps([{"shape": [8, 10], "dtype": "float32"}])
    with FakeTensorMode() as fake_mode:
        fake_input = fake_mode.from_tensor(torch.randn(8, 5))
        fake_blob = fake_mode.from_tensor(torch.zeros(50, dtype=torch.uint8))
        result = torch.ops.tensorrt.execute_engine_et.default(
            [fake_input], fake_blob, spec
        )
    assertions.assertEqual(result[0].shape, torch.Size([8, 10]))


@pytest.mark.unit
def test_fake_kernel_empty_inputs():
    from torch._subclasses.fake_tensor import FakeTensorMode

    spec = json.dumps([{"shape": [2, 3], "dtype": "float32"}])
    with FakeTensorMode() as fake_mode:
        fake_blob = fake_mode.from_tensor(torch.zeros(50, dtype=torch.uint8))
        result = torch.ops.tensorrt.execute_engine_et.default([], fake_blob, spec)
    assertions.assertEqual(result[0].shape, torch.Size([2, 3]))
    assertions.assertEqual(result[0].device.type, "cpu")


@pytest.mark.unit
def test_fake_kernel_device_propagation():
    from torch._subclasses.fake_tensor import FakeTensorMode

    spec = json.dumps([{"shape": [1, 10], "dtype": "float32"}])
    with FakeTensorMode() as fake_mode:
        fake_input = fake_mode.from_tensor(
            torch.randn(1, 5), static_shapes=True
        )
        fake_blob = fake_mode.from_tensor(torch.zeros(50, dtype=torch.uint8))
        result = torch.ops.tensorrt.execute_engine_et.default(
            [fake_input], fake_blob, spec
        )
    assertions.assertEqual(result[0].device, fake_input.device)


@pytest.mark.unit
def test_fake_kernel_bad_output_spec_raises():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode() as fake_mode:
        fake_input = fake_mode.from_tensor(torch.randn(1, 5))
        fake_blob = fake_mode.from_tensor(torch.zeros(50, dtype=torch.uint8))
        with pytest.raises((json.JSONDecodeError, KeyError)):
            torch.ops.tensorrt.execute_engine_et.default(
                [fake_input], fake_blob, "not valid json"
            )


@pytest.mark.unit
def test_real_op_raises():
    spec = json.dumps([{"shape": [1, 10], "dtype": "float32"}])
    with pytest.raises(RuntimeError, match="placeholder op"):
        torch.ops.tensorrt.execute_engine_et.default(
            [torch.randn(1, 5)], torch.zeros(50, dtype=torch.uint8), spec
        )
