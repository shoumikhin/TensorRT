import operator
import unittest

import pytest
import torch

import torch_tensorrt
from torch_tensorrt.dynamo._exporter import transform
from torch_tensorrt.executorch._backend import TensorRTBackend
from torch_tensorrt.executorch._converter import convert_engines, export_trt_module
from torch_tensorrt.executorch._operator_support import is_trt_engine_op
from torch_tensorrt.executorch._partitioner import TensorRTPartitioner

from .models import ConvSigmoidConv, SimpleConvRelu

assertions = unittest.TestCase()


def _compile_to_ep(model, inputs, **extra):
    compile_spec = {
        "inputs": inputs,
        "device": torch_tensorrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "min_block_size": 1,
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "use_python_runtime": False,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }
    compile_spec.update(extra)
    ep = torch.export.export(model, tuple(inputs))
    trt_gm = torch_tensorrt.dynamo.compile(ep, **compile_spec)
    trt_gm = transform(trt_gm)
    convert_engines(trt_gm)
    ep = export_trt_module(trt_gm, arg_inputs=inputs)
    return ep


def test_single_engine_tagged():
    model = SimpleConvRelu().eval().cuda()
    inputs = [torch.randn(1, 3, 32, 32).cuda()]
    ep = _compile_to_ep(model, inputs)

    partitioner = TensorRTPartitioner()
    result = partitioner.partition(ep)
    assertions.assertGreater(len(result.partition_tags), 0)


def test_multi_engine_separate_tags():
    model = ConvSigmoidConv().eval().cuda()
    inputs = [torch.randn(1, 3, 32, 32).cuda()]
    ep = _compile_to_ep(
        model, inputs, torch_executed_ops={"torch.ops.aten.sigmoid.default"}
    )

    partitioner = TensorRTPartitioner()
    result = partitioner.partition(ep)
    tags = list(result.partition_tags.keys())
    assertions.assertGreaterEqual(len(tags), 2, "Expected multiple engine partitions")
    assertions.assertEqual(len(tags), len(set(tags)))


def test_no_engines_no_tags():
    model = SimpleConvRelu().eval().cuda()
    inputs = [torch.randn(1, 3, 32, 32).cuda()]
    ep = _compile_to_ep(model, inputs, min_block_size=100)

    partitioner = TensorRTPartitioner()
    result = partitioner.partition(ep)
    assertions.assertEqual(len(result.partition_tags), 0)


def test_aten_ops_not_tagged():
    model = SimpleConvRelu().eval().cuda()
    inputs = [torch.randn(1, 3, 32, 32).cuda()]
    ep = _compile_to_ep(
        model, inputs, torch_executed_ops={"torch.ops.aten.relu.default"}
    )

    partitioner = TensorRTPartitioner()
    partitioner.partition(ep)

    for node in ep.graph.nodes:
        if node.op == "call_function" and not is_trt_engine_op(node):
            if node.target is not operator.getitem:
                assertions.assertNotIn("delegation_tag", node.meta)


def test_getitem_users_tagged_with_engine():
    model = SimpleConvRelu().eval().cuda()
    inputs = [torch.randn(1, 3, 32, 32).cuda()]
    ep = _compile_to_ep(model, inputs)

    partitioner = TensorRTPartitioner()
    partitioner.partition(ep)

    for node in ep.graph.nodes:
        if is_trt_engine_op(node):
            engine_tag = node.meta.get("delegation_tag")
            for user in node.users:
                if user.op == "call_function" and user.target is operator.getitem:
                    assertions.assertEqual(
                        user.meta.get("delegation_tag"), engine_tag
                    )


@pytest.mark.unit
def test_delegation_spec_backend_id():
    partitioner = TensorRTPartitioner()
    assertions.assertEqual(
        partitioner.delegation_spec.backend_id, TensorRTBackend.__name__
    )


def test_partition_result_structure():
    model = SimpleConvRelu().eval().cuda()
    inputs = [torch.randn(1, 3, 32, 32).cuda()]
    ep = _compile_to_ep(model, inputs)

    partitioner = TensorRTPartitioner()
    result = partitioner.partition(ep)

    assertions.assertIsNotNone(result.tagged_exported_program)
    assertions.assertIsInstance(result.partition_tags, dict)


def test_buffer_tagged_with_engine():
    model = SimpleConvRelu().eval().cuda()
    inputs = [torch.randn(1, 3, 32, 32).cuda()]
    ep = _compile_to_ep(model, inputs)

    partitioner = TensorRTPartitioner()
    partitioner.partition(ep)

    engine_tags = set()
    for node in ep.graph.nodes:
        if is_trt_engine_op(node):
            tag = node.meta.get("delegation_tag")
            assertions.assertIsNotNone(tag)
            engine_tags.add(tag)

    assertions.assertGreater(len(engine_tags), 0)

    tagged_placeholders = []
    for node in ep.graph.nodes:
        if node.op == "placeholder" and "delegation_tag" in node.meta:
            tagged_placeholders.append(node)

    assertions.assertGreater(len(tagged_placeholders), 0)
    for ph in tagged_placeholders:
        assertions.assertIn(ph.meta["delegation_tag"], engine_tags)
