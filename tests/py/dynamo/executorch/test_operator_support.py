import unittest

import pytest
import torch

import torch_tensorrt.executorch.register_et_ops  # noqa: F401
from torch_tensorrt.executorch._operator_support import is_trt_engine_op

assertions = unittest.TestCase()


def _make_graph_with_nodes():
    g = torch.fx.Graph()
    x = g.placeholder("x")
    blob = g.get_attr("blob")
    spec = '[{"shape": [1, 10], "dtype": "float32"}]'
    et_node = g.call_function(
        torch.ops.tensorrt.execute_engine_et.default, ([x], blob, spec)
    )
    relu_node = g.call_function(torch.ops.aten.relu.default, (x,))
    g.output((et_node, relu_node))
    return g, x, blob, et_node, relu_node


@pytest.mark.unit
def test_is_trt_engine_op_true():
    _, _, _, et_node, _ = _make_graph_with_nodes()
    assertions.assertTrue(is_trt_engine_op(et_node))


@pytest.mark.unit
def test_is_trt_engine_op_false_for_aten():
    _, _, _, _, relu_node = _make_graph_with_nodes()
    assertions.assertFalse(is_trt_engine_op(relu_node))


@pytest.mark.unit
def test_is_trt_engine_op_false_for_placeholder():
    _, x, _, _, _ = _make_graph_with_nodes()
    assertions.assertFalse(is_trt_engine_op(x))


@pytest.mark.unit
def test_is_trt_engine_op_false_for_get_attr():
    _, _, blob, _, _ = _make_graph_with_nodes()
    assertions.assertFalse(is_trt_engine_op(blob))
