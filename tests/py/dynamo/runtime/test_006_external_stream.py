import unittest

import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests

from ..testing_utilities import DECIMALS_OF_AGREEMENT


class _SampleModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax((x + 2) * 7, dim=0)


def _compile_sample(device: torch.device) -> torch.nn.Module:
    inputs = [torch_tensorrt.Input(shape=(1, 3, 5), dtype=torch.float32)]
    return torch_tensorrt.compile(
        _SampleModel().eval().to(device),
        ir="dynamo",
        inputs=inputs,
        enabled_precisions={torch.float32},
        min_block_size=1,
        device=device,
        cache_built_engines=False,
        reuse_cached_engines=False,
        use_python_runtime=False,
    )


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestExternalStream(TestCase):
    def test_set_and_clear_external_stream(self):
        device = torch.device("cuda", 0)
        trt_module = _compile_sample(device)

        stream = torch.cuda.Stream(device=device)

        # set via context manager - prior binding (none) restored on exit
        with torch_tensorrt.runtime.set_external_stream(trt_module, stream):
            engines = list(_iter_engines(trt_module))
            self.assertGreater(len(engines), 0, "expected at least one TRT engine")
            for engine in engines:
                self.assertEqual(engine.get_external_stream(), int(stream.cuda_stream))

        for engine in _iter_engines(trt_module):
            self.assertEqual(
                engine.get_external_stream(), 0, "context manager did not restore prior binding"
            )

    def test_external_stream_execution_matches_default(self):
        device = torch.device("cuda", 0)
        eager = _SampleModel().eval().to(device)
        trt_module = _compile_sample(device)

        new_input = torch.randn((1, 3, 5), dtype=torch.float32, device=device)
        eager_output = eager(new_input)
        default_output = trt_module(new_input).detach().cpu()

        stream = torch.cuda.Stream(device=device)
        with torch_tensorrt.runtime.set_external_stream(trt_module, stream):
            external_output = trt_module(new_input)
        torch.cuda.synchronize(device)
        external_output = external_output.detach().cpu()

        self.assertAlmostEqual(
            float(torch.max(torch.abs(eager_output.cpu() - external_output))),
            0,
            DECIMALS_OF_AGREEMENT,
            msg="External stream execution does not match eager output",
        )
        self.assertAlmostEqual(
            float(torch.max(torch.abs(default_output - external_output))),
            0,
            DECIMALS_OF_AGREEMENT,
            msg="External stream execution diverges from default-stream execution",
        )

    def test_swap_external_stream_between_calls(self):
        device = torch.device("cuda", 0)
        trt_module = _compile_sample(device)

        stream_a = torch.cuda.Stream(device=device)
        stream_b = torch.cuda.Stream(device=device)
        new_input = torch.randn((1, 3, 5), dtype=torch.float32, device=device)

        torch_tensorrt.runtime.set_external_stream(trt_module, stream_a)
        out_a = trt_module(new_input).detach().cpu()

        # Re-resolution must take effect immediately on the next call.
        torch_tensorrt.runtime.set_external_stream(trt_module, stream_b)
        out_b = trt_module(new_input).detach().cpu()

        torch_tensorrt.runtime.clear_external_stream(trt_module)
        out_default = trt_module(new_input).detach().cpu()

        self.assertAlmostEqual(
            float(torch.max(torch.abs(out_a - out_b))),
            0,
            DECIMALS_OF_AGREEMENT,
            msg="Swapping external streams produced different numerical results",
        )
        self.assertAlmostEqual(
            float(torch.max(torch.abs(out_a - out_default))),
            0,
            DECIMALS_OF_AGREEMENT,
            msg="Clearing external stream produced different numerical results",
        )

    def test_set_external_stream_rejects_null_handle(self):
        device = torch.device("cuda", 0)
        trt_module = _compile_sample(device)
        with self.assertRaises(ValueError):
            torch_tensorrt.runtime.set_external_stream(trt_module, 0)

    def test_set_external_stream_rejects_unsupported_type(self):
        device = torch.device("cuda", 0)
        trt_module = _compile_sample(device)
        with self.assertRaises(TypeError):
            torch_tensorrt.runtime.set_external_stream(trt_module, "not a stream")

    def test_external_stream_blocks_cudagraphs(self):
        device = torch.device("cuda", 0)
        trt_module = _compile_sample(device)
        stream = torch.cuda.Stream(device=device)
        new_input = torch.randn((1, 3, 5), dtype=torch.float32, device=device)

        torch_tensorrt.runtime.set_external_stream(trt_module, stream)
        try:
            with torch_tensorrt.runtime.enable_cudagraphs(trt_module) as cg_mod:
                with self.assertRaises(Exception):
                    cg_mod(new_input)
        finally:
            torch_tensorrt.runtime.clear_external_stream(trt_module)


def _iter_engines(module):
    if hasattr(module, "engine") and module.engine is not None:
        yield module.engine
        return
    for child in getattr(module, "children", lambda: [])():
        yield from _iter_engines(child)


if __name__ == "__main__":
    run_tests()
