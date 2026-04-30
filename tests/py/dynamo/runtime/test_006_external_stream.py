import unittest

import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.runtime._external_stream import _collect_rt_modules

from ..testing_utilities import DECIMALS_OF_AGREEMENT


class _SampleModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax((x + 2) * 7, dim=0)


def _compile_sample(device: torch.device, use_python_runtime: bool) -> torch.nn.Module:
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
        use_python_runtime=use_python_runtime,
    )


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestExternalStreamCpp(TestCase):
    """Exercises the C++ TRTEngine custom-class path (use_python_runtime=False)."""

    use_python_runtime = False

    def test_set_and_clear(self) -> None:
        device = torch.device("cuda", 0)
        trt_module = _compile_sample(device, self.use_python_runtime)
        stream = torch.cuda.Stream(device=device)

        with torch_tensorrt.runtime.set_external_stream(trt_module, stream):
            rt_mods = _collect_rt_modules(trt_module)
            self.assertGreater(len(rt_mods), 0)
            for rt_mod in rt_mods:
                self.assertEqual(rt_mod.get_external_stream(), int(stream.cuda_stream))

        for rt_mod in _collect_rt_modules(trt_module):
            self.assertEqual(
                rt_mod.get_external_stream(),
                0,
                "context manager did not restore prior binding",
            )

    def test_execution_matches_default_stream(self) -> None:
        device = torch.device("cuda", 0)
        eager = _SampleModel().eval().to(device)
        trt_module = _compile_sample(device, self.use_python_runtime)

        new_input = torch.randn((1, 3, 5), dtype=torch.float32, device=device)
        eager_output = eager(new_input).cpu()
        default_output = trt_module(new_input).detach().cpu()

        stream = torch.cuda.Stream(device=device)
        with torch_tensorrt.runtime.set_external_stream(trt_module, stream):
            external_output = trt_module(new_input)
        torch.cuda.synchronize(device)
        external_output = external_output.detach().cpu()

        self.assertAlmostEqual(
            float(torch.max(torch.abs(eager_output - external_output))),
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

    def test_swap_and_clear_between_calls(self) -> None:
        """Verifies the per-call re-resolution: set/clear take effect immediately."""
        device = torch.device("cuda", 0)
        trt_module = _compile_sample(device, self.use_python_runtime)

        stream_a = torch.cuda.Stream(device=device)
        stream_b = torch.cuda.Stream(device=device)
        new_input = torch.randn((1, 3, 5), dtype=torch.float32, device=device)

        torch_tensorrt.runtime.set_external_stream(trt_module, stream_a)
        out_a = trt_module(new_input).detach().cpu()
        for rt_mod in _collect_rt_modules(trt_module):
            self.assertEqual(rt_mod.get_external_stream(), int(stream_a.cuda_stream))

        torch_tensorrt.runtime.set_external_stream(trt_module, stream_b)
        out_b = trt_module(new_input).detach().cpu()
        for rt_mod in _collect_rt_modules(trt_module):
            self.assertEqual(rt_mod.get_external_stream(), int(stream_b.cuda_stream))

        torch_tensorrt.runtime.clear_external_stream(trt_module)
        out_default = trt_module(new_input).detach().cpu()
        for rt_mod in _collect_rt_modules(trt_module):
            self.assertEqual(rt_mod.get_external_stream(), 0)

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

    def test_rejects_null_handle(self) -> None:
        device = torch.device("cuda", 0)
        trt_module = _compile_sample(device, self.use_python_runtime)
        with self.assertRaisesRegex(ValueError, "non-null"):
            torch_tensorrt.runtime.set_external_stream(trt_module, 0)

    def test_rejects_unsupported_type(self) -> None:
        device = torch.device("cuda", 0)
        trt_module = _compile_sample(device, self.use_python_runtime)
        with self.assertRaisesRegex(TypeError, "int"):
            torch_tensorrt.runtime.set_external_stream(trt_module, "not a stream")

    def test_rejects_module_with_no_engines(self) -> None:
        eager = _SampleModel().eval().cuda()
        stream = torch.cuda.Stream()
        with self.assertRaisesRegex(ValueError, "No TRT runtime submodules"):
            torch_tensorrt.runtime.set_external_stream(eager, stream)

    def test_blocks_cudagraphs(self) -> None:
        device = torch.device("cuda", 0)
        trt_module = _compile_sample(device, self.use_python_runtime)
        stream = torch.cuda.Stream(device=device)
        new_input = torch.randn((1, 3, 5), dtype=torch.float32, device=device)

        torch_tensorrt.runtime.set_external_stream(trt_module, stream)
        try:
            with torch_tensorrt.runtime.enable_cudagraphs(trt_module) as cg_mod:
                with self.assertRaisesRegex(
                    Exception, "CUDA Graphs are not supported when an external stream"
                ):
                    cg_mod(new_input)
        finally:
            torch_tensorrt.runtime.clear_external_stream(trt_module)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestExternalStreamPython(TestExternalStreamCpp):
    """Same suite, run against the Python runtime path (use_python_runtime=True)."""

    use_python_runtime = True


if __name__ == "__main__":
    run_tests()
