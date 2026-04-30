import unittest

import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.runtime._external_stream import _iter_rt_modules

from ..testing_utilities import DECIMALS_OF_AGREEMENT


class _SampleModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax((x + 2) * 7, dim=0)


class _MultiEngineModel(torch.nn.Module):
    # The .cpu().cuda() round-trip is a stable splitter: the partitioner cannot
    # fold a device hop into a TRT engine.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.softmax(x + 1, dim=0)
        b = a.cpu().cuda()
        c = torch.softmax(b * 2, dim=0)
        d = c.cpu().cuda()
        return torch.softmax(d + 3, dim=0)


def _compile(
    model: torch.nn.Module, device: torch.device, use_python_runtime: bool
) -> torch.nn.Module:
    return torch_tensorrt.compile(
        model.eval().to(device),
        ir="dynamo",
        inputs=[torch_tensorrt.Input(shape=(1, 3, 5), dtype=torch.float32)],
        enabled_precisions={torch.float32},
        min_block_size=1,
        device=device,
        cache_built_engines=False,
        reuse_cached_engines=False,
        use_python_runtime=use_python_runtime,
    )


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestExternalStreamCpp(TestCase):
    """Exercises the C++ TRTEngine custom-class path."""

    use_python_runtime = False

    def test_swap_and_clear_between_calls(self) -> None:
        device = torch.device("cuda", 0)
        m = _compile(_SampleModel(), device, self.use_python_runtime)

        s_a = torch.cuda.Stream(device=device)
        s_b = torch.cuda.Stream(device=device)
        x = torch.randn((1, 3, 5), dtype=torch.float32, device=device)

        torch_tensorrt.runtime.set_external_stream(m, s_a)
        out_a = m(x).detach().cpu()
        for _, rt in _iter_rt_modules(m):
            self.assertEqual(rt.get_external_stream(), int(s_a.cuda_stream))

        torch_tensorrt.runtime.set_external_stream(m, s_b)
        out_b = m(x).detach().cpu()
        for _, rt in _iter_rt_modules(m):
            self.assertEqual(rt.get_external_stream(), int(s_b.cuda_stream))

        torch_tensorrt.runtime.clear_external_stream(m)
        out_d1 = m(x).detach().cpu()
        out_d2 = m(x).detach().cpu()  # second call guards the provenance fix
        for _, rt in _iter_rt_modules(m):
            self.assertEqual(rt.get_external_stream(), 0)

        for label, out in [("swap", out_b), ("clear", out_d1), ("re-default", out_d2)]:
            self.assertAlmostEqual(
                float(torch.max(torch.abs(out_a - out))),
                0,
                DECIMALS_OF_AGREEMENT,
                msg=f"output diverged after {label}",
            )

    def test_execution_matches_eager(self) -> None:
        device = torch.device("cuda", 0)
        eager = _SampleModel().eval().to(device)
        m = _compile(_SampleModel(), device, self.use_python_runtime)
        x = torch.randn((1, 3, 5), dtype=torch.float32, device=device)

        with torch_tensorrt.runtime.set_external_stream(
            m, torch.cuda.Stream(device=device)
        ):
            out = m(x)
        torch.cuda.synchronize(device)

        self.assertAlmostEqual(
            float(torch.max(torch.abs(eager(x).cpu() - out.detach().cpu()))),
            0,
            DECIMALS_OF_AGREEMENT,
        )

    def test_per_engine_binding(self) -> None:
        device = torch.device("cuda", 0)
        m = _compile(_MultiEngineModel(), device, self.use_python_runtime)
        rt_mods = _iter_rt_modules(m)
        if len(rt_mods) < 2:
            self.skipTest(
                f"need ≥2 TRT submodules to verify per-engine binding (got {len(rt_mods)})"
            )

        streams = {name: torch.cuda.Stream(device=device) for name, _ in rt_mods}
        with torch_tensorrt.runtime.set_external_stream(m, streams):
            for name, rt in rt_mods:
                self.assertEqual(
                    rt.get_external_stream(), int(streams[name].cuda_stream)
                )

        for _, rt in rt_mods:
            self.assertEqual(rt.get_external_stream(), 0)

    def test_blocks_cudagraphs(self) -> None:
        device = torch.device("cuda", 0)
        m = _compile(_SampleModel(), device, self.use_python_runtime)
        torch_tensorrt.runtime.set_external_stream(m, torch.cuda.Stream(device=device))
        try:
            with torch_tensorrt.runtime.enable_cudagraphs(m) as cg:
                with self.assertRaisesRegex(
                    Exception, "CUDA Graphs are not supported when an external stream"
                ):
                    cg(torch.randn((1, 3, 5), dtype=torch.float32, device=device))
        finally:
            torch_tensorrt.runtime.clear_external_stream(m)

    def test_validation(self) -> None:
        device = torch.device("cuda", 0)
        m = _compile(_SampleModel(), device, self.use_python_runtime)
        s = torch.cuda.Stream(device=device)
        with self.assertRaisesRegex(ValueError, "non-null"):
            torch_tensorrt.runtime.set_external_stream(m, 0)
        with self.assertRaisesRegex(TypeError, "must be int or torch.cuda.Stream"):
            torch_tensorrt.runtime.set_external_stream(m, "not a stream")
        with self.assertRaisesRegex(ValueError, "Unknown submodule keys"):
            torch_tensorrt.runtime.set_external_stream(m, {"__not_real__": s})
        with self.assertRaisesRegex(ValueError, "No TRT runtime submodules"):
            torch_tensorrt.runtime.set_external_stream(_SampleModel().eval().cuda(), s)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestExternalStreamPython(TestExternalStreamCpp):
    """Same suite against the Python runtime path."""

    use_python_runtime = True


if __name__ == "__main__":
    run_tests()
