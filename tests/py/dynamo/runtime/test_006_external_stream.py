import unittest

import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.runtime._external_stream import _collect_rt_modules

from ..testing_utilities import DECIMALS_OF_AGREEMENT


class _SampleModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax((x + 2) * 7, dim=0)


class _MultiEngineModel(torch.nn.Module):
    """Forces torch-tensorrt to produce multiple TRT submodules.

    The .cpu().cuda() round-trip between supported regions is a stable
    splitter across torch-tensorrt versions: the partitioner cannot fold a
    device hop into a TRT engine.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.softmax(x + 1, dim=0)
        b = a.cpu().cuda()
        c = torch.softmax(b * 2, dim=0)
        d = c.cpu().cuda()
        e = torch.softmax(d + 3, dim=0)
        return e


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


def _compile_multi_engine(
    device: torch.device, use_python_runtime: bool
) -> torch.nn.Module:
    inputs = [torch_tensorrt.Input(shape=(1, 3, 5), dtype=torch.float32)]
    return torch_tensorrt.compile(
        _MultiEngineModel().eval().to(device),
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

    def tearDown(self) -> None:
        # Defensively clear any external stream binding that may have leaked
        # from a failed test, so subsequent tests see a clean state.
        if hasattr(self, "_module_under_test"):
            try:
                torch_tensorrt.runtime.clear_external_stream(self._module_under_test)
            except Exception:
                pass

    def test_set_and_clear(self) -> None:
        device = torch.device("cuda", 0)
        self._module_under_test = _compile_sample(device, self.use_python_runtime)
        stream = torch.cuda.Stream(device=device)

        with torch_tensorrt.runtime.set_external_stream(
            self._module_under_test, stream
        ):
            rt_mods = _collect_rt_modules(self._module_under_test)
            self.assertGreater(len(rt_mods), 0)
            for _, rt_mod in rt_mods:
                self.assertEqual(rt_mod.get_external_stream(), int(stream.cuda_stream))

        for _, rt_mod in _collect_rt_modules(self._module_under_test):
            self.assertEqual(
                rt_mod.get_external_stream(),
                0,
                "context manager did not restore prior binding",
            )

    def test_execution_matches_default_stream(self) -> None:
        device = torch.device("cuda", 0)
        eager = _SampleModel().eval().to(device)
        self._module_under_test = _compile_sample(device, self.use_python_runtime)

        new_input = torch.randn((1, 3, 5), dtype=torch.float32, device=device)
        eager_output = eager(new_input).cpu()
        default_output = self._module_under_test(new_input).detach().cpu()

        stream = torch.cuda.Stream(device=device)
        with torch_tensorrt.runtime.set_external_stream(
            self._module_under_test, stream
        ):
            external_output = self._module_under_test(new_input)
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
        self._module_under_test = _compile_sample(device, self.use_python_runtime)

        stream_a = torch.cuda.Stream(device=device)
        stream_b = torch.cuda.Stream(device=device)
        new_input = torch.randn((1, 3, 5), dtype=torch.float32, device=device)

        torch_tensorrt.runtime.set_external_stream(self._module_under_test, stream_a)
        out_a = self._module_under_test(new_input).detach().cpu()
        for _, rt_mod in _collect_rt_modules(self._module_under_test):
            self.assertEqual(rt_mod.get_external_stream(), int(stream_a.cuda_stream))

        torch_tensorrt.runtime.set_external_stream(self._module_under_test, stream_b)
        out_b = self._module_under_test(new_input).detach().cpu()
        for _, rt_mod in _collect_rt_modules(self._module_under_test):
            self.assertEqual(rt_mod.get_external_stream(), int(stream_b.cuda_stream))

        torch_tensorrt.runtime.clear_external_stream(self._module_under_test)
        out_default = self._module_under_test(new_input).detach().cpu()
        for _, rt_mod in _collect_rt_modules(self._module_under_test):
            self.assertEqual(rt_mod.get_external_stream(), 0)

        # After clear, the next forward must NOT keep using the previously
        # bound stream wrapper. This regression guards the engine_stream_is_external
        # provenance fix.
        out_default2 = self._module_under_test(new_input).detach().cpu()

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
        self.assertAlmostEqual(
            float(torch.max(torch.abs(out_default - out_default2))),
            0,
            DECIMALS_OF_AGREEMENT,
            msg="Two consecutive default-stream calls after clear diverge",
        )

    def test_rejects_null_handle(self) -> None:
        device = torch.device("cuda", 0)
        self._module_under_test = _compile_sample(device, self.use_python_runtime)
        with self.assertRaisesRegex(ValueError, "non-null"):
            torch_tensorrt.runtime.set_external_stream(self._module_under_test, 0)

    def test_rejects_unsupported_type(self) -> None:
        device = torch.device("cuda", 0)
        self._module_under_test = _compile_sample(device, self.use_python_runtime)
        with self.assertRaisesRegex(TypeError, "int"):
            torch_tensorrt.runtime.set_external_stream(
                self._module_under_test, "not a stream"
            )

    def test_rejects_module_with_no_engines(self) -> None:
        eager = _SampleModel().eval().cuda()
        stream = torch.cuda.Stream()
        with self.assertRaisesRegex(ValueError, "No TRT runtime submodules"):
            torch_tensorrt.runtime.set_external_stream(eager, stream)

    def test_blocks_cudagraphs(self) -> None:
        device = torch.device("cuda", 0)
        self._module_under_test = _compile_sample(device, self.use_python_runtime)
        stream = torch.cuda.Stream(device=device)
        new_input = torch.randn((1, 3, 5), dtype=torch.float32, device=device)

        torch_tensorrt.runtime.set_external_stream(self._module_under_test, stream)
        with torch_tensorrt.runtime.enable_cudagraphs(
            self._module_under_test
        ) as cg_mod:
            with self.assertRaisesRegex(
                Exception, "CUDA Graphs are not supported when an external stream"
            ):
                cg_mod(new_input)

    def test_per_engine_binding_via_set_external_streams(self) -> None:
        """Multi-engine module: bind each engine to its own stream by name."""
        device = torch.device("cuda", 0)
        self._module_under_test = _compile_multi_engine(device, self.use_python_runtime)

        names = torch_tensorrt.runtime.list_trt_submodules(self._module_under_test)
        if len(names) < 2:
            self.skipTest(
                f"Multi-engine model produced only {len(names)} TRT submodule(s); "
                "test requires at least 2 to verify per-engine binding."
            )

        streams = {name: torch.cuda.Stream(device=device) for name in names}
        with torch_tensorrt.runtime.set_external_streams(
            self._module_under_test, streams
        ):
            for name, rt_mod in _collect_rt_modules(self._module_under_test):
                self.assertEqual(
                    rt_mod.get_external_stream(),
                    int(streams[name].cuda_stream),
                    f"submodule {name} not bound to its assigned stream",
                )

        # Restored on exit
        for _, rt_mod in _collect_rt_modules(self._module_under_test):
            self.assertEqual(rt_mod.get_external_stream(), 0)

    def test_set_external_streams_rejects_unknown_keys(self) -> None:
        device = torch.device("cuda", 0)
        self._module_under_test = _compile_sample(device, self.use_python_runtime)
        stream = torch.cuda.Stream(device=device)
        with self.assertRaisesRegex(ValueError, "Unknown submodule keys"):
            torch_tensorrt.runtime.set_external_streams(
                self._module_under_test, {"__not_a_real_engine__": stream}
            )

    def test_set_external_streams_rejects_non_dict(self) -> None:
        device = torch.device("cuda", 0)
        self._module_under_test = _compile_sample(device, self.use_python_runtime)
        stream = torch.cuda.Stream(device=device)
        with self.assertRaisesRegex(TypeError, "dict"):
            torch_tensorrt.runtime.set_external_streams(
                self._module_under_test, [stream]
            )

    def test_list_trt_submodules(self) -> None:
        device = torch.device("cuda", 0)
        self._module_under_test = _compile_sample(device, self.use_python_runtime)
        names = torch_tensorrt.runtime.list_trt_submodules(self._module_under_test)
        self.assertGreater(len(names), 0)
        self.assertTrue(all("_run_on_acc" in n for n in names))


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestExternalStreamPython(TestExternalStreamCpp):
    """Same suite, run against the Python runtime path (use_python_runtime=True)."""

    use_python_runtime = True


if __name__ == "__main__":
    run_tests()
