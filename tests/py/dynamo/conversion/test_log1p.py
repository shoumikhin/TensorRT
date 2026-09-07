import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestLog1pConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((10,), torch.float),
            ((1, 20), torch.float),
            ((2, 3, 4), torch.float),
            ((2, 3, 4, 5), torch.float),
        ]
    )
    def test_log1p_float(self, input_shape, dtype):
        class Log1p(nn.Module):
            def forward(self, input):
                return torch.ops.aten.log1p.default(input)

        inputs = [
            torch.randn(input_shape, dtype=dtype).abs() + 0.001
        ]  # ensure positive input
        self.run_test(
            Log1p(),
            inputs,
        )

    @parameterized.expand(
        [
            ((10,), torch.int, 0, 5),
            ((1, 20), torch.int, 0, 10),
            ((2, 3, 4), torch.int, 0, 5),
            ((2, 3, 4, 5), torch.int, 0, 5),
        ]
    )
    def test_log1p_int(self, input_shape, dtype, low, high):
        class Log1p(nn.Module):
            def forward(self, input):
                return torch.ops.aten.log1p.default(input)

        inputs = [
            torch.randint(low, high, input_shape, dtype=dtype).abs() + 0.001
        ]  # ensure positive input
        self.run_test(
            Log1p(),
            inputs,
        )

    @parameterized.expand(
        [
            (torch.full((1, 20), 2, dtype=torch.float),),
            (torch.full((2, 3, 4), 3, dtype=torch.float),),
            (torch.full((2, 3, 4, 5), 4, dtype=torch.float),),
        ]
    )
    def test_log1p_const_float(self, data):
        class Log1p(nn.Module):
            def forward(self, input):
                return torch.ops.aten.log1p.default(input)

        inputs = [data]
        self.run_test(
            Log1p(),
            inputs,
        )

    @parameterized.expand(
        [
            ((1,), (3,), (5,)),
            ((1, 20), (2, 20), (3, 20)),
            ((2, 3, 4), (3, 4, 5), (4, 5, 6)),
            ((2, 3, 4, 5), (3, 5, 5, 6), (4, 5, 6, 7)),
        ]
    )
    def test_log1p_float_dynamic_shape(self, min_shape, opt_shape, max_shape):
        class Log1p(nn.Module):
            def forward(self, input):
                return torch.ops.aten.log1p.default(input)

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(
            Log1p(),
            input_specs,
            use_dynamo_tracer=True,
        )

    @parameterized.expand(
        [
            ("int32", torch.int32),
            ("int64", torch.int64),
            ("bool", torch.bool),
        ]
    )
    def test_log1p_integer_input(self, _, dtype):
        """TensorRT's log accepts no integer type, so log1p has to cast. It also has to
        cast before adding one, or a large integer wraps and the log returns NaN."""

        class Log1p(nn.Module):
            def forward(self, input):
                return torch.ops.aten.log1p.default(input)

        if dtype is torch.bool:
            inputs = [torch.tensor([True, False, True, True])]
        else:
            inputs = [torch.tensor([3, 1, 2, 1], dtype=dtype)]
        self.run_test(
            Log1p(),
            inputs,
            use_dynamo_tracer=True,
        )

    def test_log1p_int32_near_max(self):
        """1 + INT32_MAX wraps negative in the integer domain, and the log of a negative
        number is NaN, so this returns a wrong answer rather than failing."""

        class Log1p(nn.Module):
            def forward(self, input):
                return torch.ops.aten.log1p.default(input)

        inputs = [torch.tensor([2147483647, 100, 10, 1], dtype=torch.int32)]
        self.run_test(
            Log1p(),
            inputs,
            use_dynamo_tracer=True,
        )


if __name__ == "__main__":
    run_tests()
