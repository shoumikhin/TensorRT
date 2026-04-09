import pytest
import importlib
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast tests without GPU compilation")
    config.addinivalue_line("markers", "critical: critical-path tests that must pass")


def pytest_collection_modifyitems(config, items):
    if importlib.util.find_spec("executorch") is None:
        skip = pytest.mark.skip(reason="executorch not installed")
        for item in items:
            item.add_marker(skip)
        return

    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "unit" not in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture(autouse=True)
def dynamo_reset():
    yield
    import torch._dynamo

    torch._dynamo.reset()
