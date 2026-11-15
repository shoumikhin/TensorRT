# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import glob
import inspect
import os
import tempfile

import pytest
import torch

pytest.importorskip("executorch.exir")

import torch_tensorrt  # noqa: E402
from executorch.exir._serialize._cord import FileBackedData  # noqa: E402
from executorch.exir._serialize._named_data_store import NamedDataStore  # noqa: E402
from torch_tensorrt._utils import _cache_root  # noqa: E402
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule  # noqa: E402


class _TwoEngines(torch.nn.Module):
    """Two TensorRT partitions around an op forced to PyTorch."""

    def __init__(self):
        super().__init__()
        self.a, self.b = torch.nn.Linear(64, 64), torch.nn.Linear(64, 8)

    def forward(self, x):
        return self.b(torch.sin(self.a(x)))


@pytest.mark.unit
def test_engines_stream_from_their_files_into_the_program(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    setups, streamed = [], {}
    real_setup = TorchTensorRTModule.setup_engine
    real_add = NamedDataStore.add_named_data

    def counting_setup(self):
        setups.append(self.name)
        real_setup(self)

    def recording_add(self, key, data, *args, **kwargs):
        if isinstance(data, FileBackedData):
            with open(data._path, "rb") as f:
                streamed[data._path] = f.read()
        else:
            streamed[key] = data
        return real_add(self, key, data, *args, **kwargs)

    monkeypatch.setattr(TorchTensorRTModule, "setup_engine", counting_setup)
    monkeypatch.setattr(NamedDataStore, "add_named_data", recording_add)

    torch.manual_seed(0)
    x = torch.randn(4, 64, device="cuda")
    pte_path = tmp_path / "model.pte"
    torch_tensorrt.compile_and_save(
        torch.export.export(_TwoEngines().eval().cuda(), (x,)),
        pte_path,
        output_format="executorch",
        arg_inputs=[x],
        min_block_size=1,
        torch_executed_ops={torch.ops.aten.sin.default},
    )

    # No engine was ever set up, and each one reached the program as its own file, from the cache directory.
    assert setups == []
    assert len(streamed) == 2
    offloaded = str(_cache_root() / "offloaded_engines")
    assert all(
        path.endswith(".engine") and os.path.dirname(path) == offloaded
        for path in streamed
    )
    pte = pte_path.read_bytes()
    assert all(pte.find(engine) >= 0 for engine in streamed.values())
    # The files are gone once the program is saved.
    assert not glob.glob(str(tmp_path / "**" / "*.engine"), recursive=True)


@pytest.mark.unit
def test_engines_move_into_the_program_without_a_second_copy(tmp_path, monkeypatch):
    if "consume" not in inspect.signature(FileBackedData.write_to_file).parameters:
        pytest.skip("This ExecuTorch can only copy file-backed data")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    left = []
    real_write = FileBackedData.write_to_file

    def recording_write(self, outfile, **kwargs):
        real_write(self, outfile, **kwargs)
        left.append(os.path.getsize(self._path))

    monkeypatch.setattr(FileBackedData, "write_to_file", recording_write)
    torch.manual_seed(0)
    x = torch.randn(4, 64, device="cuda")
    torch_tensorrt.compile_and_save(
        torch.export.export(_TwoEngines().eval().cuda(), (x,)),
        tmp_path / "model.pte",
        output_format="executorch",
        arg_inputs=[x],
        min_block_size=1,
        torch_executed_ops={torch.ops.aten.sin.default},
    )
    # Each engine file was emptied as it was written, so no engine was ever on disk twice.
    assert left == [0, 0]


@pytest.mark.unit
def test_an_etrecord_is_saved_from_the_engines_before_they_move(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    torch.manual_seed(0)
    x = torch.randn(4, 64, device="cuda")
    pte_path = tmp_path / "model.pte"
    torch_tensorrt.compile_and_save(
        torch.export.export(_TwoEngines().eval().cuda(), (x,)),
        pte_path,
        output_format="executorch",
        arg_inputs=[x],
        min_block_size=1,
        torch_executed_ops={torch.ops.aten.sin.default},
        save_options={"generate_etrecord": True},
    )
    assert pte_path.stat().st_size > 0
    assert (tmp_path / "model_etrecord.bin").stat().st_size > 0
