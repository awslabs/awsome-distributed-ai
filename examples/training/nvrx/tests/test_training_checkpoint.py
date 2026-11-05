"""Focused single-rank checkpoint and input checks; real restart is in test_train_gpt2."""
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import training_checkpoint as ckpt
from train_gpt2 import validate_tokens


@pytest.fixture
def group(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def components():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    return model, optimizer, scheduler


def test_new_and_required_resume(tmp_path, group):
    root = tmp_path / "run"
    assert ckpt.select_checkpoint(root, False, 0) is None
    with pytest.raises(FileExistsError):
        ckpt.select_checkpoint(root, False, 0)
    for resume, restart in [(True, 0), (False, 1)]:
        with pytest.raises(FileNotFoundError):
            ckpt.select_checkpoint(root, resume, restart)
    (root / ckpt.POINTER).write_text("broken")
    with pytest.raises(ValueError):
        ckpt.select_checkpoint(root, True, 0)


def test_selector_broadcast_pins_device(tmp_path, group, monkeypatch):
    """The object broadcast names its device: CPU on Gloo without touching CUDA, the current CUDA device on NCCL.

    The Gloo half runs the real one-rank collective; the NCCL half only checks the device choice, because no GPU or NCCL group exists in this CPU test.
    """
    calls = []
    real_broadcast = dist.broadcast_object_list
    def record(*args, **kwargs):
        calls.append(kwargs)
        return real_broadcast(*args, **kwargs)
    def no_cuda(*args, **kwargs):
        raise AssertionError("Gloo path must not query CUDA")
    with monkeypatch.context() as patch:
        patch.setattr(dist, "broadcast_object_list", record)
        patch.setattr(torch.cuda, "current_device", no_cuda)
        assert ckpt.select_checkpoint(tmp_path / "run", False, 0) is None
    assert [call.get("device") for call in calls] == [torch.device("cpu")]
    with monkeypatch.context() as patch:
        patch.setattr(dist, "get_backend", lambda *args, **kwargs: "nccl")
        patch.setattr(torch.cuda, "current_device", lambda: 3)
        assert ckpt.collective_device() == torch.device("cuda", 3)


def test_incomplete_save_and_corrupt_resume(tmp_path, group, monkeypatch):
    model, optimizer, scheduler = components()
    settings = {"strategy": "ddp", "world_size": 1}
    ckpt.save(tmp_path, model, optimizer, scheduler, 1, 1, settings)
    original = (tmp_path / "step_1/rank_0.pt").read_bytes()
    def fail_write(*args, **kwargs):
        raise OSError("injected write failure")
    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", fail_write)
        with pytest.raises(OSError, match="write failure"):
            ckpt.save(tmp_path, model, optimizer, scheduler, 2, 2, settings)
    assert ckpt.select_checkpoint(tmp_path, True, 0) == 1
    assert (tmp_path / "step_1/rank_0.pt").read_bytes() == original
    assert ckpt.load(tmp_path, 1, model, optimizer, scheduler, settings) == 1
    with pytest.raises(ValueError, match="settings differ"):
        ckpt.load(tmp_path, 1, model, optimizer, scheduler, dict(settings, world_size=2))
    (tmp_path / "step_1/rank_0.pt").unlink()
    with pytest.raises(FileNotFoundError):
        ckpt.load(tmp_path, 1, model, optimizer, scheduler, settings)
    (tmp_path / "step_1/rank_0.pt").write_bytes(b"broken")
    with pytest.raises(Exception):
        ckpt.load(tmp_path, 1, model, optimizer, scheduler, settings)


def test_token_boundary():
    for shape in [(0, 2), (1, 0), (1, 1)]:
        with pytest.raises(ValueError, match="at least two tokens"):
            validate_tokens(torch.zeros(shape, dtype=torch.int64))
    with pytest.raises(ValueError, match="int64"):
        validate_tokens(torch.zeros((1, 2)))
    validate_tokens(torch.zeros((1, 2), dtype=torch.int64))
