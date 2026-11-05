"""Real NVRx two-rank CPU SIGKILL/SIGSTOP recovery, not a scheduler simulation."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(sys.executable).parent


@pytest.fixture
def runtime():
    # NVRx sockets need a short local path; tempfile honors the caller's TMPDIR.
    with tempfile.TemporaryDirectory(prefix="ft-") as short:
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("NVRX", "TORCHELASTIC", "PET_"))}
        env.update(TMPDIR=short, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1")
        yield env


def train(root, log, env, *args, restarts=1, expected=0, fixture_args=("--tiny",)):
    command = [str(BIN / "ft_launcher"), "--standalone", "--nnodes=1", "--nproc-per-node=2",
               f"--max-restarts={restarts}", "--ft-enable-nic-monitor=false", "--ft-enable-nic-healthcheck=false",
               "--ft-rank-section-timeouts=setup:60,step:15,checkpointing:30",
               "--ft-rank-out-of-section-timeout=30", "--ft-workload-check-interval=1",
               f"--ft-checkpoint-iteration-file={root}/latest_checkpointed_iteration.txt",
               f"--log-dir={log.parent / (log.stem + '-elastic')}", "--tee=3",
               "src/train_gpt2.py", "--run-dir", str(root), "--device=cpu", "--collective-timeout=60",
               *fixture_args, *args]
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=150)
    text = log.read_text()
    assert result.returncode == expected, text
    return text


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            equal(left, right)
    else:
        assert a == b


@pytest.mark.parametrize("fault_kind", ["kill", "stop"])
def test_signal_resume_matches_fresh_control(tmp_path, runtime, fault_kind):
    normal, recovered = tmp_path / "normal", tmp_path / "recovered"
    train(normal, tmp_path / "normal.log", runtime)
    text = train(recovered, tmp_path / "recovered.log", runtime, "--fault-step=3", f"--fault-kind={fault_kind}")
    assert text.count(f"INJECT SIG{fault_kind.upper()}") == 1
    if fault_kind == "stop":
        assert "Section 'step' has been open" in text
    assert text.count("LOADED step=2 cursor=2 restart=1") == 2
    for rank in range(2):
        a = torch.load(normal / f"step_6/rank_{rank}.pt", weights_only=False)
        b = torch.load(recovered / f"step_6/rank_{rank}.pt", weights_only=False)
        equal(a, b)  # Includes optimizer moments/counters, scheduler, RNG and cursor.
        assert a["cursor"] == a["step"] == 6
        assert all(value["step"].item() == 6 for value in a["optimizer"]["state"].values())
        replay = re.findall(rf"\[Rank {rank}\] STEP step=3 .*", text)
        if fault_kind == "kill":
            assert len(replay) == 2 and replay[0] == replay[1]
        else:
            assert len(replay) == 1  # STOP is before forward/backward, not after update 3.
    # Requested resume at the target step loads but makes no additional update.
    text = train(recovered, tmp_path / "resume.log", runtime, "--resume")
    assert "STEP step=" not in text and text.count("DONE step=6") == 2


def test_before_first_save_and_no_restart(tmp_path, runtime):
    root = tmp_path / "early"
    text = train(root, tmp_path / "early.log", runtime, "--fault-step=1", expected=1)
    assert "INJECT SIGKILL" in text and "FileNotFoundError" in text
    assert not (root / "latest_checkpointed_iteration.txt").exists()
    assert "LOADED" not in text and "DONE" not in text
    assert text.count("FRESH step=0") == 2
    root = tmp_path / "bounded"
    text = train(root, tmp_path / "bounded.log", runtime, "--fault-step=3", restarts=0, expected=1)
    assert (root / "latest_checkpointed_iteration.txt").read_text() == "2\n"
    assert "LOADED" not in text and "DONE" not in text


def test_requested_missing_resume_and_local_inputs(tmp_path, runtime):
    root = tmp_path / "missing"
    text = train(root, tmp_path / "missing.log", runtime, "--resume", restarts=0, expected=1)
    assert "FileNotFoundError" in text and "STEP step=" not in text
    from transformers import GPT2Config, GPT2LMHeadModel
    model_path, tokens_path = tmp_path / "gpt2", tmp_path / "tokens.pt"
    GPT2LMHeadModel(GPT2Config(vocab_size=8, n_positions=4, n_embd=8, n_layer=1, n_head=1)).save_pretrained(model_path)
    torch.save(torch.tensor([[1, 2]], dtype=torch.int64), tokens_path)
    root = tmp_path / "local"
    inputs = ("--model-path", str(model_path), "--tokens-path", str(tokens_path))
    text = train(root, tmp_path / "local.log", runtime, "--max-steps=1", fixture_args=inputs)
    assert text.count("DONE step=1") == 2
    (root / "step_1/rank_1.pt").write_bytes(b"corrupt")
    text = train(root, tmp_path / "corrupt.log", runtime, "--max-steps=2", "--resume",
                 fixture_args=inputs, restarts=0, expected=1)
    assert "STEP step=" not in text and "DONE" not in text
