# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Targeted CPU-only configuration tests; these do not validate training."""
import importlib.util
import os
from pathlib import Path
import subprocess
import re
import tempfile
import textwrap
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ENTRYPOINT = HERE / "kimi-k2/benchmarks/bench_kimi_k2_pretrain.py"


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(
            num_layers=61, hidden_size=7168, num_moe_experts=384, moe_router_topk=8,
            num_attention_heads=64, vocab_size=163840, moe_router_num_groups=1,
            multi_latent_attention=True, cuda_graph_impl="none",
            moe_token_dispatcher_type="alltoall", moe_shared_expert_overlap=True,
            overlap_moe_expert_parallel_comm=False, delay_wgrad_compute=False,
        )
        self.cfg = SimpleNamespace(
            model=None, dataset=SimpleNamespace(seq_length=4096),
            train=SimpleNamespace(), optimizer=SimpleNamespace(lr=None),
            scheduler=SimpleNamespace(lr_warmup_iters=2000),
            tokenizer=SimpleNamespace(vocab_size=131072),
            checkpoint=SimpleNamespace(save="recipe-checkpoints", load="recipe-checkpoints"),
            logger=SimpleNamespace(tensorboard_dir="recipe-tensorboard"),
            comm_overlap=SimpleNamespace(overlap_moe_expert_parallel_comm=False),
        )
        def apply(model, backend):
            model.moe_token_dispatcher_type = "flex"
            model.moe_flex_dispatcher_backend = backend
        self.helper = apply
        def route(model, backend):
            self.helper(model, backend)
        recipe = ModuleType("megatron.bridge.recipes.deepseek.deepseek_v3")
        recipe.deepseek_v3_pretrain_config_32nodes = lambda: self.cfg
        recipe.set_deepseek_v3_pipeline_model_parallel_layout = lambda m: None
        recipe.apply_flex_dispatcher_backend = route
        auto = SimpleNamespace(from_hf_pretrained=lambda *a, **k: SimpleNamespace(
            to_megatron_provider=lambda **k: self.model))
        torch = SimpleNamespace(bfloat16="bf16", cuda=SimpleNamespace(
            get_device_properties=lambda _: SimpleNamespace(name="unsupported")))
        fused = SimpleNamespace(HAVE_DEEP_EP_V2=True)
        self.modules = {
            "torch": torch,
            "megatron.bridge": SimpleNamespace(AutoBridge=auto),
            recipe.__name__: recipe,
            "megatron.bridge.training.flex_dispatcher_backend": SimpleNamespace(apply_flex_dispatcher_backend=route),
            "megatron.core.transformer.moe": SimpleNamespace(fused_a2a=fused),
        }
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.module_patch = patch.dict(sys.modules, self.modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        spec = importlib.util.spec_from_file_location("bench_kimi_config_test", ENTRYPOINT)
        self.bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.bench)

    def test_stable_default_deepep_keeps_overlap(self):
        cfg = self.bench.build_config()
        self.assertEqual(cfg.model.moe_flex_dispatcher_backend, "deepep")
        self.assertEqual(cfg.model.moe_token_dispatcher_type, "flex")
        self.assertTrue(cfg.comm_overlap.overlap_moe_expert_parallel_comm)
        self.assertIsNone(cfg.optimizer.lr)

    def test_stable_alltoall(self):
        os.environ["MOE_DISPATCHER"] = "alltoall"
        cfg = self.bench.build_config()
        self.assertEqual(cfg.model.moe_token_dispatcher_type, "alltoall")
        self.assertIsNone(cfg.model.moe_flex_dispatcher_backend)

    def test_deepep_helper_cannot_silently_fall_back(self):
        self.helper = lambda *args: None
        with self.assertRaisesRegex(RuntimeError, "did not become flex"):
            self.bench.build_config()

    def test_deepepv2_requires_opt_in(self):
        os.environ["MOE_DISPATCHER"] = "deepepv2"
        with self.assertRaisesRegex(ValueError, "requires MEGATRON_VARIANT"):
            self.bench.build_config()

    def test_dev_comparison_has_matching_training_settings(self):
        os.environ.update(MEGATRON_VARIANT="dev-deepepv2", MOE_DISPATCHER="deepepv2")
        self.helper = lambda *args: self.fail("deepepv2 must bypass the unsupported helper")
        cfg = self.bench.build_config()
        self.assertEqual(cfg.model.moe_token_dispatcher_type, "flex")
        self.assertEqual(cfg.model.moe_flex_dispatcher_backend, "deepepv2")
        self.assertFalse(cfg.comm_overlap.overlap_moe_expert_parallel_comm)
        self.assertEqual(cfg.optimizer.lr, 1e-5)
        self.assertEqual(cfg.scheduler.lr_warmup_iters, 0)
        self.assertEqual(cfg.tokenizer.vocab_size, cfg.model.vocab_size)
        self.assertEqual(cfg.model.cross_entropy_fusion_impl, "native")
        self.assertEqual(cfg.model.num_moe_experts, 384)
        self.assertIsNone(cfg.checkpoint.save)
        self.assertIsNone(cfg.checkpoint.load)
        self.assertIsNone(cfg.logger.tensorboard_dir)
        os.environ["MOE_DISPATCHER"] = "alltoall"
        cfg = self.bench.build_config()
        self.assertEqual(cfg.model.moe_token_dispatcher_type, "alltoall")
        self.assertEqual(cfg.optimizer.lr, 1e-5)
        self.assertFalse(cfg.comm_overlap.overlap_moe_expert_parallel_comm)

    def test_deepepv2_rejects_overlap_and_missing_package(self):
        os.environ.update(MEGATRON_VARIANT="dev-deepepv2", MOE_DISPATCHER="deepepv2", MOE_A2A_OVERLAP="on")
        with self.assertRaisesRegex(ValueError, "should_free_input"):
            self.bench.build_config()
        os.environ["MOE_A2A_OVERLAP"] = "off"
        self.modules["megatron.core.transformer.moe"].fused_a2a.HAVE_DEEP_EP_V2 = False
        with self.assertRaisesRegex(RuntimeError, "ElasticBuffer"):
            self.bench.build_config()

    def test_unknown_dispatcher_is_rejected(self):
        os.environ["MOE_DISPATCHER"] = "deepep_v2"
        with self.assertRaisesRegex(ValueError, "MOE_DISPATCHER"):
            self.bench.build_config()


class LauncherTests(unittest.TestCase):
    def render(self, script, arm, **extra):
        env = dict(os.environ, CTX="render-only", IMG="review:local", RENDER_ONLY="1", **extra)
        return subprocess.run(["bash", str(HERE / script), arm, "2"],
                              env=env, check=True, capture_output=True, text=True).stdout

    def test_existing_models_and_modes_render(self):
        for model in ("dsv3", "kimi-k2", "qwen3-235b", "qwen3-30b"):
            for arm in ("alltoall", "deepep"):
                with self.subTest(model=model, arm=arm):
                    text = self.render("run-ab-rawpods.sh", arm, MODEL=model)
                    self.assertEqual(text.count("kind: Pod"), 2)
                    self.assertIn(f"MOE_DISPATCHER={arm}", text)
                    self.assertIn("MOE_A2A_OVERLAP=on", text)
                    self.assertNotIn("mountPath: /dev/gdrdrv", text)
                    self.assertNotIn("NCCL_SYM_GIN_KERNELS_ENABLE=0", text)

    def test_explicit_allocation_and_operator_device_path(self):
        text = self.render("3.run-deepep-v2.sh", "deepepv2",
                           NODE_NAMES="node-a,node-b", NODE_GROUP="assigned-group",
                           CAPACITY_RESERVATION_LABEL="assigned-reservation",
                           GDRCOPY_HOST_PATH="/run/nvidia/driver/dev/gdrdrv")
        self.assertIn("kubernetes.io/hostname: node-a", text)
        self.assertIn("kubernetes.io/hostname: node-b", text)
        self.assertIn("eks.amazonaws.com/nodegroup: assigned-group", text)
        self.assertIn("capacity-reservation: assigned-reservation", text)
        self.assertIn("path: /run/nvidia/driver/dev/gdrdrv", text)
        self.assertIn("--rdzv-conf=is_host=true --local-addr=${POD_IP}", text)
        self.assertIn("--rdzv-conf=is_host=false --local-addr=${POD_IP}", text)
        self.assertIn("fieldPath: status.podIP", text)
        with self.assertRaises(subprocess.CalledProcessError):
            self.render("3.run-deepep-v2.sh", "deepepv2", NODE_NAMES="node-a,node-a")

    def test_rank_zero_preserves_training_failure_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake = root / "torchrun"
            fake.write_text("#!/bin/sh\nexit 19\n")
            fake.chmod(0o755)
            text = self.render("3.run-deepep-v2.sh", "deepepv2", RUN_DIR=str(root / "run"))
            command = textwrap.dedent(re.search(r"- >\n(.*?)\n      resources:", text, re.S).group(1)).replace("\n", " ")
            result = subprocess.run(["bash", "-c", command],
                                    env=dict(os.environ, PATH=f"{root}:{os.environ['PATH']}"),
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 19, result.stderr)
            self.assertIn("exit=19", (root / "run/STATUS").read_text())

    def test_dev_launch_has_devices_and_baked_entrypoint(self):
        for arm in ("alltoall", "deepepv2"):
            text = self.render("3.run-deepep-v2.sh", arm)
            self.assertIn("MOE_A2A_OVERLAP=off", text)
            self.assertIn(f"MOE_DISPATCHER={arm}", text)
            self.assertIn("/opt/benchmark/bench_kimi_k2_pretrain.py", text)
            self.assertIn("mountPath: /dev/gdrdrv", text)
            self.assertIn("${PYTHONPATH:-}", text)
            self.assertIn("NCCL_SYM_GIN_KERNELS_ENABLE=0", text)


if __name__ == "__main__":
    unittest.main()
