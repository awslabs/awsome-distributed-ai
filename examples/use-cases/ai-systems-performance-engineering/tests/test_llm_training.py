"""Tiny CPU Qwen3/optimizer/DCP correctness fixtures, never hardware evidence."""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
try:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError:
    torch = None


class FixtureTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [2 + ord(character) % 29 for character in text]


@unittest.skipIf(torch is None, 'requires pinned torch and transformers')
class TrainingCorrectnessTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(347)

    def test_parallel_pretrained_loading_preserves_all_weights(self):
        import os
        from transformers import AutoModelForCausalLM
        with tempfile.TemporaryDirectory() as directory:
            self.model().save_pretrained(directory, max_shard_size='5KB')
            self.assertGreater(len(list(Path(directory).glob('*.safetensors'))), 1)
            with patch.dict(os.environ, HF_ENABLE_PARALLEL_LOADING='false'):
                serial = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True)
            with patch.dict(os.environ, HF_ENABLE_PARALLEL_LOADING='true', HF_PARALLEL_LOADING_WORKERS='3'):
                parallel = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True)
            before, after = serial.state_dict(), parallel.state_dict()
            self.assertEqual(before.keys(), after.keys())
            for name in before:
                self.assertTrue(torch.equal(before[name], after[name]), name)

    def test_long_document_selection_does_not_manufacture_padding(self):
        from llm_data import write_documents, Documents
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_documents(directory, [{'text': 'abc', 'id': 'short'},
                                                     {'text': 'abcdefghij', 'id': 'long', 'url': 'https://example.org'}],
                                       FixtureTokenizer(), records=1, sequence_length=8,
                                       minimum_document_tokens=8, provenance={})
            data = Documents(directory)
            self.assertEqual(manifest['useful_tokens'], 7)
            self.assertTrue(torch.all(data[0]['attention_mask'] == 1))
            source = json.loads((Path(directory) / 'sources.jsonl').read_text())
            self.assertEqual(source['id'], 'long')
            self.assertEqual(source['original_tokens'], 10)

    def test_skip_fill_requires_deterministic_algorithms(self):
        import os
        import train_llm
        lab = Path(__file__).resolve().parents[1]
        argv = ['trainer', '--config', str(lab / 'configs/llm.json'), '--data', '/unused',
                '--output', '/unused', '--allocation-label', 'argument-validation', '--skip-uninitialized-fill']
        with patch.object(sys, 'argv', argv), patch.dict(os.environ, RANK='0', WORLD_SIZE='1', LOCAL_RANK='0'), \
                patch.object(torch.distributed, 'init_process_group') as initialize:
            with self.assertRaisesRegex(ValueError, 'requires deterministic'):
                train_llm.main()
            initialize.assert_not_called()

    def test_hybrid_shard_rejects_invalid_mesh_before_initialization(self):
        import os
        import train_llm
        config = Path(__file__).resolve().parents[1] / 'configs/llm.json'
        for size in ('-1', '1', '3', '16'):
            argv = ['trainer', '--config', str(config), '--data', '/unused', '--output', '/unused',
                    '--allocation-label', 'validation', '--hybrid-shard-size', size]
            with patch.object(sys, 'argv', argv), patch.dict(os.environ, RANK='0', WORLD_SIZE='16', LOCAL_RANK='0'), \
                    patch.object(torch.distributed, 'init_process_group') as initialize:
                with self.assertRaisesRegex(ValueError, 'HSDP'):
                    train_llm.main()
                initialize.assert_not_called()

    def test_causal_right_padding_preserves_loss_and_gradients(self):
        from train_llm import omit_right_padding_attention_mask
        model = self.model()
        ids = torch.tensor([[2, 3, 4, 1, 0, 0], [5, 6, 1, 0, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]])
        labels = ids.clone()
        labels[mask == 0] = -100
        batch = dict(input_ids=ids, attention_mask=mask, labels=labels,
                     position_ids=torch.arange(6).expand_as(ids))
        before = model(**batch).loss
        before.backward()
        gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        reduced = omit_right_padding_attention_mask(batch)
        self.assertEqual(reduced['input_ids'].shape, ids.shape)
        after = model(**reduced).loss
        after.backward()
        torch.testing.assert_close(before, after, rtol=1e-6, atol=1e-7)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=1e-5, atol=1e-7)
        for key, value in [('attention_mask', torch.tensor([[0, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]])),
                           ('labels', ids), ('position_ids', torch.zeros_like(ids))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                omit_right_padding_attention_mask(dict(batch, **{key: value}))

    def test_llm_environment_does_not_load_legacy_settings(self):
        import os
        import subprocess
        source = Path(__file__).resolve().parents[1] / 'lib/common.sh'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'lib').mkdir()
            (root / 'lib/common.sh').write_text(source.read_text())
            (root / '.env').write_text('COMPUTE_NODES=legacy\n')
            for skip, expected in [('0', 'legacy'), ('1', 'llm')]:
                result = subprocess.run(
                    ['bash', '-c', 'source "$1/lib/common.sh"; printf "%s" "$COMPUTE_NODES"', 'bash', directory],
                    env=dict(os.environ, AIM347_SKIP_LEGACY_ENV=skip, COMPUTE_NODES='llm'),
                    capture_output=True, text=True, check=True)
                self.assertEqual(result.stdout, expected)

    def test_diagnostic_campaign_preserves_controller_events(self):
        self._diagnostic_campaign(planned=False)

    def test_profiled_planned_interruption_preserves_exit_and_traces(self):
        self._diagnostic_campaign(planned=True)

    def _diagnostic_campaign(self, planned):
        import os
        import subprocess
        from llm_data import write_documents
        lab = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_documents(root / 'tokens', [{'text': 'checkpoint fixture'}] * 12,
                            FixtureTokenizer(), records=12, sequence_length=8, provenance={})
            config = json.loads((lab / 'configs/llm-recovery.json').read_text())
            config.update(sequence_length=8, global_batch_samples=2)
            (root / 'config.json').write_text(json.dumps(config))
            command = [sys.executable, str(lab / '13.measure-recovery.py'), '--diagnostic-profile',
                       '--output', str(root / 'campaign'), '--allocated-gpus', '0',
                       '--interrupt-after-update', '5', '--', sys.executable,
                       '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=1',
                       str(lab / 'lib/train_llm.py'), '--cpu-fixture', '--workers', '0',
                       '--config', str(root / 'config.json'), '--data', str(root / 'tokens'),
                       '--checkpoint-mode', 'sync', '--deterministic', '--allocation-label', 'CPU-fixture']
            if planned:
                command.append('--drain-before-planned-interruption')
            subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                           check=True, capture_output=True, text=True, timeout=90)
            result = json.loads((root / 'campaign/campaign.json').read_text())
            self.assertTrue(result['profiled'])
            self.assertFalse(result['performance_eligible'])
            self.assertEqual([event['phase'] for event in result['events']],
                             ['interrupted_start', 'interrupted_exit', 'recovery_selected',
                              'resumed_start', 'resumed_exit'])
            self.assertAlmostEqual(result['finished_utc_seconds'] - result['started_utc_seconds'],
                                   result['elapsed_seconds'], delta=0.1)
            for phase in ('interrupted', 'resumed'):
                for kind in ('setup-trace', 'trace'):
                    events = json.loads((root / f'campaign/{phase}/{kind}-rank-0.json').read_text())['traceEvents']
                    self.assertTrue(events)

    def test_checkpoint_profile_records_save_and_restore(self):
        from llm_checkpoint import Checkpoints
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        progress = {'completed_updates': 1, 'useful_tokens': 3, 'workload': {'fixture': True}}
        with tempfile.TemporaryDirectory() as directory:
            checkpoints = Checkpoints(directory, model, optimizer, scheduler)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                checkpoints.save(progress)
                checkpoints.restore(Path(directory) / 'update-000001', expected_workload=progress['workload'])
            names = {event.key for event in profile.key_averages()}
            self.assertTrue({'checkpoint/sidecar_flush', 'checkpoint/state_dict',
                             'checkpoint/dcp_sync_write', 'checkpoint/restore_state_dict',
                             'checkpoint/read', 'checkpoint/apply'} <= names)

    def test_staging_fence_does_not_publish_pending_upload(self):
        from concurrent.futures import Future
        from unittest.mock import Mock
        from llm_checkpoint import Checkpoints
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Checkpoints(Path(directory) / 'checkpoints', model, optimizer, scheduler)
            staging, upload = Future(), Future()
            staging.set_result({'model': {'weight': torch.ones(2)}})
            checkpoint.stager = Mock()
            checkpoint.staging_future = staging
            checkpoint.snapshot_sources = {'model': model}
            checkpoint.pending = (checkpoint.root / 'update-000001', {'completed_updates': 1}, upload)
            checkpoint.before_source_reuse()
            checkpoint.stager.synchronize_staging.assert_called_once()
            self.assertIsNone(checkpoint.staging_future)
            self.assertIsNone(checkpoint.snapshot_sources)
            self.assertFalse(upload.done())
            self.assertFalse((checkpoint.root / 'update-000001/COMPLETED.json').exists())
            failed = Future()
            failed.set_exception(RuntimeError('copy failed'))
            checkpoint.staging_future = failed
            with self.assertRaisesRegex(RuntimeError, 'copy failed'):
                checkpoint.before_source_reuse()
            self.assertIs(checkpoint.staging_future, failed)
            checkpoint.staging_future = None
            upload.set_result(None)
            stager = checkpoint.stager
            checkpoint.close()
            stager.close.assert_called_once()
            self.assertIsNone(checkpoint.stager)

    def test_cleanup_attempts_all_resources_and_propagates_errors(self):
        from concurrent.futures import Future
        from unittest.mock import Mock
        import llm_checkpoint
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = llm_checkpoint.Checkpoints(directory, model, optimizer, scheduler)
            staging, upload = Future(), Future()
            staging.set_exception(RuntimeError('copy failed'))
            upload.set_exception(RuntimeError('upload failed'))
            stager = Mock()
            stager.close.side_effect = RuntimeError('close failed')
            checkpoint.stager = stager
            checkpoint.staging_future = staging
            checkpoint.pending = (Path(directory), {}, upload)
            checkpoint.snapshot_sources = {'model': model}
            checkpoint.source_storage_refs[1] = torch.ones(1).untyped_storage()
            with self.assertRaisesRegex(RuntimeError, 'copy failed; upload failed; close failed'):
                checkpoint.close()
            stager.close.assert_called_once()
            for name in ('stager', 'staging_future', 'pending', 'snapshot_sources'):
                self.assertIsNone(getattr(checkpoint, name))
            self.assertEqual(checkpoint.source_storage_refs, {})
            self.assertFalse((Path(directory) / 'COMPLETED.json').exists())
        first, second = Mock(), Mock()
        second.close.side_effect = RuntimeError('second failed')
        with patch.object(llm_checkpoint, '_OPEN_CHECKPOINTERS', [first, second]):
            with self.assertRaisesRegex(RuntimeError, 'second failed'):
                llm_checkpoint.close_checkpointers()
            first.close.assert_called_once()
            second.close.assert_called_once()
            self.assertEqual(llm_checkpoint._OPEN_CHECKPOINTERS, [])

    def test_campaign_rejects_failure_at_update_without_intent(self):
        import importlib.util
        from types import SimpleNamespace
        path = Path(__file__).resolve().parents[1] / '13.measure-recovery.py'
        spec = importlib.util.spec_from_file_location('recovery_controller', path)
        assert spec is not None and spec.loader is not None
        controller = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(controller)
        for intent in (False, True):
            with self.subTest(intent=intent), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'campaign'

                def interrupted(command, **kwargs):
                    root = Path(command[command.index('--output') + 1])
                    root.mkdir()
                    row = {'update': 5}
                    if intent:
                        row['controlled_interruption'] = True
                    (root / 'rank-0-updates.jsonl').write_text(json.dumps(row) + '\n')
                    return SimpleNamespace(returncode=1)

                argv = ['controller', '--output', str(output), '--allocated-gpus', '1',
                        '--interrupt-after-update', '5', '--', 'fixture']
                expected = 'no completed checkpoint' if intent else 'controlled interruption'
                with patch.object(sys, 'argv', argv), patch.object(controller.subprocess, 'run', side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError, expected):
                        controller.main()
                self.assertFalse((output / 'campaign.json').exists())

    def test_campaign_rejects_missing_rank_and_rank_intent(self):
        import importlib.util
        from types import SimpleNamespace
        path = Path(__file__).resolve().parents[1] / '13.measure-recovery.py'
        spec = importlib.util.spec_from_file_location('recovery_controller', path)
        assert spec is not None and spec.loader is not None
        controller = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(controller)
        for fault in ('missing_rank', 'missing_intent'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'campaign'

                def interrupted(command, **kwargs):
                    root = Path(command[command.index('--output') + 1])
                    checkpoint = root / 'checkpoints/update-000004'
                    checkpoint.mkdir(parents=True)
                    (checkpoint / 'COMPLETED.json').write_text(json.dumps({
                        'completed_updates': 4, 'workload': {'data_parallel_size': 2}}))
                    (checkpoint / '.metadata').touch()
                    for rank in range(1 if fault == 'missing_rank' else 2):
                        row = {'update': 5, 'controlled_interruption': rank == 0}
                        (root / f'rank-{rank}-updates.jsonl').write_text(json.dumps(row) + '\n')
                    return SimpleNamespace(returncode=1)

                argv = ['controller', '--output', str(output), '--allocated-gpus', '2',
                        '--interrupt-after-update', '5', '--', 'fixture']
                expected = 'every rank' if fault == 'missing_rank' else 'controlled interruption'
                with patch.object(sys, 'argv', argv), patch.object(controller.subprocess, 'run', side_effect=interrupted) as run:
                    with self.assertRaisesRegex(RuntimeError, expected):
                        controller.main()
                    self.assertEqual(run.call_count, 1)
                self.assertFalse((output / 'campaign.json').exists())

    def test_grouped_fsdp_rejects_unsupported_combinations(self):
        import os
        import train_llm
        config = Path(__file__).resolve().parents[1] / 'configs/llm.json'
        for flags in (['--fsdp-blocks-per-group', '0'],
                      ['--fsdp-blocks-per-group', '2', '--cpu-fixture'],
                      ['--fsdp-blocks-per-group', '2', '--compile-loss'],
                      ['--fsdp-blocks-per-group', '2', '--compile-pointwise'],
                      ['--fsdp-blocks-per-group', '2', '--compile-blocks'],
                      ['--fsdp-blocks-per-group', '2', '--forward-prefetch']):
            argv = ['trainer', '--config', str(config), '--data', '/unused',
                    '--output', '/unused', '--allocation-label', 'validation', *flags]
            with self.subTest(flags=flags), patch.object(sys, 'argv', argv), patch.dict(
                    os.environ, RANK='0', WORLD_SIZE='1', LOCAL_RANK='0'):
                with self.assertRaisesRegex(ValueError, 'FSDP'):
                    train_llm.main()

    def model(self):
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=8,
                             tie_word_embeddings=True, attention_dropout=0.0)
        config._attn_implementation = 'sdpa'
        return Qwen3ForCausalLM(config)

    def dataset(self, root):
        from llm_data import Documents, write_documents
        write_documents(root, [{'text': value} for value in ['ab', 'abcdefg', 'x', '12345']],
                         FixtureTokenizer(), records=4, sequence_length=8,
                         provenance={'kind': 'synthetic arithmetic fixture'})
        return Documents(root)

    def test_mask_accounting_and_update_membership(self):
        from llm_data import UpdateBatches
        with tempfile.TemporaryDirectory() as root:
            data = self.dataset(root)
            self.assertEqual(data.update_tokens(0, 4), 15)
            self.assertEqual(sum(int((data[i]['labels'][1:] != -100).sum()) for i in range(4)), 15)
            self.assertEqual(data[0]['labels'].tolist(), [12, 13, 1, -100, -100, -100, -100, -100])
            for rank in range(2):
                members = []
                for microbatch in (1, 2):
                    batches = UpdateBatches(start_update=0, stop_update=1, global_batch=4,
                                            microbatch=microbatch, rank=rank, dp_size=2)
                    members.append([i for batch in batches for i in batch])
                self.assertEqual(*members)

    def test_microbatch_loss_gradient_and_parameter_update_equivalence(self):
        with tempfile.TemporaryDirectory() as root:
            data = self.dataset(root)
            initial = self.model()
            results = []
            for microbatch in (1, 2, 4):
                model = copy.deepcopy(initial)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                losses = []
                for start in range(0, 4, microbatch):
                    batch = {key: torch.stack([data[i][key] for i in range(start, start + microbatch)])
                             for key in data[0]}
                    loss = model(**batch, num_items_in_batch=15).loss
                    loss.backward()
                    losses.append(loss.item())
                gradients = [p.grad.clone() for p in model.parameters()]
                optimizer.step()
                results.append((sum(losses), gradients, [p.detach().clone() for p in model.parameters()]))
            for loss, gradients, parameters in results[1:]:
                self.assertAlmostEqual(loss, results[0][0], places=6)
                for actual, expected in zip(gradients, results[0][1]):
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
                for actual, expected in zip(parameters, results[0][2]):
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)

    def test_trim_padding_preserves_loss_gradients_and_update(self):
        from llm_data import trim_padding
        with tempfile.TemporaryDirectory() as root:
            data = self.dataset(root)
            initial = self.model()
            results = []
            for trim in (False, True):
                model = copy.deepcopy(initial)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                total_loss = 0.0
                tokens = 0
                for i in range(len(data)):
                    batch = {key: value.unsqueeze(0) for key, value in data[i].items()}
                    if trim:
                        batch = trim_padding(batch)
                    tokens += int((batch['labels'][:, 1:] != -100).sum())
                    loss = model(**batch, num_items_in_batch=15).loss
                    loss.backward()
                    total_loss += loss.item()
                gradients = [p.grad.clone() for p in model.parameters()]
                optimizer.step()
                results.append((total_loss, tokens, gradients,
                                [p.detach().clone() for p in model.parameters()]))
            self.assertEqual(results[0][1], results[1][1])
            self.assertAlmostEqual(results[0][0], results[1][0], places=6)
            for collection in (2, 3):
                for actual, expected in zip(results[1][collection], results[0][collection]):
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)

    def test_full_tensor_difference_detects_gradient_scale_error(self):
        from llm_numerics import tensor_differences
        reference = {'weight': torch.tensor([1., -2., 3.])}
        identical = tensor_differences(reference, reference)
        self.assertEqual(identical['error_squared'], 0)
        scaled = tensor_differences(reference, {'weight': 2 * reference['weight']})
        self.assertEqual(scaled['error_squared'], scaled['reference_squared'])
        with self.assertRaises(ValueError):
            tensor_differences(reference, {'weight': torch.tensor([float('nan'), 0., 0.])})

    def test_numerical_report_rejects_incomplete_matching_evidence(self):
        from llm_numerics import numerical_report
        policy = json.loads((Path(__file__).resolve().parents[1] / 'configs/llm-numerics-policy.json').read_text())
        with tempfile.TemporaryDirectory() as root:
            paths = [Path(root) / name for name in ('before', 'after')]
            summary = dict(workload={'fixture': True, 'updates': 2}, allocation={'dp_size': 2},
                           completed_updates=2, initial_progress_tokens=0,
                           configuration={'deterministic': True}, status='completed',
                           numerical_verification=True, profiled=False)
            state = {key: {'weight': torch.tensor([1., 2.]), 'bias': torch.tensor([3.])}
                     for key in ('parameters', 'gradients', 'parameter_deltas')}
            updates = [dict(update=i, global_mean_loss=1., global_useful_tokens=4,
                            local_useful_tokens=2) for i in (1, 2)]

            def write_evidence():
                for path in paths:
                    path.mkdir(exist_ok=True)
                    (path / 'summary.json').write_text(json.dumps(summary))
                    for rank in range(2):
                        torch.save(state, path / f'numerics-rank-{rank}.pt')
                        (path / f'rank-{rank}-updates.jsonl').write_text(
                            ''.join(json.dumps(row) + '\n' for row in updates))

            write_evidence()
            self.assertTrue(numerical_report(*paths, policy)['passed'])

            summary['allocation']['dp_size'] = 0
            write_evidence()
            with self.assertRaisesRegex(ValueError, 'positive rank count'):
                numerical_report(*paths, policy)
            summary['allocation']['dp_size'] = 2
            del state['gradients']['bias']
            write_evidence()
            with self.assertRaisesRegex(ValueError, 'every trainable parameter'):
                numerical_report(*paths, policy)
            state['gradients']['bias'] = torch.tensor([3.])
            updates.pop()
            write_evidence()
            with self.assertRaisesRegex(ValueError, 'every expected update'):
                numerical_report(*paths, policy)
            summary['completed_updates'] = 1
            write_evidence()
            with self.assertRaisesRegex(ValueError, 'declared workload'):
                numerical_report(*paths, policy)
            summary['completed_updates'] = 2
            updates.append(dict(updates[0], update=2))
            write_evidence()
            changed = [dict(row, global_mean_loss=2.) for row in updates]
            for path in paths:
                (path / 'rank-1-updates.jsonl').write_text(
                    ''.join(json.dumps(row) + '\n' for row in changed))
            with self.assertRaisesRegex(ValueError, 'across ranks'):
                numerical_report(*paths, policy)

    def test_aot_backend_requires_active_correctness_only_compile(self):
        import train_llm
        with tempfile.TemporaryDirectory() as root:
            config = Path(root) / 'config.json'
            config.write_text(json.dumps({'warmup_updates': 0, 'updates': 2}))
            command = ['train_llm.py', '--config', str(config), '--data', root,
                       '--output', root, '--allocation-label', 'CPU argument validation',
                       '--compile-backend', 'aot_eager']
            for flags in (['--compile-loss'], ['--verify-numerics'],
                          ['--verify-numerics', '--compile-loss', '--compile-preserve-casts'],
                          ['--verify-numerics', '--compile-loss', '--cpu-fixture']):
                with self.subTest(flags=flags), patch.object(sys, 'argv', command + flags), \
                        patch.dict('os.environ', RANK='0', WORLD_SIZE='1', LOCAL_RANK='0'), \
                        patch.object(torch.distributed, 'init_process_group') as initialize:
                    with self.assertRaisesRegex(ValueError, 'aot_eager requires'):
                        train_llm.main()
                    initialize.assert_not_called()

    def test_pointwise_scope_rejects_missing_or_overriding_compilation(self):
        import train_llm
        with tempfile.TemporaryDirectory() as root:
            config = Path(root) / 'config.json'
            config.write_text(json.dumps({'warmup_updates': 0, 'updates': 2}))
            command = ['train_llm.py', '--config', str(config), '--data', root,
                       '--output', root, '--allocation-label', 'CPU argument validation',
                       '--compile-pointwise-scope']
            cases = [(scope, flags) for scope in ('norms', 'activations')
                     for flags in ([], ['--compile-pointwise', '--compile-blocks'])]
            for scope, flags in cases:
                with self.subTest(scope=scope, flags=flags), patch.object(sys, 'argv', command + [scope] + flags), \
                        patch.dict('os.environ', RANK='0', WORLD_SIZE='1', LOCAL_RANK='0'), \
                        patch.object(torch.distributed, 'init_process_group') as initialize:
                    with self.assertRaisesRegex(ValueError, 'pointwise scope requires'):
                        train_llm.main()
                    initialize.assert_not_called()

    def test_pointwise_compile_dispatch_excludes_projections_and_blocks(self):
        from train_llm import compile_pointwise_modules
        model = self.model()
        norms = [model.model.norm]
        activations = []
        for block in model.model.layers:
            norms.extend([block.input_layernorm, block.post_attention_layernorm,
                          block.self_attn.q_norm, block.self_attn.k_norm])
            activations.append(block.mlp.act_fn)
        for scope, expected in [('norms', norms), ('activations', activations),
                                ('all', norms + activations)]:
            with self.subTest(scope=scope), patch.object(torch.nn.Module, 'compile', autospec=True) as compile_call:
                options = {'emulate_precision_casts': True}
                compile_pointwise_modules(model, scope, 'inductor', options)
                self.assertCountEqual([call.args[0] for call in compile_call.call_args_list], expected)
                for call in compile_call.call_args_list:
                    self.assertEqual(call.kwargs, dict(dynamic=True, backend='inductor', options=options))

    def test_microbatch_retention_rejects_inapplicable_configuration(self):
        import train_llm
        with tempfile.TemporaryDirectory() as root:
            config = Path(root) / 'config.json'
            config.write_text(json.dumps(dict(warmup_updates=0, updates=2,
                                              checkpoint_updates=[2], global_batch_samples=2)))
            command = ['train_llm.py', '--config', str(config), '--data', root,
                       '--output', root, '--allocation-label', 'CPU argument validation',
                       '--retain-between-microbatches']
            for flags in ([], ['--no-reshard-after-forward', '--cpu-fixture'],
                          ['--no-reshard-after-forward', '--microbatch', '2']):
                with self.subTest(flags=flags), patch.object(sys, 'argv', command + flags), \
                        patch.dict('os.environ', RANK='0', WORLD_SIZE='1', LOCAL_RANK='0'), \
                        patch.object(torch.distributed, 'init_process_group') as initialize:
                    with self.assertRaisesRegex(ValueError, 'microbatch retention requires'):
                        train_llm.main()
                    initialize.assert_not_called()

    def test_config_only_fresh_restore_preserves_next_update(self):
        from transformers import AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights
        from llm_checkpoint import Checkpoints
        with tempfile.TemporaryDirectory() as directory:
            model = self.model()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

            def update(model, optimizer, scheduler):
                optimizer.zero_grad(set_to_none=True)
                ids = torch.tensor([[2, 3, 4, 1]])
                model(input_ids=ids, labels=ids).loss.backward()
                optimizer.step()
                scheduler.step()

            update(model, optimizer, scheduler)
            saves = Checkpoints(Path(directory) / 'checkpoints', model, optimizer, scheduler)
            progress = dict(completed_updates=1, useful_tokens=3, workload={'fixture': True})
            saves.save(progress)
            expected_random = (random.random(), np.random.rand(), torch.rand(1))
            update(model, optimizer, scheduler)
            expected = copy.deepcopy(model.state_dict())
            with no_init_weights():
                fresh = AutoModelForCausalLM.from_config(model.config, torch_dtype=torch.float32)
            fresh.tie_weights()
            original_buffers = dict(model.named_buffers())
            reconstructed_buffers = dict(fresh.named_buffers())
            self.assertEqual(original_buffers.keys(), reconstructed_buffers.keys())
            for name in original_buffers:
                torch.testing.assert_close(original_buffers[name], reconstructed_buffers[name], rtol=0, atol=0)
            fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=1e-4)
            fresh_scheduler = torch.optim.lr_scheduler.LambdaLR(fresh_optimizer, lambda _: 1.0)
            restore = Checkpoints(Path(directory) / 'resumed', fresh, fresh_optimizer, fresh_scheduler)
            restore.restore(Path(directory) / 'checkpoints/update-000001', expected_workload=progress['workload'])
            self.assertEqual(random.random(), expected_random[0])
            self.assertEqual(np.random.rand(), expected_random[1])
            torch.testing.assert_close(torch.rand(1), expected_random[2], rtol=0, atol=0)
            update(fresh, fresh_optimizer, fresh_scheduler)
            for name, value in fresh.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    def test_sync_and_async_restore_same_next_update_and_rng(self):
        from llm_checkpoint import Checkpoints, latest_completed
        cases = [(asynchronous, threads, ahead) for asynchronous, threads in
                 ((False, 1), (True, 1), (False, 4), (True, 4))
                 for ahead in (10_000_000, 64 * 1024 * 1024)]
        for asynchronous, writer_threads, copy_ahead_bytes in cases:
            with self.subTest(asynchronous=asynchronous, writer_threads=writer_threads,
                              copy_ahead_bytes=copy_ahead_bytes), tempfile.TemporaryDirectory() as root:
                model = self.model()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

                def update():
                    optimizer.zero_grad()
                    ids = torch.tensor([[2, 3, 4, 1]])
                    loss = model(input_ids=ids, labels=ids).loss
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                update()
                checkpoints = Checkpoints(root, model, optimizer, scheduler, asynchronous=asynchronous,
                                          writer_threads=writer_threads, copy_ahead_bytes=copy_ahead_bytes)
                progress = dict(completed_updates=1, useful_tokens=3, next_sample=1,
                                workload={'kind': 'tiny CPU Qwen3'})
                import llm_checkpoint
                with patch.object(llm_checkpoint.dcp, 'FileSystemWriter',
                                  wraps=llm_checkpoint.dcp.FileSystemWriter) as writer:
                    checkpoints.save(progress)
                    self.assertEqual(writer.call_args.kwargs['per_thread_copy_ahead'], copy_ahead_bytes)
                    self.assertEqual(writer.call_args.kwargs['thread_count'], writer_threads)
                    self.assertTrue(writer.call_args.kwargs['sync_files'])
                if asynchronous:
                    self.assertIsNone(latest_completed(root))
                    self.assertIsNone(checkpoints.durable_progress)
                # Mutate the live model while the staged snapshot is being saved.
                update()
                expected = copy.deepcopy(model.state_dict())
                expected_lr = scheduler.get_last_lr()
                checkpoints.finish()
                path = latest_completed(root)
                self.assertIsNotNone(path)
                restored = checkpoints.restore(path, expected_workload=progress['workload'])
                expected_random = (random.random(), np.random.rand(), torch.rand(1))
                checkpoints.restore(path, expected_workload=progress['workload'])
                self.assertEqual(random.random(), expected_random[0])
                self.assertEqual(np.random.rand(), expected_random[1])
                torch.testing.assert_close(torch.rand(1), expected_random[2])
                self.assertEqual(restored, progress)
                update()
                for name, parameter in model.state_dict().items():
                    torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
                self.assertEqual(scheduler.get_last_lr(), expected_lr)
                pending = Path(root) / 'update-000999'
                pending.mkdir()
                (pending / '.metadata').write_text('incomplete fixture')
                self.assertEqual(latest_completed(root), path)
                with self.assertRaises(ValueError):
                    checkpoints.restore(path, expected_workload={'different': 'work'})

    def test_archived_campaign_is_not_a_local_restore_candidate(self):
        from llm_checkpoint import Checkpoints, latest_completed
        for index_location in ('campaign', 'segment'):
            with self.subTest(index_location=index_location), tempfile.TemporaryDirectory() as directory:
                campaign = Path(directory) / 'campaign'
                segment = campaign / 'interrupted'
                model = self.model()
                optimizer = torch.optim.AdamW(model.parameters())
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
                checkpoints = Checkpoints(segment / 'checkpoints', model, optimizer, scheduler)
                path = checkpoints.root / 'update-000004'
                path.mkdir()
                (path / '.metadata').write_text('retained metadata fixture')
                (path / 'COMPLETED.json').write_text(json.dumps({'completed_updates': 4}))
                self.assertEqual(latest_completed(checkpoints.root), path)
                index_parent = campaign if index_location == 'campaign' else segment
                (index_parent / 'S3_ARCHIVE.json').write_text(json.dumps({'prefix': 's3://fixture/verified/'}))
                self.assertIsNone(latest_completed(checkpoints.root))
                with patch('llm_checkpoint.dcp.load') as load:
                    with self.assertRaisesRegex(ValueError, 'checkpoint tensors were archived'):
                        checkpoints.restore(path, expected_workload={})
                    load.assert_not_called()

    def test_failed_checkpoint_never_publishes_completion(self):
        from concurrent.futures import Future
        from llm_checkpoint import Checkpoints, latest_completed
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous), tempfile.TemporaryDirectory() as root:
                model = self.model()
                optimizer = torch.optim.AdamW(model.parameters())
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
                checkpoints = Checkpoints(root, model, optimizer, scheduler, asynchronous=asynchronous)
                progress = dict(completed_updates=1, useful_tokens=3, next_sample=1, workload={'fixture': True})
                if asynchronous:
                    future = Future()
                    future.set_exception(OSError('injected writer failure'))
                    with patch('llm_checkpoint.dcp.async_save', return_value=future):
                        checkpoints.save(progress)
                    with self.assertRaises(OSError):
                        checkpoints.finish()
                else:
                    with patch('llm_checkpoint.dcp.save', side_effect=OSError('injected writer failure')):
                        with self.assertRaises(OSError):
                            checkpoints.save(progress)
                self.assertIsNone(checkpoints.durable_progress)
                self.assertIsNone(latest_completed(root))


if __name__ == '__main__':
    unittest.main()
