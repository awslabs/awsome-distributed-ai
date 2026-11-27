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

    def test_restore_gives_the_same_next_update_and_rng(self):
        from llm_checkpoint import Checkpoints, latest_completed
        with tempfile.TemporaryDirectory() as root:
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
            checkpoints = Checkpoints(root, model, optimizer, scheduler)
            progress = dict(completed_updates=1, useful_tokens=3, next_sample=1,
                            workload={'kind': 'tiny CPU Qwen3'})
            checkpoints.save(progress)
            path = latest_completed(root)
            self.assertEqual(path, Path(root) / 'update-000001')
            # The update after the save, on the live model, is what a restored model must repeat.
            update()
            expected = copy.deepcopy(model.state_dict())
            expected_lr = scheduler.get_last_lr()
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
            with self.assertRaises(ValueError):
                checkpoints.restore(path, expected_workload={'different': 'work'})

    def test_failed_write_never_marks_the_checkpoint_complete(self):
        from llm_checkpoint import Checkpoints, latest_completed
        with tempfile.TemporaryDirectory() as root:
            model = self.model()
            optimizer = torch.optim.AdamW(model.parameters())
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
            checkpoints = Checkpoints(root, model, optimizer, scheduler)
            progress = dict(completed_updates=1, useful_tokens=3, next_sample=1, workload={'fixture': True})
            with patch('llm_checkpoint.dcp.save', side_effect=OSError('injected writer failure')):
                with self.assertRaises(OSError):
                    checkpoints.save(progress)
            self.assertIsNone(checkpoints.durable_progress)
            self.assertIsNone(latest_completed(root))

    def _fixture_job(self, root, *, updates, warmup, checkpoint_updates):
        """Workshop config shrunk to the CPU fixture: 2 sequences of 8 positions."""
        from llm_data import write_documents
        lab = Path(__file__).resolve().parents[1]
        config = json.loads((lab / 'configs/smol-2048-40.json').read_text())
        config.update(sequence_length=8, global_batch_samples=2, updates=updates,
                      warmup_updates=warmup, checkpoint_updates=checkpoint_updates)
        (root / 'config.json').write_text(json.dumps(config))
        write_documents(root / 'tokens', [{'text': 'checkpoint fixture'}] * (2 * updates),
                        FixtureTokenizer(), records=2 * updates, sequence_length=8, provenance={})
        return root / 'config.json', root / 'tokens'

    def _train(self, config, data, output, *extra):
        """Run one CPU-fixture rank directly, so its own exit status is visible."""
        import os
        import socket
        import subprocess
        lab = Path(__file__).resolve().parents[1]
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        env = dict(os.environ, RANK='0', WORLD_SIZE='1', LOCAL_RANK='0', LOCAL_WORLD_SIZE='1',
                   MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), CUDA_VISIBLE_DEVICES='')
        command = [sys.executable, str(lab / 'lib/train_llm.py'), '--cpu-fixture', '--workers', '0',
                   '--config', str(config), '--data', str(data),
                   '--output', str(output), '--allocation-label', 'CPU-fixture', *extra]
        return subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)

    def test_profile_marks_attention_core_and_every_update(self):
        for attention in ('eager', 'sdpa'):
            with self.subTest(attention=attention), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config, data = self._fixture_job(root, updates=5, warmup=1, checkpoint_updates=[5])
                result = self._train(config, data, root / 'train', '--profile',
                                     '--attention-implementation', attention)
                self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                events = json.loads((root / 'train/trace-rank-0.json').read_text())['traceEvents']
                ranges = [e for e in events if e.get('name') == 'attention/core' and e.get('ph') == 'X']
                # 5 updates x 2 microbatches x 2 decoder layers.
                self.assertEqual(len(ranges), 20)
                steps = {e['name'] for e in events if e.get('name', '').startswith('ProfilerStep#')}
                self.assertTrue({f'ProfilerStep#{n}' for n in range(5)} <= steps)
                inside = set()
                for op in events:
                    number = op.get('args', {}).get('Sequence number')
                    if op.get('cat') != 'cpu_op' or number is None or number < 0:
                        continue
                    if any(r['tid'] == op['tid'] and r['ts'] <= op['ts'] <= r['ts'] + r['dur'] for r in ranges):
                        inside.add(number)
                backward = [e for e in events if e.get('cat') == 'cpu_op'
                            and e['name'].startswith('autograd::engine::evaluate_function:')
                            and e.get('args', {}).get('Sequence number') in inside]
                self.assertTrue(inside)
                self.assertTrue(backward)
                # tools/profile-table.py on the same trace: update u is ProfilerStep#<u-1>,
                # and its window holds that update's input wait and attention operators.
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    'profile_table', Path(__file__).resolve().parents[1] / 'tools/profile-table.py')
                table_tool = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(table_tool)
                rows = [json.loads(line) for line in (root / 'train/rank-0-updates.jsonl').read_text().splitlines()]
                complete = table_tool.complete(events)
                for update in range(1, 6):
                    span = next(e for e in complete if e['name'] == f'ProfilerStep#{update - 1}')
                    table = table_tool.update_table(events, update)
                    self.assertAlmostEqual(table['update'], span['dur'] / 1e6)
                    self.assertGreater(table['input_wait'], 0)
                    main = (span['pid'], span['tid'])
                    self.assertTrue(table_tool.attention_ids(complete, main, span['ts'], span['ts'] + span['dur']))
                    if update > 1:
                        # The span ends after the row is stamped, close to the row interval.
                        gap = rows[update - 1]['monotonic_seconds'] - rows[update - 2]['monotonic_seconds']
                        self.assertLess(abs(table['update'] - gap), 0.5 * gap + 0.05)

    def test_restore_from_saved_update_runs_next_updates_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, data = self._fixture_job(root, updates=8, warmup=2, checkpoint_updates=[2, 6, 8])
            saved = self._train(config, data, root / 'saved', '--checkpoint-mode', 'sync')
            self.assertEqual(saved.returncode, 0, saved.stderr[-2000:])
            restored = self._train(config, data, root / 'restore', '--checkpoint-mode', 'sync',
                                   '--resume', str(root / 'saved/checkpoints/update-000002'),
                                   '--interrupt-after-update', '4')
            self.assertEqual(restored.returncode, 75, restored.stderr[-2000:])
            self.assertFalse((root / 'restore/summary.json').exists())
            rows = [json.loads(line) for line in (root / 'restore/rank-0-updates.jsonl').read_text().splitlines()]
            self.assertEqual([row['update'] for row in rows], [3, 4])
            self.assertTrue(rows[-1]['controlled_interruption'])
            original = {json.loads(line)['update']: json.loads(line)['global_mean_loss']
                        for line in (root / 'saved/rank-0-updates.jsonl').read_text().splitlines()}
            for row in rows:
                self.assertEqual(row['global_mean_loss'], original[row['update']])

    def test_pending_upload_is_published_only_after_it_completes(self):
        from concurrent.futures import Future
        from llm_checkpoint import Checkpoints
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoints = Checkpoints(Path(directory) / 'checkpoints', model, optimizer, scheduler)
            path = checkpoints.root / 'update-000020'
            path.mkdir()
            progress = {'completed_updates': 20, 'useful_tokens': 1310080, 'workload': {'fixture': True}}
            upload = Future()
            checkpoints.pending = (path, dict(progress), upload)
            checkpoints.poll()
            self.assertFalse((path / 'COMPLETED.json').exists())
            self.assertIsNone(checkpoints.durable_progress)
            upload.set_result(None)
            checkpoints.poll()
            self.assertEqual(json.loads((path / 'COMPLETED.json').read_text()), progress)
            self.assertEqual(checkpoints.durable_progress, progress)
            self.assertIsNone(checkpoints.pending)

    def test_latest_completed_skips_a_save_without_completion_marker(self):
        from llm_checkpoint import latest_completed
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'checkpoints'
            for update, completed in ((5, True), (20, False)):
                path = root / f'update-{update:06d}'
                path.mkdir(parents=True)
                (path / '.metadata').write_text('fixture metadata')
                if completed:
                    (path / 'COMPLETED.json').write_text(json.dumps({'completed_updates': update}))
            self.assertEqual(latest_completed(root), root / 'update-000005')


if __name__ == '__main__':
    unittest.main()
