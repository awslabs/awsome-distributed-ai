# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Final findings 4/5: actual recovery decisions on the existing recording bench.

Strings are source-derived test inputs, NOT a real supported memlock capture.
No scheduler, cloud, or hardware operation is executed by this bench.
"""
import test_device_session_rework as bench
import test_safety_rework3 as warnings

session = bench.session
PIN = 'a4ba07eb15e6f277063b4000346f9109c98de843'
SKIPPED = (
    '[DEBUG] rdma statistic show failed -- EFA statistics skipped\n',
    '[DEBUG] rdma tool not found -- EFA statistics skipped\n',
)


class QualificationBench(bench.Base):
    def start_with_baseline(self, baseline):
        self.executor.check_output['6'] = (
            warnings.WARN6_RETRANS_2 if baseline == 'WARN'
            else warnings.PASS6_STATISTICS_CLEAN)
        self.make().start('efa')
        state = self.make().store.read()
        key = 'round-1/baseline/check-6'
        if baseline == 'absent':
            del state.baseline_check_results[key]
        elif baseline == 'NOT-COLLECTED':
            state.baseline_check_results[key] = {'verdict': 'NOT-COLLECTED'}
        self.make().store.write(state)

    def attempt(self, deadline=False):
        try:
            handler = self.make(now=9999.0 if deadline else 1000.0)
            result = handler.expire() if deadline else handler.recover()
            refusal = None
        except session.Refusal as error:
            result, refusal = None, error
        print('scheduler:', self.executor.slurm_calls)
        print('maintenance:', self.executor.maintenance_calls)
        print('phase:', self.make().store.read().phase, 'refusal:', refusal)
        return result, refusal

    def assert_isolated(self, result, refusal, phase):
        self.assertEqual([], self.executor.resumes(), 'unsafe RESUME was issued')
        self.assertIsNone(result)
        self.assertIsNotNone(refusal)
        state = self.make().store.read()
        self.assertEqual(phase, state.phase)
        self.assertIsNone(state.recovered_at)
        self.assertIsNone(state.recovered_by)
        self.assertNotIn('facilitator', str(refusal).lower())
        self.assertIn('drain', state.prepared)
        self.assertTrue(state.check_results)


class IncompletePass(QualificationBench):
    def skipped_against(self, baseline):
        for skipped in SKIPPED:
            with self.subTest(baseline=baseline, skipped=skipped):
                self.make().store.path.unlink(missing_ok=True)
                self.executor = bench.ReworkExecutor()
                self.start_with_baseline(baseline)
                self.executor.check_output['6'] = bench.PASS_6 + skipped
                result, refusal = self.attempt()
                self.assert_isolated(result, refusal, 'recovery-failed')
                self.assertIn('statistics collection was skipped', str(refusal))

    def test_skipped_against_pass(self):
        self.skipped_against('PASS')

    def test_skipped_against_absent(self):
        self.skipped_against('absent')

    def test_skipped_against_not_collected(self):
        self.skipped_against('NOT-COLLECTED')

    def test_skipped_against_warn(self):
        self.skipped_against('WARN')

    def test_skipped_cannot_be_overridden_by_clean_sentence(self):
        self.start_with_baseline('WARN')
        self.executor.check_output['6'] = warnings.PASS6_STATISTICS_CLEAN + SKIPPED[0]
        result, refusal = self.attempt()
        self.assert_isolated(result, refusal, 'recovery-failed')

    def test_skipped_debug_on_stderr_is_refused(self):
        self.start_with_baseline('PASS')
        original = self.executor.run_maintenance

        def stderr_capture(argv, timeout=None):
            result = original(argv, timeout=timeout)
            if argv == ['collect', '6']:
                return session.Completed(0, bench.PASS_6, SKIPPED[0])
            return result

        self.executor.run_maintenance = stderr_capture
        result, refusal = self.attempt()
        self.assert_isolated(result, refusal, 'recovery-failed')

    def test_deadline_skipped_pass_does_not_resume(self):
        self.start_with_baseline('PASS')
        self.executor.check_output['6'] = bench.PASS_6 + SKIPPED[1]
        result, refusal = self.attempt(deadline=True)
        self.assert_isolated(result, refusal, 'recovery-failed')

    def test_complete_clean_pass_all_baselines(self):
        for baseline in ('PASS', 'absent', 'NOT-COLLECTED', 'WARN'):
            with self.subTest(baseline=baseline):
                # Each baseline gets a separate real round/state on the same bench.
                self.make().store.path.unlink(missing_ok=True)
                self.executor = bench.ReworkExecutor()
                self.start_with_baseline(baseline)
                self.executor.check_output['6'] = warnings.PASS6_STATISTICS_CLEAN
                result, refusal = self.attempt()
                self.assertIsNone(refusal)
                assert result is not None
                self.assertEqual('runtime-ready', result.state.phase)
                self.assertEqual(1, len(self.executor.resumes()))


class UnsupportedWarn(QualificationBench):
    def warn_round(self, text):
        self.executor.check_output['2'] = text
        self.executor.check_output['6'] = warnings.PASS6_STATISTICS_CLEAN
        self.make().start('efa')
        result, refusal = self.attempt()
        self.assert_isolated(result, refusal, 'replacement-required')
        self.assertIn('Replacement is required', str(refusal))
        self.assertEqual([], self.executor.reboots())

    def test_matching_name_without_revision_does_not_qualify(self):
        self.warn_round(warnings.WARN2_MEMLOCK_COMPLETE)

    def test_missing_embedded_names_do_not_qualify(self):
        self.warn_round(warnings.WARN2_MEMLOCK_COMPLETE.replace('2-efa-enumeration: ', ''))

    def test_matching_printed_pin_is_not_supported_capture(self):
        self.warn_round('# suite_revision=' + PIN + '\n' + warnings.WARN2_MEMLOCK_COMPLETE)

    def test_wrong_printed_pin_does_not_qualify(self):
        self.warn_round('# suite_revision=' + '0' * 40 + '\n' + warnings.WARN2_MEMLOCK_COMPLETE)

    def test_wrong_embedded_name_stays_refused(self):
        self.warn_round(warnings.WARN2_MEMLOCK_COMPLETE.replace('2-efa-enumeration', '6-efa-loopback'))

    def test_replacement_required_holds_pair_and_evidence_on_retry(self):
        self.warn_round(warnings.WARN2_MEMLOCK_COMPLETE)
        state = self.make().store.read()
        result, refusal = self.attempt()
        self.assert_isolated(result, refusal, 'replacement-required')
        self.assertEqual(state.operation, self.make().store.read().operation)
        self.assertEqual(state.baseline_check_results, self.make().store.read().baseline_check_results)
        with self.assertRaises(session.Refusal):
            self.make(assignment='table-2').start('efa')

    def test_deadline_memlock_warn_requires_replacement(self):
        self.executor.check_output['2'] = warnings.WARN2_MEMLOCK_COMPLETE
        self.make().start('efa')
        result, refusal = self.attempt(deadline=True)
        self.assert_isolated(result, refusal, 'replacement-required')

    def test_clean_pass_after_memlock_baseline_still_resumes(self):
        self.executor.check_output['2'] = warnings.WARN2_MEMLOCK_COMPLETE
        self.executor.check_output['6'] = warnings.PASS6_STATISTICS_CLEAN
        self.make().start('efa')
        self.executor.check_output['2'] = bench.PASS_2
        result, refusal = self.attempt()
        self.assertIsNone(refusal)
        assert result is not None
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertEqual(1, len(self.executor.resumes()))
