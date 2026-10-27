"""report.py renders the answer, verdict table, maps and glossary from a fixture."""
import re
import unittest
import report


def make_grid(ttft_axis, tpot_axis, by_arch):
    return {(ttft, tpot): {"winner": "unified", "by_arch": by_arch, "top": None, "lead": []} for ttft in ttft_axis for tpot in tpot_axis}


class ReportTests(unittest.TestCase):
    def test_the_report_holds_the_verdict_one_map_per_rule_and_a_definition_per_term(self):
        ttft_axis, tpot_axis = (1000, 2000), (50, 100)
        definitions = ("mean", "itl-p99", "worst-itl")
        by_arch = {"unified": {"offered": 2, "useful_per_gpu": 0.5, "files_met": 1, "files_measured": 2}, "disaggregated": None}
        qualified = {(arch, d): v for d in definitions for arch, v in by_arch.items()}
        evidence = [{"architecture": "unified", "offered": 2, "seed": 345, "planned_calls": 10, "completed_calls": 10,
                     "measurement_including_drain_s": 20.0, "gpu_count": 2, "instance_type": "g7.48xlarge",
                     "mean_pass": True, "mean_attainment": 1.0, "mean_goodput": 0.5,
                     "itl-p99_pass": True, "itl-p99_attainment": 1.0, "itl_ms_p99": 20.0,
                     "worst-itl_pass": False, "worst-itl_attainment": 0.6, "longest_gap_ms_p90": 150.0}]
        page = report.render_report({
            "shape": "B-long-context", "archs": ("unified", "disaggregated"), "objective": (2000, 100),
            "attainment": 0.9, "ttft_metric": "per-call", "qualified": qualified,
            "grid": {d: make_grid(ttft_axis, tpot_axis, by_arch) for d in definitions},
            "ttft_axis": ttft_axis, "tpot_axis": tpot_axis, "evidence": evidence,
        })
        self.assertIn('<table class="verdict">', page)
        self.assertIn("only unified has a qualifying rate. Its qualifying rate is 2 tasks per second, with a goodput of 0.5 useful calls per second per GPU.", page)
        maps = re.findall(r'<div class="pmap".*?</div></div>', page, re.S)
        self.assertEqual(len(maps), 3)
        for m in maps:
            self.assertEqual(m.count('class="cell '), len(ttft_axis) * len(tpot_axis))
        self.assertIn('data-detail="TTFT at most 2000 ms (judged per-call), TPOT at most 100 ms, mean TPOT rule. disaggregated: did not qualify at any measured rate under this objective. '
                      'unified: qualifying rate 2 tasks per second, where 1 of 2 files met the objective; goodput 0.5 useful calls per second per GPU. Unified has the higher qualifying rate."', page)
        for key in set(re.findall(r'data-term="([^"]+)"', page)):
            self.assertIn(f'<template id="g-{key}">', page)


if __name__ == "__main__":
    unittest.main()


class FormatTests(unittest.TestCase):
    def test_shares_and_latencies_keep_the_digits_a_verdict_depends_on(self):
        self.assertEqual([report.fmt_share(x) for x in (0.9, 0.9996, 0.0001, 1.0)], ['90%', '99.96%', '0.01%', '100%'])
        self.assertEqual([report.fmt_ms(x) for x in (0.04, 12.34, 100.3, 2500.0)], ['0.04', '12.3', '100.3', '2,500'])
