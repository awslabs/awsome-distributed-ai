#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Two-directional checks on the leak patterns in test_case_bundle.py.

The bundle checker no longer spells out the internal names it forbids, because it
lives in a public-bound repository and a checker that hardcodes a forbidden
string republishes it. It matches on shape instead.

A shape pattern can fail in two opposite ways, and one direction of testing hides
the other:

  UNDER-MATCH  it no longer catches the identifier it was written for, so the
               leak guard passes vacuously.
  OVER-MATCH   it fires on ordinary text the cleared bundle legitimately
               contains, so the guard is unusable and gets disabled.

Each pattern below is therefore given both a positive and a negative sample.

Every positive sample is an authored fixture, not a transformed source record.
A reversible split or encoding is not redaction. Shapes cannot recognize every
restricted literal; the optional external deny-list supplies that separate check.

Run with the rest of the suite; this module needs no staged bundle.
"""
import unittest

from test_case_bundle import ANSWER_PATTERNS, INTERNAL_PATTERNS


# label -> (must match, must not match)
#
# Every positive sample is INVENTED to have the forbidden shape. None is a real
# identifier, and none is a real identifier that has been split, reordered or
# otherwise encoded: a reversible transformation of a real value would still be
# that value in a public file. Negative samples are the strings the cleared
# bundle and this lab really do contain, so a pattern that fires on one of them
# is unusable and gets caught here rather than in the field.
SAMPLES = {
    'ticket id': (
        ['P11111111', 'D22222222', 'V333333333'],
        ['P4d', 'D1', 'PyTorch 1.8.1', 'rank 12345', 'V100'],
    ),
    # The publisher's corporate mail domain. The domain is public and appears all
    # over this tree; what the pattern forbids is an address AT it, so the
    # positive samples are invented local parts.
    'corporate domain': (
        ['someone@amazon.com', 'a.b@team.amazon.com'],
        ['someone@example.com', 'https://aws.amazon.com/'],
    ),
    'instance id': (
        ['i-0123456789abcdef0', 'i-0fedcba987654321f'],
        ['i-0123', 'i-', 'instance-0123456789abcdef0'],
    ),
    'hyperpod host': (
        ['hyperpod-i-0abc123'],
        ['hyperpod cluster', 'i-0abc123'],
    ),
    'correspondence uuid': (
        ['00000000-1111-4222-8333-444444444444',
         'abcdef01-2345-4678-89ab-cdef01234567'],
        ['00000000-1111-4222-8333', 'GPU-1234', '2026-09-17T12:00:00Z'],
    ),
    # Invented prefix: the letter/digit pair below belongs to no team. It
    # exercises the `[a-z]\d_` and `[A-Z]\d[A-Z][a-z]+ormers` shapes.
    'internal project': (
        ['q7_dataloaders/data_provider.py', 'Q7Transformers', 'q7_datasets'],
        ['dataloaders/data_provider.py', 'multiprocessing/queues.py',
         'my_dataloaders', 'Transformers'],
    ),
    # Invented model family and parameter count, same reasoning.
    'internal model or dataset': (
        ['q7-large', 'q7_44b', 'z1-base'],
        # These are the strings the cleared bundle and this lab really contain.
        # If the shape pattern fires on any of them the guard is unusable.
        ['p4d.24xlarge', 'pretraining.yaml', 'NCCL 2.7.8', 'python3.8',
         'g7.48xlarge', 'p6-b200', 'casing', 'basin', 'against'],
    ),
    'internal workspace root': (
        ['/opt/amazon/openmpi/bin/mpirun'],
        ['/opt/aim344/cases', '/opt/venv/lib/python3.6', '/opt/aws/pcs'],
    ),
    # Invented node numbers on an invented instance-type digit count: the shape
    # is role word + Slurm static marker + instance-type token, which is what the
    # pattern is for.
    'real cluster hostname': (
        ['compute-st-p4d99xlarge-7', 'compute-st-p4d99xlarge-8'],
        ['node-a', 'node-b', 'gpu-g7-1', 'compute-st-g7-1', 'trainer-worker-0'],
    ),
    'gpu serial number': (
        ['S/N 1111111111111', 'S/N2222222222222'],
        ['SN', 'serial number removed', 'S/N/A'],
    ),
}

ANSWER_SAMPLES = {
    'stated answer': (
        ['Answer: the interconnect', '  answer:  x'],
        ['answers-template.md', 'Record your answers in your own copy',
         'A defensible answer may end in "insufficient evidence, collect X next".'],
    ),
    'stated root cause': (
        ['the root cause is a failed link', 'The root cause was the switch'],
        ['name a root cause', 'Do not force a component-level verdict.',
         'root cause analysis is out of scope'],
    ),
    'solution section': (
        ['## Solution', '# Answers', '### answer'],
        ['## Questions', '## Your files', 'Record your answers'],
    ),
}


class PatternsCatchWhatTheyAreFor(unittest.TestCase):
    """Under-match direction: a guard that no longer fires is not a guard."""

    def test_every_internal_pattern_has_samples(self):
        self.assertEqual(sorted(INTERNAL_PATTERNS), sorted(SAMPLES),
                         'a leak pattern has no positive/negative samples')

    def test_every_answer_pattern_has_samples(self):
        self.assertEqual(sorted(ANSWER_PATTERNS), sorted(ANSWER_SAMPLES))

    def test_internal_patterns_match_their_identifiers(self):
        for label, (positives, _) in SAMPLES.items():
            pattern = INTERNAL_PATTERNS[label]
            for sample in positives:
                with self.subTest(pattern=label, sample=sample):
                    self.assertRegex(sample, pattern,
                                     f'{label} no longer catches what it is for')

    def test_answer_patterns_match_stated_answers(self):
        for label, (positives, _) in ANSWER_SAMPLES.items():
            pattern = ANSWER_PATTERNS[label]
            for sample in positives:
                with self.subTest(pattern=label, sample=sample):
                    self.assertRegex(sample, pattern)


class PatternsDoNotFireOnClearedText(unittest.TestCase):
    """Over-match direction: a guard that fires on the real bundle is unusable."""

    def test_internal_patterns_leave_cleared_strings_alone(self):
        for label, (_, negatives) in SAMPLES.items():
            pattern = INTERNAL_PATTERNS[label]
            for sample in negatives:
                with self.subTest(pattern=label, sample=sample):
                    self.assertNotRegex(sample, pattern,
                                        f'{label} over-matches ordinary text')

    def test_answer_patterns_leave_question_prose_alone(self):
        for label, (_, negatives) in ANSWER_SAMPLES.items():
            pattern = ANSWER_PATTERNS[label]
            for sample in negatives:
                with self.subTest(pattern=label, sample=sample):
                    self.assertNotRegex(sample, pattern)


if __name__ == '__main__':
    unittest.main()
