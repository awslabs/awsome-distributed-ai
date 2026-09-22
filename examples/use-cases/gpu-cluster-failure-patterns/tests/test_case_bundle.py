#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tests for the staged diagnosis case bundle.

These assert the properties plan task 5 requires of the bundle as it will be
read by a participant: read-only, no leaked internal identifiers, no answers, no
cross-case identifier confusion, and every case answerable from files that are
actually present.

They run against a staged directory, so point AIM344_CASES at one:

    AIM344_CASES=/opt/aim344/cases python3 -m unittest test_case_bundle -v

With no environment variable the tests use the repository's own build output if
it is present, and skip otherwise rather than passing vacuously.
"""
import os
import re
import stat
import unittest
from pathlib import Path

CASES = Path(os.environ.get('AIM344_CASES', '/opt/aim344/cases'))
EXPECTED_CASES = ('case-a', 'case-b', 'case-c')

# Identifier-shape checks do not establish distribution clearance. Literal
# checks require a separately supplied list; restricted literals must never be
# embedded here, including reversibly split or encoded forms. Positive/negative
# samples in test_bundle_patterns.py exercise each shape without source records.
INTERNAL_PATTERNS = {
    'ticket id': re.compile(r'\b[PDV]\d{6,}\b'),
    # Any address at the publisher's corporate mail domain. The domain itself is
    # public; a person's address at it is not, so the shape is the address form.
    'corporate domain': re.compile(r'@[A-Za-z0-9.-]*amazon\.com'),
    'instance id': re.compile(r'\bi-[0-9a-f]{17}\b'),
    'hyperpod host': re.compile(r'hyperpod-i-[0-9a-f]+'),
    'correspondence uuid': re.compile(
        r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b'),
    # An internal python package whose name is a team prefix (one letter, one
    # digit) plus an underscore plus a module word, and the CamelCase workspace
    # of the same shape.
    'internal project': re.compile(r'\b[a-z]\d_(dataloaders|datasets|transformers)\b'
                                   r'|\b[A-Z]\d[A-Z][a-z]+ormers\b'),
    # Model-name shapes, not restricted literals.
    'internal model or dataset': re.compile(
        r'\b[a-z]\d-(?:large|xl|base)\b|\b[a-z]\d_\d+b\b'),
    'internal workspace root': re.compile(r'/opt/amazon/'),
    # The scheduler-generated hostname of the real reporting cluster: a role
    # word, the Slurm static marker, and an instance-type token.
    'real cluster hostname': re.compile(r'\bcompute-st-p4d\d+xlarge\b'),
    'gpu serial number': re.compile(r'S/N\s*\d+'),
}

# Exact internal literals cannot live in this file. A facilitator who has the
# internal list can supply it at run time:
#
#     AIM344_INTERNAL_DENYLIST=/path/to/list python3 -m unittest test_case_bundle
#
# One literal per line, blank lines and #-comments ignored. When the variable is
# absent the corresponding test SKIPS and says so: the literal-level check is
# then simply not performed, which is a stated gap and not a pass.
DENYLIST_ENV = 'AIM344_INTERNAL_DENYLIST'


def internal_denylist():
    path = os.environ.get(DENYLIST_ENV)
    if not path:
        return None
    lines = Path(path).read_text().splitlines()
    return [line.strip() for line in lines
            if line.strip() and not line.strip().startswith('#')]

ANSWER_PATTERNS = {
    'stated answer': re.compile(r'^\s*answer\s*:', re.I | re.M),
    'stated root cause': re.compile(r'\broot cause (is|was)\b', re.I),
    'solution section': re.compile(r'^#+\s*(solution|answers?)\b', re.I | re.M),
}


def bundle_files():
    return sorted(p for p in CASES.rglob('*') if p.is_file())


@unittest.skipUnless(CASES.is_dir(), f'no staged bundle at {CASES}')
class StagedBundle(unittest.TestCase):
    """Properties of the bundle as staged."""

    def test_every_expected_case_is_present_with_a_case_file(self):
        for name in EXPECTED_CASES:
            with self.subTest(case=name):
                self.assertTrue((CASES / name).is_dir(), f'{name} directory missing')
                self.assertTrue((CASES / name / 'CASE.md').is_file(),
                                f'{name}/CASE.md missing')

    def test_the_shared_readme_and_answer_template_are_present(self):
        self.assertTrue((CASES / 'README.md').is_file())
        self.assertTrue((CASES / 'answers-template.md').is_file())

    def test_every_file_is_read_only(self):
        for path in bundle_files():
            with self.subTest(path=path.relative_to(CASES)):
                mode = stat.S_IMODE(path.stat().st_mode)
                self.assertFalse(mode & stat.S_IWOTH, 'world-writable')
                self.assertFalse(mode & stat.S_IWGRP, 'group-writable')
                self.assertFalse(mode & stat.S_IWUSR, 'owner-writable')

    def test_every_file_is_readable(self):
        for path in bundle_files():
            with self.subTest(path=path.relative_to(CASES)):
                self.assertTrue(stat.S_IMODE(path.stat().st_mode) & stat.S_IROTH,
                                'not world-readable, participants could not read it')

    def test_no_internal_identifier_reaches_a_participant_file(self):
        for path in bundle_files():
            body = path.read_text(errors='replace')
            for label, pattern in INTERNAL_PATTERNS.items():
                match = pattern.search(body)
                with self.subTest(path=path.relative_to(CASES), leak=label):
                    self.assertIsNone(
                        match,
                        f'{label} present: {match.group(0)!r}' if match else '')

    def test_no_internal_literal_from_the_facilitator_denylist_is_present(self):
        """Run literal checks only when an owner supplies the external list."""
        denylist = internal_denylist()
        if denylist is None:
            self.skipTest(
                f'{DENYLIST_ENV} not set: the literal-level internal-name check '
                f'did NOT run. Shape patterns above still ran. A facilitator with '
                f'the internal list should set it before staging.')
        self.assertTrue(denylist, f'{DENYLIST_ENV} names an empty list')
        for path in bundle_files():
            body = path.read_text(errors='replace').lower()
            for literal in denylist:
                with self.subTest(path=path.relative_to(CASES)):
                    self.assertNotIn(
                        literal.lower(), body,
                        f'a facilitator-listed internal literal reached '
                        f'{path.relative_to(CASES)}')

    def test_no_answers_are_staged_with_the_cases(self):
        for path in bundle_files():
            if path.name == 'answers-template.md':
                continue  # the blank template legitimately says "answers"
            body = path.read_text(errors='replace')
            for label, pattern in ANSWER_PATTERNS.items():
                match = pattern.search(body)
                with self.subTest(path=path.relative_to(CASES), kind=label):
                    self.assertIsNone(
                        match, f'{label} present: {match.group(0)!r}' if match else '')

    def test_no_credential_shaped_material_is_staged(self):
        secrets = re.compile(
            r'AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY|aws_secret_access_key'
            r'|password\s*[:=]', re.I)
        for path in bundle_files():
            with self.subTest(path=path.relative_to(CASES)):
                self.assertIsNone(secrets.search(path.read_text(errors='replace')))

    def test_no_whole_record_json_is_staged(self):
        """A dumped ticket export would carry far more than the case needs."""
        for path in bundle_files():
            with self.subTest(path=path.relative_to(CASES)):
                self.assertNotEqual(path.suffix, '.json',
                                    'raw JSON record staged for participants')

    def test_each_case_file_references_only_files_that_exist(self):
        """A question pointing at a missing file cannot be answered."""
        referenced = re.compile(r'`([a-z0-9][a-z0-9._/-]*\.(?:log|txt|md))`')
        for name in EXPECTED_CASES:
            case_dir = CASES / name
            body = (case_dir / 'CASE.md').read_text()
            for target in set(referenced.findall(body)):
                if target in ('answers-template.md', 'CASE.md'):
                    continue
                with self.subTest(case=name, target=target):
                    self.assertTrue((case_dir / target).is_file(),
                                    f'{name}/CASE.md references missing {target}')

    def test_no_case_references_another_cases_files(self):
        """Cross-case identifier confusion is an explicit requirement to avoid."""
        others = {'case-a': ('case-b', 'case-c'),
                  'case-b': ('case-a', 'case-c'),
                  'case-c': ('case-a', 'case-b')}
        for name, foreign in others.items():
            for path in (CASES / name).rglob('*'):
                if not path.is_file():
                    continue
                body = path.read_text(errors='replace')
                for other in foreign:
                    with self.subTest(path=path.relative_to(CASES), other=other):
                        self.assertNotIn(other, body)

    def test_every_case_asks_for_evidence_and_a_next_observation(self):
        """Require evidence-led questions independent of line wrapping."""
        for name in EXPECTED_CASES:
            body = ' '.join((CASES / name / 'CASE.md').read_text().lower().split())
            with self.subTest(case=name):
                self.assertIn('quote', body)
                self.assertTrue('rule out' in body or 'next observation' in body
                                or 'would you collect' in body,
                                'no question asks what to collect next')

    def test_the_readme_states_that_no_command_is_faked(self):
        body = (CASES / 'README.md').read_text().lower()
        self.assertTrue('alias' in body,
                        'README does not tell participants tools are real')

    def test_the_readme_tells_participants_how_to_get_an_editable_copy(self):
        """Measured on the pair: the bundle is 0444, and `cp` preserves that mode,
        so a plain `cp` hands the participant a template they cannot edit and a
        second `cp` fails with Permission denied. The instructions must not do
        that."""
        body = (CASES / 'README.md').read_text()
        self.assertIn('install -m 0644', body,
                      'README does not give a copy command that yields a writable file')
        self.assertNotRegex(
            body, r'cp /opt/aim344/cases/answers-template\.md\s+~/aim344-answers/\s*$',
            'README still recommends a plain cp, which leaves a read-only copy')

    def test_the_answer_template_has_a_block_per_case(self):
        body = (CASES / 'answers-template.md').read_text()
        for label in ('Case A', 'Case B', 'Case C'):
            with self.subTest(label=label):
                self.assertIn(label, body)


class RepositoryContainment(unittest.TestCase):
    """Properties of the companion repository itself.

    Deliberately NOT gated on a staged bundle: the point of review finding F12
    is that a public-bound repository must not carry or link an internal-derived
    answer key, and that must be asserted on every run, including a run with no
    staged tree present. A skip here would be the same vacuous pass that made
    the earlier tampered-bundle control unfalsifiable.
    """

    REPO_ROOT = Path(__file__).resolve().parent.parent

    def test_no_answer_key_file_is_in_the_repository(self):
        answers = self.REPO_ROOT / 'facilitator' / 'cases' / 'ANSWERS.md'
        self.assertFalse(answers.exists(),
                         f'internal-derived answer key present at {answers}')

    def test_no_participant_facing_document_links_an_answer_key(self):
        for name in ('README.md', 'RUNBOOK.md', '14.case-exercise.sh'):
            path = self.REPO_ROOT / name
            if not path.exists():
                continue
            body = path.read_text()
            with self.subTest(file=name):
                self.assertNotIn('cases/ANSWERS.md', body)
                self.assertNotIn('ANSWERS.md', body)

    def test_the_readme_says_where_the_key_is_not(self):
        """A reader who needs the key must be routed somewhere, or they will
        assume it was lost and rebuild it in the repository.
        """
        body = (self.REPO_ROOT / 'README.md').read_text()
        self.assertRegex(body, r'(?i)no answer key is in this repository')
        self.assertRegex(body, r'(?i)workshop owner')


if __name__ == '__main__':
    unittest.main()
