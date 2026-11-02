#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Pass a job's output through, and append the lab's own lines to the lab log.

common.sh puts this between srun and the job log when job.sh runs an incident.
A line that starts with workload.py's LAB_LINE_MARKER goes to the lab log named
by the argument; every other line goes to standard output unchanged. Each line is
flushed as it arrives, so the terminal and the job log keep pace with the job.
"""
import sys

MARKER = b'AIM344 lab: '


def main(lab_log):
    output = sys.stdout.buffer
    with open(lab_log, 'ab') as lab:
        for line in iter(sys.stdin.buffer.readline, b''):
            if line.startswith(MARKER):
                lab.write(line)
                lab.flush()
            else:
                output.write(line)
                output.flush()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: lab-lines.py LAB_LOG')
    main(sys.argv[1])
