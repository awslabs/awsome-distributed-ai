#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Participant entry point for the log-investigation exercise. Prints where the
# cases are and how to keep your own answers; it reads nothing privileged and
# changes nothing.
#
#   bash 14.case-exercise.sh            list the cases
#   bash 14.case-exercise.sh start      set up your own answers file
set -euo pipefail

cases=${AIM344_CASES:-/opt/aim344/cases}
answers=${AIM344_ANSWERS:-$HOME/aim344-answers}

if [[ $# -gt 1 ]]; then
    echo 'One command at a time: bash 14.case-exercise.sh [start]' >&2
    exit 2
fi
command=${1:-list}
case $command in
    list|start) ;;
    *) echo "Unknown command '$command'. Commands: list | start" >&2; exit 2 ;;
esac

if [[ ! -d $cases ]]; then
    cat >&2 <<EOF
The case material is not on this machine yet at $cases.
Ask your facilitator to stage it; nothing you can do from here will fix it.
EOF
    exit 3
fi

if [[ $command == list ]]; then
    printf 'Case material (read-only) is in %s\n\n' "$cases"
    for dir in "$cases"/case-*; do
        [[ -d $dir ]] || continue
        printf '%s\n' "$(basename "$dir")"
        # Show the symptom line, not the cause: the first heading of each case.
        head -1 "$dir/CASE.md" | sed 's/^# /   /'
        for file in "$dir"/*; do
            [[ -f $file ]] || continue
            [[ $(basename "$file") == CASE.md ]] && continue
            printf '     %s\n' "${file#"$dir"/}"
        done
        for sub in "$dir"/*/; do
            [[ -d $sub ]] || continue
            for file in "$sub"*; do
                [[ -f $file ]] && printf '     %s\n' "${file#"$dir"/}"
            done
        done
        printf '\n'
    done
    printf 'Start with %s/README.md, then each case'"'"'s CASE.md.\n' "$cases"
    printf 'To set up your own answers file: bash 14.case-exercise.sh start\n'
    exit 0
fi

# start
mkdir -p "$answers"
target=$answers/my-answers.md
if [[ -e $target ]]; then
    printf 'You already have %s; leaving it alone.\n' "$target"
else
    # install rather than cp: the staged template is read-only, and cp would
    # preserve that mode and hand you a file you cannot edit.
    install -m 0644 "$cases/answers-template.md" "$target"
    printf 'Created %s\n' "$target"
fi
printf 'Edit it with your usual editor. It is yours; no one else can read it.\n'
printf 'The case files stay read-only in %s.\n' "$cases"
