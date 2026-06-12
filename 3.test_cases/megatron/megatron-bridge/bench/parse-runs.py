#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Parse a megatron-bridge-bench campaign tree into a per-run loss curve + a summary index.

Walks ``<campaign>/<model>/<arm>-mb<m>-ovl<on|off>/logs/`` and, for each run, parses the
HIGHEST-numbered ``rank-<r>.log`` (the last PP stage is where Megatron prints the
per-iteration training line: ``iteration N/M | elapsed time per iteration (ms) | lm loss |
throughput per GPU (TFLOP/s/GPU)``) plus ``rank-0.log`` for the EFA/UCCL init signals, and emits:

  - ``<run_dir>/loss_curve.csv``  : iter, lm_loss, iter_time_s   (the loss-equivalence source)
  - one row appended to ``<campaign>/index.csv``                : perf + validity summary

Reads only; never deletes. Idempotent — re-running rewrites loss_curve.csv and rebuilds
index.csv from scratch so it always reflects the current tree.

Usage:  parse-runs.py <campaign_dir> [--warmup N]
The Megatron per-iteration log format varies by build; the regexes below are deliberately
tolerant and the script warns (does not crash) when a field is missing, so the first real
rank-0.log can be eyeballed against `--debug` output before trusting the numbers.
"""
import csv
import glob
import os
import re
import sys

WARMUP_DEFAULT = 4

# Tolerant patterns for the standard Megatron-LM / Megatron-Bridge training log line, e.g.:
#  " iteration       5/      24 | ... | elapsed time per iteration (ms): 6263.0 | ... |
#    lm loss: 1.189735E+01 | ... | throughput per GPU (TFLOP/s/GPU): 170.5 | ..."
RE_ITER = re.compile(r"\biteration\s+(\d+)\s*/\s*(\d+)")
RE_TIME_MS = re.compile(r"elapsed time per iteration \(ms\):\s*([\d.]+)")
RE_LOSS = re.compile(r"\blm loss:\s*([0-9.]+[eE]?[+-]?[0-9]*)")
RE_TFLOPS = re.compile(r"(?:throughput per GPU \(TFLOP/s/GPU\)|TFLOP/s/GPU|tflops?)[^\d-]*([\d.]+)", re.I)
RE_GBS = re.compile(r"global batch size:\s*(\d+)")
RE_EFA = re.compile(r"Selected provider is efa")
RE_UCCL = re.compile(r"Registered proxies|high-throughput mode")


def parse_run(run_dir, warmup, debug=False):
    # Megatron prints the per-iteration training line (lm loss / elapsed time / TFLOP) on
    # the LAST rank (last PP stage) — i.e. the HIGHEST-numbered node log. rank-0.log only
    # carries the Bridge "Step Time" logger plus the EFA/UCCL init lines. Verified against
    # the preserved 2026-06-01 logs (abrun-*-31.log has the iteration lines; rank0 does not).
    logs = glob.glob(os.path.join(run_dir, "logs", "rank-*.log"))
    if not logs:
        return None
    def _rank(p):
        try:
            return int(os.path.basename(p).split("-")[1].split(".")[0])
        except ValueError:
            return -1
    iter_log = max(logs, key=_rank)              # last node: per-iteration training lines
    rank0_log = min(logs, key=_rank)             # first node: EFA / UCCL-proxy init signals
    iters = []  # (iter, loss, time_s, tflops, gbs)
    efa_ok = uccl_ok = False
    for sig_log in {rank0_log, iter_log}:
        with open(sig_log, errors="replace") as f:
            for line in f:
                if RE_EFA.search(line):
                    efa_ok = True
                if RE_UCCL.search(line):
                    uccl_ok = True
    with open(iter_log, errors="replace") as f:
        for line in f:
            mi = RE_ITER.search(line)
            mt = RE_TIME_MS.search(line)
            ml = RE_LOSS.search(line)
            if mi and (mt or ml):
                it = int(mi.group(1))
                t = float(mt.group(1)) / 1000.0 if mt else float("nan")
                loss = float(ml.group(1)) if ml else float("nan")
                tf = float(RE_TFLOPS.search(line).group(1)) if RE_TFLOPS.search(line) else float("nan")
                gbs = int(RE_GBS.search(line).group(1)) if RE_GBS.search(line) else 0
                iters.append((it, loss, t, tf, gbs))
                if debug and len(iters) <= 3:
                    sys.stderr.write("  [debug] %s %s\n" % (os.path.basename(iter_log), iters[-1]))

    # write the loss curve (all iters, no drop — the curve IS the equivalence evidence)
    with open(os.path.join(run_dir, "loss_curve.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iter", "lm_loss", "iter_time_s", "tflops_per_gpu"])
        for it, loss, t, tf, _ in iters:
            w.writerow([it, "%.6f" % loss, "%.4f" % t, "%.2f" % tf])

    # perf summary over steady state (drop the first `warmup` iters incl. compile/init)
    steady = [r for r in iters if r[0] > warmup]
    times = [r[2] for r in steady if r[2] == r[2]]
    tfs = [r[3] for r in steady if r[3] == r[3]]
    gbs = next((r[4] for r in iters if r[4]), 0)
    seq = 4096
    n = len(times)
    mean_t = sum(times) / n if n else float("nan")
    med_t = sorted(times)[n // 2] if n else float("nan")
    mean_tf = sum(tfs) / len(tfs) if tfs else float("nan")
    # stalls: steady iters > 3x the median (the untrained-router hang signature)
    stalls = sum(1 for t in times if t == t and med_t == med_t and t > 3 * med_t) if n else 0
    tok_s = (gbs * seq / mean_t) if (mean_t == mean_t and mean_t and gbs) else float("nan")
    return {
        "n_iters_total": len(iters),
        "warmup_dropped": warmup,
        "n_steady": n,
        "mean_iter_s": mean_t,
        "median_iter_s": med_t,
        "tflops_per_gpu": mean_tf,
        "global_batch": gbs,
        "tok_s": tok_s,
        "stalls": stalls,
        "efa_ok": efa_ok,
        "uccl_ok": uccl_ok,
    }


def read_env(run_dir):
    env = {}
    p = os.path.join(run_dir, "env.txt")
    if os.path.isfile(p):
        for line in open(p, errors="replace"):
            for tok in line.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    env[k] = v
    return env


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    warmup = WARMUP_DEFAULT
    debug = "--debug" in sys.argv
    if "--warmup" in sys.argv:
        warmup = int(sys.argv[sys.argv.index("--warmup") + 1])
    if not args:
        sys.exit("usage: parse-runs.py <campaign_dir> [--warmup N] [--debug]")
    campaign = args[0].rstrip("/")

    rows = []
    for logdir in sorted(glob.glob(os.path.join(campaign, "*", "*", "logs"))):
        run_dir = os.path.dirname(logdir)
        model = os.path.basename(os.path.dirname(run_dir))
        tag = os.path.basename(run_dir)  # arm-mb<m>-ovl<on|off>
        s = parse_run(run_dir, warmup, debug)
        if s is None:
            continue
        env = read_env(run_dir)
        status = ""
        sp = os.path.join(run_dir, "STATUS")
        if os.path.isfile(sp):
            status = open(sp).read().strip().replace("\n", " ")
        rows.append({
            "model": model, "run": tag,
            "arm": env.get("arm", tag.split("-")[0]),
            "mb": env.get("mb", ""), "overlap": env.get("overlap", ""),
            "mean_iter_s": "%.4f" % s["mean_iter_s"],
            "median_iter_s": "%.4f" % s["median_iter_s"],
            "tflops_per_gpu": "%.2f" % s["tflops_per_gpu"],
            "tok_s": "%.1f" % s["tok_s"],
            "stalls": s["stalls"], "n_steady": s["n_steady"],
            "efa_ok": s["efa_ok"], "uccl_ok": s["uccl_ok"],
            "git_rev": env.get("git_rev", ""), "status": status,
        })

    if not rows:
        sys.exit("no runs with logs/rank-0.log found under %s" % campaign)
    cols = ["model", "run", "arm", "mb", "overlap", "mean_iter_s", "median_iter_s",
            "tflops_per_gpu", "tok_s", "stalls", "n_steady", "efa_ok", "uccl_ok",
            "git_rev", "status"]
    idx = os.path.join(campaign, "index.csv")
    with open(idx, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    sys.stderr.write("wrote %s (%d runs)\n" % (idx, len(rows)))
    # echo a readable table to stdout
    print("%-8s %-22s %10s %10s %8s %6s %7s %7s" %
          ("model", "run", "mean_s", "tflop/gpu", "tok/s", "stalls", "efa", "uccl"))
    for r in rows:
        print("%-8s %-22s %10s %10s %8s %6s %7s %7s" %
              (r["model"], r["run"], r["mean_iter_s"], r["tflops_per_gpu"],
               r["tok_s"], r["stalls"], r["efa_ok"], r["uccl_ok"]))


if __name__ == "__main__":
    main()
