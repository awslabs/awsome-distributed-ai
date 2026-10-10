#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Fail-closed gates for run_nccl_ep_efa.sh. A gate whose inputs are missing is NOT_RUN, never PASS.

  check_run.py gpu-requests             stdin: pod list JSON -> GPUs requested by live pods
  check_run.py pods --dir D --p0 P --p1 P          prints "<ip0> <ip1>"
  check_run.py preflight --dir D --p0 P --p1 P --gpus N --efa N [--build-in-pod 0|1]
  check_run.py correctness --dir D [--world 16] [--expect-argv "-a ht -L fl"]
  check_run.py timing --dir D [--world 16]
  check_run.py summarize --dir D --out RESULT.json

Layout written by the launcher: D/<case>/<pod>/mpi/1/rank.N/{stdout,stderr}, D/<case>/<pod>/
{argv.txt,env.txt,mpirun.exit} (the mpirun pod), D/<case>/efa-{before,after}-<pod>.json,
D/<case>/reap-<pod>.txt, D/gpu-after-<pod>.txt, D/sshd-stop.txt, D/preflight-<pod>.txt.
argv.txt holds one word per line: the program mpirun started, then its arguments; spec.txt holds np= (pod_tools.sh
mpirun). The launcher passes --world "$NP" (2 x GPU_PER_NODE), and np= must equal it.
Source references are to NVIDIA/nccl-extensions at 901c4e65, the commit the image builds.
"""
import argparse
import datetime
import glob
import json
import os
import re
import sys

WANT_NCCL = "2.32.3"
# The programs run_nccl_ep_efa.sh starts (its two run_case lines); argv.txt's first word must be exactly one of these.
EP_TEST = "/opt/nccl-ep/test/nccl_ep/ep_test"
EP_BENCH = "/opt/nccl-ep/test/nccl_ep/ep_bench"
# HT GIN contexts per rank: the library refuses fewer than 1 reserved + 2 N2N warps per dispatch SM and, with
# num_qp_per_rank AUTO (what ep_test and ep_bench pass for HT), creates exactly that many (nccl_ep.cc:1849-1856,
# 2319-2323; device/ht_ep_configs.cuh:17,26,58). The dispatch SM budget defaults to 16, so 33 contexts.
# ep_test has no SM option (ep_test.cu:257). ep_bench prints its --dispatch-num-sms and --max-num-sms requests
# (ep_bench.cu:5800-5806), which the library resolves in that order before the default (nccl_ep.cc:2175-2206).
# The NCCL_EP_DISPATCH_SMS / NCCL_EP_COMM_SMS overrides win over both (nccl_ep.cc:2213-2231) and are not printed;
# the launcher does not set them, and a run that lowers the budget through them fails this gate.
HT_RESERVED_CTXS, HT_N2N_WARPS, HT_DEFAULT_SMS = 1, 2, 16
# The configuration ep_bench prints on rank 0 (ep_bench.cu:5789-5842). getopt keeps the last value of a repeated
# flag (5294-5540), so this block, not argv, says what ran. "Ranks" must also equal --world.
BENCH_CONFIG = {"Algorithm": "HIGH_THROUGHPUT", "Layout": "flat", "Validate mode": "enabled",
                "Dispatch recipe": "none", "Combine recipe": "none"}
FAIL_PATTERNS = [r"verification FAILED", r"check failed", r"Exiting test due to", r"Failed: (MPI|Cuda|NCCL) error",
                 r"illegal memory access", r"Segmentation fault", r"are incompatible", r"Recv_count check failed",
                 r"GIN: DevComm setup failed", r"validation failure"]
BYTE_KEYS = ["libnccl_sha256", "libnccl_ep_sha256", "ep_test_sha256", "ep_bench_sha256", "aws_ofi_nccl_sha256"]
SEMANTIC_KEYS = ["verdict", "nccl_extensions_commit", "nccl_ep_version", "nccl_wheel_sha256",
                 "nccl_header_version_code", "efa_installer_version", "gdrcopy_commit", "nvcc_gencode"]


def read(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def verdict_of(errors, have_inputs):
    if not have_inputs:
        return "NOT_RUN"
    return "PASS" if not errors else "FAIL"


def rank_texts(case_dir):
    """rank -> stdout+stderr text, from every pod's mpirun --output-filename tree."""
    out = {}
    for p in glob.glob(os.path.join(case_dir, "*", "mpi", "**", "rank.*", "std*"), recursive=True):
        m = re.search(r"rank\.(\d+)[/\\](stdout|stderr)$", p)
        if m:
            out.setdefault(int(m.group(1)), []).append(read(p))
    return {r: "\n".join(v) for r, v in out.items()}


def mpirun_files(case_dir):
    """(exit code, argv words, launch env, spec.txt keys) from the one mpirun pod directory; all None otherwise."""
    exits = glob.glob(os.path.join(case_dir, "*", "mpirun.exit"))
    if len(exits) != 1:
        return None, None, None, None
    base = os.path.dirname(exits[0])
    argv = [a for a in read(os.path.join(base, "argv.txt")).splitlines() if a != ""]
    env = dict(line.split("=", 1) for line in read(os.path.join(base, "env.txt")).splitlines() if "=" in line)
    return read(exits[0]).strip(), argv, env, parse_kv(os.path.join(base, "spec.txt"))


def split_program(argv, path):
    """(program, arguments) from argv.txt's words; program is None unless the first word is exactly <path>."""
    if argv and argv[0] == path:
        return argv[0], argv[1:]
    return None, argv or []


def launch_errors(rc, argv, spec, program, world):
    """The mpirun pod's records: exit code, the program word, and spec.txt's np= against --world."""
    if rc is None:
        return ["mpirun.exit missing or ambiguous"]
    errors = [] if rc == "0" else [f"mpirun exit {rc}"]
    if split_program(argv, program)[0] is None:
        errors.append(f"argv.txt program {argv[0] if argv else None!r} is not {program}")
    if spec.get("np") != str(world):
        errors.append(f"spec.txt np={spec.get('np')}, want {world} (--world)")
    return errors


def ht_min_contexts(dispatch_sms):
    return HT_RESERVED_CTXS + HT_N2N_WARPS * dispatch_sms


def printed(text, key):
    """Value of ep_bench's '  <key>: <value>' configuration line; None unless it appears exactly once."""
    found = re.findall(r"^  " + re.escape(key) + r":[ \t]+(\S[^\n]*?)[ \t]*$", text, re.M)
    return found[0] if len(found) == 1 else None


def transport_row(r, txt):
    contexts = [int(n) for n in re.findall(r"devCommCreate: creating (\d+) contexts", txt)]
    fabric = re.search(r"NET/OFI Selected provider is efa, fabric is (efa-direct|efa)\b", txt)
    return {
        "gin_type_env_2": bool(re.search(r"NCCL_GIN_TYPE set by environment to 2\b", txt)),
        "nccl_version": bool(re.search(r"NCCL version " + re.escape(WANT_NCCL) + r"\b", txt)),
        "using_network_libfabric": "Using network Libfabric" in txt,
        "rma_plugin_libfabric": "RMA/Plugin: Assigned plugin Libfabric to comm" in txt,
        "gin_proxy_line": "GIN Proxy will not be using GDRCopy" in txt,
        "efa_provider": fabric is not None,
        "max_gin_contexts": max(contexts) if contexts else 0,
        "no_socket_channels": "via NET/Socket" not in txt,
        "fabric": fabric.group(1) if fabric else None,
    }


def transport_errors(r, row, min_contexts):
    need = ["gin_type_env_2", "nccl_version", "using_network_libfabric", "rma_plugin_libfabric", "gin_proxy_line",
            "efa_provider", "no_socket_channels"]
    errs = [f"rank {r}: {k}" for k in need if not row[k]]
    if row["max_gin_contexts"] < min_contexts:
        errs.append(f"rank {r}: GIN contexts {row['max_gin_contexts']} < {min_contexts}")
    return errs


def efa_deltas(case_dir):
    rows, errors = {}, []
    befores = sorted(glob.glob(os.path.join(case_dir, "efa-before-*.json")))
    for b in befores:
        pod = os.path.basename(b)[len("efa-before-"):-len(".json")]
        before, after = load_json(b), load_json(os.path.join(case_dir, f"efa-after-{pod}.json"))
        if not before or not after:
            errors.append(f"{pod}: unreadable counter snapshot")
            continue
        delta = after["total_bytes"] - before["total_bytes"]
        rows[pod] = {"host": after.get("host"), "delta_total_bytes": delta, "n_devices": after.get("n_devices")}
        if before.get("host") != after.get("host") or not before.get("n_devices") or delta <= 0:
            errors.append(f"{pod}: EFA delta {delta} over {after.get('n_devices')} devices")
    if len(rows) + len(errors) < 2:
        errors.append(f"expected counter snapshots for 2 pods, found {len(befores)}")
    return {"verdict": verdict_of(errors, bool(befores)), "per_pod": rows, "errors": errors}


def correctness(case_dir, world, expect_argv):
    rc, argv, env, spec = mpirun_files(case_dir)
    texts = rank_texts(case_dir)
    errors, rows = launch_errors(rc, argv, spec, EP_TEST, world), []
    prog, args = split_program(argv, EP_TEST)
    if prog is not None and args != expect_argv:
        errors.append(f"argv {args} != {expect_argv} (HT + FLAT explicit; random mode skips the oracle)")
    if env is not None and env.get("NCCL_GIN_TYPE") != "2":
        errors.append("NCCL_GIN_TYPE=2 not in the launch environment")
    for r in range(world):
        txt = texts.get(r, "")
        row = {"rank": r, "has_output": bool(txt),
               "algorithm_ht": f"Rank {r}: Testing ncclEpCreateGroup with algorithm: HIGH_THROUGHPUT" in txt,
               "layout_flat": f"Rank {r}: Verifying recv_topk_weights and recv_topk_idx (HT+FLAT, 2D)" in txt,
               "dispatch_passed": f"Rank {r}: HIGH_THROUGHPUT Dispatch flow passed successfully" in txt,
               "combine_passed": f"Rank {r}: Combine verification PASSED! All" in txt,
               "success_line": f"[MPI Rank {r}] Success" in txt,
               "failure_markers": [p for p in FAIL_PATTERNS if re.search(p, txt)]}
        row.update(transport_row(r, txt))
        miss = [k for k in ("algorithm_ht", "layout_flat", "dispatch_passed", "combine_passed", "success_line") if not row[k]]
        if miss:
            errors.append(f"rank {r}: missing {miss}")
        if row["failure_markers"]:
            errors.append(f"rank {r}: failure markers {row['failure_markers']}")
        errors += transport_errors(r, row, ht_min_contexts(HT_DEFAULT_SMS))  # ep_test has no SM option
        rows.append(row)
    efa = efa_deltas(case_dir)
    errors += [f"efa: {e}" for e in efa["errors"]]
    return {"verdict": verdict_of(errors, rc is not None or bool(texts)), "argv": argv, "mpirun_exit": rc,
            "oracle": "NVIDIA ep_test built-in checks (dispatch recv counts/topk, combine weighted sums)",
            "efa_delta": efa, "rows": rows, "errors": errors[:40]}


SUMMARY_RE = {
    "header": r"=== Summary \(High Throughput recipe (\S+), across (\d+) ranks\) ===",
    "dispatch": r"Dispatch:\s+total=([\d.]+) us \(min=([\d.]+), max=([\d.]+)\)",
    "combine": r"Combine:\s+total=([\d.]+) us \(min=([\d.]+), max=([\d.]+)\)",
    "total": r"Total \(D\+C\): avg=([\d.]+) us, min=([\d.]+) us, max=([\d.]+) us",
}


def timing(case_dir, world):
    rc, argv, env, spec = mpirun_files(case_dir)
    texts = rank_texts(case_dir)
    if rc is None and not texts:
        return {"verdict": "NOT_RUN", "mpirun_exit": None, "observed": None, "printed_config": None,
                "efa_delta": {"verdict": "NOT_RUN", "per_pod": {}, "errors": []},
                "errors": ["timing not run: no mpirun.exit and no rank output"]}
    errors = launch_errors(rc, argv, spec, EP_BENCH, world)
    rank0 = texts.get(0, "")
    # What ep_bench ran is what it printed (BENCH_CONFIG); argv may repeat a flag, and the last value wins.
    blocks = rank0.count("=== NCCL EP Performance Benchmark ===")
    if blocks != 1:
        errors.append(f"ep_bench configuration block printed {blocks} times on rank 0, want 1")
    config = {k: printed(rank0, k) for k in list(BENCH_CONFIG) + ["Ranks", "Shared SMs", "Dispatch SMs"]}
    for k, want in list(BENCH_CONFIG.items()) + [("Ranks", str(world))]:
        if config[k] != want:
            errors.append(f"ep_bench printed {k}: {config[k]}, want {want}")
    sms = next((int(config[k]) for k in ("Dispatch SMs", "Shared SMs") if (config[k] or "").isdigit()), HT_DEFAULT_SMS)
    parsed = {}
    for k, pat in SUMMARY_RE.items():
        m = re.search(pat, rank0)
        parsed[k] = m.groups() if m else None
        if not m:
            errors.append(f"ep_bench summary line '{k}' not found on rank 0")
    if parsed["header"] and parsed["header"][0] != "none":
        errors.append(f"summary header recipe {parsed['header'][0]}, want none")
    if parsed["header"] and int(parsed["header"][1]) != world:
        errors.append(f"summary covers {parsed['header'][1]} ranks, want {world}")
    if "Global validation: Dispatch=PASSED, Combine=PASSED" not in rank0:
        errors.append("ep_bench --validate did not report Dispatch=PASSED, Combine=PASSED")
    for r in range(world):
        txt = texts.get(r, "")
        errors += transport_errors(r, transport_row(r, txt), ht_min_contexts(sms))
        bad = [p for p in FAIL_PATTERNS if re.search(p, txt)]
        if bad:
            errors.append(f"rank {r}: failure markers {bad}")
    efa = efa_deltas(case_dir)
    errors += [f"efa: {e}" for e in efa["errors"]]
    observed = None
    if all(parsed.values()):
        observed = {"label": "ep_bench host-observed averages, one launch; not a validated performance measurement",
                    "dispatch_total_us": float(parsed["dispatch"][0]), "combine_total_us": float(parsed["combine"][0]),
                    "dc_avg_us": float(parsed["total"][0]), "dc_min_us": float(parsed["total"][1]),
                    "dc_max_us": float(parsed["total"][2]), "argv": argv}
    return {"verdict": verdict_of(errors, rc is not None or bool(texts)), "mpirun_exit": rc, "observed": observed,
            "printed_config": config, "efa_delta": efa, "errors": errors[:40]}


def lifecycle(d):
    errors, info = [], {}
    reaps = sorted(glob.glob(os.path.join(d, "*", "reap-*.txt")))
    for p in reaps:
        t = read(p)
        info[os.path.relpath(p, d)] = " ".join(t.split())
        if "leftover: none" not in t:
            errors.append(f"{os.path.relpath(p, d)}: processes outlived mpirun ({' '.join(t.split())})")
    gpus = sorted(glob.glob(os.path.join(d, "gpu-after-*.txt")))
    for p in gpus:
        t = read(p).split()
        info[os.path.basename(p)] = " ".join(t)
        if len(t) != 3 or t[0] != "0" or t[1] != "0":
            errors.append(f"{os.path.basename(p)}: GPUs not idle after the run ({' '.join(t)})")
    stop = read(os.path.join(d, "sshd-stop.txt"))
    if stop.count("keys removed") < 2 or "still alive" in stop:
        errors.append("sshd not stopped with keys removed in both pods")
    if len(gpus) < 2 or not reaps:
        errors.append("lifecycle records incomplete")
    return {"verdict": verdict_of(errors, bool(reaps or gpus or stop)), "records": info, "errors": errors}


def cmd_gpu_requests(_a):
    pods = json.load(sys.stdin).get("items", [])
    total = 0
    for p in pods:
        if p.get("status", {}).get("phase") in ("Succeeded", "Failed"):
            continue
        for c in p.get("spec", {}).get("containers", []):
            total += int(c.get("resources", {}).get("requests", {}).get("nvidia.com/gpu", 0) or 0)
    print(total)
    return 0


def cmd_pods(a):
    errs, ips, nodes = [], [], []
    now = datetime.datetime.now(datetime.timezone.utc)
    for p in (a.p0, a.p1):
        d = load_json(os.path.join(a.dir, f"pod-{p}.json"))
        if not d:
            errs.append(f"{p}: no pod record")
            continue
        s, spec = d.get("status", {}), d.get("spec", {})
        if s.get("phase") != "Running":
            errs.append(f"{p}: phase {s.get('phase')}")
        if spec.get("hostNetwork") is not True:
            errs.append(f"{p}: hostNetwork is not true")
        ips.append(s.get("podIP"))
        nodes.append(spec.get("nodeName"))
        start = s.get("startTime")
        if start:
            age_h = (now - datetime.datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds() / 3600
            if age_h > 6:
                print(f"WARN {p}: pod is {age_h:.1f} h old; fresh pods are the trusted baseline", file=sys.stderr)
    if len(set(nodes)) != 2 or None in nodes:
        errs.append(f"pods must be on two different nodes (got {nodes})")
    if len(set(ips)) != 2 or None in ips:
        errs.append(f"need two distinct pod IPs (got {ips})")
    if errs:
        print("\n".join(errs), file=sys.stderr)
        return 1
    print(ips[0], ips[1])
    return 0


def parse_kv(path):
    out = {}
    for line in read(path).splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def cmd_preflight(a):
    errs, rows = [], {}
    for p in (a.p0, a.p1):
        kv = parse_kv(os.path.join(a.dir, f"preflight-{p}.txt"))
        rows[p] = kv
        want = {"gpus": str(a.gpus), "gpu_compute_apps": "0", "efa_devices": str(a.efa), "ofi_plugin_present": "1",
                "sshd_present": "1", "libnccl_files": "/opt/nccl/lib/libnccl.so.2", "record.verdict": "PASS"}
        for k, v in want.items():
            if kv.get(k) != v:
                errs.append(f"{p}: {k}={kv.get(k)!r}, want {v!r}")
        if int(kv.get("fi_info_efa_providers", "0") or 0) < 1:
            errs.append(f"{p}: fi_info -p efa reports no efa provider")
        if int(kv.get("zombies", "0") or 0) > 5:
            print(f"WARN {p}: {kv.get('zombies')} zombie processes; prefer a fresh pod", file=sys.stderr)
    r0, r1 = rows.get(a.p0, {}), rows.get(a.p1, {})
    for k in SEMANTIC_KEYS + ([] if a.build_in_pod == 1 else BYTE_KEYS):
        if r0.get("record." + k) != r1.get("record." + k):
            errs.append(f"build records differ on {k}: {r0.get('record.' + k)} vs {r1.get('record.' + k)}")
    result = {"verdict": "PASS" if not errs else "FAIL", "errors": errs, "pods": rows}
    with open(os.path.join(a.dir, "PREFLIGHT.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({"preflight": result["verdict"], "errors": errs[:10]}))
    return 0 if not errs else 1


def cmd_correctness(a):
    res = correctness(os.path.join(a.dir, "correctness"), a.world, a.expect_argv.split())
    print(json.dumps({"correctness": res["verdict"], "errors": res["errors"][:8]}))
    return 0 if res["verdict"] == "PASS" else 1


def cmd_timing(a):
    res = timing(os.path.join(a.dir, "timing"), a.world)
    print(json.dumps({"timing": res["verdict"], "observed": res["observed"], "errors": res["errors"][:8]}))
    return 0 if res["verdict"] == "PASS" else 1


def cmd_summarize(a):
    corr = correctness(os.path.join(a.dir, "correctness"), a.world, a.expect_argv.split())
    tim = timing(os.path.join(a.dir, "timing"), a.world)
    life = lifecycle(a.dir)
    pre = load_json(os.path.join(a.dir, "PREFLIGHT.json")) or {"verdict": "NOT_RUN"}
    gates = {"preflight": pre.get("verdict"), "correctness_and_transport": corr["verdict"],
             "timing": tim["verdict"], "lifecycle": life["verdict"]}
    ok = all(v == "PASS" for v in gates.values())
    out = {"schema": "nccl-ep-efa-run/1", "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "verdict": "PASS" if ok else "FAIL", "gates": gates,
           "correctness": corr, "timing": tim, "lifecycle": life,
           "identity": {p: {k[7:]: v for k, v in kv.items() if k.startswith("record.")}
                        for p, kv in (pre.get("pods") or {}).items()},
           "not_established": ["performance beyond this one launch (no replication, no comparison)",
                               "framework binding (Megatron-LM, NeMo RL, TRT-LLM) on this library",
                               "EFA-GDA (NCCL_GIN_TYPE=5)", "LL mode (not exercised)", "any customer workload"]}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"verdict": out["verdict"], "gates": gates}))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gpu-requests")
    p = sub.add_parser("pods")
    p.add_argument("--dir", required=True)
    p.add_argument("--p0", required=True)
    p.add_argument("--p1", required=True)
    p = sub.add_parser("preflight")
    for k in ("--dir", "--p0", "--p1"):
        p.add_argument(k, required=True)
    p.add_argument("--gpus", type=int, required=True)
    p.add_argument("--efa", type=int, required=True)
    p.add_argument("--build-in-pod", type=int, default=0)
    for name in ("correctness", "timing", "summarize"):
        p = sub.add_parser(name)
        p.add_argument("--dir", required=True)
        p.add_argument("--world", type=int, default=16)
        p.add_argument("--expect-argv", default="-a ht -L fl")
        if name == "summarize":
            p.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"gpu-requests": cmd_gpu_requests, "pods": cmd_pods, "preflight": cmd_preflight,
            "correctness": cmd_correctness, "timing": cmd_timing, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
