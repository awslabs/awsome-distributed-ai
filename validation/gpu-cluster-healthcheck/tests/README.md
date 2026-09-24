# Check 5 regression fixtures

Run `python3 -B -m unittest discover -s tests -v` from the suite directory.
The existing command-mock bench tests the actual shell entrypoints without
launching Slurm, MPI, containers, or GPU work.

`nccl-captured-252.txt` and `nccl-captured-294.txt` contain selected lines from
two captured nccl-tests runs: the table header, all result rows, the completion
summary, and provider-selection statements. Numeric results are unchanged; trailing whitespace in the header is trimmed.
Hostnames and process/thread identifiers were removed by keeping only the
`NCCL INFO NET/OFI` suffix of each provider line; all other diagnostic lines
were omitted. Private originals and provenance remain outside this checkout.
These excerpts test parsing, not hardware qualification or rank attribution.

`nccl-valid-rows.txt` is the pre-existing launch-test fixture. The test bench
adds a source-format completion summary to this command mock. Mutation tests
explicitly alter captured fields to represent invalid inputs; those mutations
are not measured results. The large-output regression adds labeled mock log
padding to reproduce the early-reader SIGPIPE condition without publishing
private diagnostic logs.

The default table and completion format is independently defined by
[nccl-tests v2.18.3 util.cu](https://github.com/NVIDIA/nccl-tests/blob/v2.18.3/src/util.cu):
`writeBenchmarkLineBody`, `writeResultHeader`, and `writeResultFooter`.
Disabled correctness prints `N/A`, not evidence of zero errors.

## Check 6 counter diagnostics

The same command-mock bench also invokes Check 6 with synthetic `fi_info`,
`fi_pingpong`, and `rdma` output; it never exercises an EFA device. The fixtures
cover all-returned-link absolute totals, multiline link output, missing/malformed
counters, duplicate or invalid links, u64 totals, and unavailable statistics.
The malformed-counter method includes the independent review's exact trailing
`link` and `link  ` records, missing identities, broken headers and counter fields,
and malformed sections before, between, or after good records. It requires an
unknown warning with no clean marker. Every nonblank line must be a complete
header and/or counter-name/u64 pairs; fields cannot borrow a missing identity
or value from a later line. Any malformed record invalidates both totals.
Valid indented/multiline records, blank lines, CRLF, and other u64 counters remain
accepted; only the two monitored counters contribute to their absolute totals.
Separate executions with rising, falling, equal, or wrap-shaped values verify
that no before/after delta or driver-reset epoch is inferred. They are not
measured before/after snapshots and do not establish per-interface pinning.
Nonzero absolute totals continue to warn; incomplete counters are unknown,
not zero. This local composition includes the owner changes from #1270
(result accumulation) and the Check 6-only portion of #1221 (device pinning).
The existing mock now rejects unknown FI_EFA_IFACE values, records both sides'
selected interface and SHM setting, and can deliberately ignore the pin to
verify the owner's negative-control rejection. These are mock contract tests,
not proof of physical per-interface traffic. Saved JSON must preserve WARN
after a later PASS, including every malformed-counter case. The owner's
published severity test sequence also exercises the real existing aggregate
and Kubernetes consumers, richer DCGM fields, and per-execution reset. No
runtime implementation or new test framework is added beyond the owner patches.
