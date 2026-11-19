# Regression fixtures

The three `slurm-*` files are Slurm 25.05.9-shaped regression inputs with
**synthetic identifiers**, not raw operational records or hardware evidence.
Instance, boot, operation, account, reservation, AMI, volume, network, placement,
profile and resource identifiers are test values. IPv4 addresses use the
RFC 5737 documentation range `192.0.2.0/24`; MAC addresses are locally
administered test values. Names and identity relationships are kept consistent
across the node rows, dispatching state and consumer tests.

The Reason body, serializer annotations, state flags, field order and recovery
relationships preserve the parser and replacement regressions. Timestamps and
counters are fixture data, not claims of a live deployment. These files contain
no source-to-test identifier map, private archive or operational provenance.

`enroot-3.5.0-50-slurm-pmi.sh` retains its upstream copyright and license notice.
