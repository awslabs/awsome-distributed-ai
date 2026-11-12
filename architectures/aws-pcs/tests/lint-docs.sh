#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Docs ⇄ template consistency lint for the AWS PCS reference architecture.
#
# Catches the most common drift after a parameter rename/removal: docs (and
# other templates) still referencing an old parameter name, a removed parameter
# presented as current, or a known-stale phrase. Run from anywhere; paths are
# resolved relative to this script's location (architectures/aws-pcs/).
#
#   bash architectures/aws-pcs/tests/lint-docs.sh [path/to/ws-pcs-cluster.yaml]
# GPU contract checks require Python 3 and PyYAML (also used by publish staging).
#
# Exit code is non-zero if any check fails, so it can gate a PR in CI.
#
# When you intentionally rename/remove a parameter, update the BANNED list below
# in the same change — that is the point: the lint forces docs to keep up.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # architectures/aws-pcs
cd "$ROOT"

# Files that describe the user-facing interface (must not mention removed params).
DOC_GLOBS=(README.md docs/*.md tests/*.md)

fail=0
report() { echo "FAIL: $1"; fail=1; }

# 1. Removed / renamed parameters must not appear in docs as if current.
#    Each entry is TAB-separated: <banned-regex>\t<allowed-context-regex>\t<message>.
#    A hit is a failure UNLESS the line also matches the allowed-context regex
#    (used to permit explicit "(renamed from X)" / "→" / "internally" history
#    notes). Use NEVERMATCH as the allow-regex when no exception applies.
BANNED=(
  $'OnDemandEnableEfa\tremoved\tOnDemandEnableEfa was replaced by OnDemandEfaInterfaceCount (0/1/2)'
  $'GrafanaPublicAccessCidr\t[Rr]enamed|→ ?`?GrafanaAccessCidr\tGrafanaPublicAccessCidr was renamed to GrafanaAccessCidr'
  $'DeployMonitoring=true\tMonitoringStack|internally|nested\tDeployMonitoring (bool) was replaced by MonitoringStack at the deploy-all layer'
  $'S3 (public )?hosting is not allowed\tNEVERMATCH\tthe Enroot/Pyxis installer is fetched from S3 by the PCS agent'
  $'PostInstallScript(Url|Args)\t[Rr]enamed|replaced\tPostInstallScriptUrl/Args were replaced by InstallEnrootPyxis (custom scripts: use PCS node lifecycle actions directly)'
  $'architectures/aws-pcs/iam/\tNEVERMATCH\tthe iam/ directory was removed; use docs/IAM.md + assets/cluster-*-iam.yaml'
  $'aws:pcs:compute-node-group-name\tNEVERMATCH\ttag key does not exist — use `aws pcs list-compute-node-groups` + `tag:aws:pcs:compute-node-group-id`'
  $'Name=tag:Name,Values=PCS-\tNEVERMATCH\tthe Name tag is <ClusterName>-<CngName>; resolve the login node via the PCS API instead'
)
for entry in "${BANNED[@]}"; do
  IFS=$'\t' read -r pat allow msg <<<"$entry"
  hits=$(grep -rnE "$pat" "${DOC_GLOBS[@]}" 2>/dev/null | grep -vE "$allow" || true)
  if [ -n "$hits" ]; then
    report "stale reference ($msg):"
    echo "$hits" | sed 's/^/    /'
  fi
done


# 3. Every parameter the deploy-all template declares should be documented in
#    PARAMETERS.md (catches a new param added without a docs row).
params=$(awk '/^Parameters:/{p=1;next} /^[A-Za-z]/{p=0} p&&/^  [A-Z][A-Za-z0-9]+:/{gsub(/[: ]/,"");print}' assets/pcs-ml-cluster-deploy-all.yaml | sort -u)
for prm in $params; do
  grep -q "\`$prm\`" docs/PARAMETERS.md || report "deploy-all parameter '$prm' is not documented in docs/PARAMETERS.md"
done

# 4. The NodeLifecycleActions block (needrestart guard + FSx mounts) must be
#    byte-identical across the four CNG templates. It is hand-duplicated (no
#    shared include), so an edit that lands in only some of the copies is
#    exactly the drift this catches.
lifecycle_extract() {  # print the block: NodeLifecycleActions through the end of the resource
  # Leading whitespace is stripped because the four templates legitimately nest
  # the block at different depths. The block ends where the next 2-space-indented
  # resource ID starts (NodeLifecycleActions is the last property of the CNG).
  # Side effect: the check cannot see RELATIVE indentation drift inside the block.
  awk '/^      NodeLifecycleActions:/{p=1; print; next}
       p&&/^  [A-Za-z0-9]+:/{exit}
       p{print}' "$1" | sed -E 's/^[[:space:]]+//'
}
ref=$(lifecycle_extract assets/add-cng.yaml)
if [ -z "$ref" ]; then
  report "NodeLifecycleActions block not found in assets/add-cng.yaml"
else
  for t in add-cng-p5 add-cng-p6-b200 add-cng-p6-b300; do
    other=$(lifecycle_extract "assets/$t.yaml")
    if [ "$other" != "$ref" ]; then
      report "NodeLifecycleActions block in assets/$t.yaml differs from assets/add-cng.yaml (keep the four copies byte-identical):"
      diff <(printf '%s\n' "$ref") <(printf '%s\n' "$other") | sed 's/^/    /'
    fi
  done
fi

# 5. Every lifecycle script referenced by a template OR by another boot script
#    must exist in assets/scripts/ (a typo'd ScriptLocation only fails at node
#    boot, which costs a 30-minute deploy to discover) — AND be listed in the
#    publish manifest. The manifest is an explicit allowlist: a script
#    missing from it never reaches the production bucket, so post-merge
#    deploys would fail at the agent's script download (mount scripts
#    TERMINATE — a node replace loop). Scan the scripts too, not just the
#    templates: ldap-add-user.sh is fetched at boot by setup-directory.sh, not
#    by a ScriptLocation, so a template-only glob would miss it. Skipped when
#    the manifest is absent (e.g. a partial checkout).
MANIFEST="../../.github/template-publish-manifest.yml"
for scr in $(grep -hoE 'scripts/[a-z0-9-]+\.sh' assets/*.yaml assets/scripts/*.sh | sort -u); do
  [ -f "assets/$scr" ] || report "lifecycle script assets/$scr is referenced by a template or boot script but does not exist"
  if [ -f "$MANIFEST" ]; then
    grep -q "key: aws-pcs/$scr" "$MANIFEST" \
      || report "assets/$scr is referenced by a template or boot script but missing from .github/template-publish-manifest.yml (would not be published to the production bucket)"
  fi
done

# 6. Same-file Markdown anchor links in README.md resolve to a real heading.
#    (Cross-file and external links are out of scope — kept simple on purpose.)
while IFS= read -r anchor; do
  # build the set of heading slugs in README
  slugs=$(grep -E '^#{1,6} ' README.md \
    | sed -E 's/^#{1,6} //; s/`//g' \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9 -]//g; s/ /-/g')
  grep -qx "$anchor" <<<"$slugs" || report "README.md internal anchor '#$anchor' has no matching heading"
done < <(grep -oE '\]\(#[a-z0-9-]+\)' README.md | sed -E 's/\]\(#//; s/\)//' | sort -u)

# 7. Nested GPU caller contracts and evaluated launch options (local, no AWS).
# PyYAML is also required by the existing template-publish staging check.
python3 - "${1:-}" <<'PY' || report "GPU nested-template contract regression"
from pathlib import Path
import itertools
import sys
import yaml

class CfnLoader(yaml.SafeLoader):
    pass

def intrinsic(loader, tag, node):
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    else:
        value = loader.construct_sequence(node, deep=True)
    return {tag if tag in ('Ref', 'Condition') else 'Fn::' + tag: value}

CfnLoader.add_multi_constructor('!', intrinsic)
def load(path):
    return yaml.load(Path(path).read_text(), Loader=CfnLoader)

def evaluate(value, params, conditions):
    if isinstance(value, list):
        return [evaluate(v, params, conditions) for v in value]
    if not isinstance(value, dict):
        return value
    if len(value) == 1:
        key, arg = next(iter(value.items()))
        if key == 'Ref':
            return params.get(arg, arg)
        if key == 'Condition':
            return evaluate(conditions[arg], params, conditions)
        if key == 'Fn::If':
            branch = 1 if evaluate(conditions[arg[0]], params, conditions) else 2
            return evaluate(arg[branch], params, conditions)
        if key in ('Fn::Equals', 'Fn::Not', 'Fn::And', 'Fn::Or'):
            args = evaluate(arg, params, conditions)
            return (args[0] == args[1] if key == 'Fn::Equals' else
                    not args[0] if key == 'Fn::Not' else
                    any(args) if key == 'Fn::Or' else all(args))
    return {k: evaluate(v, params, conditions) for k, v in value.items()}

parent = load('assets/pcs-ml-cluster-deploy-all.yaml')
parameters = parent['Parameters']
assert parameters['PseriesInstanceType']['Default'] == 'p5.48xlarge'
assert parameters['CapacityReservationType']['Default'] == 'capacity-block'
assert parameters['PseriesPlacementGroupName']['Default'] == ''
expected = {
    'p5.48xlarge': ('PseriesCNGStack', 32),
    'p5e.48xlarge': ('PseriesCNGStack', 32),
    'p5en.48xlarge': ('PseriesCNGStack', 16),
    'p6-b200.48xlarge': ('P6B200CNGStack', 8),
    'p6-b300.48xlarge': ('P6B300CNGStack', 17),
    'g7.48xlarge': ('G7CNGStack', 2),
}
assert set(parameters['PseriesInstanceType']['AllowedValues']) == set(expected)
# Validate every nested call, including login, CPU, prerequisites and cluster.
for name, resource in parent['Resources'].items():
    if resource['Type'] != 'AWS::CloudFormation::Stack':
        continue
    props = resource['Properties']
    filename = props['TemplateURL']['Fn::Sub'].split('${S3KeyPrefix}')[1]
    child = load('assets/' + filename)
    assert not (set(props['Parameters']) - set(child['Parameters'])), name
    assert all('Default' in spec or key in props['Parameters']
               for key, spec in child['Parameters'].items()), name

# Optionally check a real external caller, without making Workshop assets a dependency.
if sys.argv[1]:
    caller = load(sys.argv[1])
    call = caller['Resources']['PcsMlCluster']['Properties']['Parameters']
    assert not (set(call) - set(parameters)), 'external caller names'
    defaults = {k: v.get('Default') for k, v in caller['Parameters'].items()}
    for key, value in evaluate(call, defaults, caller.get('Conditions', {})).items():
        if 'AllowedValues' in parameters[key]:
            assert value in parameters[key]['AllowedValues'], (key, value)
    for output in caller['Outputs'].values():
        value = output.get('Value', {})
        if isinstance(value, dict) and 'Fn::GetAtt' in value:
            target = value['Fn::GetAtt']
            if isinstance(target, str) and target.startswith('PcsMlCluster.Outputs.'):
                assert target.split('.')[-1] in parent['Outputs'], target
    print('External Workshop caller parameters/defaults/outputs: PASS')

# Generic login/CPU defaults still omit reservation, market and purchase settings.
generic = load('assets/add-cng.yaml')
gp = {k: v.get('Default') for k, v in generic['Parameters'].items()}
gc = generic['Conditions']
glt = evaluate(generic['Resources']['PCSLaunchTemplate']['Properties']['LaunchTemplateData'], gp, gc)
assert all(glt.get(k, 'AWS::NoValue') == 'AWS::NoValue' for k in ('InstanceMarketOptions', 'CapacityReservationSpecification', 'Placement', 'NetworkInterfaces'))
assert evaluate(generic['Resources']['PCSNodeGroupCompute']['Properties'].get('PurchaseOption', 'AWS::NoValue'), gp, gc) == 'AWS::NoValue'
for count in (1, 2):
    gp['EfaInterfaceCount'] = count
    nics = evaluate(generic['Resources']['PCSLaunchTemplate']['Properties']['LaunchTemplateData']['NetworkInterfaces'], gp, gc)
    nics = [n for n in nics if n != 'AWS::NoValue']
    assert len(nics) == count and all(n['InterfaceType'] == 'efa' for n in nics)

# Keep targeted CPU ODCR and explicit placement independent of EFA (#1267).
assert generic['Parameters']['CapacityReservationType']['Default'] == 'targeted-odcr'
for count, placement, reservation in itertools.product((0, 1, 2), ('', 'existing-pg'), ('', 'cr-test')):
    cp = {k: v.get('Default') for k, v in generic['Parameters'].items()}
    cp.update(EfaInterfaceCount=count, PlacementGroupName=placement, CapacityReservationId=reservation)
    lt = evaluate(generic['Resources']['PCSLaunchTemplate']['Properties']['LaunchTemplateData'], cp, gc)
    assert lt.get('InstanceMarketOptions', 'AWS::NoValue') == 'AWS::NoValue'
    assert lt['CapacityReservationSpecification'] == ({'CapacityReservationTarget': {'CapacityReservationId': reservation}} if reservation else 'AWS::NoValue')
    assert lt['Placement'] == ({'GroupName': placement or 'PlacementGroup'} if count or placement else 'AWS::NoValue')
    assert evaluate(gc['CreatePlacementGroup'], cp, gc) == bool(count and not placement)

cases = 0
for instance, enabled, reservation, mode, placement in itertools.product(
        expected, ('true', 'false'), ('', 'cr-test'),
        ('capacity-block', 'targeted-odcr'), ('', 'existing-pg')):
    params = {k: v.get('Default') for k, v in parameters.items()}
    params.update(PseriesInstanceType=instance, DeployPseriesCNG=enabled,
                  CapacityReservationId=reservation, CapacityReservationType=mode,
                  PseriesPlacementGroupName=placement)
    cond = parent['Conditions']
    selected = [name for name in {v[0] for v in expected.values()}
                if evaluate(cond[parent['Resources'][name]['Condition']], params, cond)]
    assert selected == ([] if enabled == 'false' else [expected[instance][0]])
    rule = parent['Rules']['G7ReservationType']
    accepted = (not evaluate(rule['RuleCondition'], params, cond) or
                evaluate(rule['Assertions'][0]['Assert'], params, cond))
    assert accepted == (not (enabled == 'true' and instance == 'g7.48xlarge'
                            and reservation and mode == 'capacity-block'))
    if not selected or not accepted:
        continue
    call = parent['Resources'][selected[0]]['Properties']
    child = load('assets/' + call['TemplateURL']['Fn::Sub'].split('${S3KeyPrefix}')[1])
    cp = {k: v.get('Default') for k, v in child['Parameters'].items()}
    cp.update(evaluate(call['Parameters'], params, cond))
    assert cp['CapacityReservationType'] == mode
    assert cp['PlacementGroupName'] == placement
    cc = child['Conditions']
    lt = evaluate(child['Resources']['PCSLaunchTemplate']['Properties']['LaunchTemplateData'], cp, cc)
    block = bool(reservation and mode == 'capacity-block')
    assert lt.get('InstanceMarketOptions', 'AWS::NoValue') == ({'MarketType': 'capacity-block'} if block else 'AWS::NoValue')
    assert evaluate(child['Resources']['PCSNodeGroupCompute']['Properties'].get('PurchaseOption', 'AWS::NoValue'), cp, cc) == ('CAPACITY_BLOCK' if block else 'AWS::NoValue')
    assert lt['CapacityReservationSpecification'] == ({'CapacityReservationTarget': {'CapacityReservationId': reservation}} if reservation else 'AWS::NoValue')
    assert lt['Placement'] == ('AWS::NoValue' if block else {'GroupName': placement or 'PlacementGroup'})
    assert evaluate(cc['CreatePlacementGroup'], cp, cc) == (not block and not placement)
    nics = [nic for nic in lt['NetworkInterfaces'] if nic != 'AWS::NoValue']
    assert len(nics) == expected[instance][1], instance
    if instance == 'g7.48xlarge':
        assert [(nic['NetworkCardIndex'], nic['DeviceIndex'], nic['InterfaceType']) for nic in nics] == [(0, 0, 'efa'), (1, 1, 'efa-only')]
        assert nics[1]['SubnetId'] == 'AWS::NoValue'
    cases += 1
print(f'GPU caller/launch contracts: PASS ({cases} enabled valid combinations; disabled and rejected combinations checked)')
PY

if [ "$fail" -eq 0 ]; then
  echo "docs lint: PASS (no stale parameter references, all deploy-all params documented, NodeLifecycleActions in lock-step, lifecycle scripts exist, README anchors resolve)"
fi
exit $fail
