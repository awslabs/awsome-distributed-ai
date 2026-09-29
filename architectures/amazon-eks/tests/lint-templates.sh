#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Mechanical checks for the amazon-eks CloudFormation deploy path.
#
#   bash architectures/amazon-eks/tests/lint-templates.sh
#
# Every check prints one line per problem saying which file failed, what was
# expected and what was found; the run ends with a count and exits non-zero if
# any check failed.
#
# Checks 1 and 5 call the AWS API and SKIP without credentials, so the same
# script runs in CI, where the lint step runs before credentials are
# configured, and on a laptop. A SKIP is never counted as a pass: skips are
# counted and printed separately.
#
# Chart versions are read out of the templates, never hardcoded here — a lint
# that carries its own copy of a pinned version stops checking the moment the
# template moves on. If a version cannot be read, the check fails; it never
# falls back to a default.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # architectures/amazon-eks
cd "$ROOT"

PREREQ_T="assets/eks-cluster-prerequisites.yaml"
CLUSTER_T="assets/eks-cluster.yaml"
GPU_T="assets/eks-add-gpu-nodegroup.yaml"
ROOT_T="assets/eks-gpu-cluster-deploy-all.yaml"
AMI_T="assets/eks-gpu-node-ami.yaml"
ALL_T=("$PREREQ_T" "$CLUSTER_T" "$GPU_T" "$ROOT_T" "$AMI_T")
PUBLISH_MANIFEST="../../.github/template-publish-manifest.yml"
RENDER="tests/render-nic-block.py"
PARAMS_DOC="docs/PARAMETERS.md"

# Instance types NicLayout has to carry, and the
# Kubernetes-visible values the templates and the test procedures share.
REQUIRED_NIC_TYPES=(
  g7e.12xlarge g7e.24xlarge g7e.48xlarge
  p4d.24xlarge p4de.24xlarge
  p5.48xlarge p5en.48xlarge
  p6-b200.48xlarge p6-b300.48xlarge
  g6e.12xlarge g6e.48xlarge
  g5.12xlarge g4dn.8xlarge
)
NIC_KEYS=(Cards PrimaryEfa SecondaryDeviceIndex EfaInterfaces)
GPU_TAINT_KEY="nvidia.com/gpu"

FAILURES=0
SKIPS=0

fail() { printf 'FAIL  [%s] %s\n' "$1" "$2"; FAILURES=$((FAILURES + 1)); }
skip() { printf 'SKIP  [%s] %s\n' "$1" "$2"; SKIPS=$((SKIPS + 1)); }
pass() { printf 'PASS  [%s] %s\n' "$1" "$2"; }
head_() { printf '\n=== check %s: %s\n' "$1" "$2"; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# --------------------------------------------------------------------------
# Interpreter discovery. The YAML checks need PyYAML; the text checks do not.
# --------------------------------------------------------------------------
PY_ANY=""
for cand in "${PYTHON:-python3}" python3 python; do
  [ -n "$cand" ] || continue
  if command -v "$cand" >/dev/null 2>&1; then PY_ANY="$cand"; break; fi
done
PY_YAML=""
for cand in "${PYTHON:-python3}" python3 python3.13 python3.12 python3.11 python; do
  [ -n "$cand" ] || continue
  command -v "$cand" >/dev/null 2>&1 || continue
  if "$cand" -c 'import yaml' >/dev/null 2>&1; then PY_YAML="$cand"; break; fi
done

CFNQ="$TMP/cfnq.py"
TEXTQ="$TMP/textq.py"

# --------------------------------------------------------------------------
# cfnq.py — reads a CloudFormation template. PyYAML's SafeLoader rejects the
# short intrinsic tags (!Ref, !If, ...), so the loader below accepts every
# "!Something" tag and keeps its argument. Exit 3 = no PyYAML, 4 = the section
# the caller asked for is missing.
# --------------------------------------------------------------------------
cat > "$CFNQ" <<'PYEOF'
import sys

try:
    import yaml
except ImportError:
    sys.exit(3)


class CfnLoader(yaml.SafeLoader):
    pass


def _keep_tag(loader, tag_suffix, node):
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:
        value = loader.construct_mapping(node, deep=True)
    return {tag_suffix if tag_suffix == "Ref" else "Fn::" + tag_suffix: value}


CfnLoader.add_multi_constructor("!", _keep_tag)


def load(path):
    with open(path, encoding="utf-8") as handle:
        doc = yaml.load(handle, Loader=CfnLoader)
    if not isinstance(doc, dict):
        sys.stderr.write("%s: not a YAML mapping\n" % path)
        sys.exit(4)
    return doc


def section(doc, path, name):
    part = doc.get(name)
    if not isinstance(part, dict):
        sys.stderr.write("%s: no %s section\n" % (path, name))
        sys.exit(4)
    return part


cmd = sys.argv[1]
path = sys.argv[2]
doc = load(path)

if cmd == "params":
    for name in section(doc, path, "Parameters"):
        print(name)

elif cmd == "allowed":
    param = section(doc, path, "Parameters").get(sys.argv[3])
    if not isinstance(param, dict):
        sys.stderr.write("%s: no parameter %s\n" % (path, sys.argv[3]))
        sys.exit(4)
    values = param.get("AllowedValues")
    if not isinstance(values, list):
        sys.stderr.write("%s: parameter %s has no AllowedValues list\n" % (path, sys.argv[3]))
        sys.exit(4)
    for value in values:
        print(value)

elif cmd == "default":
    param = section(doc, path, "Parameters").get(sys.argv[3], {})
    print(param.get("Default", ""))

elif cmd == "resolve":
    # The literal this template gives a name, through the paths a version can
    # legitimately travel: a parameter default, or a CodeBuild environment
    # variable whose value is a Mappings lookup. Anything else (a Ref to another
    # parameter, a Sub, a nested intrinsic) is reported as unresolvable rather
    # than guessed, because the whole point of reading the template is to test
    # the version it really installs.
    name = sys.argv[3]

    def find_map(value):
        if not (isinstance(value, dict) and list(value) == ["Fn::FindInMap"]):
            return None
        keys = value["Fn::FindInMap"]
        if not (isinstance(keys, list) and len(keys) == 3):
            return None
        if not all(isinstance(k, str) for k in keys):
            return None                      # e.g. !FindInMap [M, !Ref P, K]
        table = (doc.get("Mappings") or {}).get(keys[0]) or {}
        entry = table.get(keys[1])
        if isinstance(entry, dict) and isinstance(entry.get(keys[2]), (str, int, float)):
            return str(entry[keys[2]])
        return None

    param = (doc.get("Parameters") or {}).get(name)
    if isinstance(param, dict) and param.get("Default") not in (None, ""):
        print(param["Default"])
        raise SystemExit(0)

    def walk(node):
        if isinstance(node, dict):
            variables = None
            environment = node.get("Environment")
            if isinstance(environment, dict):
                variables = environment.get("EnvironmentVariables")
            if isinstance(variables, list):
                for item in variables:
                    if not isinstance(item, dict) or item.get("Name") != name:
                        continue
                    value = item.get("Value")
                    if isinstance(value, (str, int, float)):
                        return str(value)
                    literal = find_map(value)
                    if literal is not None:
                        return literal
            for child in node.values():
                found = walk(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = walk(child)
                if found is not None:
                    return found
        return None

    literal = walk(doc.get("Resources") or {})
    if literal is None:
        sys.stderr.write("%s: cannot resolve %s to a literal\n" % (path, name))
        sys.exit(4)
    print(literal)

elif cmd == "mapping":
    mappings = section(doc, path, "Mappings")
    table = mappings.get(sys.argv[3])
    if not isinstance(table, dict):
        sys.stderr.write("%s: no mapping %s (present: %s)\n"
                         % (path, sys.argv[3], ", ".join(sorted(mappings)) or "none"))
        sys.exit(4)
    for key in sorted(table):
        entry = table[key]
        if not isinstance(entry, dict):
            print("%s\t!not-a-mapping\t%r\t%s" % (key, entry, type(entry).__name__))
            continue
        for sub in sorted(entry):
            print("%s\t%s\t%s\t%s" % (key, sub, entry[sub], type(entry[sub]).__name__))

elif cmd == "mapping-names":
    for name in sorted(doc.get("Mappings") or {}):
        print(name)

elif cmd == "outputs":
    for name in section(doc, path, "Outputs"):
        print(name)

else:
    sys.stderr.write("unknown command %s\n" % cmd)
    sys.exit(2)
PYEOF

# --------------------------------------------------------------------------
# textq.py — the checks that read the files as text, so they work without
# PyYAML: the generated NetworkInterfaces block, Markdown links and anchors,
# and the version-pinning scan. Every subcommand prints one problem per line
# and exits 1 when it found problems, 2 on a usage/structural error.
# --------------------------------------------------------------------------
cat > "$TEXTQ" <<'PYEOF'
import os
import re
import sys

BEGIN = re.compile(r"#\s*BEGIN\s+(?:GENERATED|generated)\b|#\s*BEGIN\b.*render-nic-block\.py", re.I)
END = re.compile(r"#\s*END\s+(?:GENERATED|generated)\b|#\s*END\b.*render-nic-block\.py", re.I)


def read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def nicblock(path):
    """Print the committed NetworkInterfaces block, with and without its key line.

    Two forms are printed, with and without the `NetworkInterfaces:` key, and
    the caller accepts an exact match against either. Preferred is a marked
    region:

        # BEGIN render-nic-block.py
        ...
        # END render-nic-block.py
    """
    lines = read(path).splitlines()
    begin = [i for i, line in enumerate(lines) if BEGIN.search(line)]
    end = [i for i, line in enumerate(lines) if END.search(line)]
    if begin and end:
        if len(begin) != 1 or len(end) != 1 or end[0] < begin[0]:
            sys.stderr.write("%s: render-nic-block.py BEGIN/END markers are not a single "
                             "ordered pair (BEGIN at %s, END at %s)\n"
                             % (path, begin, end))
            return 2
        block = lines[begin[0] + 1:end[0]]
        sys.stdout.write("MARKED\n")
    else:
        starts = [i for i, line in enumerate(lines)
                  if re.match(r"^\s*NetworkInterfaces:\s*$", line)]
        if not starts:
            sys.stderr.write("%s: no `NetworkInterfaces:` key and no render-nic-block.py "
                             "BEGIN/END markers found\n" % path)
            return 2
        if len(starts) > 1:
            sys.stderr.write("%s: `NetworkInterfaces:` appears %d times (lines %s) — the "
                             "comparison needs one block; mark the generated region with "
                             "`# BEGIN render-nic-block.py` / `# END render-nic-block.py`\n"
                             % (path, len(starts), [i + 1 for i in starts]))
            return 2
        start = starts[0]
        indent = len(lines[start]) - len(lines[start].lstrip())
        block = [lines[start]]
        for line in lines[start + 1:]:
            if line.strip() and (len(line) - len(line.lstrip())) <= indent:
                break
            block.append(line)
        while block and not block[-1].strip():
            block.pop()
        sys.stdout.write("KEYED\n")
    with open(sys.argv[3], "w", encoding="utf-8") as handle:      # with the key line
        handle.write("\n".join(block) + "\n")
    body = block[1:] if re.match(r"^\s*NetworkInterfaces:\s*$", block[0]) else block
    with open(sys.argv[4], "w", encoding="utf-8") as handle:      # without the key line
        handle.write("\n".join(body) + "\n")
    return 0


def slugs(path):
    """GitHub heading anchors for a Markdown file, including duplicate suffixes."""
    out = []
    seen = {}
    in_fence = False
    for line in read(path).splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line.startswith("#"):
            continue
        text = re.sub(r"^#{1,6}\s+", "", line)
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)      # links -> their text
        text = text.replace("`", "").replace("*", "").replace("_", "_")
        slug = re.sub(r"[^\w\- ]", "", text.lower(), flags=re.UNICODE).strip()
        slug = re.sub(r"\s+", "-", slug)
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        out.append(slug if count == 0 else "%s-%d" % (slug, count))
    return out


LINK = re.compile(r"(?<!\\)\[[^\]]*\]\(\s*([^)\s]+)(?:\s+\"[^\"]*\")?\s*\)")


def links(paths):
    problems = 0
    cache = {}
    for path in paths:
        base = os.path.dirname(path)
        in_fence = False
        for number, line in enumerate(read(path).splitlines(), 1):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            for target in LINK.findall(line):
                if re.match(r"^(https?:|mailto:|ftp:|tel:)", target) or "${" in target:
                    continue
                relpath, _, anchor = target.partition("#")
                if relpath:
                    resolved = os.path.normpath(os.path.join(base, relpath))
                    if not os.path.exists(resolved):
                        print("%s:%d: link target `%s` resolves to %s, which does not exist"
                              % (path, number, target, resolved))
                        problems += 1
                        continue
                else:
                    resolved = path
                if not anchor:
                    continue
                if not resolved.endswith(".md"):
                    continue
                if resolved not in cache:
                    cache[resolved] = slugs(resolved)
                if anchor.lower() not in cache[resolved]:
                    print("%s:%d: anchor `#%s` has no matching heading in %s"
                          % (path, number, anchor, resolved))
                    problems += 1
    return 1 if problems else 0


UNPINNED = [
    (re.compile(r":latest(?![\w.-])"), "image or chart reference tagged `latest`"),
    (re.compile(r"/latest(?=[/\"'\s]|$)"), "`latest` in a download path — pin the version"),
    (re.compile(r"stable\.txt"), "version resolved from `stable.txt` at run time — pin it"),
    (re.compile(r"--version[= ]\s*(latest|stable)"), "`--version latest` is not a pin"),
]
IMAGE_KEY = re.compile(r"^\s*[Ii]mage:\s*(\S.*?)\s*$")
HELM_INSTALL = re.compile(r"helm\s+(?:upgrade|install)\b")


def pins(paths):
    problems = 0
    for path in paths:
        raw = read(path).splitlines()
        # Join backslash continuations so a helm command split over lines is
        # scanned as one command, keeping the first line's number.
        logical = []
        buffer, first = "", None
        for number, line in enumerate(raw, 1):
            if first is None:
                first = number
            stripped = line.rstrip()
            if stripped.endswith("\\"):
                buffer += stripped[:-1] + " "
                continue
            logical.append((first, buffer + stripped))
            buffer, first = "", None
        if buffer:
            logical.append((first, buffer))
        for number, line in logical:
            # Prose about a command is not the command. A comment naming helm or an image tag would
            # otherwise have to be written around the check rather than for the reader.
            if line.lstrip().startswith("#"):
                continue
            # A documentation URL is not a download. AWS service guides live under a literal
            # /latest/ path segment, which is the current documentation rather than a version to pin.
            line = re.sub(r"https://docs\.aws\.amazon\.com/\S+", "", line)
            for pattern, why in UNPINNED:
                if pattern.search(line):
                    print("%s:%d: %s: %s" % (path, number, why, line.strip()[:160]))
                    problems += 1
            if HELM_INSTALL.search(line) and "--version" not in line:
                print("%s:%d: `helm upgrade/install` without `--version`: %s"
                      % (path, number, line.strip()[:160]))
                problems += 1
            found = IMAGE_KEY.match(line)
            if found:
                value = found.group(1)
                if value.startswith(("!", "{", "$", "'${", '"${')) or "${" in value:
                    continue
                bare = value.strip("'\"")
                if "@sha256:" in bare:
                    continue
                tail = bare.rsplit("/", 1)[-1]
                if ":" not in tail:
                    print("%s:%d: image `%s` has no tag or digest" % (path, number, bare))
                    problems += 1
    return 1 if problems else 0


cmd = sys.argv[1]
if cmd == "nicblock":
    sys.exit(nicblock(sys.argv[2]))
elif cmd == "links":
    sys.exit(links(sys.argv[2:]))
elif cmd == "pins":
    sys.exit(pins(sys.argv[2:]))
sys.stderr.write("unknown command %s\n" % cmd)
sys.exit(2)
PYEOF

printf 'lint-templates.sh — %s\n' "$ROOT"

# --------------------------------------------------------------------------
# check 0 — the files the other checks read
# --------------------------------------------------------------------------
head_ 0 "expected files are present"
for f in "${ALL_T[@]}" "$RENDER" "$PARAMS_DOC" README.md; do
  if [ -f "$f" ]; then pass 0 "$f"; else fail 0 "$f: expected to exist, not found"; fi
done
if [ -z "$PY_ANY" ]; then
  fail 0 "no python3 on PATH: checks 2 to 10 cannot run"
elif [ -z "$PY_YAML" ]; then
  fail 0 "no python3 with PyYAML on PATH (install: python3 -m pip install pyyaml): the checks that parse the templates cannot run"
else
  pass 0 "$PY_YAML can import yaml"
fi

cfnq() { "$PY_YAML" "$CFNQ" "$@"; }

# --------------------------------------------------------------------------
# check 1 — aws cloudformation validate-template on every template
# --------------------------------------------------------------------------
head_ 1 "aws cloudformation validate-template on every template"
HAVE_AWS=no
if ! command -v aws >/dev/null 2>&1; then
  skip 1 "aws CLI not on PATH"
elif ! aws sts get-caller-identity >/dev/null 2>&1; then
  skip 1 "no usable AWS credentials (validate-template is an authenticated call)"
else
  HAVE_AWS=yes
fi
if [ "$HAVE_AWS" = yes ]; then
  for t in "${ALL_T[@]}"; do
    [ -f "$t" ] || { skip 1 "$t: file missing"; continue; }
    size=$(wc -c < "$t" | tr -d ' ')
    if [ "$size" -gt 51200 ]; then
      skip 1 "$t: $size bytes exceeds the 51,200-byte --template-body limit; validate with --template-url from S3"
      continue
    fi
    if out=$(aws cloudformation validate-template --template-body "file://$t" 2>&1); then
      pass 1 "$t"
    else
      fail 1 "$t: validate-template rejected it: $(printf '%s' "$out" | tr '\n' ' ' | cut -c1-200)"
    fi
  done
fi

# --------------------------------------------------------------------------
# check 2 — GpuInstanceType.AllowedValues, NicLayout and GpuCount agree
# --------------------------------------------------------------------------
head_ 2 "GpuInstanceType, NicLayout and GpuCount cover the same types, with all keys"
if [ -z "$PY_YAML" ] || [ ! -f "$GPU_T" ]; then
  skip 2 "needs PyYAML and $GPU_T"
else
  before2=$FAILURES
  for table in NicLayout GpuCount; do
    if ! cfnq mapping "$GPU_T" "$table" > "$TMP/$table.tsv" 2>"$TMP/e2$table"; then
      fail 2 "$GPU_T: mapping $table could not be read: $(tr '\n' ' ' < "$TMP/e2$table")"
      : > "$TMP/$table.tsv"
    fi
  done

  nic_types=$(cut -f1 "$TMP/NicLayout.tsv" | sort -u)
  gpucount_types=$(cut -f1 "$TMP/GpuCount.tsv" | sort -u)

  # The three sets have to be the same: a type offered by GpuInstanceType with no NicLayout entry
  # renders a launch template with no interfaces, and one with no GpuCount entry leaves the node check
  # with no expected GPU count. The root and the GPU template offer the same list because the root
  # passes the value down.
  for t_file in "$GPU_T" "$ROOT_T"; do
    if ! offered=$(cfnq allowed "$t_file" GpuInstanceType 2>"$TMP/e2a"); then
      fail 2 "$t_file: could not read GpuInstanceType.AllowedValues: $(tr '\n' ' ' < "$TMP/e2a")"
      continue
    fi
    while read -r t; do
      [ -n "$t" ] || continue
      grep -qx "$t" <<<"$nic_types" || fail 2 "$t_file: GpuInstanceType offers '$t' with no NicLayout entry"
    done <<<"$offered"
    while read -r t; do
      [ -n "$t" ] || continue
      grep -qx "$t" <<<"$offered" || fail 2 "$t_file: NicLayout has '$t' and GpuInstanceType does not offer it"
    done <<<"$nic_types"
  done
  while read -r t; do
    [ -n "$t" ] || continue
    grep -qx "$t" <<<"$gpucount_types" || fail 2 "$GPU_T: NicLayout has '$t' and GpuCount does not, so the bootstrap cannot assert an exact nvidia.com/gpu count for it"
  done <<<"$nic_types"
  while read -r t; do
    [ -n "$t" ] || continue
    grep -qx "$t" <<<"$nic_types" || fail 2 "$GPU_T: GpuCount has '$t' and NicLayout does not, so the launch template would render with no interfaces for it"
  done <<<"$gpucount_types"

  # Dropping a type from all three tables at once would otherwise pass: the sets still agree.
  for t in "${REQUIRED_NIC_TYPES[@]}"; do
    grep -qx "$t" <<<"$nic_types" \
      || fail 2 "$GPU_T: '$t' is required by this architecture and NicLayout does not carry it"
  done

  while IFS=$'\t' read -r t key value vtype; do
    [ -n "$t" ] || continue
    case " ${NIC_KEYS[*]} " in *" $key "*) ;; *) fail 2 "$GPU_T: NicLayout.$t has an unexpected key '$key'"; continue ;; esac
    # The values decide which conditional interface blocks render, so a typo here is a launch
    # template with the wrong interfaces rather than a template that fails to deploy.
    case "$key" in
      Cards|EfaInterfaces)
        case "$value" in ''|*[!0-9]*) fail 2 "$GPU_T: NicLayout.$t.$key is '$value', which is not a count" ;; esac
        ;;
      PrimaryEfa)
        case "$value" in true|false) ;; *) fail 2 "$GPU_T: NicLayout.$t.PrimaryEfa is '$value'; the template compares it to \"true\", so anything else reads as false" ;; esac
        ;;
      SecondaryDeviceIndex)
        case "$value" in 0|1) ;; *) fail 2 "$GPU_T: NicLayout.$t.SecondaryDeviceIndex is '$value'; the template only distinguishes 0 from 1" ;; esac
        ;;
    esac
  done < "$TMP/NicLayout.tsv"
  while IFS=$'\t' read -r t key value vtype; do
    [ -n "$t" ] || continue
    [ "$key" = Gpus ] || fail 2 "$GPU_T: GpuCount.$t has an unexpected key '$key'"
    case "$value" in ''|*[!0-9]*|0) fail 2 "$GPU_T: GpuCount.$t.Gpus is '$value', which is not a GPU count" ;; esac
  done < "$TMP/GpuCount.tsv"
  for t in $nic_types; do
    for key in "${NIC_KEYS[@]}"; do
      awk -F'\t' -v t="$t" -v k="$key" '$1==t && $2==k {found=1} END {exit !found}' "$TMP/NicLayout.tsv" \
        || fail 2 "$GPU_T: NicLayout.$t has no $key"
    done
  done
  # The taint the node group sets, the toleration the device plugins carry and the label the bootstrap
  # waits on are the same string in several places; one of them diverging leaves pods Pending on nodes
  # the deploy then reports as unverified.
  taint_refs=$(grep -c "$GPU_TAINT_KEY" "$GPU_T" || true)
  [ "$taint_refs" -ge 4 ] \
    || fail 2 "$GPU_T: '$GPU_TAINT_KEY' appears $taint_refs time(s); the taint, the tolerations and the resource name should all use it"
  grep -q "nvidia\.com/[a-z]" "$GPU_T" \
    && grep -Fxvq "$GPU_TAINT_KEY" <(grep -o 'nvidia\.com/[a-z-]*' "$GPU_T" | sort -u) \
    && fail 2 "$GPU_T: a resource or taint under nvidia.com/ is not '$GPU_TAINT_KEY'" || true

  [ "$FAILURES" -eq "$before2" ] \
    && pass 2 "$(printf '%s\n' "$nic_types" | grep -c .) type(s) in both tables, every NicLayout entry complete, every value in range, one taint key"
fi

# --------------------------------------------------------------------------
# check 3 — the committed NetworkInterfaces block is what the generator emits
# --------------------------------------------------------------------------
head_ 3 "$RENDER output equals the NetworkInterfaces block in $GPU_T"
if [ ! -f "$RENDER" ]; then
  fail 3 "$RENDER: expected to exist (the committed block is generated by it); nothing to compare against"
elif [ ! -f "$GPU_T" ]; then
  skip 3 "$GPU_T missing"
elif [ -z "$PY_ANY" ]; then
  skip 3 "no python3 on PATH"
else
  if ! "$PY_ANY" "$RENDER" > "$TMP/render.out" 2>"$TMP/render.err"; then
    fail 3 "$RENDER exited non-zero: $(head -3 "$TMP/render.err" | tr '\n' ' ')"
  elif [ ! -s "$TMP/render.out" ]; then
    fail 3 "$RENDER produced no output"
  else
    if mode=$("$PY_ANY" "$TEXTQ" nicblock "$GPU_T" "$TMP/block-keyed.txt" "$TMP/block-body.txt" 2>"$TMP/nic.err"); then
      if cmp -s "$TMP/render.out" "$TMP/block-body.txt"; then
        pass 3 "byte-identical to the committed block ($mode, key line excluded)"
      elif cmp -s "$TMP/render.out" "$TMP/block-keyed.txt"; then
        pass 3 "byte-identical to the committed block ($mode, key line included)"
      else
        fail 3 "$GPU_T: the committed NetworkInterfaces block is not what $RENDER emits — regenerate it (the diff below is the committed block against the render; < is the render)"
        { diff "$TMP/render.out" "$TMP/block-body.txt" || true; } | sed 's/^/        /' | head -40
      fi
    else
      fail 3 "$GPU_T: could not locate the generated block: $(tr '\n' ' ' < "$TMP/nic.err")"
    fi
  fi

  # The generator's thresholds, the template's Cards* conditions and the card counts in NicLayout are
  # three copies of the same set. Check 3 compares only the rendered interfaces, so a condition that
  # exists in one place and not the others survives it.
  cat > "$TMP/thresholds.py" <<'PYEOF'
import re
import sys

render, template = sys.argv[1], sys.argv[2]
problems = []

gen = set(re.findall(r'^\s*\((\d+), "Cards\w+"\),', open(render, encoding="utf-8").read(), re.M))
text = open(template, encoding="utf-8").read()
cond = set()
for name in re.findall(r"^  (Cards(\d+)(?:Plus)?):", text, re.M):
    cond.add(name[1])
cards = set(re.findall(r"^      Cards: \"(\d+)\"$", text, re.M))

if not gen:
    problems.append("%s: no thresholds found; the check cannot run" % render)
if not cond:
    problems.append("%s: no Cards* conditions found; the check cannot run" % template)
if gen and cond and gen != cond:
    problems.append("the generator's thresholds %s and the template's Cards* conditions %s are "
                    "different sets" % (",".join(sorted(gen, key=int)), ",".join(sorted(cond, key=int))))
# The names say which threshold a condition stands for; the bodies decide. Evaluate each body for
# every Cards value: CardsNPlus has to be true exactly when the card count is at least N.
for name, n, body in re.findall(r"^  (Cards(\d+)(?:Plus)?):(.*?)(?=^  \S|\Z)", text, re.M | re.S):
    listed = set(re.findall(r'Cards\], "(\d+)"\]', body))
    negated = body.lstrip().startswith("!Not")
    for value in sorted(cards, key=int):
        got = (value not in listed) if negated else (value in listed)
        if got != (int(value) >= int(n)):
            problems.append("%s: condition %s is %s for Cards: %s and should be %s"
                            % (template, name, str(got).lower(), value, str(not got).lower()))
# Every card count above 1 needs a condition that gates its cards.
for value in sorted(cards - {"1"}, key=int):
    if value not in gen:
        problems.append("NicLayout has Cards: %s and the generator has no threshold for it, so cards "
                        "beyond the last threshold would never render" % value)

for problem in problems:
    print("PROBLEM %s" % problem)
if not problems:
    print("OK thresholds %s match the template's conditions and cover every Cards value"
          % ",".join(sorted(gen, key=int)))
PYEOF
  if [ -n "$PY_ANY" ] && [ -f "$RENDER" ] && [ -f "$GPU_T" ]; then
    if ! "$PY_ANY" "$TMP/thresholds.py" "$RENDER" "$GPU_T" > "$TMP/thr.out" 2>"$TMP/thr.err"; then
      fail 3 "could not compare the thresholds: $(head -3 "$TMP/thr.err" | tr '\n' ' ')"
    elif grep -q '^PROBLEM ' "$TMP/thr.out"; then
      while IFS= read -r line; do
        case "$line" in "PROBLEM "*) fail 3 "${line#PROBLEM }" ;; esac
      done < "$TMP/thr.out"
    else
      pass 3 "$(sed -n 's/^OK //p' "$TMP/thr.out")"
    fi
  fi
fi

# --------------------------------------------------------------------------
# check 3b — a template past the 51,200-byte --template-body limit cannot be
# deployed the way the README shows. The limit belongs to the CLI, not to
# CloudFormation, so the fix is --template-url; the point of the check is that
# nobody finds out from a ValidationError.
# --------------------------------------------------------------------------
head_ 3 "the README does not deploy a template too large for --template-body"
body_problems=0
for t in "${ALL_T[@]}"; do
  [ -f "$t" ] || continue
  size=$(wc -c < "$t" | tr -d ' ')
  base=$(basename "$t")
  if grep -q -- "--template-body file://assets/$base" README.md 2>/dev/null; then
    if [ "$size" -gt 51200 ]; then
      fail 3 "README.md deploys $base with --template-body, but it is $size bytes and the limit is 51,200; use --template-url"
      body_problems=$((body_problems + 1))
    elif [ "$size" -gt 48640 ]; then
      fail 3 "$base is $size bytes, within 5% of the 51,200-byte --template-body limit the README relies on"
      body_problems=$((body_problems + 1))
    fi
  fi
done
[ "$body_problems" -eq 0 ] && pass 3 "every template the README passes with --template-body is under the limit"

# --------------------------------------------------------------------------
# check 4 — every template's parameters and $PARAMS_DOC are the same set
# --------------------------------------------------------------------------
head_ 4 "the parameters of every template and $PARAMS_DOC are the same set"
if [ -z "$PY_YAML" ] || [ ! -f "$ROOT_T" ]; then
  skip 4 "needs PyYAML and $ROOT_T"
elif [ ! -f "$PARAMS_DOC" ]; then
  fail 4 "$PARAMS_DOC: expected to exist"
else
  cfnq params "$ROOT_T" | sort -u > "$TMP/root-params.txt" || true
  : > "$TMP/all-params.txt"
  : > "$TMP/params-by-template.txt"
  for t in "${ALL_T[@]}"; do
    [ -f "$t" ] || continue
    cfnq params "$t" >> "$TMP/all-params.txt" 2>/dev/null || true
    cfnq params "$t" 2>/dev/null | while read -r prm; do
      [ -n "$prm" ] && printf '%s %s\n' "$t" "$prm" >> "$TMP/params-by-template.txt"
    done
  done
  sort -u -o "$TMP/all-params.txt" "$TMP/all-params.txt"
  # A documented parameter is a backticked name in the first cell of a table
  # row, the shape aws-pcs/docs/PARAMETERS.md uses. A row may name several,
  # which is how parameters that mean the same thing everywhere are grouped.
  sed -n 's/^|\([^|]*\)|.*/\1/p' "$PARAMS_DOC" \
    | grep -oE '`[A-Za-z][A-Za-z0-9]*`' | tr -d '`' | sort -u > "$TMP/doc-params.txt"
  if [ ! -s "$TMP/doc-params.txt" ]; then
    fail 4 "$PARAMS_DOC: no parameter rows found; expected Markdown table rows whose first cell is a backticked parameter name, as in architectures/aws-pcs/docs/PARAMETERS.md"
  fi
  while read -r t prm; do
    [ -n "$prm" ] || continue
    grep -qx "$prm" "$TMP/doc-params.txt" || fail 4 "$t declares parameter '$prm' but $PARAMS_DOC has no row for it"
  done < "$TMP/params-by-template.txt"
  while read -r prm; do
    [ -n "$prm" ] || continue
    grep -qx "$prm" "$TMP/all-params.txt" || fail 4 "$PARAMS_DOC documents '$prm', which no template declares (renamed or removed?)"
  done < "$TMP/doc-params.txt"
  pass 4 "$(wc -l < "$TMP/all-params.txt" | tr -d ' ') distinct parameter(s) across the templates, all documented"
fi

# --------------------------------------------------------------------------
# check 5 — NicLayout agrees with the EC2 API
#
# SecondaryDeviceIndex is deliberately not checked here: describe-instance-types
# does not report which DeviceIndex the secondary cards must use, which is the
# reason that value is carried as data and verified on hardware instead.
# --------------------------------------------------------------------------
head_ 5 "NicLayout agrees with aws ec2 describe-instance-types"
if [ "$HAVE_AWS" != yes ]; then
  skip 5 "no usable AWS credentials (describe-instance-types is an authenticated call)"
elif [ ! -s "$TMP/NicLayout.tsv" ]; then
  skip 5 "NicLayout could not be read (see check 2)"
else
  REGION="$(aws configure get region 2>/dev/null || true)"
  if [ -n "$REGION" ]; then REGION=" in $REGION"; fi
  PRIMARY_UNVERIFIED=""
  while read -r t; do
    [ -n "$t" ] || continue
    if ! api=$(aws ec2 describe-instance-types --instance-types "$t" \
        --query 'InstanceTypes[0].[NetworkInfo.MaximumNetworkCards,NetworkInfo.EfaSupported,NetworkInfo.EfaInfo.MaximumEfaInterfaces]' \
        --output text 2>"$TMP/dit.err"); then
      skip 5 "$t: not offered${REGION} or not described: $(tr '\n' ' ' < "$TMP/dit.err" | cut -c1-140)"
      continue
    fi
    api_cards=$(printf '%s' "$api" | cut -f1)
    api_efa=$(printf '%s' "$api" | cut -f2)
    api_ifaces=$(printf '%s' "$api" | cut -f3)
    if [ "$api_ifaces" = None ]; then api_ifaces=0; fi
    map_cards=$(awk -F'\t' -v t="$t" '$1==t && $2=="Cards" {print $3; exit}' "$TMP/NicLayout.tsv")
    map_ifaces=$(awk -F'\t' -v t="$t" '$1==t && $2=="EfaInterfaces" {print $3; exit}' "$TMP/NicLayout.tsv")
    map_primary=$(awk -F'\t' -v t="$t" '$1==t && $2=="PrimaryEfa" {print $3; exit}' "$TMP/NicLayout.tsv")
    [ "$map_cards" = "$api_cards" ] || fail 5 "$t: NicLayout Cards=$map_cards but MaximumNetworkCards=$api_cards"
    [ "$map_ifaces" = "$api_ifaces" ] || fail 5 "$t: NicLayout EfaInterfaces=$map_ifaces but MaximumEfaInterfaces=$api_ifaces (EfaSupported=$api_efa)"
    card0=$(aws ec2 describe-instance-types --instance-types "$t" \
        --query 'InstanceTypes[0].NetworkInfo.NetworkCards[?NetworkCardIndex==`0`].EfaSupported' \
        --output text 2>/dev/null || true)
    case "$card0" in
      True|true)  [ "$map_primary" = true ]  || fail 5 "$t: NicLayout PrimaryEfa=$map_primary but network card 0 reports EfaSupported=true" ;;
      False|false) [ "$map_primary" = false ] || fail 5 "$t: NicLayout PrimaryEfa=$map_primary but network card 0 reports EfaSupported=false" ;;
      *) PRIMARY_UNVERIFIED="$PRIMARY_UNVERIFIED $t" ;;
    esac
    pass 5 "$t: Cards=$map_cards, EfaInterfaces=$map_ifaces"
  done < <(cut -f1 "$TMP/NicLayout.tsv" | sort -u)
  if [ -n "$PRIMARY_UNVERIFIED" ]; then
    skip 5 "PrimaryEfa is unverified for:$PRIMARY_UNVERIFIED — describe-instance-types does not return EfaSupported per network card, so this key and SecondaryDeviceIndex are both outside what the EC2 API can confirm; they are proven by tests/gpu-efa-test.md on hardware"
  fi
fi

# --------------------------------------------------------------------------
# check 6 — every template is published, where the root expects to find it
#
# A template that is not in the publish manifest exists in the repository and not
# at the URL a launch link uses, so a deploy from the published copy fails when
# the root fetches that child. The root's S3BucketName and S3KeyPrefix defaults
# have to name the bucket and prefix the manifest publishes to, or the root reads
# children from somewhere they were never put.
# --------------------------------------------------------------------------
head_ 6 "$PUBLISH_MANIFEST publishes every template, and $ROOT_T defaults to it"
if [ ! -f "$PUBLISH_MANIFEST" ]; then
  skip 6 "$PUBLISH_MANIFEST not reachable from $ROOT"
elif [ -z "$PY_YAML" ]; then
  skip 6 "needs PyYAML"
else
  cat > "$TMP/publish.py" <<'PYEOF'
import posixpath
import re
import sys

import yaml


class Loader(yaml.SafeLoader):
    pass


def _keep(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_mapping(node, deep=True)


Loader.add_multi_constructor("!", _keep)

manifest_path, arch_dir, root_template = sys.argv[1], sys.argv[2], sys.argv[3]
templates = sys.argv[4:]

with open(manifest_path, encoding="utf-8") as handle:
    manifest = yaml.safe_load(handle) or {}
entries = {}
for entry in manifest.get("entries") or []:
    entries[entry.get("source")] = entry.get("key")
bucket = manifest.get("bucket", "")
prefix = manifest.get("prefix", "")

problems = []
keys = []
for template in templates:
    source = posixpath.join(arch_dir, template)
    key = entries.get(source)
    if key is None:
        problems.append("%s has no entry, so it is never published" % source)
    else:
        keys.append(key)

# The root fetches each child by name, so the key has to end in the file name the root asks for.
for template in templates:
    source = posixpath.join(arch_dir, template)
    key = entries.get(source)
    if key and posixpath.basename(key) != posixpath.basename(source):
        problems.append("%s is published as %s, but the root fetches children by their own file name"
                        % (source, key))

prefixes = sorted({posixpath.dirname(key) for key in keys})
if len(prefixes) > 1:
    problems.append("the keys span more than one prefix (%s), so one S3KeyPrefix cannot reach them all"
                    % ", ".join(prefixes))

with open(root_template, encoding="utf-8") as handle:
    root_text = handle.read()
root_doc = yaml.load(root_text, Loader=Loader) or {}
params = root_doc.get("Parameters", {})
root_bucket = params.get("S3BucketName", {}).get("Default", "")
root_prefix = params.get("S3KeyPrefix", {}).get("Default", "")

# The names the root actually asks S3 for, rather than an assumption about them. A child renamed in
# one place and not the other publishes fine and fails at deploy with a 404 on a nested stack.
asked = set(re.findall(r"\$\{S3BucketName\}\.s3\.amazonaws\.com/\$\{S3KeyPrefix\}([A-Za-z0-9._-]+)", root_text))
children = {posixpath.basename(t) for t in templates} - {posixpath.basename(root_template)}
if not asked:
    problems.append("%s: no nested-stack TemplateURL found, so the names it fetches cannot be checked"
                    % root_template)
for name in sorted(asked - children):
    problems.append("%s fetches '%s', which is not one of the published templates" % (root_template, name))
for name in sorted(children - asked):
    problems.append("%s is published but the root never fetches it; deploy it on its own or drop it"
                    % name)
if root_bucket != bucket:
    problems.append("S3BucketName defaults to '%s' and the manifest publishes to '%s'" % (root_bucket, bucket))
if len(prefixes) == 1:
    published = prefix + prefixes[0] + "/"
    if root_prefix != published:
        problems.append("S3KeyPrefix defaults to '%s' and the manifest publishes under '%s'"
                        % (root_prefix, published))

for problem in problems:
    print("PROBLEM %s" % problem)
if not problems:
    print("OK %d template(s) published under %s%s/, which is what the root defaults to"
          % (len(keys), prefix, prefixes[0] if prefixes else ""))
PYEOF
  if ! "$PY_YAML" "$TMP/publish.py" "$PUBLISH_MANIFEST" architectures/amazon-eks "$ROOT_T" "${ALL_T[@]}" \
      > "$TMP/publish.out" 2>"$TMP/publish.err"; then
    fail 6 "could not read $PUBLISH_MANIFEST: $(head -3 "$TMP/publish.err" | tr '\n' ' ')"
  elif grep -q '^PROBLEM ' "$TMP/publish.out"; then
    while IFS= read -r line; do
      case "$line" in "PROBLEM "*) fail 6 "${line#PROBLEM }" ;; esac
    done < "$TMP/publish.out"
  else
    pass 6 "$(sed -n 's/^OK //p' "$TMP/publish.out")"
  fi
fi

# --------------------------------------------------------------------------
# check 7 — the templates agree on KubernetesVersion's default and validation, and the
# GPU node group's default kubectl is within one minor of it. A caller who sets
# one and not the others gets a node that cannot join, and neither template can
# see the other's value.
# --------------------------------------------------------------------------
if [ -z "$PY_YAML" ]; then
  skip 7 "needs PyYAML"
else
  cat > "$TMP/kver.py" <<'PYEOF'
import re
import sys

paths = sys.argv[1:]
problems = []


def default(path, name):
    """Default of a parameter, "" when it has none, None when the template does not declare it."""
    text = open(path, encoding="utf-8").read()
    block = re.search(r"^  %s:\n(?:    .*\n|\n)*" % name, text, re.M)
    if not block:
        return None
    found = re.search(r'^    Default: "?([^"\n]+)"?$', block.group(0), re.M)
    return found.group(1).strip() if found else ""


defaults = {}
for path in paths:
    value = default(path, "KubernetesVersion")
    if value is None:
        continue
    if not value:
        problems.append("%s: KubernetesVersion has no Default, so the templates cannot be checked "
                        "against each other" % path)
    else:
        defaults[path] = value

if len(set(defaults.values())) > 1:
    problems.append("the templates default KubernetesVersion to different values: %s"
                    % "; ".join("%s=%s" % (p, v) for p, v in sorted(defaults.items())))

# The same parameter validated differently accepts a value in one template that another refuses.
def rule(path, name):
    text = open(path, encoding="utf-8").read()
    block = re.search(r"^  %s:\n(?:    .*\n|\n)*" % name, text, re.M)
    if not block:
        return None
    return tuple(re.findall(r"^    (Allowed(?:Pattern|Values): .*)$", block.group(0), re.M))


for name in ("KubernetesVersion", "KubectlVersion"):
    rules = {p: rule(p, name) for p in paths if rule(p, name) is not None}
    if len(set(rules.values())) > 1:
        problems.append("the templates validate %s differently: %s" % (name, "; ".join(
            "%s=%s" % (p, " ".join(r) or "(none)") for p, r in sorted(rules.items()))))
# The root passes these through, so a root default that drifts from the child's changes what a
# root deploy installs without the child's test ever seeing it.
for name in ("KubectlVersion", "HelmVersion", "NvidiaDevicePluginChartVersion", "EfaDevicePluginChartVersion"):
    found = {p: default(p, name) for p in paths if default(p, name)}
    if len(set(found.values())) > 1:
        problems.append("the templates default %s to different values: %s"
                        % (name, "; ".join("%s=%s" % (p, v) for p, v in sorted(found.items()))))

gpu = [p for p in paths if p.endswith("eks-add-gpu-nodegroup.yaml")]
if gpu and gpu[0] in defaults:
    kubectl = default(gpu[0], "KubectlVersion")
    if not kubectl:
        problems.append("%s: KubectlVersion has no Default" % gpu[0])
    else:
        want = defaults[gpu[0]].split(".")
        got = kubectl.split(".")
        if len(got) < 2 or got[0] != want[0] or abs(int(got[1]) - int(want[1])) > 1:
            problems.append("%s: default KubectlVersion %s is not within one minor of the default "
                            "KubernetesVersion %s" % (gpu[0], kubectl, defaults[gpu[0]]))

for problem in problems:
    print("PROBLEM %s" % problem)
if not problems:
    print("OK KubernetesVersion defaults to %s and is validated the same way in all %d templates, "
          "and KubectlVersion is within one minor of it" % (next(iter(set(defaults.values()))), len(defaults)))
PYEOF
  if ! "$PY_YAML" "$TMP/kver.py" "${ALL_T[@]}" > "$TMP/kver.out" 2>"$TMP/kver.err"; then
    fail 7 "could not read the templates: $(head -3 "$TMP/kver.err" | tr '\n' ' ')"
  elif grep -q '^PROBLEM ' "$TMP/kver.out"; then
    while IFS= read -r line; do
      case "$line" in "PROBLEM "*) fail 7 "${line#PROBLEM }" ;; esac
    done < "$TMP/kver.out"
  else
    pass 7 "$(sed -n 's/^OK //p' "$TMP/kver.out")"
  fi
fi

# --------------------------------------------------------------------------
# check 8 — a security group a managed service validates carries the rules
# that service requires, including the self-referencing ones
#
# FSx reads the filesystem's security group when it creates the filesystem and
# refuses a group that does not carry LNET between the filesystem's own network
# interfaces. A group whose only rules name a client group passes every static
# read of the template and fails 10 minutes into a deploy, so the rule set is
# checked here instead.
# --------------------------------------------------------------------------
head_ 8 "the FSx and client security groups carry the rules FSx for Lustre requires, and every security group text is one EC2 accepts"
if [ -z "$PY_YAML" ]; then
  skip 8 "needs PyYAML"
else
  cat > "$TMP/sgrules.py" <<'PYEOF'
import re
import sys

import yaml


class Loader(yaml.SafeLoader):
    """SafeLoader plus CloudFormation's short-form tags, so !Ref survives as a mapping."""


def _keep(loader, tag_suffix, node):
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:
        value = loader.construct_mapping(node, deep=True)
    return {tag_suffix if tag_suffix == "Ref" else "Fn::" + tag_suffix: value}


Loader.add_multi_constructor("!", _keep)

path = sys.argv[1]
doc = yaml.load(open(path, encoding="utf-8").read(), Loader=Loader) or {}
resources = doc.get("Resources") or {}
problems = []

# Ports FSx for Lustre requires between the filesystem's own network interfaces.
REQUIRED = [(988, 988), (1018, 1023)]


def ref(value):
    """The logical id in a !Ref, whichever way the tag survived the loader."""
    if isinstance(value, str):
        found = re.match(r"^!?Ref\s+(\w+)$", value.strip())
        return found.group(1) if found else None
    if isinstance(value, dict):
        target = value.get("Ref")
        return target if isinstance(target, str) else None
    return None


filesystems = [(name, body) for name, body in resources.items()
               if body.get("Type") == "AWS::FSx::FileSystem"]
if not filesystems:
    # The check exists because this template creates a filesystem. If it stops doing so, the check
    # has lost its subject and must say so rather than pass.
    print("PROBLEM %s: no AWS::FSx::FileSystem; this check has no subject and cannot pass" % path)
    raise SystemExit(0)

for name, body in filesystems:
    groups = [g for g in (ref(v) for v in (body.get("Properties") or {}).get("SecurityGroupIds") or [])
              if g]
    if not groups:
        problems.append("%s: %s names no SecurityGroupIds by !Ref" % (path, name))
    depends = body.get("DependsOn") or []
    if isinstance(depends, str):
        depends = [depends]

    for group in groups:
        have = {}
        for rname, rbody in resources.items():
            if rbody.get("Type") != "AWS::EC2::SecurityGroupIngress":
                continue
            props = rbody.get("Properties") or {}
            if ref(props.get("GroupId")) != group:
                continue
            if ref(props.get("SourceSecurityGroupId")) != group:
                continue
            try:
                ports = (int(props.get("FromPort")), int(props.get("ToPort")))
            except (TypeError, ValueError):
                problems.append("%s: %s has no numeric FromPort/ToPort" % (path, rname))
                continue
            proto = str(props.get("IpProtocol", "")).lower()
            if proto != "tcp":
                problems.append("%s: %s carries IpProtocol '%s'; LNET is tcp" % (path, rname, proto))
                continue
            have[ports] = rname

        for ports in REQUIRED:
            rname = have.get(ports)
            if rname is None:
                problems.append(
                    "%s: security group %s has no self-referencing tcp ingress on %d-%d; FSx refuses "
                    "to create a filesystem whose group does not carry LNET between its own interfaces"
                    % (path, group, ports[0], ports[1]))
            elif rname not in depends:
                problems.append(
                    "%s: %s does not depend on %s, so the filesystem can be created before that rule "
                    "exists and FSx reads the group as it is at that moment"
                    % (path, name, rname))

        # FSx also opens connections back to its clients, so every client group it admits has to
        # admit it on the same ports.
        sg = resources.get(group) or {}
        clients = {ref(r.get("SourceSecurityGroupId")) for r in
                   (sg.get("Properties") or {}).get("SecurityGroupIngress") or []} - {None, group}
        for client in sorted(clients):
            back = set()
            for rbody in resources.values():
                props = rbody.get("Properties") or {}
                if (rbody.get("Type") == "AWS::EC2::SecurityGroupIngress"
                        and ref(props.get("GroupId")) == client
                        and ref(props.get("SourceSecurityGroupId")) == group):
                    back.add((int(props.get("FromPort")), int(props.get("ToPort"))))
            for ports in REQUIRED:
                if ports not in back:
                    problems.append("%s: client group %s has no tcp ingress on %d-%d from %s; FSx "
                                    "connects back to its clients on those ports"
                                    % (path, client, ports[0], ports[1], group))
# EC2 accepts a narrow character set in security group descriptions and rejects the rest at create
# time, which validate-template does not check.
EC2_TEXT = re.compile(r"^[a-zA-Z0-9. _\-:/()#,@\[\]+=&;{}!$*]{0,255}$")
for tpath in sys.argv[2:]:
    tres = (yaml.load(open(tpath, encoding="utf-8").read(), Loader=Loader) or {}).get("Resources") or {}
    for rname, rbody in tres.items():
        props = rbody.get("Properties") or {}
        texts = []
        if rbody.get("Type") == "AWS::EC2::SecurityGroup":
            texts.append(props.get("GroupDescription"))
            texts += [r.get("Description") for r in (props.get("SecurityGroupIngress") or [])
                      + (props.get("SecurityGroupEgress") or [])]
        elif rbody.get("Type") in ("AWS::EC2::SecurityGroupIngress", "AWS::EC2::SecurityGroupEgress"):
            texts.append(props.get("Description"))
        for text in texts:
            if isinstance(text, str) and not EC2_TEXT.match(text):
                problems.append("%s: %s has a description EC2 rejects (allowed: a-zA-Z0-9. _-:/()#,@[]+=&;{}!$*): %s"
                                % (tpath, rname, text))
for problem in problems:
    print("PROBLEM %s" % problem)
if not problems:
    print("OK %s: %s carries the self-referencing LNET rules, waits for each of them, and every client group admits it"
          % (path, ", ".join(sorted({g for _, b in filesystems
                                     for g in (ref(v) for v in (b.get("Properties") or {}).get("SecurityGroupIds") or [])
                                     if g}))))
PYEOF
  if ! "$PY_YAML" "$TMP/sgrules.py" "$PREREQ_T" "${ALL_T[@]}" > "$TMP/sgrules.out" 2>"$TMP/sgrules.err"; then
    fail 8 "could not read $PREREQ_T: $(head -3 "$TMP/sgrules.err" | tr '\n' ' ')"
  elif grep -q '^PROBLEM ' "$TMP/sgrules.out"; then
    while IFS= read -r line; do
      case "$line" in "PROBLEM "*) fail 8 "${line#PROBLEM }" ;; esac
    done < "$TMP/sgrules.out"
  else
    pass 8 "$(sed -n 's/^OK //p' "$TMP/sgrules.out")"
  fi
fi

# --------------------------------------------------------------------------
# check 9 — every Markdown link in the documents resolves, anchors included
# --------------------------------------------------------------------------
head_ 9 "Markdown links and anchors in README.md, $PARAMS_DOC and tests/*.md resolve"
DOCS=(README.md "$PARAMS_DOC")
for d in tests/*.md; do [ -f "$d" ] && DOCS+=("$d"); done
if [ -z "$PY_ANY" ]; then
  skip 9 "no python3 on PATH"
else
  set +e
  "$PY_ANY" "$TEXTQ" links "${DOCS[@]}" > "$TMP/links.out" 2>"$TMP/links.err"
  rc=$?
  set -e
  if [ "$rc" -eq 2 ]; then
    fail 9 "could not read the documents: $(head -3 "$TMP/links.err" | tr '\n' ' ')"
  elif [ "$rc" -ne 0 ]; then
    while IFS= read -r line; do [ -n "$line" ] && fail 9 "$line"; done < "$TMP/links.out"
  else
    pass 9 "${#DOCS[@]} document(s), every relative link and anchor resolves"
  fi
fi

# --------------------------------------------------------------------------
# check 10 — nothing installs an unpinned version
# --------------------------------------------------------------------------
head_ 10 "no unpinned versions in the templates or the documents"
PINNED=("${ALL_T[@]}" README.md "$PARAMS_DOC")
for d in tests/*.md; do [ -f "$d" ] && PINNED+=("$d"); done
if [ -z "$PY_ANY" ]; then
  skip 10 "no python3 on PATH"
else
  set +e
  "$PY_ANY" "$TEXTQ" pins "${PINNED[@]}" > "$TMP/pins.out" 2>"$TMP/pins.err"
  rc=$?
  set -e
  if [ "$rc" -eq 2 ]; then
    fail 10 "could not read the files: $(head -3 "$TMP/pins.err" | tr '\n' ' ')"
  elif [ "$rc" -ne 0 ]; then
    while IFS= read -r line; do [ -n "$line" ] && fail 10 "$line"; done < "$TMP/pins.out"
  else
    pass 10 "${#PINNED[@]} file(s), every chart, image and package carries a version"
  fi
fi

# --------------------------------------------------------------------------
printf '\n'
if [ "$FAILURES" -eq 0 ]; then
  printf '[OK] lint-templates.sh: 0 failures, %d skipped\n' "$SKIPS"
  printf '     A skip is not a pass. Re-run with AWS credentials to close the skips.\n'
  exit 0
fi
printf '[FAILED] lint-templates.sh: %d failure(s), %d skipped\n' "$FAILURES" "$SKIPS"
printf '         Every FAIL line above names the file, what was expected and what was found.\n'
exit 1
