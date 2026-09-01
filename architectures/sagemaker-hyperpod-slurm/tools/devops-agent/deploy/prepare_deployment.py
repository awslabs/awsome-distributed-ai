#!/usr/bin/env python3
"""Prepare the single-template HyperPod x DevOps Agent deployment.

Four subcommands (mostly driven by deploy.sh) that turn the source tree into
inputs 'aws cloudformation deploy' can consume:

  embed    Inline each deploy/lambda/*.py into hyperpod_devops_agent.template.yaml
           at its "# <NAME>_CODE_PLACEHOLDER" marker, producing the deployable
           hyperpod_devops_agent.yaml, handling every placeholder in one pass.

  sync-skills
           For each skill directory under ../skills/, stage its contents
           (bundling docs/hyperpod-mental-model.md into references/ when the
           SKILL.md references it), zip it, upload to s3://<bucket>/skills/<name>.zip,
           and print a JSON object with the manifest + a content-hash version:
               {"manifest": [...], "version": "<sha256>"}
           deploy.sh captures that and passes it to CloudFormation.

  build-skill-uploader
           Package the skill-uploader Lambda (handler + bundled boto3) as an S3
           zip and upload it; print its {bucket, key} for the template.

  check-boto3
           Preflight: verify a suitable boto3 is importable (see BOTO3_MIN); exit
           nonzero with an install hint otherwise. deploy.sh runs this up front.

The `embed` and `sync-skills` subcommands are stdlib-only. `build-skill-uploader`
and `check-boto3` require a suitable boto3 (see BOTO3_MIN) to already be installed
in the environment running this script — `build-skill-uploader` bundles that copy
into the Lambda and never pip-installs, so nothing is added to the caller's
environment silently.
"""
import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile


HERE = os.path.dirname(os.path.abspath(__file__))
SOLUTION_ROOT = os.path.dirname(HERE)              # devops-agent
MENTAL_MODEL = os.path.join(SOLUTION_ROOT, "docs", "hyperpod-mental-model.md")

LAMBDA_DIR = os.path.join(HERE, "lambda")
SKILLS_DIR = os.path.join(SOLUTION_ROOT, "skills")

# placeholder marker -> lambda source file. These are inlined into the template
# as ZipFile code. The skill uploader is NOT here: its Asset API is newer than
# the Lambda runtime's bundled boto3, so it ships as an S3 zip with a current
# boto3 bundled (see build_skill_uploader).
PLACEHOLDERS = {
    "WEBHOOK_BRIDGE_CODE_PLACEHOLDER": "webhook_bridge.py",
    "PERIODIC_AUDIT_CODE_PLACEHOLDER": "periodic_audit.py",
    "EMAIL_NOTIFIER_CODE_PLACEHOLDER": "email_notifier.py",
    "CR_WEBHOOK_PROVISIONER_CODE_PLACEHOLDER": "cr_webhook_provisioner.py",
    "HMA_CW_BRIDGE_CODE_PLACEHOLDER": "hma_cw_bridge.py",
}

# Minimum boto3 that includes the DevOps Agent Asset API (list_assets etc.).
# The skill-uploader Lambda needs this bundled because the Lambda runtime's
# built-in boto3 predates the Asset API. We do NOT pip-install it during deploy;
# it must already be present in the environment running this script (see
# _require_boto3), and we bundle that installed copy.
BOTO3_MIN_VERSION = (1, 43, 25)
BOTO3_MIN = "boto3>=" + ".".join(str(n) for n in BOTO3_MIN_VERSION)


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" "))]


def embed(template_in: str, template_out: str) -> None:
    with open(template_in) as f:
        lines = f.readlines()

    out: list[str] = []
    replaced: set[str] = set()
    for line in lines:
        stripped = line.strip()
        marker = stripped.lstrip("# ").strip()
        if stripped.startswith("#") and marker in PLACEHOLDERS:
            indent = _indent_of(line)
            src_path = os.path.join(LAMBDA_DIR, PLACEHOLDERS[marker])
            with open(src_path) as sf:
                for code_line in sf.read().splitlines():
                    out.append((indent + code_line).rstrip() + "\n" if code_line else "\n")
            replaced.add(marker)
        else:
            out.append(line)

    # Every placeholder MUST be replaced. A renamed/typo'd marker comment would
    # otherwise leave a ZipFile body that is just the stale comment line, and
    # 'aws cloudformation deploy' would ship it — for a custom-resource Lambda
    # that means a comment-only handler that never signals CFN, hanging the stack
    # for up to an hour before rollback. Fail loudly here instead.
    missing = set(PLACEHOLDERS) - replaced
    if missing:
        sys.exit(
            f"embed: {len(missing)} placeholder(s) not found in {template_in}: "
            f"{sorted(missing)}. Check the '# <NAME>_CODE_PLACEHOLDER' markers."
        )

    with open(template_out, "w") as f:
        f.writelines(out)
    print(f"embedded {len(replaced)} Lambda sources -> {template_out}")


def _parse_skill_frontmatter(skill_md: str) -> tuple[str, list[str]]:
    text = open(skill_md).read()
    m = re.search(r"^---\s*\n(.*?)\n---", text, re.S)
    if not m:
        sys.exit(f"{skill_md}: missing frontmatter")
    fm = m.group(1)
    name_match = re.search(r"^name:\s*(\S+)", fm, re.M)
    if not name_match:
        sys.exit(f"{skill_md}: frontmatter missing 'name:'")
    name = name_match.group(1).strip()
    at_match = re.search(r"agent_types:\s*\[([^\]]*)\]", fm)
    if at_match:
        agent_types = [
            t.strip().strip('"').strip("'")
            for t in at_match.group(1).split(",")
            if t.strip()
        ]
    else:
        agent_types = ["GENERIC"]
    return name, agent_types


def _build_skill_zip(skill_path: str, skill_md: str) -> bytes:
    """Zip a skill dir deterministically; bundle the mental-model doc if referenced.

    Fixed member timestamps + sorted order so identical content always yields
    identical bytes — the content hash only changes when a skill actually changes.
    """
    references_needed = "hyperpod-mental-model.md" in open(skill_md).read()

    members: list[tuple[str, str]] = []  # (arcname, abs_path)
    for root, dirs, files in os.walk(skill_path):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for fn in sorted(files):
            if fn == ".DS_Store":
                continue
            abs_path = os.path.join(root, fn)
            members.append((os.path.relpath(abs_path, skill_path), abs_path))
    if references_needed:
        if not os.path.exists(MENTAL_MODEL):
            sys.exit(
                f"SKILL.md references hyperpod-mental-model.md but "
                f"{MENTAL_MODEL} not found"
            )
        members.append(("references/hyperpod-mental-model.md", MENTAL_MODEL))
    members.sort()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, abs_path in members:
            info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(abs_path, "rb") as fp:
                zf.writestr(info, fp.read())
    return buf.getvalue()


def sync_skills(bucket: str, region: str) -> None:
    if not os.path.isdir(SKILLS_DIR):
        sys.exit(f"skills dir not found: {SKILLS_DIR}")

    manifest: list[dict] = []
    hasher = hashlib.sha256()
    for entry in sorted(os.listdir(SKILLS_DIR)):
        skill_path = os.path.join(SKILLS_DIR, entry)
        skill_md = os.path.join(skill_path, "SKILL.md")
        if not os.path.isdir(skill_path) or not os.path.isfile(skill_md):
            continue  # skip 'upstream/' clone and non-skill dirs
        name, agent_types = _parse_skill_frontmatter(skill_md)
        blob = _build_skill_zip(skill_path, skill_md)
        key = f"skills/{name}.zip"
        # Deterministic hash: name + agent_types + zip bytes.
        hasher.update(name.encode())
        hasher.update(json.dumps(agent_types, sort_keys=True).encode())
        hasher.update(blob)
        with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
            tmp.write(blob)
            tmp.flush()
            subprocess.run(
                ["aws", "s3api", "put-object",
                 "--bucket", bucket, "--key", key,
                 "--body", tmp.name, "--region", region],
                check=True,
                stdout=subprocess.DEVNULL,
            )
        print(f"uploaded s3://{bucket}/{key} ({len(blob)} bytes)", file=sys.stderr)
        manifest.append({"name": name, "zipKey": key, "agentTypes": agent_types})

    if not manifest:
        sys.exit(f"no skills found under {SKILLS_DIR}")

    print(json.dumps({"manifest": manifest, "version": hasher.hexdigest()[:32]}))


# boto3's runtime dependency closure — the importable packages we bundle into
# the skill-uploader Lambda. Copied from the already-installed environment; never
# fetched from the network.
BOTO3_BUNDLE_MODULES = (
    "boto3", "botocore", "jmespath", "s3transfer", "dateutil", "urllib3", "six",
)


def _require_boto3() -> "module":  # noqa: F821 - returns the imported boto3 module
    """Import boto3 from the current environment and enforce the minimum version.

    We deliberately do NOT pip-install boto3 during deploy — that would silently
    pull an arbitrary version from the network into the customer's environment.
    Instead, require a suitable boto3 to already be installed and fail loudly with
    remediation if it is missing or too old.
    """
    try:
        import boto3  # noqa: E402 - imported lazily so `embed` needs no boto3
    except ImportError:
        sys.exit(
            f"boto3 is required to build the skill-uploader Lambda ({BOTO3_MIN}) "
            f"but is not installed in this environment ({sys.executable}).\n"
            f"Install it first, e.g.:  {sys.executable} -m pip install '{BOTO3_MIN}'"
        )

    def _parse(v: str) -> tuple:
        parts = []
        for token in v.split(".")[:3]:
            num = "".join(c for c in token if c.isdigit())
            parts.append(int(num) if num else 0)
        return tuple(parts)

    installed = getattr(boto3, "__version__", "0")
    if _parse(installed) < BOTO3_MIN_VERSION:
        sys.exit(
            f"boto3 {installed} is installed, but the skill-uploader Lambda needs "
            f"{BOTO3_MIN} (the DevOps Agent Asset API — list_assets etc. — is not in "
            f"older versions).\n"
            f"Upgrade it first, e.g.:  {sys.executable} -m pip install --upgrade '{BOTO3_MIN}'"
        )
    return boto3


def _copy_installed_module(module_name: str, stage: str) -> bool:
    """Copy an already-imported module's package dir (or single .py file) into
    the Lambda staging dir. Returns True if copied, False if not importable."""
    import importlib.util

    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return False
    if spec.submodule_search_locations:  # a package
        pkg_dir = list(spec.submodule_search_locations)[0]
        shutil.copytree(
            pkg_dir, os.path.join(stage, module_name),
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
    elif spec.origin and spec.origin.endswith(".py"):  # a single-file module
        shutil.copy2(spec.origin, os.path.join(stage, os.path.basename(spec.origin)))
    else:
        return False
    return True


def build_skill_uploader(bucket: str, region: str) -> None:
    """Package the skill uploader Lambda (handler + cfnresponse + current boto3)
    as an S3 zip and upload it. The Lambda runtime's bundled boto3 predates the
    DevOps Agent Asset API (list_assets etc.), so we bundle the current one from
    this environment (requiring — never installing — a suitable boto3).

    Prints {"bucket": ..., "key": ...} for the template's Code.S3Bucket/S3Key.
    """
    _require_boto3()

    stage = tempfile.mkdtemp(prefix="skilluploader-")
    try:
        # index.py = handler; cfnresponse.py = the S3-package shim.
        with open(os.path.join(LAMBDA_DIR, "cr_skill_uploader.py")) as f:
            src = f.read()
        with open(os.path.join(stage, "index.py"), "w") as f:
            f.write(src)
        with open(os.path.join(LAMBDA_DIR, "cfnresponse.py")) as f:
            shim = f.read()
        with open(os.path.join(stage, "cfnresponse.py"), "w") as f:
            f.write(shim)

        print(f"bundling installed {BOTO3_MIN} into the package...", file=sys.stderr)
        missing = [
            m for m in BOTO3_BUNDLE_MODULES if not _copy_installed_module(m, stage)
        ]
        # botocore/jmespath/s3transfer are hard boto3 deps; six only matters for
        # some dateutil builds. Fail only if a core dependency is absent.
        core_missing = [m for m in missing if m in ("boto3", "botocore", "jmespath", "s3transfer")]
        if core_missing:
            sys.exit(
                "Could not locate installed boto3 dependencies "
                f"{core_missing} to bundle. Reinstall boto3: "
                f"{sys.executable} -m pip install '{BOTO3_MIN}'"
            )

        # Deterministic-ish zip (member order sorted; fixed timestamps).
        members: list[tuple[str, str]] = []
        for root, dirs, files in os.walk(stage):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            for fn in sorted(files):
                if fn.endswith(".pyc") or fn == ".DS_Store":
                    continue
                abs_path = os.path.join(root, fn)
                members.append((os.path.relpath(abs_path, stage), abs_path))
        members.sort()

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for arcname, abs_path in members:
                info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                with open(abs_path, "rb") as fp:
                    zf.writestr(info, fp.read())
        blob = buf.getvalue()

        # Content-address the key: CloudFormation does NOT auto-detect changes to
        # an S3 deployment package during a stack update (AWS::Lambda::Function
        # Code docs). A fixed key therefore leaves the deployed function running
        # stale code after the first deploy — including the CR whose Delete path
        # gates teardown. The zip is deterministic (fixed timestamps, sorted
        # members), so its hash moves only when the code moves; a changed hash
        # yields a new key -> a template diff -> a real function update.
        digest = hashlib.sha256(blob).hexdigest()[:16]
        key = f"lambda/skill_uploader-{digest}.zip"
        with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
            tmp.write(blob)
            tmp.flush()
            subprocess.run(
                ["aws", "s3api", "put-object",
                 "--bucket", bucket, "--key", key,
                 "--body", tmp.name, "--region", region],
                check=True,
                stdout=subprocess.DEVNULL,
            )
        print(f"uploaded s3://{bucket}/{key} ({len(blob)} bytes)", file=sys.stderr)
        print(json.dumps({"bucket": bucket, "key": key}))
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed")
    e.add_argument("--in", dest="template_in", required=True)
    e.add_argument("--out", dest="template_out", required=True)

    s = sub.add_parser("sync-skills")
    s.add_argument("--bucket", required=True)
    s.add_argument("--region", required=True)

    b = sub.add_parser("build-skill-uploader")
    b.add_argument("--bucket", required=True)
    b.add_argument("--region", required=True)

    # Preflight: fail fast (before any AWS calls) if the deploy environment can't
    # build the skill-uploader Lambda. Used by deploy.sh up front.
    sub.add_parser("check-boto3")

    args = ap.parse_args()
    if args.cmd == "embed":
        embed(args.template_in, args.template_out)
    elif args.cmd == "sync-skills":
        sync_skills(args.bucket, args.region)
    elif args.cmd == "build-skill-uploader":
        build_skill_uploader(args.bucket, args.region)
    elif args.cmd == "check-boto3":
        boto3 = _require_boto3()
        min_str = ".".join(str(n) for n in BOTO3_MIN_VERSION)
        print(f"boto3 {boto3.__version__} OK (>= {min_str})", file=sys.stderr)


if __name__ == "__main__":
    main()
