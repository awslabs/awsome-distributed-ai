# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The monitoring stack's CodeBuild commands, run with stand-ins for curl, aws and kubectl, and its trigger
Lambda, run with stand-ins for boto3 and the response URL.

Covers what no template lint can see: a bundle whose digest differs from the pinned one is neither
unpacked nor run; a failed Create removes only an install it started; a failed Update never removes
the install it was updating; an answer that could not be delivered is sent again by post_build; and the
trigger answers for a build that cannot start, and for the Delete of an install that was never created.
"""
import gzip
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

TEMPLATE = Path(__file__).resolve().parents[1] / "assets" / "eks-monitoring.yaml"
STAGE_SCRIPT = Path(__file__).resolve().parents[3] / ".github" / "scripts" / "stage-template-publish.py"


def phase(name):
    """The shell text of one buildspec phase: the block scalar after `<name>:` and `- |`."""
    lines = TEMPLATE.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == f"{name}:")
    begin = next(i for i in range(start, len(lines)) if lines[i].strip() == "- |") + 1
    indent = len(lines[begin]) - len(lines[begin].lstrip())
    body = []
    for l in lines[begin:]:
        if l.strip() and len(l) - len(l.lstrip()) < indent:
            break
        body.append(l[indent:])
    return "\n".join(body) + "\n"


STUBS = r'''
curl() {
  echo "curl $*" >> "$CALLS"
  case "$*" in
    *"-X PUT"*)
      n=$(cat "$WORK/puts" 2>/dev/null || echo 0); echo $((n + 1)) > "$WORK/puts"
      [ "$((n + 1))" -le "${PUT_FAILURES:-0}" ] && return 7
      cp "$WORK/resp.json" "$WORK/delivered.$((n + 1)).json"; return 0 ;;
    *"$WORK/bundle.tar.gz"*) cp "$WORK/served.tar.gz" "$WORK/bundle.tar.gz"; return 0 ;;
    *) : > "$WORK/kubectl"; return 0 ;;
  esac
}
aws() { echo "aws $*" >> "$CALLS"; return 0; }
kubectl() {
  echo "kubectl $*" >> "$CALLS"
  case "$*" in
    "get namespace "*)
      [ -n "${NAMESPACE_READ_ERROR:-}" ] && { echo "$NAMESPACE_READ_ERROR" >&2; return 1; }
      [ -n "${NAMESPACE_OWNER:-}" ] && { printf '%s' "$NAMESPACE_OWNER"; return 0; }
      echo 'Error from server (NotFound): namespaces "mon" not found' >&2; return 1 ;;
  esac
  return 0
}
'''

# A bundle whose install.py records each call and exits with the status the test sets for that command.
RECORDING_INSTALL = """import os, sys
with open(os.environ["CALLS"], "a") as f:
    f.write("install.py " + " ".join(sys.argv[1:]) + "\\n")
print("install.py " + sys.argv[1] + " output")
sys.exit(int(os.environ.get("INSTALL_" + sys.argv[1].upper() + "_RC", "0")))
"""


def bundle(files):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return raw.getvalue()


class BuildspecGates(unittest.TestCase):
    def run_phase(self, name, work, **env):
        script = phase(name).replace("/tmp/", f"{work}/").replace("/usr/local/bin/kubectl", f"{work}/kubectl")
        full = dict(os.environ, WORK=work, CALLS=f"{work}/calls", CFN_RESPONSE_URL="https://example.invalid/response",
                    CFN_PHYSICAL_ID="installed:mon", CFN_STACK_ID="stack", CFN_REQUEST_ID="request", CFN_LOGICAL_ID="Monitoring",
                    CODEBUILD_BUILD_ID="build:1", NAMESPACE="mon", OWNER="owner", REQUEST_TYPE="Create", CLUSTER_NAME="c",
                    AWS_REGION="us-east-1", KUBECTL_VERSION="1.36.4", S3_BUCKET="b", S3_PREFIX="p/", NODE_SELECTOR="",
                    SCRAPE_SERVICES="", SCRAPE_NAMESPACES="", BUNDLE_URL="https://example.invalid/bundle.tar.gz")
        full.update(env)
        return subprocess.run(["bash", "-c", STUBS + script], env=full, capture_output=True, text=True, timeout=60)

    def serve(self, work, files):
        tar = bundle(files)
        Path(work, "served.tar.gz").write_bytes(gzip.compress(tar))
        return hashlib.sha256(tar).hexdigest()

    def test_a_bundle_with_another_digest_is_neither_unpacked_nor_run(self):
        with tempfile.TemporaryDirectory() as work:
            self.serve(work, {"install.py": "raise SystemExit('the bundle ran')\n"})
            result = self.run_phase("build", work, BUNDLE_SHA256="0" * 64)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(Path(work, "bundle", "install.py").exists(), "the bundle was unpacked")
            self.assertNotIn("the bundle ran", result.stdout + result.stderr)
            answer = json.loads(Path(work, "delivered.1.json").read_text())
            self.assertEqual(answer["Status"], "FAILED")
            self.assertIn("bundle digest is", answer["Reason"])
            self.assertEqual(answer["PhysicalResourceId"], "not-created:mon")

    def test_an_answer_that_was_not_delivered_is_sent_again_by_post_build(self):
        with tempfile.TemporaryDirectory() as work:
            self.serve(work, {"install.py": ""})
            # The build's own PUT (with curl's retries, one call) fails.
            self.run_phase("build", work, BUNDLE_SHA256="0" * 64, PUT_FAILURES="1")
            self.assertFalse(Path(work, "responded").exists(), "a failed delivery was marked as delivered")
            self.assertFalse(Path(work, "delivered.1.json").exists())
            result = self.run_phase("post_build", work, BUNDLE_SHA256="0" * 64, PUT_FAILURES="1")
            self.assertEqual(result.returncode, 0, result.stderr)
            answer = json.loads(Path(work, "delivered.2.json").read_text())
            self.assertEqual(answer["Status"], "FAILED")
            self.assertIn("bundle digest is", answer["Reason"], "post_build sent another answer than the prepared one")

    def answered(self, work, n=1):
        return json.loads(Path(work, f"delivered.{n}.json").read_text())

    def calls(self, work):
        return Path(work, "calls").read_text().splitlines()

    def test_a_bundle_with_the_pinned_digest_is_unpacked_run_and_answered(self):
        with tempfile.TemporaryDirectory() as work:
            digest = self.serve(work, {"install.py": RECORDING_INSTALL})
            result = self.run_phase("build", work, BUNDLE_SHA256=digest)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual([c.split()[1] for c in self.calls(work) if c.startswith("install.py")], ["apply", "check"])
            answer = self.answered(work)
            self.assertEqual((answer["Status"], answer["PhysicalResourceId"], answer["Reason"]), ("SUCCESS", "installed:mon", "installed in mon"))

    def test_an_exporter_that_is_not_up_is_reported_and_the_install_succeeds(self):
        with tempfile.TemporaryDirectory() as work:
            digest = self.serve(work, {"install.py": RECORDING_INSTALL})
            self.run_phase("build", work, BUNDLE_SHA256=digest, INSTALL_CHECK_RC="4")
            answer = self.answered(work)
            self.assertEqual(answer["Status"], "SUCCESS")
            self.assertIn("has no target up yet", answer["Reason"])
            self.assertNotIn("delete", " ".join(self.calls(work)))

    def test_a_failed_create_removes_the_install_it_started_and_answers_not_created(self):
        with tempfile.TemporaryDirectory() as work:
            digest = self.serve(work, {"install.py": RECORDING_INSTALL})
            result = self.run_phase("build", work, BUNDLE_SHA256=digest, INSTALL_CHECK_RC="1")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("monitoring Create failed: check after apply", result.stdout)
            self.assertEqual([c.split()[1] for c in self.calls(work) if c.startswith("install.py")], ["apply", "check", "delete"])
            answer = self.answered(work)
            self.assertEqual((answer["Status"], answer["PhysicalResourceId"]), ("FAILED", "not-created:mon"))

    def test_a_failed_create_whose_cleanup_fails_keeps_its_id_for_the_rollback_delete(self):
        with tempfile.TemporaryDirectory() as work:
            digest = self.serve(work, {"install.py": RECORDING_INSTALL})
            self.run_phase("build", work, BUNDLE_SHA256=digest, INSTALL_APPLY_RC="1", INSTALL_DELETE_RC="1")
            answer = self.answered(work)
            self.assertEqual((answer["Status"], answer["PhysicalResourceId"]), ("FAILED", "installed:mon"))

    def test_a_failed_update_never_removes_the_install_it_was_updating(self):
        with tempfile.TemporaryDirectory() as work:
            digest = self.serve(work, {"install.py": RECORDING_INSTALL})
            self.run_phase("build", work, BUNDLE_SHA256=digest, REQUEST_TYPE="Update", NAMESPACE_OWNER="owner", INSTALL_APPLY_RC="1")
            self.assertEqual([c.split()[1] for c in self.calls(work) if c.startswith("install.py")], ["apply"])
            answer = self.answered(work)
            self.assertEqual((answer["Status"], answer["PhysicalResourceId"]), ("FAILED", "installed:mon"))

    def test_a_namespace_that_cannot_be_read_stops_before_anything_is_written(self):
        with tempfile.TemporaryDirectory() as work:
            digest = self.serve(work, {"install.py": RECORDING_INSTALL})
            self.run_phase("build", work, BUNDLE_SHA256=digest, NAMESPACE_READ_ERROR="Error from server (Forbidden): namespaces is forbidden")
            self.assertFalse([c for c in self.calls(work) if c.startswith("install.py")])
            answer = self.answered(work)
            self.assertEqual(answer["Status"], "FAILED")
            self.assertIn("cannot read namespace mon: Error from server (Forbidden)", answer["Reason"])

    def test_a_failed_delete_answers_success_and_writes_why_to_the_build_log(self):
        with tempfile.TemporaryDirectory() as work:
            digest = self.serve(work, {"install.py": RECORDING_INSTALL})
            result = self.run_phase("build", work, BUNDLE_SHA256=digest, REQUEST_TYPE="Delete", NAMESPACE_OWNER="owner", INSTALL_DELETE_RC="3")
            self.assertIn("install.py delete output", result.stdout)
            answer = self.answered(work)
            self.assertEqual(answer["Status"], "SUCCESS")
            self.assertIn("the install may remain", answer["Reason"])

    def test_a_success_that_was_not_delivered_removes_nothing_and_post_build_sends_it(self):
        with tempfile.TemporaryDirectory() as work:
            digest = self.serve(work, {"install.py": RECORDING_INSTALL})
            self.run_phase("build", work, BUNDLE_SHA256=digest, PUT_FAILURES="1")
            self.assertNotIn("delete", " ".join(c for c in self.calls(work) if c.startswith("install.py")))
            self.run_phase("post_build", work, BUNDLE_SHA256=digest, PUT_FAILURES="1")
            answer = self.answered(work, 2)
            self.assertEqual((answer["Status"], answer["PhysicalResourceId"]), ("SUCCESS", "installed:mon"))

    def test_post_build_answers_when_the_build_prepared_nothing(self):
        with tempfile.TemporaryDirectory() as work:
            result = self.run_phase("post_build", work, BUNDLE_SHA256="0" * 64, REQUEST_TYPE="Delete")
            self.assertEqual(result.returncode, 0, result.stderr)
            answer = json.loads(Path(work, "delivered.1.json").read_text())
            self.assertEqual(answer["Status"], "SUCCESS")
            self.assertEqual(answer["Data"]["Namespace"], "mon")


def lambda_source():
    """The trigger function's ZipFile text, as the template carries it."""
    lines = TEMPLATE.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == "ZipFile: |") + 1
    indent = len(lines[start]) - len(lines[start].lstrip())
    body = []
    for l in lines[start:]:
        if l.strip() and len(l) - len(l.lstrip()) < indent:
            break
        body.append(l[indent:])
    return "\n".join(body)


class TriggerTests(unittest.TestCase):
    """The Lambda that starts the build: what it answers by itself, and what it passes to the build."""
    def run_handler(self, kind, physical="", start_error=None):
        started, answers = [], []
        codebuild = mock.Mock()
        codebuild.start_build.side_effect = start_error or (lambda **kw: started.append(kw))
        fake_boto3 = types.SimpleNamespace(client=lambda name: codebuild)
        module = types.ModuleType("index")
        with mock.patch.dict(sys.modules, {"boto3": fake_boto3}):
            exec(compile(lambda_source(), "index.py", "exec"), module.__dict__)
        event = {"RequestType": kind, "ResponseURL": "https://example.invalid/response",
                 "StackId": "arn:aws:cloudformation:us-east-1:111122223333:stack/mon/abc", "RequestId": "r", "LogicalResourceId": "Monitoring",
                 "ResourceProperties": {"Namespace": "mon", "ProjectName": "p", "ClusterName": "c", "BundleSha256": "0" * 64,
                                        "S3BucketName": "b", "S3KeyPrefix": "p/", "KubectlVersion": "1.36.4"}}
        if physical:
            event["PhysicalResourceId"] = physical
        with redirect_stdout_to_null(), mock.patch("urllib.request.urlopen", lambda req: answers.append(json.loads(req.data))):
            module.handler(event, None)
        env = {v["name"]: v["value"] for v in started[0]["environmentVariablesOverride"]} if started else None
        return env, answers

    def test_the_delete_of_an_install_never_created_answers_without_a_build(self):
        env, answers = self.run_handler("Delete", "not-created:mon")
        self.assertIsNone(env)
        self.assertEqual((answers[0]["Status"], answers[0]["PhysicalResourceId"]), ("SUCCESS", "not-created:mon"))

    def test_a_delete_of_any_other_id_runs_the_removal_with_that_id(self):
        for physical in ("installed:mon", "mon-Monitoring-1A2B3C"):
            env, answers = self.run_handler("Delete", physical)
            self.assertEqual((env["REQUEST_TYPE"], env["CFN_PHYSICAL_ID"]), ("Delete", physical))
            self.assertEqual(answers, [])

    def test_a_build_that_cannot_start_is_answered_by_the_trigger(self):
        expected = {"Create": ("FAILED", "not-created:mon"), "Update": ("FAILED", "installed:mon"), "Delete": ("SUCCESS", "installed:mon")}
        for kind, (status, physical) in expected.items():
            env, answers = self.run_handler(kind, "installed:mon" if kind != "Create" else "", start_error=RuntimeError("throttled"))
            self.assertEqual((answers[0]["Status"], answers[0]["PhysicalResourceId"]), (status, physical), kind)
            self.assertIn("throttled", answers[0]["Reason"])


class redirect_stdout_to_null:
    def __enter__(self):
        self.real, sys.stdout = sys.stdout, io.StringIO()

    def __exit__(self, *exc):
        sys.stdout = self.real


class BundleTests(unittest.TestCase):
    def test_a_symbolic_link_in_the_bundle_is_refused(self):
        spec = importlib.util.spec_from_file_location("stage_template_publish", STAGE_SCRIPT)
        stage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage)
        with tempfile.TemporaryDirectory() as d:
            Path(d, "install.py").write_text("")
            Path(d, "link.py").symlink_to(Path(d, "install.py"))
            with self.assertRaises(ValueError) as e:
                stage.bundle_tar(Path(d), ["*.py"])
            self.assertIn("link.py", str(e.exception))


if __name__ == "__main__":
    unittest.main()
