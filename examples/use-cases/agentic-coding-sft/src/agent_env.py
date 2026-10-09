# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Sandboxed tool environment for one bug-fix task + parser for Qwen3 tool calls.

The model is told the repo lives at /testbed. Each task really lives in its own directory;
paths are rewritten in both directions so the model only ever sees /testbed.

Tools mirror SWE-agent's (`bash`, `str_replace_editor`, `submit`), the tools in the training
traces; output strings follow SWE-agent's wording.

SECURITY: commands come from the model. `bash` runs as an unprivileged uid (default 65534,
nobody) in a new process group with a timeout, inside a disposable pod; the editor refuses paths
outside the task dir. That keeps a stray `rm -rf` away from mounted volumes (owned by root), but
it is not a hardened sandbox (no network isolation).
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys

VIRTUAL = "/testbed"
MAX_OBS_CHARS = 8000
BASH_TIMEOUT = 60

TC_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def parse_assistant(text):
    """Model output -> (content, [(name, args)], n_invalid). Thinking (if any) is dropped."""
    for tok in ("<|im_end|>", "<|endoftext|>"):
        text = text.replace(tok, "")
    body = text.split("</think>")[-1]
    content = body.split("<tool_call>")[0].strip()
    calls, bad = [], 0
    for m in TC_RE.finditer(body):
        try:
            obj = json.loads(m.group(1), strict=False)  # allow raw newlines in strings
            args = obj.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args, strict=False)
            if not isinstance(obj.get("name"), str) or not isinstance(args, dict):
                raise ValueError("bad shape")
            calls.append((obj["name"], args))
        except Exception:
            bad += 1
    if body.count("<tool_call>") > body.count("</tool_call>"):
        bad += 1
    return content, calls, bad


def truncate(s):
    if len(s) <= MAX_OBS_CHARS:
        return s
    half = MAX_OBS_CHARS // 2
    return (s[:half] + f"\n<response clipped: {len(s) - MAX_OBS_CHARS} characters omitted>\n"
            + s[-half:])


def shim_bin(work_root):
    """A bin dir with `python` -> this interpreter (some images only ship `python3`)."""
    d = f"{work_root}/.bin"
    os.makedirs(d, exist_ok=True)
    for name in ("python", "python3"):
        p = f"{d}/{name}"
        if not os.path.exists(p):
            os.symlink(sys.executable, p)
    return d


class TaskEnv:
    def __init__(self, task, tasks_root, work_root, uid=65534):
        self.task = task
        self.pristine = f"{tasks_root}/{task['id']}/repo"
        os.makedirs(work_root, exist_ok=True)
        self.dir = f"{os.path.realpath(work_root)}/{task['id']}"  # realpath: /tmp may be a symlink
        shutil.rmtree(self.dir, ignore_errors=True)
        shutil.copytree(self.pristine, self.dir)
        self.bin = shim_bin(work_root)
        # A VS Code workspace is normally a git repo, and the training traces use git.
        subprocess.run(["bash", "-c", "git init -q && git add -A && git -c user.name=dev "
                        "-c user.email=dev@example.com commit -qm init"], cwd=self.dir,
                       capture_output=True)
        self.uid = uid if (uid is not None and uid >= 0 and os.geteuid() == 0) else None
        self._own(self.dir)
        self.undo = {}
        self.stats = {"bash": 0, "view": 0, "create": 0, "str_replace": 0, "insert": 0,
                      "undo_edit": 0, "submit": 0, "edit_errors": 0, "edits_applied": 0,
                      "invalid_calls": 0, "pytest_runs": 0}

    # ------------------------------------------------------------ path mapping
    def _own(self, path):
        if self.uid is None:
            return
        for root, dirs, files in os.walk(path):
            os.chown(root, self.uid, self.uid)
            for f in files:
                os.chown(os.path.join(root, f), self.uid, self.uid)

    def to_real(self, s):
        return s.replace(VIRTUAL, self.dir)

    def to_virtual(self, s):
        return s.replace(self.dir, VIRTUAL)

    def _resolve(self, path):
        if not isinstance(path, str) or not path.startswith("/"):
            return None, (f"The path {path} is not an absolute path, it should start with `/`. "
                          f"Maybe you meant {VIRTUAL}/{str(path).lstrip('./')}?")
        real = os.path.realpath(self.to_real(path))
        if real != self.dir and not real.startswith(self.dir + os.sep):
            return None, f"The path {path} is outside the repository {VIRTUAL}."
        return real, None

    # ------------------------------------------------------------ tools
    def call(self, name, args):
        """Run one tool call. Returns (observation, done)."""
        try:
            if name == "bash":
                cmd = args.get("command")
                if not isinstance(cmd, str) or not cmd.strip():
                    self.stats["invalid_calls"] += 1
                    return "Parameter `command` is required for tool `bash`.", False
                self.stats["bash"] += 1
                if "pytest" in cmd or "tests/" in cmd:
                    self.stats["pytest_runs"] += 1
                return self.bash(cmd), False
            if name == "str_replace_editor":
                return self.editor(args), False
            if name == "submit":
                self.stats["submit"] += 1
                return "Submitted.", True
            self.stats["invalid_calls"] += 1
            return (f"Unknown tool `{name}`. Available tools: bash, str_replace_editor, submit.",
                    False)
        except Exception as e:  # never let a tool crash the episode
            return f"Tool error: {type(e).__name__}: {e}", False

    def bash(self, cmd):
        env = {"PATH": self.bin + ":" + os.environ.get("PATH", "/usr/bin:/bin"), "HOME": self.dir,
               "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8",
               "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "safe.directory",
               "GIT_CONFIG_VALUE_0": "*"}
        kw = {"user": self.uid, "group": self.uid} if self.uid is not None else {}
        p = subprocess.Popen(["bash", "-c", self.to_real(cmd)], cwd=self.dir, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True, **kw)
        try:
            out, _ = p.communicate(timeout=BASH_TIMEOUT)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.communicate()
            return f"The command took too long to execute (>{BASH_TIMEOUT}s) and was killed."
        out = out.decode(errors="replace").strip()
        if not out:
            return "Your command ran successfully and did not produce any output."
        return truncate(self.to_virtual(out))

    def editor(self, a):
        cmd = a.get("command")
        if cmd not in ("view", "create", "str_replace", "insert", "undo_edit"):
            self.stats["invalid_calls"] += 1
            return (f"Unrecognized command {cmd}. The allowed commands for the str_replace_editor "
                    "tool are: view, create, str_replace, insert, undo_edit")
        self.stats[cmd] += 1
        path = a.get("path")
        real, err = self._resolve(path)
        if err:
            self.stats["edit_errors"] += cmd != "view"
            return err

        if cmd == "view":
            if os.path.isdir(real):
                lines = [path]
                for root, dirs, files in os.walk(real):
                    dirs[:] = sorted(d for d in dirs if not d.startswith((".", "__pycache__")))
                    depth = root[len(real):].count(os.sep)
                    if depth >= 2:
                        dirs[:] = []
                    for n in sorted(dirs) + sorted(f for f in files if not f.startswith(".")):
                        lines.append(self.to_virtual(os.path.join(root, n)))
                return (f"Here's the files and directories up to 2 levels deep in {path}, "
                        "excluding hidden items:\n" + "\n".join(sorted(set(lines))))
            if not os.path.exists(real):
                return f"The path {path} does not exist. Please provide a valid path."
            lines = open(real, errors="replace").read().split("\n")
            start, end = 1, len(lines)
            vr = a.get("view_range")
            if isinstance(vr, list) and len(vr) == 2:
                start = max(1, int(vr[0]))
                end = len(lines) if int(vr[1]) == -1 else min(len(lines), int(vr[1]))
            body = "\n".join(f"{i:6}\t{lines[i - 1]}" for i in range(start, end + 1))
            return truncate(f"Here's the result of running `cat -n` on {path}:\n{body}\n")

        if cmd == "create":
            if os.path.exists(real):
                self.stats["edit_errors"] += 1
                return (f"File already exists at: {path}. Cannot overwrite files using command "
                        "`create`.")
            if not isinstance(a.get("file_text"), str):
                self.stats["edit_errors"] += 1
                return "Parameter `file_text` is required for command: create"
            os.makedirs(os.path.dirname(real), exist_ok=True)
            open(real, "w").write(a["file_text"])
            self._own(os.path.dirname(real))
            self.stats["edits_applied"] += 1
            return f"File created successfully at: {path}"

        if not os.path.isfile(real):
            self.stats["edit_errors"] += 1
            return f"The path {path} does not exist or is not a file."
        text = open(real, errors="replace").read()

        if cmd == "undo_edit":
            if not self.undo.get(real):
                self.stats["edit_errors"] += 1
                return f"No edit history found for {path}."
            open(real, "w").write(self.undo[real].pop())
            return f"Last edit to {path} undone successfully."

        if cmd == "str_replace":
            old = a.get("old_str")
            new = a.get("new_str") or ""
            if not isinstance(old, str) or old == "":
                self.stats["edit_errors"] += 1
                return "Parameter `old_str` is required for command: str_replace"
            n = text.count(old)
            if n == 0:
                self.stats["edit_errors"] += 1
                return (f"No replacement was performed, old_str `{old}` did not appear verbatim "
                        f"in {path}.")
            if n > 1:
                self.stats["edit_errors"] += 1
                lines = [i + 1 for i, l in enumerate(text.split("\n")) if old.split("\n")[0] in l]
                return (f"No replacement was performed. Multiple occurrences of old_str `{old}` "
                        f"in lines {lines}. Please ensure it is unique")
            self.undo.setdefault(real, []).append(text)
            new_text = text.replace(old, new)
            open(real, "w").write(new_text)
            self.stats["edits_applied"] += 1
            line = text.split(old)[0].count("\n") + 1
            return self._snippet(path, new_text, line, new.count("\n"), "edited")

        # insert
        try:
            at = int(a.get("insert_line"))
        except Exception:
            self.stats["edit_errors"] += 1
            return "Parameter `insert_line` is required for command: insert"
        lines = text.split("\n")
        if not 0 <= at <= len(lines):
            self.stats["edit_errors"] += 1
            return (f"Invalid `insert_line` parameter: {at}. It should be within the range of "
                    f"lines of the file: [0, {len(lines)}]")
        new = a.get("new_str") or ""
        self.undo.setdefault(real, []).append(text)
        new_text = "\n".join(lines[:at] + new.split("\n") + lines[at:])
        open(real, "w").write(new_text)
        self.stats["edits_applied"] += 1
        return self._snippet(path, new_text, at + 1, new.count("\n"), "edited")

    def _snippet(self, path, text, line, extra, verb):
        lines = text.split("\n")
        s, e = max(1, line - 4), min(len(lines), line + extra + 4)
        body = "\n".join(f"{i:6}\t{lines[i - 1]}" for i in range(s, e + 1))
        return (f"The file {path} has been {verb}. Here's the result of running `cat -n` on a "
                f"snippet of {path}:\n{body}\nReview the changes and make sure they are as "
                "expected. Edit the file again if necessary.")

    # ------------------------------------------------------------ grading
    def grade(self, timeout=20):
        """Copy the final repo, restore pristine tests, run every test file directly."""
        g = self.dir + ".grade"
        shutil.rmtree(g, ignore_errors=True)
        shutil.copytree(self.dir, g, ignore=shutil.ignore_patterns("__pycache__", ".git"))
        shutil.rmtree(f"{g}/tests", ignore_errors=True)
        shutil.copytree(f"{self.pristine}/tests", f"{g}/tests")
        self._own(g)
        kw = {"user": self.uid, "group": self.uid} if self.uid is not None else {}
        res = {}
        for n in self.task["modules"]:
            try:
                r = subprocess.run([sys.executable, f"tests/test_{n}.py"], cwd=g,
                                   capture_output=True, timeout=timeout,
                                   stdin=subprocess.DEVNULL, **kw)
                res[n] = r.returncode == 0
            except subprocess.TimeoutExpired:
                res[n] = False
        shutil.rmtree(g, ignore_errors=True)
        target = self.task["entry_point"]

        def _read(p):
            try:
                return open(p).read()
            except (FileNotFoundError, IsADirectoryError, OSError):
                return None  # file missing/unreadable (e.g. model deleted it)

        final = _read(f"{self.dir}/{self.task['module']}")
        pristine = _read(f"{self.pristine}/{self.task['module']}")
        return {"target_fixed": res[target],
                "others_intact": all(v for k, v in res.items() if k != target),
                "resolved": all(res.values()),
                "touched_target": final != pristine}