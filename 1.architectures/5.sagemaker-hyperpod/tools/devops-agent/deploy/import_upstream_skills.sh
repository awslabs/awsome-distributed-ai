#!/usr/bin/env bash
# Stage curated upstream hyperpod-* skills from awslabs/agent-plugins into the
# solution's skills/ directory so the next `make deploy` uploads them (via
# prepare_deployment.py sync-skills + the SkillUploader custom resource).
#
# This does NOT upload anything itself — in the unified single-template model,
# `make deploy` is the only thing that talks to the Agent Space. This script
# only prepares the skills/ tree:
#   1. clone (or pull) awslabs/agent-plugins into skills/upstream/ (git-ignored)
#   2. for each curated skill, copy it to skills/<name>/ with scripts/ stripped
#      (DevOps Agent skills are non-executable documents only)
#
# Then run `make deploy` to push them. Re-runnable.
#
# Env overrides:
#   UPSTREAM_REPO_URL  (default awslabs/agent-plugins)
#   UPSTREAM_REF       (default: a pinned commit SHA — see below)
#   SKILLS='name1 name2'  (default: the curated in-guardrail set below)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLUTION_ROOT="$(cd "${HERE}/.." && pwd)"

: "${UPSTREAM_REPO_URL:=https://github.com/awslabs/agent-plugins.git}"
# Pin to a specific commit, NOT a moving branch: these staged skills are prose
# instructions the deployed agent executes against the cluster, so whatever
# upstream contains at import time becomes deployed agent behavior with no review
# step. A SHA keeps imports reproducible. To bump: pick a new commit from
# awslabs/agent-plugins that carries the curated hyperpod-* skills, verify it,
# and update this default. (The 1.0.0 tag predates these skills, so it is not a
# usable pin.)
: "${UPSTREAM_REF:=fe6b948e86dc8f15fe4c3095d13f461d0f4666f1}"
UPSTREAM_DIR="${SOLUTION_ROOT}/skills/upstream"
UPSTREAM_SKILLS_DIR="${UPSTREAM_DIR}/plugins/sagemaker-ai/skills"
SKILLS_DIR="${SOLUTION_ROOT}/skills"

# Curated default — only upstream skills whose in-guardrail (API + kubectl)
# portion is useful inside DevOps Agent. Skills whose entire procedure depends on
# SSM (issue-report, version-checker, ssm) are excluded because the DevOps Agent
# permission guardrail blocks ssm:StartSession; slurm-debugger needs controller
# SSM. See the README's "Impact on the imported skills" section.
DEFAULT_SKILLS=(
    hyperpod-cluster-debugger
    hyperpod-nccl
    hyperpod-node-debugger
    hyperpod-performance-debugger
)

if [[ -n "${SKILLS:-}" ]]; then
    read -r -a SKILLS_TO_IMPORT <<< "${SKILLS}"
else
    SKILLS_TO_IMPORT=("${DEFAULT_SKILLS[@]}")
fi

echo "==> Upstream: ${UPSTREAM_REPO_URL} (${UPSTREAM_REF})"
echo "    Skills to stage: ${SKILLS_TO_IMPORT[*]}"
echo

# ---- fetch upstream at the pinned ref --------------------------------------
# Use init + fetch + checkout FETCH_HEAD rather than 'git clone --branch', which
# does NOT accept a commit SHA. This form works for a SHA, tag, or branch, so an
# override of UPSTREAM_REF with any of them behaves.
if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
    rm -rf "${UPSTREAM_DIR}"
    mkdir -p "${UPSTREAM_DIR}"
    git -C "${UPSTREAM_DIR}" init --quiet
    git -C "${UPSTREAM_DIR}" remote add origin "${UPSTREAM_REPO_URL}"
fi
echo "==> Fetching upstream at ${UPSTREAM_REF}"
git -C "${UPSTREAM_DIR}" fetch --depth 1 origin "${UPSTREAM_REF}"
git -C "${UPSTREAM_DIR}" checkout --quiet --force FETCH_HEAD
RESOLVED_SHA="$(git -C "${UPSTREAM_DIR}" rev-parse HEAD)"
echo "    upstream HEAD: ${RESOLVED_SHA}"
# Guard against a moving ref drifting from the intended pin: if UPSTREAM_REF is a
# full 40-char SHA, the checked-out commit must match it exactly.
if [[ "${UPSTREAM_REF}" =~ ^[0-9a-f]{40}$ && "${RESOLVED_SHA}" != "${UPSTREAM_REF}" ]]; then
    echo "Error: resolved ${RESOLVED_SHA} != pinned ${UPSTREAM_REF}" >&2
    exit 1
fi
echo

# ---- stage each skill into skills/<name>/ ----------------------------------
for skill in "${SKILLS_TO_IMPORT[@]}"; do
    src="${UPSTREAM_SKILLS_DIR}/${skill}"
    dst="${SKILLS_DIR}/${skill}"
    if [[ ! -d "${src}" ]]; then
        echo "    SKIP ${skill}: not found at ${src}"
        continue
    fi
    rm -rf "${dst}"
    cp -R "${src}" "${dst}"
    if [[ -d "${dst}/scripts" ]]; then
        rm -rf "${dst}/scripts"
        echo "    staged ${skill} (scripts/ stripped)"
    else
        echo "    staged ${skill}"
    fi
    if [[ ! -f "${dst}/SKILL.md" ]]; then
        echo "    WARNING ${skill}: no SKILL.md after staging — deploy will skip it" >&2
    fi
done

echo
echo "Done. Curated upstream skills are staged under skills/."
echo "Run 'make deploy' to upload them to the Agent Space(s)."
