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
#   UPSTREAM_REF       (default main)
#   SKILLS='name1 name2'  (default: the curated in-guardrail set below)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLUTION_ROOT="$(cd "${HERE}/.." && pwd)"

: "${UPSTREAM_REPO_URL:=https://github.com/awslabs/agent-plugins.git}"
: "${UPSTREAM_REF:=main}"
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

# ---- clone or pull upstream ------------------------------------------------
if [[ -d "${UPSTREAM_DIR}/.git" ]]; then
    echo "==> Pulling latest upstream"
    git -C "${UPSTREAM_DIR}" fetch --depth 1 origin "${UPSTREAM_REF}"
    git -C "${UPSTREAM_DIR}" reset --hard "origin/${UPSTREAM_REF}"
else
    echo "==> Cloning upstream"
    rm -rf "${UPSTREAM_DIR}"
    git clone --depth 1 --branch "${UPSTREAM_REF}" "${UPSTREAM_REPO_URL}" "${UPSTREAM_DIR}"
fi
echo "    upstream HEAD: $(git -C "${UPSTREAM_DIR}" rev-parse --short HEAD)"
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
