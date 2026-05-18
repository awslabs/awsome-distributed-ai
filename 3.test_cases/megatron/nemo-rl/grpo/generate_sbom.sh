#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Generate Software Bill of Materials (SBOM) for the container image.
# Produces:
#   /THIRD-PARTY-LICENSES  — per-package license text (Python + system)
#   /SBOM.txt              — machine-readable package inventory

set -euo pipefail

VENV="${UV_PROJECT_ENVIRONMENT:-/opt/nemo_rl_venv}"
PYTHON="${VENV}/bin/python"
PIP_LICENSES="${VENV}/bin/pip-licenses"
OUT_LICENSES="/THIRD-PARTY-LICENSES"
OUT_SBOM="/SBOM.txt"

echo "==> Installing pip-licenses into venv"
"${VENV}/bin/pip" install --quiet --no-deps pip-licenses prettytable

# ── Python packages ──────────────────────────────────────────────
echo "==> Generating Python package licenses"

{
    echo "============================================================"
    echo " THIRD-PARTY LICENSES — Python Packages"
    echo " Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "============================================================"
    echo ""
    "${PIP_LICENSES}" --python "${PYTHON}" \
        --format=plain-vertical --with-license-file --no-license-path \
        2>/dev/null || "${PIP_LICENSES}" --python "${PYTHON}" --format=plain-vertical 2>/dev/null
} > "${OUT_LICENSES}"

# ── System packages (dpkg) ──────────────────────────────────────
echo "==> Generating system package licenses"

{
    echo ""
    echo "============================================================"
    echo " THIRD-PARTY LICENSES — System Packages (dpkg)"
    echo "============================================================"
    echo ""
    dpkg-query -W -f='${Package}\t${Version}\t${License}\n' 2>/dev/null \
        || dpkg-query -W -f='${Package}\t${Version}\n'
} >> "${OUT_LICENSES}"

# ── SBOM (machine-readable inventory) ───────────────────────────
echo "==> Generating SBOM inventory"

{
    echo "# Software Bill of Materials (SBOM)"
    echo "# Format: name | version | license"
    echo "# Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "#"
    echo "# ── Python packages ──"
    "${PIP_LICENSES}" --python "${PYTHON}" --format=csv 2>/dev/null
    echo ""
    echo "# ── System packages ──"
    dpkg-query -W -f='${Package},${Version},${License:-unknown}\n' 2>/dev/null \
        || dpkg-query -W -f='${Package},${Version},unknown\n'
} > "${OUT_SBOM}"

# ── Cleanup ─────────────────────────────────────────────────────
# Only remove pip-licenses. Keep prettytable — it's a runtime dep of pyecharts (via swanlab).
"${VENV}/bin/pip" uninstall --quiet -y pip-licenses 2>/dev/null || true

echo "==> SBOM generation complete"
echo "    ${OUT_LICENSES} ($(wc -l < "${OUT_LICENSES}") lines)"
echo "    ${OUT_SBOM} ($(wc -l < "${OUT_SBOM}") lines)"

