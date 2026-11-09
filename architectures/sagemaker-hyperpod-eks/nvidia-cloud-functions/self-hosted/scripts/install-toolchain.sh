#!/usr/bin/env bash
#
# Install the exact toolchain the NVCF 1.0.4 bundle requires, into a local
# directory. The system helm/helmfile are left alone.
#
# The bundle's manifest.json pins helm 3.21.4 and helmfile 1.1.9, and
# deploy/stacks/self-managed/Makefile enforces both in `check-versions`:
#
#   * Helm must be 3.x. "Helm 4 is not supported" — the check hard-fails on 4.x.
#   * Helmfile must be 1.1.x. The Makefile warns that "helmfile v1.2.0+ ...
#     breaks deployment ordering", which matters because the control plane has
#     to come up in dependency order (Cassandra and OpenBao before the services
#     that migrate against them).
#   * The helm-diff plugin is needed for helmfile diff/apply.
#
# Homebrew currently installs helm 4.x and helmfile 1.8.x, so `brew install`
# produces a toolchain this stack refuses. Hence the pinned download.
#
# Usage:
#   ./install-toolchain.sh          # installs under <repo>/.toolchain
#   source <repo>/.toolchain/env.sh # prepends it to PATH for the current shell
set -euo pipefail

HELM_VERSION="${HELM_VERSION:-3.21.4}"
HELMFILE_VERSION="${HELMFILE_VERSION:-1.1.9}"
HELM_DIFF_VERSION="${HELM_DIFF_VERSION:-3.14.1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TC="${ROOT}/.toolchain"
BIN="${TC}/bin"
PLUGINS="${TC}/helm-plugins"
mkdir -p "$BIN" "$PLUGINS" "${TC}/dl"

case "$(uname -s)" in Darwin) OS=darwin ;; Linux) OS=linux ;; *) echo "unsupported OS" >&2; exit 1 ;; esac
case "$(uname -m)" in arm64|aarch64) ARCH=arm64 ;; x86_64) ARCH=amd64 ;; *) echo "unsupported arch" >&2; exit 1 ;; esac

# --- helm 3.x --------------------------------------------------------------
if [ ! -x "${BIN}/helm" ] || ! "${BIN}/helm" version --short 2>/dev/null | grep -q "v${HELM_VERSION}"; then
  echo ">>> helm ${HELM_VERSION} (${OS}/${ARCH})"
  TGZ="${TC}/dl/helm-v${HELM_VERSION}-${OS}-${ARCH}.tar.gz"
  curl -fsSL -o "$TGZ" "https://get.helm.sh/helm-v${HELM_VERSION}-${OS}-${ARCH}.tar.gz"
  # Official published SHA256, verified before extracting.
  EXPECTED="$(curl -fsSL "https://get.helm.sh/helm-v${HELM_VERSION}-${OS}-${ARCH}.tar.gz.sha256sum" | awk '{print $1}')"
  ACTUAL="$(shasum -a 256 "$TGZ" | awk '{print $1}')"
  [ "$EXPECTED" = "$ACTUAL" ] || { echo "helm checksum mismatch: want ${EXPECTED} got ${ACTUAL}" >&2; exit 1; }
  tar -xzf "$TGZ" -C "${TC}/dl"
  install -m 0755 "${TC}/dl/${OS}-${ARCH}/helm" "${BIN}/helm"
fi
"${BIN}/helm" version --short

# --- helmfile 1.1.x --------------------------------------------------------
if [ ! -x "${BIN}/helmfile" ] || ! "${BIN}/helmfile" version --short 2>/dev/null | grep -q "${HELMFILE_VERSION}"; then
  echo ">>> helmfile ${HELMFILE_VERSION} (${OS}/${ARCH})"
  TGZ="${TC}/dl/helmfile_${HELMFILE_VERSION}_${OS}_${ARCH}.tar.gz"
  curl -fsSL -o "$TGZ" \
    "https://github.com/helmfile/helmfile/releases/download/v${HELMFILE_VERSION}/helmfile_${HELMFILE_VERSION}_${OS}_${ARCH}.tar.gz"
  mkdir -p "${TC}/dl/helmfile"
  tar -xzf "$TGZ" -C "${TC}/dl/helmfile"
  install -m 0755 "${TC}/dl/helmfile/helmfile" "${BIN}/helmfile"
fi
"${BIN}/helmfile" version --short 2>/dev/null || "${BIN}/helmfile" version 2>/dev/null | head -2

# --- helm-diff plugin -----------------------------------------------------
# Installed into the toolchain's own HELM_PLUGINS so it is not registered
# against the system helm.
if [ ! -d "${PLUGINS}/helm-diff" ]; then
  echo ">>> helm-diff ${HELM_DIFF_VERSION}"
  HELM_PLUGINS="$PLUGINS" "${BIN}/helm" plugin install \
    https://github.com/databus23/helm-diff --version "v${HELM_DIFF_VERSION}"
fi
HELM_PLUGINS="$PLUGINS" "${BIN}/helm" plugin list

# --- env ------------------------------------------------------------------
cat > "${TC}/env.sh" <<EOF
# source this to put the pinned NVCF toolchain ahead of the system one
export PATH="${BIN}:\$PATH"
export HELM_PLUGINS="${PLUGINS}"
EOF
echo
echo "toolchain ready. run:  source ${TC}/env.sh"
