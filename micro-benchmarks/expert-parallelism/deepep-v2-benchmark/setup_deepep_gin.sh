#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Build DeepEP V2 with the NCCL GIN backend for AWS EFA clusters.
#
# This script only builds/installs the DeepEP Python package.

set -euo pipefail

log() {
    printf '%s\n' "$*" >&2
}

warn() {
    printf 'WARNING: %s\n' "$*" >&2
}

die() {
    printf 'ERROR: %s\n' "$*" >&2 || true
    exit 1
}

PROG_NAME="${0##*/}"
if [[ -z "$PROG_NAME" || "$PROG_NAME" == "bash" || "$PROG_NAME" == "-bash" ]]; then
    PROG_NAME="setup_deepep_gin.sh"
fi

print_help() {
    cat <<EOF
Usage: ${PROG_NAME} [options]

Build DeepEP V2 with the NCCL GIN / EFA-GDA backend and install it into the
active Python environment.

With no arguments, clones DeepEP from https://github.com/amazon-contributing/DeepEP.git
at the pinned ref, points the build at NCCL under --nccl-root, and installs DeepEP.

Options:
  --deepep-ref <ref>        DeepEP git ref to checkout (commit, branch, tag,
                            or PR ref like refs/pull/N/head)
                            (default: main)
  --deepep-prefix <path>    Directory to clone/install DeepEP into
                            (default: /opt/amazon/deepep)
  --deepep-src <path>       Build DeepEP from this existing source tree instead
                            of cloning (default: unset; clone). Useful for
                            rebuilding an already-baked tree against a different
                            torch, and for local development checkouts.
  --nccl-root <path>        Root of the GIN-capable NCCL install (must contain
                            include/nccl.h, include/nccl_device.h, lib/libnccl.so*).
                            Exported to the build as EP_NCCL_ROOT_DIR.
                            (default: /opt/nccl/build)
  --python <cmd|path>       Python interpreter (default: python3)
  --pip <cmd|path>          pip command (default: pip3)
  --skip-checks             Skip Python prerequisite validation (torch, pip,
                            ninja) (default: off)
  -h, --help                Print this usage information and exit

Environment:
  TORCH_CUDA_ARCH_LIST      GPU architectures, dotted and ;-separated
                            (default: 9.0;10.0). 9.0=Hopper, 10.0/10.3=Blackwell.
  EP_NCCL_ROOT_DIR          Used when --nccl-root is not given (an explicit
                            --nccl-root wins).

EOF
    exit 0
}

# DeepEP is pinned to the amazon-contributing fork; the benchmark supports no
# other source. Test a DeepEP change by pointing --deepep-ref at its ref.
readonly DEEPEP_REPO="https://github.com/amazon-contributing/DeepEP.git"

parse_args() {
    DEEPEP_REF="main"
    DEEPEP_PREFIX="/opt/amazon/deepep"
    DEEPEP_SRC=""
    NCCL_ROOT="/opt/nccl/build"
    NCCL_ROOT_FLAG_SET="false"
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
    SKIP_CHECKS="false"
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST-9.0;10.0}"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --deepep-ref)
                [[ $# -ge 2 ]] || die "option '$1' requires a value"
                DEEPEP_REF="$2"; shift 2 ;;
            --deepep-prefix)
                [[ $# -ge 2 ]] || die "option '$1' requires a value"
                DEEPEP_PREFIX="$2"; shift 2 ;;
            --deepep-src)
                [[ $# -ge 2 ]] || die "option '$1' requires a value"
                DEEPEP_SRC="$2"; shift 2 ;;
            --nccl-root)
                [[ $# -ge 2 ]] || die "option '$1' requires a value"
                NCCL_ROOT="$2"; NCCL_ROOT_FLAG_SET="true"; shift 2 ;;
            --python)
                [[ $# -ge 2 ]] || die "option '$1' requires a value"
                PYTHON_CMD="$2"; shift 2 ;;
            --pip)
                [[ $# -ge 2 ]] || die "option '$1' requires a value"
                PIP_CMD="$2"; shift 2 ;;
            --skip-checks)
                SKIP_CHECKS="true"; shift ;;
            -h|--help)
                print_help ;;
            *)
                die "unrecognized argument: '$1'" ;;
        esac
    done

    # An explicit --nccl-root wins; otherwise EP_NCCL_ROOT_DIR from the
    # environment overrides the default (matching DeepEP's own precedence).
    if [[ "$NCCL_ROOT_FLAG_SET" != "true" && -n "${EP_NCCL_ROOT_DIR:-}" ]]; then
        log "Using EP_NCCL_ROOT_DIR from the environment: ${EP_NCCL_ROOT_DIR}"
        NCCL_ROOT="$EP_NCCL_ROOT_DIR"
    fi
}

validate_arch_list() {
    local arch_list="$1"
    if [[ ! "$arch_list" =~ ^[0-9]+\.[0-9]+(;[0-9]+\.[0-9]+)*$ ]]; then
        die "invalid TORCH_CUDA_ARCH_LIST: '$arch_list' (expected non-empty, dotted, ;-separated architectures, e.g. 9.0;10.0)"
    fi
}

compute_disable_ptx() {
    # DeepEP's setup.py requires DISABLE_AGGRESSIVE_PTX_INSTRS=1 for any arch
    # list other than a bare "9.0" (pure Hopper). Mirror that here.
    local arch_list="${1-${TORCH_CUDA_ARCH_LIST:-}}"
    if [[ "$arch_list" == "9.0" ]]; then
        printf '0\n'
    else
        printf '1\n'
    fi
}


check_python_prereqs() {
    local python_cmd="${PYTHON_CMD:-python3}"
    local pip_cmd="${PIP_CMD:-pip3}"

    if [[ "$python_cmd" == /* ]]; then
        [[ -x "$python_cmd" ]] || die "python command not found: '$python_cmd' is not an executable file"
    else
        command -v "$python_cmd" >/dev/null 2>&1 || die "python command not found: '$python_cmd' is not on PATH"
    fi

    if [[ "$pip_cmd" == /* ]]; then
        [[ -x "$pip_cmd" ]] || die "pip command not found: '$pip_cmd' is not an executable file"
    else
        command -v "$pip_cmd" >/dev/null 2>&1 || die "pip command not found: '$pip_cmd' is not on PATH"
    fi

    local missing=()
    if ! "$python_cmd" -c "import torch" >/dev/null 2>&1; then
        missing+=("torch (Python module)")
    fi
    if ! "$python_cmd" -m pip --version >/dev/null 2>&1; then
        missing+=("pip")
    fi
    if ! command -v ninja >/dev/null 2>&1; then
        missing+=("ninja")
    fi

    if (( ${#missing[@]} > 0 )); then
        local IFS=', '
        die "missing prerequisites: ${missing[*]} (install them before running this script)"
    fi
}

validate_nccl_root() {
    [[ -d "$NCCL_ROOT" ]] || die "NCCL root does not exist or is not a directory: '$NCCL_ROOT' (build the GIN-capable NCCL first, or pass --nccl-root)"
    [[ -f "$NCCL_ROOT/include/nccl.h" ]] || die "NCCL root missing include/nccl.h: '$NCCL_ROOT' does not look like an NCCL build/install tree"
    # nccl_device.h is the GIN header DeepEP V2 links against; its absence means
    # this NCCL predates GIN and DeepEP V2 will not compile.
    [[ -f "$NCCL_ROOT/include/nccl_device.h" ]] || \
        die "NCCL root missing include/nccl_device.h: '$NCCL_ROOT' is not a GIN-capable NCCL (DeepEP V2 requires the GIN backend)"
    if ! ls "$NCCL_ROOT"/lib/libnccl.so* >/dev/null 2>&1; then
        die "NCCL root missing lib/libnccl.so*: '$NCCL_ROOT' does not contain a built libnccl"
    fi
    log "Using GIN-capable NCCL at ${NCCL_ROOT}"
}

acquire_deepep() {
    if [[ -z "${DEEPEP_SRC:-}" ]]; then
        log "Cloning DeepEP from ${DEEPEP_REPO} into ${DEEPEP_PREFIX} at ${DEEPEP_REF}..."

        rm -rf "$DEEPEP_PREFIX"
        mkdir -p "$(dirname "$DEEPEP_PREFIX")"

        git clone "$DEEPEP_REPO" "$DEEPEP_PREFIX" \
            || die "failed to clone DeepEP from ${DEEPEP_REPO}"
        # If the ref is a pull-request ref (refs/pull/N/head) it is not fetched
        # by a standard clone; fetch it explicitly before checking out.
        if [[ "$DEEPEP_REF" == refs/pull/* ]]; then
            git -C "$DEEPEP_PREFIX" fetch origin "$DEEPEP_REF" \
                || die "failed to fetch PR ref '${DEEPEP_REF}'"
            git -C "$DEEPEP_PREFIX" checkout FETCH_HEAD \
                || die "failed to checkout PR ref '${DEEPEP_REF}'"
        else
            git -C "$DEEPEP_PREFIX" checkout "$DEEPEP_REF" \
                || die "failed to checkout DeepEP ref '${DEEPEP_REF}'"
        fi
        # third-party/fmt is a submodule DeepEP's setup.py includes.
        git -C "$DEEPEP_PREFIX" submodule update --init --recursive \
            || die "failed to init DeepEP submodules (third-party/fmt)"

        log "DeepEP cloned and checked out at ${DEEPEP_REF}"
    else
        log "Using existing DeepEP source tree at ${DEEPEP_SRC}..."
        [[ -d "$DEEPEP_SRC" ]] || die "DeepEP source path does not exist or is not a directory: '${DEEPEP_SRC}'"
        # V2 marker: the NCCL GIN backend source. Its presence distinguishes a
        # V2 tree from a legacy V1 (NVSHMEM-only) tree.
        [[ -f "$DEEPEP_SRC/csrc/kernels/backend/nccl.cu" ]] || \
            die "DeepEP source path does not contain csrc/kernels/backend/nccl.cu: '${DEEPEP_SRC}' is not a V2 (NCCL GIN) tree"
        DEEPEP_PREFIX="$DEEPEP_SRC"
        log "DeepEP V2 source tree validated at ${DEEPEP_SRC}"
    fi
}

build_deepep() {
    local disable_ptx
    disable_ptx="$(compute_disable_ptx)"

    export EP_NCCL_ROOT_DIR="$NCCL_ROOT"
    export LD_LIBRARY_PATH="${NCCL_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LIBRARY_PATH="${NCCL_ROOT}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

    cd "$DEEPEP_PREFIX"

    log "Building and installing DeepEP V2..."
    "$PIP_CMD" uninstall -y deep_ep 2>/dev/null || true

    DISABLE_AGGRESSIVE_PTX_INSTRS="$disable_ptx" \
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    EP_NCCL_ROOT_DIR="$NCCL_ROOT" \
        "$PIP_CMD" install -vv --no-build-isolation . \
        || die "DeepEP build/install failed"

    log "DeepEP V2 installed successfully"
}

print_completion_notes() {
    log ""
    log "=== DeepEP V2 (GIN) setup complete ==="
    log ""
    log "Runtime environment for the NCCL GIN / EFA-GDA backend:"
    log ""
    log "  export LD_LIBRARY_PATH=${NCCL_ROOT}/lib:\${LD_LIBRARY_PATH}"
    log "  export FI_PROVIDER=efa"
    log "  export NCCL_NET_PLUGIN=/path/to/libnccl-net-ofi.so   # aws-ofi-nccl (also set NCCL_GIN_PLUGIN)"
    log ""
    log "EFA-GDA also requires the HOST EFA kernel driver to support the"
    log "comp-counter API (efadv_create_comp_cntr). The container ships only the"
    log "userspace stack; the node's EFA driver must provide the kernel side."
    log ""
}

main() {
    parse_args "$@"

    validate_arch_list "$TORCH_CUDA_ARCH_LIST"
    if [[ "$SKIP_CHECKS" != "true" ]]; then
        check_python_prereqs
    fi

    validate_nccl_root
    acquire_deepep
    build_deepep
    print_completion_notes
}

main "$@"
