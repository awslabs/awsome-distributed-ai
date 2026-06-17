#!/bin/bash
# =============================================================================
# Install optional extras for RLinf-on-EKS container image
#
# Called by Dockerfile with EXTRAS arg (comma-separated list).
# Each extra is a self-contained install block.
#
# Supported extras:
#   rlinf       - RLinf source (editable install in all venvs)
#
# DreamZero deps are no longer installed here -- they come from upstream
# install.sh --env wan (stage 1) plus the DreamZero tree on PYTHONPATH.
#
# Usage (from Dockerfile):
#   RUN /tmp/install_extras.sh "rlinf"
# =============================================================================
set -euo pipefail

EXTRAS="${1:-rlinf}"
VENVS="/opt/venv/openvla /opt/venv/openvla-oft /opt/venv/openpi"

echo "=== Installing extras: ${EXTRAS} ==="

install_in_venvs() {
    local pkg_dir=$1
    local pkg_name=$2
    shift 2
    # Remaining args are extra pip packages to install
    local extra_pkgs=("$@")

    for venv in ${VENVS}; do
        if [ -d "$venv" ]; then
            echo "  Installing ${pkg_name} into $(basename $venv) venv..."
            # Venv activate scripts reference PYTHONPATH/CPATH which may be unset
            set +u
            # shellcheck disable=SC1091
            . "$venv/bin/activate"
            set -u
            pip install --no-deps -e "$pkg_dir" 2>/dev/null
            for pkg in "${extra_pkgs[@]}"; do
                if [ -n "$pkg" ]; then
                    echo "    Extra: $pkg"
                    MAX_JOBS=4 pip install --no-build-isolation "$pkg" 2>/dev/null || \
                        echo "    WARNING: $pkg install failed in $(basename $venv), skipping"
                fi
            done
            set +u
            deactivate
            set -u
        fi
    done
}

# Parse comma-separated EXTRAS into array
IFS=',' read -ra EXTRA_LIST <<< "$EXTRAS"

for extra in "${EXTRA_LIST[@]}"; do
    extra=$(echo "$extra" | tr -d ' ')  # trim whitespace
    case "$extra" in
        rlinf)
            echo ""
            echo "--- Installing: RLinf source ---"
            if [ -f /workspace/RLinf/pyproject.toml ]; then
                echo "  RLinf source found, installing..."
                install_in_venvs /workspace/RLinf "RLinf"
            else
                echo "  ERROR: RLinf source not found at /workspace/RLinf."
                echo "  Ensure buildspec clones RLinf and Dockerfile COPY's it."
                exit 1
            fi
            ;;

        "")
            # Empty string from trailing comma, ignore
            ;;

        *)
            echo "  WARNING: Unknown extra '$extra', skipping."
            ;;
    esac
done

echo ""
echo "=== Extras installation complete ==="
