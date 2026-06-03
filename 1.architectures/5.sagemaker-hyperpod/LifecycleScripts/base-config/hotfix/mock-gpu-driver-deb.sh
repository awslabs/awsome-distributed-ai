#!/bin/bash

set -exuo pipefail

DRV_VERSION=$(modinfo -F version nvidia)
DRV_VERSION_MAJOR=${DRV_VERSION%%.*}
MOCK_PKG=libnvidia-compute-${DRV_VERSION_MAJOR}

#apt-get -y -o DPkg::Lock::Timeout=120 update   # Most likely this has been been done by another script.
apt-get -y -o DPkg::Lock::Timeout=120 install equivs

# 1) Try exact-patch match first (preferred: mock matches the running kmod exactly).
for SUFFIX in "-1ubuntu1" "-0ubuntu1" ""; do
    if apt-cache show ${MOCK_PKG}=${DRV_VERSION}${SUFFIX} 2>/dev/null | egrep '^Package|^Version|^Provides' &> ${MOCK_PKG}; then
        echo "Found exact match for ${DRV_VERSION} with suffix: ${SUFFIX}"
        break
    fi
done

# 2) Fallback: kmod patch version isn't in apt-cache (e.g. AMI kmod 595.58.03 but
#    apt only has 595.71.05). Use the latest candidate for the same major and
#    rewrite Version: to the kmod's value so pins on the kmod patch still resolve.
if [ ! -s ${MOCK_PKG} ]; then
    LATEST_VERSION=$(apt-cache policy ${MOCK_PKG} 2>/dev/null | awk '/Candidate:/ {print $2}')
    if [ -n "${LATEST_VERSION}" ] && [ "${LATEST_VERSION}" != "(none)" ]; then
        echo "Exact ${DRV_VERSION} not in apt-cache; falling back to candidate ${LATEST_VERSION} and rewriting Version to ${DRV_VERSION}"
        apt-cache show ${MOCK_PKG}=${LATEST_VERSION} \
            | egrep '^Package|^Version|^Provides' \
            | sed "s/^Version:.*/Version: ${DRV_VERSION}/" \
            > ${MOCK_PKG}
    fi
fi

if [ ! -s ${MOCK_PKG} ]; then
    echo "Error: Could not find any ${MOCK_PKG} in apt-cache (kmod version ${DRV_VERSION})"
    exit 1
fi

equivs-build ${MOCK_PKG}
apt install -y -o DPkg::Lock::Timeout=120 ./${MOCK_PKG}_*.deb

dpkg_hold_with_retry() {
    # Retry when dpkg frontend is locked
    for (( i=0; i<=20; i++ )); do
        echo "$1 hold" | sudo dpkg --set-selections && break || { echo To retry... ; sleep 6 ; }
    done
}
dpkg_hold_with_retry ${MOCK_PKG}

