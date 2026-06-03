#!/bin/bash

set -exuo pipefail

DRV_VERSION=$(modinfo -F version nvidia)
DRV_VERSION_MAJOR=${DRV_VERSION%%.*}
MOCK_PKG=libnvidia-compute-${DRV_VERSION_MAJOR}

#apt-get -y -o DPkg::Lock::Timeout=120 update   # Most likely this has been been done by another script.
apt-get -y -o DPkg::Lock::Timeout=120 install equivs

# Try different package revision suffixes to find the correct one
for SUFFIX in "-1ubuntu1" "-0ubuntu1" ""; do
    if apt-cache show ${MOCK_PKG}=${DRV_VERSION}${SUFFIX} 2>/dev/null | egrep '^Package|^Version|^Provides' &> ${MOCK_PKG}; then
        echo "Found package with suffix: ${SUFFIX}"
        break
    fi
done

# Check if we successfully created the control file
if [ ! -s ${MOCK_PKG} ]; then
    echo "Error: Could not find package ${MOCK_PKG} with version ${DRV_VERSION} in apt-cache"
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

