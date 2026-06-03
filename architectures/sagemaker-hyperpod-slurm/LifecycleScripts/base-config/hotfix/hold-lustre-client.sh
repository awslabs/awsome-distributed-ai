#!/bin/bash

set -exuo pipefail

dpkg_hold_with_retry() {
    # Retry when dpkg frontend is locked
    for (( i=0; i<=20; i++ )); do
        echo "$1 hold" | sudo dpkg --set-selections && break || { echo To retry... ; sleep 6 ; }
    done
}

# Don't let new lustre client module brings in new kernel.
dpkg_hold_with_retry lustre-client-modules-aws
