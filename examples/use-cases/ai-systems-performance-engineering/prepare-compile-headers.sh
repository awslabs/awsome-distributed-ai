#!/usr/bin/env bash
# Extract exact-version headers for the recorded Ubuntu 22.04/Python 3.10 image.
set -euo pipefail
: "${1:?Usage: bash prepare-compile-headers.sh EMPTY_OUTPUT_DIRECTORY}"
output=$1
[[ ! -e "$output" ]] || { echo 'output already exists; preserve prior preparation' >&2; exit 2; }
mkdir -p "$output"
python3 - "$output" <<'PY'
import hashlib
from pathlib import Path
import sys
import urllib.request
root = Path(sys.argv[1])
name = 'libpython3.10-dev_3.10.12-1~22.04.17_amd64.deb'
url = 'https://launchpad.net/ubuntu/+archive/primary/+files/' + name
with urllib.request.urlopen(url, timeout=60) as response:
    body = response.read()
if hashlib.sha256(body).hexdigest() != '37295d7d2e142044ac76476b7876f84f5f33397cd4d95ef918d352573afc1247':
    raise ValueError('official Ubuntu header package digest mismatch')
(root / name).write_bytes(body)
(root / 'source-url.txt').write_text(url + '\n')
PY
dpkg-deb --extract "$output/libpython3.10-dev_3.10.12-1~22.04.17_amd64.deb" "$output"
test -f "$output/usr/include/python3.10/Python.h"
test -f "$output/usr/include/x86_64-linux-gnu/python3.10/pyconfig.h"
