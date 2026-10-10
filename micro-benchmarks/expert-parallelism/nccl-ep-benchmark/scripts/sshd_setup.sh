#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# sshd_setup.sh - a per-run sshd so mpirun can start ranks in the second pod. Key-only, one
# ephemeral key generated in the launcher pod, bound to one address on a non-standard port,
# accepting only the launcher pod's address, and stopped with its keys removed by "stop".
#
#   sshd_setup.sh keygen                                  (launcher pod) prints the public key
#   sshd_setup.sh server <listen-ip> <port> <from-ip> <pubkey...>   (remote pod)
#   sshd_setup.sh check <host> <port>                     (launcher pod) ssh round trip
#   sshd_setup.sh stop                                    (both pods)
set -euo pipefail
D="${RUN_DIR:?RUN_DIR must name the per-run directory}/ssh"
mkdir -p "$D"
chmod 700 "$D"
case "${1:?mode}" in
  keygen)
    # Regenerate unless BOTH halves exist and are non-empty (a stray empty file must not block it).
    if [ ! -s "$D/id_ed25519" ] || [ ! -s "$D/id_ed25519.pub" ]; then
      rm -f "$D/id_ed25519" "$D/id_ed25519.pub"
      ssh-keygen -q -t ed25519 -N '' -C "nccl-ep-run-$(date -u +%Y%m%dT%H%M%SZ)" -f "$D/id_ed25519"
    fi
    cat "$D/id_ed25519.pub" ;;
  server)
    LISTEN="${2:?listen ip}"; PORT="${3:?port}"; FROM="${4:?from ip}"; shift 4; PUB="$*"
    [ -n "$PUB" ] || { echo "public key required" >&2; exit 2; }
    [ -x /usr/sbin/sshd ] || { echo "openssh-server missing (setup_nccl_ep_efa.sh deps installs it)" >&2; exit 3; }
    [ -f "$D/ssh_host_ed25519_key" ] || ssh-keygen -q -t ed25519 -N '' -f "$D/ssh_host_ed25519_key"
    echo "from=\"$FROM\",no-agent-forwarding,no-port-forwarding,no-pty,no-X11-forwarding,no-user-rc $PUB" > "$D/authorized_keys"
    chmod 600 "$D/authorized_keys"
    cat > "$D/sshd_config" <<EOF
ListenAddress $LISTEN:$PORT
HostKey $D/ssh_host_ed25519_key
PidFile $D/sshd.pid
AuthorizedKeysFile $D/authorized_keys
PermitRootLogin prohibit-password
AllowUsers root
PubkeyAuthentication yes
PasswordAuthentication no
KbdInteractiveAuthentication no
UsePAM no
X11Forwarding no
AllowTcpForwarding no
AllowAgentForwarding no
PermitTunnel no
StrictModes no
LogLevel VERBOSE
EOF
    mkdir -p /run/sshd
    /usr/sbin/sshd -t -f "$D/sshd_config"
    /usr/sbin/sshd -f "$D/sshd_config" -E "$D/sshd.log"
    sleep 1
    pid=$(cat "$D/sshd.pid")
    kill -0 "$pid"
    echo "sshd pid $pid listening on $LISTEN:$PORT, key-only, from=$FROM" ;;
  check)
    HOST="${2:?host}"; PORT="${3:?port}"
    ssh -p "$PORT" -i "$D/id_ed25519" -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=10 "root@$HOST" 'echo SSH-OK $(hostname)' ;;
  stop)
    if [ -f "$D/sshd.pid" ]; then
      pid=$(cat "$D/sshd.pid"); kill "$pid" 2>/dev/null || true; sleep 1
      if kill -0 "$pid" 2>/dev/null; then echo "sshd $pid still alive" >&2; exit 3; fi
      echo "sshd $pid stopped"
    fi
    rm -f "$D/id_ed25519" "$D/id_ed25519.pub" "$D/ssh_host_ed25519_key" "$D/ssh_host_ed25519_key.pub" "$D/authorized_keys"
    echo "keys removed" ;;
  *) echo "unknown mode $1" >&2; exit 2 ;;
esac
