#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# ldap-add-user.sh — Helper to add a POSIX user to the OpenLDAP directory.
# Run on the login node as root (or with LDAP admin credentials).
#
# Usage: ./ldap-add-user.sh <username> [uid] [gid] [ssh-pub-key]
#
# Example:
#   ./ldap-add-user.sh alice 10001 3000
#   ./ldap-add-user.sh bob 10002 3000 "ssh-ed25519 AAAA..."

set -euo pipefail

USERNAME="${1:?Usage: $0 <username> [uid] [gid] [ssh-pub-key]}"
USER_UID="${2:?uid required — pick a cluster-unique uidNumber (see USER-MANAGEMENT.md). Auto-random was removed: bash RANDOM tops out at 32767 so it could not span the intended range and risked UID collisions (two users sharing a uidNumber = same POSIX principal on shared /home,/fsx)}"
USER_GID="${3:-3000}"
SSH_PUBKEY="${4:-}"

# Validate before these values flow into an LDAP DN and (for the SSH-key path)
# into filesystem paths used by cp/mkdir/chown -R. A username with "/" or ".."
# would make HOME_DIR resolve outside /home (e.g. USERNAME=../../etc →
# chown -R on /etc); a non-numeric uid/gid would corrupt the LDIF. Turns a
# malformed batch-add row into a clean error instead of a destructive op.
if [[ ! "${USERNAME}" =~ ^[a-z_][a-z0-9_-]*$ ]]; then
    echo "Invalid username '${USERNAME}': lowercase letters/digits/_/- only, starting with a letter or '_'." >&2
    exit 1
fi
if [[ ! "${USER_UID}" =~ ^[0-9]+$ ]]; then
    echo "Invalid uid '${USER_UID}': must be a positive integer." >&2
    exit 1
fi
if [[ ! "${USER_GID}" =~ ^[0-9]+$ ]]; then
    echo "Invalid gid '${USER_GID}': must be a positive integer." >&2
    exit 1
fi

# Auto-detect LDAP config from sssd.conf or environment
LDAP_DOMAIN_SUFFIX="${LDAP_DOMAIN_SUFFIX:-$(sed -n 's/^ldap_search_base[[:space:]]*=[[:space:]]*//p' /etc/sssd/sssd.conf 2>/dev/null || echo 'dc=cluster,dc=internal')}"
LDAP_DOMAIN_SUFFIX="${LDAP_DOMAIN_SUFFIX:-dc=cluster,dc=internal}"
LDAP_ADMIN_DN="cn=admin,${LDAP_DOMAIN_SUFFIX}"

echo "Adding user: ${USERNAME} (uid=${USER_UID}, gid=${USER_GID})"
echo "LDAP base: ${LDAP_DOMAIN_SUFFIX}"

# Prompt for admin password if not set
if [ -z "${LDAP_ADMIN_PASSWORD:-}" ]; then
    read -sp "LDAP admin password: " LDAP_ADMIN_PASSWORD
    echo
fi

# Create user entry
LDIF=$(cat <<EOF
dn: uid=${USERNAME},ou=People,${LDAP_DOMAIN_SUFFIX}
objectClass: inetOrgPerson
objectClass: posixAccount
objectClass: shadowAccount
uid: ${USERNAME}
cn: ${USERNAME}
sn: ${USERNAME}
uidNumber: ${USER_UID}
gidNumber: ${USER_GID}
homeDirectory: /home/${USERNAME}
loginShell: /bin/bash
userPassword: {SSHA}placeholder
EOF
)

echo "$LDIF" | ldapadd -x -H ldap://localhost -D "${LDAP_ADMIN_DN}" -w "${LDAP_ADMIN_PASSWORD}" 2>&1

# Set a random initial password (user should change via ldappasswd)
INITIAL_PW=$(openssl rand -base64 12)
ldappasswd -x -H ldap://localhost -D "${LDAP_ADMIN_DN}" -w "${LDAP_ADMIN_PASSWORD}" \
    -s "${INITIAL_PW}" "uid=${USERNAME},ou=People,${LDAP_DOMAIN_SUFFIX}"

# SSH public key (optional 4th arg): install into the user's authorized_keys
# on shared /home (visible to every node). This intentionally does NOT use the
# LDAP openssh-lpk schema — slapd here doesn't load it, and authorized_keys on
# the shared filesystem gives the same result with zero sshd/SSSD wiring.
# Requires root (this script already needs root to be useful).
if [ -n "${SSH_PUBKEY}" ]; then
    HOME_DIR="/home/${USERNAME}"
    if [ ! -d "${HOME_DIR}" ]; then
        # Creating the home dir here pre-empts pam_mkhomedir, so copy the
        # skeleton files it would have provided (.bashrc, .profile, ...).
        cp -rT /etc/skel "${HOME_DIR}"
    fi
    mkdir -p "${HOME_DIR}/.ssh"
    echo "${SSH_PUBKEY}" >> "${HOME_DIR}/.ssh/authorized_keys"
    chmod 700 "${HOME_DIR}/.ssh"
    chmod 600 "${HOME_DIR}/.ssh/authorized_keys"
    chown -R "${USER_UID}:${USER_GID}" "${HOME_DIR}"
    chmod 750 "${HOME_DIR}"
fi

echo ""
echo "User '${USERNAME}' created successfully."
echo "  UID: ${USER_UID}"
echo "  GID: ${USER_GID}"
if [ -n "${SSH_PUBKEY}" ]; then
    echo "  Home: /home/${USERNAME} (created; SSH public key installed)"
else
    echo "  Home: /home/${USERNAME} (auto-created on first login via pam_mkhomedir)"
fi
echo "  Initial password: ${INITIAL_PW}"
echo ""
echo "To add to Slurm accounting:"
echo "  sacctmgr add user ${USERNAME} account=default"
