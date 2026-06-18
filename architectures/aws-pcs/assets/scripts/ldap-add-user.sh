#!/bin/bash
# ldap-add-user.sh — Helper to add a POSIX user to the OpenLDAP directory.
# Run on the login node as root (or with LDAP admin credentials).
#
# Usage: ./ldap-add-user.sh <username> [uid] [gid] [ssh-pub-key]
#
# Example:
#   ./ldap-add-user.sh alice 10001 3000
#   ./ldap-add-user.sh bob 10002 3000 "ssh-rsa AAAA..."

set -euo pipefail

USERNAME="${1:?Usage: $0 <username> [uid] [gid] [ssh-pub-key]}"
USER_UID="${2:-$((10000 + RANDOM % 50000))}"
USER_GID="${3:-3000}"
SSH_PUBKEY="${4:-}"

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

# Add SSH key if provided (uses openssh-lpk schema if available)
if [ -n "${SSH_PUBKEY}" ]; then
    LDIF="${LDIF}
objectClass: ldapPublicKey
sshPublicKey: ${SSH_PUBKEY}"
fi

echo "$LDIF" | ldapadd -x -H ldap://localhost -D "${LDAP_ADMIN_DN}" -w "${LDAP_ADMIN_PASSWORD}" 2>&1

# Set a random initial password (user should change via ldappasswd)
INITIAL_PW=$(openssl rand -base64 12)
ldappasswd -x -H ldap://localhost -D "${LDAP_ADMIN_DN}" -w "${LDAP_ADMIN_PASSWORD}" \
    -s "${INITIAL_PW}" "uid=${USERNAME},ou=People,${LDAP_DOMAIN_SUFFIX}"

echo ""
echo "User '${USERNAME}' created successfully."
echo "  UID: ${USER_UID}"
echo "  GID: ${USER_GID}"
echo "  Home: /home/${USERNAME} (auto-created on first login via pam_mkhomedir)"
echo "  Initial password: ${INITIAL_PW}"
echo ""
echo "To add to Slurm accounting:"
echo "  sacctmgr add user ${USERNAME} account=default"
