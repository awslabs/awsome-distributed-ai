#!/bin/bash
# setup-ldap-client.sh — Configure SSSD LDAP client on compute nodes.
# Called from CNG UserData when DeployDirectory=true and MonitoringRole=compute (or none).
#
# Environment variables (passed from UserData):
#   LDAP_SERVER_URI     — e.g. "ldap://10.1.x.x" (login node private IP)
#   LDAP_DOMAIN_SUFFIX  — e.g. "dc=cluster,dc=internal"
#   LDAP_DOMAIN         — e.g. "cluster.internal"

set -euo pipefail

LDAP_SERVER_URI="${LDAP_SERVER_URI:?LDAP_SERVER_URI must be set}"
LDAP_DOMAIN_SUFFIX="${LDAP_DOMAIN_SUFFIX:-dc=cluster,dc=internal}"
LDAP_DOMAIN="${LDAP_DOMAIN:-cluster.internal}"

export DEBIAN_FRONTEND=noninteractive

echo "[ldap-client] Running apt-get update..."
apt-get update -qq

echo "[ldap-client] Installing SSSD + LDAP client packages..."
apt-get install -y sssd libpam-sss libnss-sss ldap-utils

echo "[ldap-client] Configuring SSSD..."
cat > /etc/sssd/sssd.conf <<EOF
[sssd]
config_file_version = 2
services = nss, pam
domains = ${LDAP_DOMAIN}

[domain/${LDAP_DOMAIN}]
id_provider = ldap
auth_provider = ldap
ldap_uri = ${LDAP_SERVER_URI}
ldap_search_base = ${LDAP_DOMAIN_SUFFIX}
ldap_user_search_base = ou=People,${LDAP_DOMAIN_SUFFIX}
ldap_group_search_base = ou=Groups,${LDAP_DOMAIN_SUFFIX}
ldap_default_bind_dn = cn=admin,${LDAP_DOMAIN_SUFFIX}
ldap_default_authtok_type = password
ldap_id_use_start_tls = false
cache_credentials = true
enumerate = true
default_shell = /bin/bash
fallback_homedir = /home/%u

# UID/GID range for LDAP users (avoid collision with system users)
min_id = 10000
max_id = 60000
EOF

chmod 600 /etc/sssd/sssd.conf

# Enable pam_mkhomedir for auto home directory creation on first login
if ! grep -q pam_mkhomedir /etc/pam.d/common-session; then
    echo "session optional pam_mkhomedir.so skel=/etc/skel umask=0022" >> /etc/pam.d/common-session
fi

# Configure NSS to use sss
if ! grep -q sss /etc/nsswitch.conf; then
    sed -i 's/^passwd:.*/passwd:         files systemd sss/' /etc/nsswitch.conf
    sed -i 's/^group:.*/group:          files systemd sss/' /etc/nsswitch.conf
    sed -i 's/^shadow:.*/shadow:         files sss/' /etc/nsswitch.conf
fi

systemctl enable sssd
systemctl restart sssd

echo "[ldap-client] SSSD LDAP client configured."
echo "[ldap-client] Server: ${LDAP_SERVER_URI}"
echo "[ldap-client] Search base: ${LDAP_DOMAIN_SUFFIX}"
echo "[ldap-client] Verify: getent passwd (should show LDAP users after they are created)"
