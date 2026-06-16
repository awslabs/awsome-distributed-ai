#!/bin/bash
# setup-openldap-server.sh — Install and configure OpenLDAP (slapd) on the login node.
# Called from CNG UserData when DeployDirectory=true and MonitoringRole=login.
#
# LDAP DB is stored on /home/ldap-db (shared OpenZFS) so it survives login
# node CNG recreation. The slapd service on a fresh node detects existing DB
# and reuses it (idempotent).
#
# Environment variables (passed from UserData):
#   LDAP_DOMAIN_SUFFIX  — e.g. "dc=cluster,dc=internal" (from DirectoryDomainSuffix)
#   LDAP_DOMAIN         — e.g. "cluster.internal" (derived)
#   LDAP_ADMIN_PASSWORD — from Secrets Manager or auto-generated
#   LDAP_BASE_GROUPS    — comma-separated default groups (default: "clusterusers,clusteradmins")

set -euo pipefail

LDAP_DOMAIN_SUFFIX="${LDAP_DOMAIN_SUFFIX:-dc=cluster,dc=internal}"
LDAP_DOMAIN="${LDAP_DOMAIN:-cluster.internal}"
LDAP_ADMIN_PASSWORD="${LDAP_ADMIN_PASSWORD:-$(openssl rand -base64 16)}"
LDAP_BASE_GROUPS="${LDAP_BASE_GROUPS:-clusterusers,clusteradmins}"
LDAP_DB_DIR="/home/ldap-db"

export DEBIAN_FRONTEND=noninteractive

echo "[ldap-server] Installing slapd + ldap-utils..."
debconf-set-selections <<EOF
slapd slapd/internal/adminpw password ${LDAP_ADMIN_PASSWORD}
slapd slapd/internal/generated_adminpw password ${LDAP_ADMIN_PASSWORD}
slapd slapd/password2 password ${LDAP_ADMIN_PASSWORD}
slapd slapd/password1 password ${LDAP_ADMIN_PASSWORD}
slapd slapd/domain string ${LDAP_DOMAIN}
slapd shared/organization string PCSCluster
slapd slapd/purge_database boolean false
slapd slapd/move_old_database boolean false
slapd slapd/no_configuration boolean false
EOF
apt-get install -y slapd ldap-utils

echo "[ldap-server] Configuring LDAP DB on shared storage (${LDAP_DB_DIR})..."
systemctl stop slapd || true

# Move DB to shared OpenZFS (/home) if not already there
if [ ! -d "${LDAP_DB_DIR}" ]; then
    mkdir -p "${LDAP_DB_DIR}"
    # Copy existing DB from default location
    if [ -d /var/lib/ldap ] && [ "$(ls -A /var/lib/ldap 2>/dev/null)" ]; then
        cp -a /var/lib/ldap/* "${LDAP_DB_DIR}/"
    fi
    chown -R openldap:openldap "${LDAP_DB_DIR}"
fi

# Point slapd at shared storage via apparmor + config override
# Update slapd DB directory in the config
if [ -d /etc/ldap/slapd.d ]; then
    # cn=config style — find the mdb backend config and update olcDbDirectory
    MDB_LDIF=$(find /etc/ldap/slapd.d -name "olcDatabase*mdb*" -o -name "olcDatabase*hdb*" | head -1)
    if [ -n "$MDB_LDIF" ] && grep -q "olcDbDirectory" "$MDB_LDIF"; then
        sed -i "s|olcDbDirectory:.*|olcDbDirectory: ${LDAP_DB_DIR}|" "$MDB_LDIF"
    fi
fi

# AppArmor: allow slapd to access /home/ldap-db
if [ -f /etc/apparmor.d/usr.sbin.slapd ]; then
    if ! grep -q "${LDAP_DB_DIR}" /etc/apparmor.d/usr.sbin.slapd; then
        sed -i "/\/var\/lib\/ldap\/ r,/a\\  ${LDAP_DB_DIR}/ r," /etc/apparmor.d/usr.sbin.slapd
        sed -i "/\/var\/lib\/ldap\/\*\* rwk,/a\\  ${LDAP_DB_DIR}/** rwk," /etc/apparmor.d/usr.sbin.slapd
        apparmor_parser -r /etc/apparmor.d/usr.sbin.slapd 2>/dev/null || true
    fi
fi

# Ensure correct ownership
chown -R openldap:openldap "${LDAP_DB_DIR}"

systemctl start slapd
systemctl enable slapd

# Wait for slapd to be ready
for i in $(seq 1 10); do
    ldapsearch -x -H ldap://localhost -b "" -s base namingContexts >/dev/null 2>&1 && break
    sleep 1
done

echo "[ldap-server] Creating base groups..."
for GROUP in $(echo "${LDAP_BASE_GROUPS}" | tr ',' ' '); do
    GID=$((3000 + RANDOM % 1000))
    ldapadd -x -H ldap://localhost -D "cn=admin,${LDAP_DOMAIN_SUFFIX}" -w "${LDAP_ADMIN_PASSWORD}" <<EOF 2>/dev/null || true
dn: cn=${GROUP},${LDAP_DOMAIN_SUFFIX}
objectClass: posixGroup
cn: ${GROUP}
gidNumber: ${GID}
EOF
done

# Create an OU for users if not exists
ldapadd -x -H ldap://localhost -D "cn=admin,${LDAP_DOMAIN_SUFFIX}" -w "${LDAP_ADMIN_PASSWORD}" <<EOF 2>/dev/null || true
dn: ou=People,${LDAP_DOMAIN_SUFFIX}
objectClass: organizationalUnit
ou: People
EOF

ldapadd -x -H ldap://localhost -D "cn=admin,${LDAP_DOMAIN_SUFFIX}" -w "${LDAP_ADMIN_PASSWORD}" <<EOF 2>/dev/null || true
dn: ou=Groups,${LDAP_DOMAIN_SUFFIX}
objectClass: organizationalUnit
ou: Groups
EOF

echo "[ldap-server] OpenLDAP server ready."
echo "[ldap-server] Admin DN: cn=admin,${LDAP_DOMAIN_SUFFIX}"
echo "[ldap-server] DB stored on: ${LDAP_DB_DIR} (shared OpenZFS, survives CNG delete)"
echo "[ldap-server] Add users with:"
echo "  ldapadd -x -H ldap://localhost -D 'cn=admin,${LDAP_DOMAIN_SUFFIX}' -w '<password>'"
