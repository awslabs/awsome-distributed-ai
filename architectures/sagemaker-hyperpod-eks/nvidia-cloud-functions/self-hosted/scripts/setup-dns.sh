#!/usr/bin/env bash
#
# Create the private DNS the multi-cluster topology needs.
#
# WHY THIS IS REQUIRED
#
# NVCF derives every service hostname from `global.domain`: api.<domain>,
# api-keys.<domain>, invocation.<domain>, sis.<domain>, reval.<domain>,
# nats.<domain>, worker-api.<domain>, worker-tasks.<domain>, tasks.<domain>.
#
# The upstream CSP example sets `global.domain` to the Gateway load balancer
# hostname. For HTTP routing alone that can work, because a client may dial the
# bare load balancer and pass the per-service name only as a Host header. But
# `global.workerEndpoints.*` are URLs, and clients DNS-resolve them. AWS ELB DNS
# names resolve only exactly — there is no record for
# `worker-api.<something>.elb.amazonaws.com` — so grpc-proxy comes up, fails its
# own /health with 503, and is killed by its liveness probe in a loop:
#
#   Using fixed bearer token authentication for NVCF client
#     {"nvcf_host": "worker-api.<nlb>.elb.us-east-1.amazonaws.com:80"}
#   Liveness probe failed: HTTP probe failed with statuscode: 503
#
# A wildcard record in a Route 53 *private* hosted zone attached to the VPC
# fixes this: both clusters resolve *.<domain> to the internal NLB, and nothing
# is published to public DNS.
#
# NOTE: a wildcard does not cover the zone apex, so every URL must use a
# subdomain. The templates in infra/ are written that way.
#
# Usage:
#   ./setup-dns.sh
#
# Env (with defaults):
#   DOMAIN                private zone name            (default: nvcf.internal)
#   VPC_ID                VPC to associate              (default: HyperPod VPC)
#   AWS_REGION            region                        (default: us-east-1)
#   CONTROL_PLANE_CONTEXT kubectl context               (default: nvcf-cp)
#   GATEWAY_NAMESPACE     namespace of nvcf-gateway     (default: envoy-gateway)
set -euo pipefail

DOMAIN="${DOMAIN:-nvcf.internal}"
VPC_ID="${VPC_ID:-vpc-0123456789abcdef0}"
AWS_REGION="${AWS_REGION:-us-east-1}"
CONTROL_PLANE_CONTEXT="${CONTROL_PLANE_CONTEXT:-nvcf-cp}"
GATEWAY_NAMESPACE="${GATEWAY_NAMESPACE:-envoy-gateway}"

GATEWAY_ADDR="$(kubectl --context "$CONTROL_PLANE_CONTEXT" get gateway nvcf-gateway \
  -n "$GATEWAY_NAMESPACE" -o jsonpath='{.status.addresses[0].value}' 2>/dev/null || true)"
[ -n "$GATEWAY_ADDR" ] || { echo "nvcf-gateway has no address yet" >&2; exit 1; }
echo "gateway: ${GATEWAY_ADDR}"

ZID="$(aws route53 list-hosted-zones-by-name --dns-name "${DOMAIN}." \
  --query "HostedZones[?Name=='${DOMAIN}.'].Id|[0]" --output text 2>/dev/null || true)"
if [ "$ZID" = "None" ] || [ -z "$ZID" ]; then
  ZID="$(aws route53 create-hosted-zone --name "$DOMAIN" \
    --vpc "VPCRegion=${AWS_REGION},VPCId=${VPC_ID}" \
    --caller-reference "nvcf-$(date +%s)" \
    --hosted-zone-config "Comment=NVCF self-hosted control plane,PrivateZone=true" \
    --query 'HostedZone.Id' --output text)"
  echo "created private hosted zone ${DOMAIN} (${ZID})"
else
  echo "reusing private hosted zone ${DOMAIN} (${ZID})"
fi
ZID="${ZID##*/}"

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
# Two records are required.
#   *.${DOMAIN}             - api, api-keys, sis, reval, nats, worker-api, ...
#   *.invocation.${DOMAIN}  - the invocation callback uses a PER-POD hostname
#                             derived from the pod IP, e.g.
#                             10-0-1-23.invocation.${DOMAIN}. That is two
#                             labels deep, and a wildcard matches only one
#                             label, so *.${DOMAIN} does not cover it. Without
#                             this record the worker picks up the request and
#                             then fails request-attach with "no such host",
#                             and the caller sees a 504.
cat > "$TMP" <<EOF
{"Comment":"wildcards to internal nvcf-gateway NLB",
 "Changes":[
   {"Action":"UPSERT","ResourceRecordSet":{
     "Name":"*.${DOMAIN}","Type":"CNAME","TTL":60,
     "ResourceRecords":[{"Value":"${GATEWAY_ADDR}"}]}},
   {"Action":"UPSERT","ResourceRecordSet":{
     "Name":"*.invocation.${DOMAIN}","Type":"CNAME","TTL":60,
     "ResourceRecords":[{"Value":"${GATEWAY_ADDR}"}]}}]}
EOF
CID="$(aws route53 change-resource-record-sets --hosted-zone-id "$ZID" \
  --change-batch "file://${TMP}" --query 'ChangeInfo.Id' --output text)"
echo "upserted *.${DOMAIN} and *.invocation.${DOMAIN} -> ${GATEWAY_ADDR}"
aws route53 wait resource-record-sets-changed --id "$CID" 2>/dev/null \
  && echo "record INSYNC" || echo "record still propagating"

echo
echo "DOMAIN=${DOMAIN}"
echo "verify from inside the VPC (not from your laptop — the zone is private):"
echo "  kubectl --context ${CONTROL_PLANE_CONTEXT} run dnscheck --rm -it --restart=Never \\"
echo "    --image=public.ecr.aws/docker/library/busybox:1.36 -- nslookup api.${DOMAIN}"
