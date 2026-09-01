#!/bin/bash
set +e

echo "=== GuardDuty VPC Endpoint Cleanup Starting ==="
echo "Region: $1"
echo "VPC ID: $2"

ENDPOINTS=$(aws ec2 describe-vpc-endpoints \
  --region $1 \
  --filters "Name=vpc-id,Values=$2" "Name=service-name,Values=*guardduty*" \
  --query 'VpcEndpoints[].VpcEndpointId' --output text 2>/dev/null || echo "")

if [ -n "$ENDPOINTS" ]; then
  echo "Found GuardDuty endpoints: $ENDPOINTS"
  echo "Retrieving security group IDs..."

  SG_IDS=$(aws ec2 describe-vpc-endpoints \
    --region $1 \
    --filters "Name=vpc-id,Values=$2" "Name=service-name,Values=*guardduty*" \
    --query 'VpcEndpoints[].Groups[].GroupId' --output text 2>/dev/null || echo "")
  
  echo "Deleting GuardDuty VPC endpoints..."
  for endpoint in $ENDPOINTS; do
    echo "Deleting endpoint: $endpoint"
    aws ec2 delete-vpc-endpoints --region $1 --vpc-endpoint-ids $endpoint 2>/dev/null || true
  done

  if [ -n "$SG_IDS" ]; then
    echo "Found security groups: $SG_IDS"
  else
    echo "No security groups found for GuardDuty endpoints"
  fi
  
  if [ -n "$SG_IDS" ]; then
    echo "Waiting for ENIs to be deleted..."
    while true; do
      ENI_COUNT=$(aws ec2 describe-network-interfaces \
        --region $1 \
        --filters "Name=group-id,Values=$SG_IDS" \
        --query 'length(NetworkInterfaces)' --output text 2>/dev/null || echo "0")
      
      if [ "$ENI_COUNT" = "0" ]; then
        echo "All ENIs deleted"
        break
      fi
      
      echo "Still $ENI_COUNT ENIs attached, waiting..."
      sleep 5
    done
    
    echo "Deleting security groups..."
    for sg in $SG_IDS; do
      echo "Deleting security group: $sg"
      aws ec2 delete-security-group --region $1 --group-id $sg 2>/dev/null || true
    done
  fi
else
  echo "No GuardDuty VPC endpoints found"
fi

echo "=== GuardDuty VPC Endpoint Cleanup Complete ==="
