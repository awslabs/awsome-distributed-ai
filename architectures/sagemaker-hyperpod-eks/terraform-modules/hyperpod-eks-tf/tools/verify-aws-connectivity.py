#!/usr/bin/env python3
"""
AWS Service Connectivity Verification Script

This script verifies network connectivity to essential AWS service endpoints
for HyperPod EKS deployment in closed network environments.
"""

import boto3
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict

def get_aws_region() -> str:
    """Get the current AWS region from boto3 session."""
    try:
        session = boto3.Session()
        return session.region_name or 'us-east-1'
    except Exception:
        return 'us-east-1'

def test_endpoint_connectivity(endpoint: str, port: int = 443, timeout: int = 10) -> Tuple[str, bool, str]:
    """
    Test connectivity to an endpoint.
    
    Args:
        endpoint: The endpoint hostname to test
        port: Port to test (default: 443 for HTTPS)
        timeout: Connection timeout in seconds
        
    Returns:
        Tuple of (endpoint, success, message)
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((endpoint, port))
        sock.close()
        
        if result == 0:
            return (endpoint, True, f"✅ Connected successfully")
        else:
            return (endpoint, False, f"❌ Connection failed (error code: {result})")
            
    except socket.gaierror as e:
        return (endpoint, False, f"❌ DNS resolution failed: {e}")
    except Exception as e:
        return (endpoint, False, f"❌ Connection error: {e}")

def test_boto3_connectivity(region: str) -> Dict[str, Tuple[bool, str]]:
    """
    Test AWS service connectivity using boto3 clients.
    
    Args:
        region: AWS region to test
        
    Returns:
        Dictionary of service results
    """
    results = {}
    
    # Test S3
    try:
        s3_client = boto3.client('s3', region_name=region)
        s3_client.list_buckets()
        results['S3'] = (True, "✅ S3 API accessible")
    except Exception as e:
        results['S3'] = (False, f"❌ S3 API error: {str(e)[:100]}...")
    
    # Test ECR
    try:
        ecr_client = boto3.client('ecr', region_name=region)
        ecr_client.describe_repositories(maxResults=1)
        results['ECR'] = (True, "✅ ECR API accessible")
    except Exception as e:
        results['ECR'] = (False, f"❌ ECR API error: {str(e)[:100]}...")
    
    # Test SSM
    try:
        ssm_client = boto3.client('ssm', region_name=region)
        ssm_client.describe_parameters(MaxResults=1)
        results['SSM'] = (True, "✅ SSM API accessible")
    except Exception as e:
        results['SSM'] = (False, f"❌ SSM API error: {str(e)[:100]}...")
    
    # Test EC2
    try:
        ec2_client = boto3.client('ec2', region_name=region)
        ec2_client.describe_regions(RegionNames=[region])
        results['EC2'] = (True, "✅ EC2 API accessible")
    except Exception as e:
        results['EC2'] = (False, f"❌ EC2 API error: {str(e)[:100]}...")
    
    # Test STS
    try:
        sts_client = boto3.client('sts', region_name=region)
        sts_client.get_caller_identity()
        results['STS'] = (True, "✅ STS API accessible")
    except Exception as e:
        results['STS'] = (False, f"❌ STS API error: {str(e)[:100]}...")
    
    # Test CloudWatch Logs
    try:
        logs_client = boto3.client('logs', region_name=region)
        logs_client.describe_log_groups(limit=1)
        results['CloudWatch Logs'] = (True, "✅ CloudWatch Logs API accessible")
    except Exception as e:
        results['CloudWatch Logs'] = (False, f"❌ CloudWatch Logs API error: {str(e)[:100]}...")
    
    # Test CloudWatch Monitoring
    try:
        cloudwatch_client = boto3.client('cloudwatch', region_name=region)
        # Use list_metrics without parameters for better compatibility
        cloudwatch_client.list_metrics()
        results['CloudWatch Monitoring'] = (True, "✅ CloudWatch Monitoring API accessible")
    except Exception as e:
        results['CloudWatch Monitoring'] = (False, f"❌ CloudWatch Monitoring API error: {str(e)[:100]}...")
    
    return results

def main():
    """Main function to run connectivity tests."""
    print("🔍 AWS Service Connectivity Verification")
    print("=" * 50)
    
    # Get AWS region
    region = get_aws_region()
    print(f"📍 Testing region: {region}")
    print()
    
    # Define endpoints to test
    endpoints = [
        f"s3.{region}.amazonaws.com",
        f"ecr.{region}.amazonaws.com", 
        f"123456789012.dkr.ecr.{region}.amazonaws.com",  # ECR Docker registry (example account)
        f"ssm.{region}.amazonaws.com",
        f"ec2messages.{region}.amazonaws.com",
        f"ssmmessages.{region}.amazonaws.com", 
        f"ec2.{region}.amazonaws.com",
        f"sts.{region}.amazonaws.com",
        f"logs.{region}.amazonaws.com",
        f"monitoring.{region}.amazonaws.com"
    ]
    
    print("🌐 Testing Network Connectivity (TCP/443)")
    print("-" * 50)
    
    # Test network connectivity with threading for speed
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_endpoint = {
            executor.submit(test_endpoint_connectivity, endpoint): endpoint 
            for endpoint in endpoints
        }
        
        connectivity_results = []
        for future in as_completed(future_to_endpoint):
            endpoint, success, message = future.result()
            connectivity_results.append((endpoint, success, message))
            print(f"{endpoint:<40} {message}")
    
    print()
    print("🔌 Testing AWS API Connectivity (boto3)")
    print("-" * 50)
    
    # Test boto3 API connectivity
    api_results = test_boto3_connectivity(region)
    for service, (success, message) in api_results.items():
        print(f"{service:<25} {message}")
    
    print()
    print("📊 Summary")
    print("-" * 50)
    
    # Network connectivity summary
    network_success = sum(1 for _, success, _ in connectivity_results if success)
    network_total = len(connectivity_results)
    print(f"Network Connectivity: {network_success}/{network_total} endpoints reachable")
    
    # API connectivity summary  
    api_success = sum(1 for success, _ in api_results.values() if success)
    api_total = len(api_results)
    print(f"API Connectivity:     {api_success}/{api_total} services accessible")
    
    # Overall status
    overall_success = network_success == network_total and api_success == api_total
    if overall_success:
        print("\n🎉 All connectivity tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  Some connectivity issues detected.")
        print("   Check network configuration, security groups, and VPC endpoints.")
        sys.exit(1)

if __name__ == "__main__":
    main()