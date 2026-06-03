#!/usr/bin/env python3

import json
import argparse
import boto3
from jsonschema import validate

# This function checks that provisioning_parameter.json meets schema
def validate_provisioning_parameters(provisioning_parameters):

    # {
    #     "version": "1.0.0",
    #     "workload_manager": "slurm",
    #     "controller_group": "controller-machine",
    #     "worker_groups": [
    #         {
    #         "instance_group_name": "worker-group-1",
    #         "partition_name": "ml.p5.48xlarge"
    #         }
    #     ],
    #     "fsx_dns_name": "fs-05dac34e835f2c47f.fsx.us-west-2.amazonaws.com",
    #     "fsx_mountname": "4owupbev"
    # }

    schema = {
        "type": "object",
        "properties": {
            "version": {"type": "string"},
            "workload_manager": {"type": "string"},
            "controller_group": {"type": "string"},
            "worker_groups": {
                "list": [
                {"type": "string"},
                {"type": "string"}
            ]},
            "fsx_dns_name": {"type": "string"},
            "fsx_mountname": {"type": "string"}
        },
        "required": ["version", "workload_manager", "controller_group", "worker_groups", "fsx_dns_name", "fsx_mountname" ]
    }

    try:
        validate(instance=provisioning_parameters, schema=schema)  # No error means valid
        print(f"✔️  Validated provisioning_parameter.json is valid json ...")
        return True
    except Exception as e:
        print(f"❌ Invalid provisioning_parameter.json, {e}")
        return False

# This function checks that all instance group names match
# between the cluster config and provisioning parameters.
def validate_instance_groups(cluster_config, provisioning_parameters):
    for group in provisioning_parameters.get('worker_groups'):
        instance_group_name = group.get('instance_group_name')
        if not [instance_group for instance_group in cluster_config.get('InstanceGroups') if instance_group.get('InstanceGroupName') == instance_group_name]:
            print(f"❌ Invalid instance group name in file provisioning_parameters.json: {instance_group_name}")
            return False
        else:
            print(f"✔️  Validated instance group name {instance_group_name} is correct ...")
    return True

# Check if Subnet is private
def validate_subnet(ec2_client, cluster_config):
    if cluster_config.get('VpcConfig'):
        subnet_id = cluster_config.get('VpcConfig').get('Subnets')[0]
        response = ec2_client.describe_subnets(SubnetIds=[subnet_id])
        if 'Subnets' in response and response.get('Subnets')[0].get('MapPublicIpOnLaunch'):
            print(f"❌ Subnet {subnet_id} is public which will fail cluster creation ...")
            return False
        else:
            print(f"✔️  Validated subnet {subnet_id} ...")
    else:
        print("⭕️ No subnet found in cluster_config.json ... skipping check")
    return True

# Check if Security Group supports EFA.
# See https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html#efa-start-security
def validate_sg(ec2_client, cluster_config):
    if cluster_config.get('VpcConfig'):
        security_group = cluster_config.get('VpcConfig').get('SecurityGroupIds')[0]
        response = ec2_client.describe_security_groups(GroupIds=[security_group])

        ingress = response.get('SecurityGroups')[0].get('IpPermissions')
        egress = response.get('SecurityGroups')[0].get('IpPermissionsEgress')

        for rule in ingress:
            if rule.get('IpProtocol') == '-1':
                user_id_group_pairs = rule.get('UserIdGroupPairs')
                if not user_id_group_pairs:
                    print(f"❌ No EFA egress rule found in security group {security_group} ...")
                    return False
                else:
                    if not ('GroupId' in user_id_group_pairs[0] or security_group == user_id_group_pairs[0].get('GroupId')):
                        print(f"❌ No EFA egress rule found in security group {security_group} ...")
                        return False
                    else:
                        print(f"✔️  Validated security group {security_group} ingress rules ...")

        for rule in egress:
            if rule.get('IpProtocol') == '-1':
                user_id_group_pairs = rule.get('UserIdGroupPairs')
                if not user_id_group_pairs:
                    print(f"❌ No EFA egress rule found in security group {security_group} ...")
                    return False
                else:
                    if not ('GroupId' in user_id_group_pairs[0] or security_group == user_id_group_pairs[0].get('GroupId')):
                        print(f"❌ No EFA egress rule found in security group {security_group} ...")
                        return False
                    else:
                        print(f"✔️  Validated security group {security_group} egress rules ...")
    else:
        print("⭕️ No security group found in cluster_config.json ... skipping check.")

    return True

def validate_fsx_lustre(fsx_client, cluster_config, provisioning_parameters):
    fsx_dns_name = provisioning_parameters.get('fsx_dns_name')
    if fsx_dns_name:
        try:
            response = fsx_client.describe_file_systems(FileSystemIds=[fsx_dns_name.split('.', 1)[0]])
            returned_fsx_mount = response['FileSystems'][0]['LustreConfiguration']['MountName']
            returned_fsx_dns_name = response['FileSystems'][0]['DNSName']

            if returned_fsx_dns_name != fsx_dns_name:
                print(f"❌ Incorrect FSx DNS name: {fsx_dns_name}. Did you mean {returned_fsx_dns_name}?")
                return False

            fsx_mountname = provisioning_parameters.get('fsx_mountname')
            if returned_fsx_mount != fsx_mountname:
                print(f"❌ Mismatch FSx mount name: {fsx_mountname}. Did you mean {returned_fsx_mount}?")
                return False

            print(f"✔️  Validated FSx Lustre DNS name {fsx_dns_name} ...")
            print(f"✔️  Validated FSx Lustre mount name {fsx_mountname} ...")

        except fsx_client.exceptions.FileSystemNotFound:
            print(f"❌ File system {fsx_dns_name} does not exist.")
            return False
    else:
        print("⭕️ No FSx Lustre found in provisioning_parameters.json ... skipping check.")

    return True


def main():
    parser = argparse.ArgumentParser(description="Validate cluster config.")
    parser.add_argument("--region", help="AWS Region where the cluster will be created", default=None)
    parser.add_argument("--profile", help="AWS Profile to use for creating the cluster", default=None)
    parser.add_argument("--cluster-config", help="Path to the cluster config JSON file")
    parser.add_argument("--provisioning-parameters", help="Path to the provisioning parameters JSON file")
    args = parser.parse_args()

    with open(args.cluster_config, "r") as cluster_config_file:
        try:
            cluster_config = json.load(cluster_config_file)
        except json.decoder.JSONDecodeError:
            print(f"❌ cluster-config.json is invalid.")
            return False

    with open(args.provisioning_parameters, "r") as provisioning_parameters_file:
        try:
            provisioning_parameters = json.load(provisioning_parameters_file)
        except json.decoder.JSONDecodeError:
            print(f"❌ provisioning_parameters.json is invalid.")
            return False
    
    session = boto3.Session(profile_name=args.profile, region_name=args.region)

    ec2_client = session.client('ec2')

    # check instance group name
    valid = validate_instance_groups(cluster_config, provisioning_parameters)

    # Validate Subnet
    valid = validate_subnet(ec2_client, cluster_config) and valid

    # Validate Security Group
    valid = validate_sg(ec2_client, cluster_config) and valid

    # Validate FSx Lustre
    valid = validate_fsx_lustre(session.client('fsx'), cluster_config, provisioning_parameters) and valid

    # validate provisioning_parameters
    valid = validate_provisioning_parameters(provisioning_parameters)

    if valid:
        # All good!
        print(f"✅ Cluster Validation succeeded")
    else:
        print(f"❌ Cluster Validation failed")

if __name__ == "__main__":
    main()
