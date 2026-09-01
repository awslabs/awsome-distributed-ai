# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import argparse
import csv

import boto3


def list_cluster_nodes_all(sagemaker_client, cluster_name):

    nodes = []
    next_token = None

    while True:
        
        params = {
            "ClusterName" : cluster_name
        }
        if next_token:
            params["NextToken"] = next_token

        response = sagemaker_client.list_cluster_nodes(**params)

        nodes += response["ClusterNodeSummaries"]

        if "NextToken" in response and response["NextToken"]:
            next_token = response["NextToken"]
            continue

        break

    return nodes


def dump_nodes(cluster_name):
    
    sagemaker_client = boto3.client("sagemaker")
    
    nodes = list_cluster_nodes_all( sagemaker_client, cluster_name )

    with open("nodes.csv", "w") as fd:
        csv_writer = csv.writer(fd)
        csv_writer.writerow([ "instance-id", "ip-address", "status", "hostname", "instance-group", "launch-time" ])

        for node in nodes:
            # For each node, we need to call the 'describe_cluster_node' API
            # Ref: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker/client/describe_cluster_node.html
            instance_id = node['InstanceId']
            node_details = sagemaker_client.describe_cluster_node(ClusterName=cluster_name, NodeId=instance_id)['NodeDetails']
            # ...and write necessary data in the CSV...
            csv_writer.writerow([node_details['InstanceId'],
                                 node_details['PrivatePrimaryIp'],
                                 node_details['InstanceStatus']['Status'],
                                 node_details['PrivateDnsHostname'],
                                 node_details['InstanceGroupName'],
                                 node_details['LaunchTime']])

    print(f"Details of all nodes in cluster '{cluster_name}' have been saved in nodes.csv")


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Dump all HyperPod cluster nodes and their details in a CSV")
    argparser.add_argument("--cluster-name", action="store", required=True, help="Name of cluster to dump")
    args = argparser.parse_args()

    dump_nodes(args.cluster_name)

