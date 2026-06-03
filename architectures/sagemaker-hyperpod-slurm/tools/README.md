# SMHP Tools <!-- omit from toc -->

The “tools” directory contains utility scripts for common tasks to help debug and troubleshoot issues.
Here are the details of each script, along with its usage and the expected output.

### [`dump_cluster_nodes_info.py`](./dump_cluster_nodes_info.py) 

Utility to dump details of all nodes in a cluster, into a csv file. 

**Usage:** `python dump_cluster_nodes_info.py –cluster-name <name-of-cluster-whose-node-details-are-needed>`

**Output:** “nodes.csv” file in the current directory, containing details of all nodes in the cluster 
