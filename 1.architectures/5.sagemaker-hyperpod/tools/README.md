# SMHP Tools <!-- omit from toc -->

The “tools” directory contains utility scripts and solutions for common tasks to help debug and troubleshoot issues.
Here are the details of each, along with its usage and the expected output.

### [`devops-agent`](./devops-agent)

**HyperPod x AWS DevOps Agent** — a solution that keeps 24/7 watch over a large-scale SageMaker HyperPod GPU fleet by wiring it into [AWS DevOps Agent](https://docs.aws.amazon.com/devopsagent/). It complements HyperPod's built-in resiliency by auto-detecting the operational conditions that still call for a human decision (configuration issues, capacity conditions, recurring hardware faults, workload-level conditions), then triaging, root-causing, and delivering a human-readable verdict email (`Monitor` / `Escalate` / `Resolved`). It deploys as one CloudFormation stack per cluster and is extensible via plain-English detection rules (*skills*) with no pipeline code change.

See the [`devops-agent/README.md`](./devops-agent/README.md) for architecture, deployment, and usage details.

### [`dump_cluster_nodes_info.py`](./dump_cluster_nodes_info.py) 

Utility to dump details of all nodes in a cluster, into a csv file. 

**Usage:** `python dump_cluster_nodes_info.py –cluster-name <name-of-cluster-whose-node-details-are-needed>`

**Output:** “nodes.csv” file in the current directory, containing details of all nodes in the cluster 
