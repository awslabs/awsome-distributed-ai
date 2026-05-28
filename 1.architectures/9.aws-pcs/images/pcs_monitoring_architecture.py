#!/usr/bin/env python3
"""
AWS PCS Monitoring Architecture Diagram
Generates architecture diagram showing monitoring stack deployment
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import EC2, EC2Instance
from diagrams.aws.network import VPC, PublicSubnet, PrivateSubnet, NATGateway, InternetGateway
from diagrams.aws.storage import FSx, FsxForLustre
from diagrams.aws.management import SystemsManager, Cloudwatch, ParameterStore
from diagrams.aws.cost import CostExplorer
from diagrams.onprem.monitoring import Prometheus, Grafana
from diagrams.onprem.client import User
from diagrams.custom import Custom

# Custom icons for components not in standard library
graph_attr = {
    "fontsize": "14",
    "bgcolor": "white",
    "pad": "0.5",
}

with Diagram("AWS PCS Cluster with Monitoring Stack",
             filename="/tmp/pcs_monitoring_architecture",
             direction="TB",
             graph_attr=graph_attr,
             show=False):

    user = User("User\n(Local)")

    with Cluster("AWS Region (us-east-1a)"):
        ssm = SystemsManager("Session Manager\n(Port Forward)")
        param_store = ParameterStore("Parameter Store\n/pcs/*/grafana/*")
        cloudwatch = Cloudwatch("CloudWatch\nMetrics")
        pricing = CostExplorer("Pricing API\nCost Data")

        with Cluster("VPC"):
            igw = InternetGateway("Internet Gateway")
            nat = NATGateway("NAT Gateway")

            with Cluster("Public Subnet"):
                with Cluster("Login Node\n(m6i.4xlarge)"):
                    login = EC2Instance("PCS Login Node")

                    with Cluster("Monitoring Stack (Docker Compose)"):
                        prometheus = Prometheus("Prometheus\n:9090")
                        grafana = Grafana("Grafana\n:443 (HTTPS)")
                        nginx = EC2("Nginx\nReverse Proxy")
                        cw_exporter = EC2("CloudWatch\nExporter :9106")
                        custom_exporters = EC2("Custom Exporters\n(cost, slurm)")

            with Cluster("Private Subnet"):
                with Cluster("Compute Node Group - cpu1\n(c6i.4xlarge, 0-4 instances)"):
                    cpu_node1 = EC2Instance("CPU Node 1")
                    cpu_exporter1 = EC2("Node Exporter\n:9100")
                    cpu_node1 - cpu_exporter1

                with Cluster("Compute Node Group - GPU\n(g6.12xlarge / p5.48xlarge)"):
                    gpu_node1 = EC2Instance("GPU Node 1")
                    dcgm_exporter1 = EC2("DCGM Exporter\n:9400\n(GPU metrics)")
                    gpu_node1 - dcgm_exporter1

            with Cluster("Shared Storage"):
                fsx_openzfs = FSx("FSx OpenZFS\n/home (NFS)\n512 GiB")
                fsx_lustre = FsxForLustre("FSx Lustre\n/fsx (shared)\n1200 GiB")

    # User access flow
    user >> Edge(label="1. Port forward\n8443→443") >> ssm
    ssm >> Edge(label="SSM tunnel") >> login
    user >> Edge(label="2. Get password") >> param_store
    user >> Edge(label="3. Access Grafana\nhttps://localhost:8443") >> grafana

    # Monitoring data flow
    nginx >> grafana
    grafana >> prometheus
    prometheus << Edge(label="scrape\nmetrics", style="dashed") << cpu_exporter1
    prometheus << Edge(label="scrape\nGPU metrics", style="dashed") << dcgm_exporter1
    prometheus << Edge(label="scrape", style="dashed") << cw_exporter

    # External API access
    login >> Edge(label="Grafana password") >> param_store
    cw_exporter >> Edge(label="fetch metrics") >> cloudwatch
    custom_exporters >> Edge(label="pricing data") >> pricing

    # Storage connections
    login >> Edge(label="mount") >> fsx_openzfs
    login >> Edge(label="mount") >> fsx_lustre
    cpu_node1 >> Edge(label="mount", style="dotted") >> fsx_openzfs
    cpu_node1 >> Edge(label="mount", style="dotted") >> fsx_lustre
    gpu_node1 >> Edge(label="mount", style="dotted") >> fsx_openzfs
    gpu_node1 >> Edge(label="mount", style="dotted") >> fsx_lustre

    # Internet connectivity
    igw >> nat
    nat >> Edge(style="dotted") >> login
    nat >> Edge(style="dotted") >> cpu_node1
    nat >> Edge(style="dotted") >> gpu_node1

print("Architecture diagram generated: /tmp/pcs_monitoring_architecture.png")
