# In-cluster Prometheus and Grafana for Amazon EKS

This directory installs Prometheus, Pushgateway and Grafana inside an EKS cluster, in one namespace, with plain Kubernetes objects and one Python script. It is for a cluster you work on directly: Grafana is reached through `kubectl port-forward`, nothing is exposed outside the cluster, and nothing outside its namespace changes except the read-only grants you ask for.

Choose [`../eks-managed-observability`](../eks-managed-observability/) instead when metrics have to outlive the cluster, be collected from several clusters, or be retained beyond a day: it sends them to Amazon Managed Service for Prometheus and Amazon Managed Grafana through the ADOT collector, and needs those workspaces, cert-manager and the ADOT add-on. This directory needs none of them, and it can run on the same cluster as `eks-managed-observability`.

The [amazon-eks architecture](../../../architectures/amazon-eks/) installs it with `DeployMonitoring=true` ([section 11](../../../architectures/amazon-eks/README.md#11-optional-in-cluster-monitoring)). Everything below also applies to a cluster created another way.

## What is installed

| Object | Where | Purpose |
|---|---|---|
| Namespace | `--namespace`, default `in-cluster-monitoring` | Holds everything else; enforces the `restricted` Pod Security level |
| Prometheus v3.15.1 | Deployment and Service `prometheus:9090` | Scrapes the targets below; keeps 24 hours or 1 GB on an `emptyDir` |
| Pushgateway v1.11.5 | Deployment and Service `pushgateway:9091` | Receives metrics that a job pushes once, for example the results of a benchmark |
| Grafana 13.3.0 | Deployment and Service `grafana:80` | Dashboards; the Prometheus data source and the bundled dashboards are provisioned read-only |
| Secret `grafana-admin` | the namespace | Grafana's admin password, generated once and printed only by `check --show-password` |
| ServiceAccount `prometheus`; ConfigMaps `prometheus-config`, `grafana-provisioning` and `grafana-dashboards` | the namespace | Prometheus's identity for discovery; the Prometheus configuration, Grafana's provisioning and the bundled dashboards |
| Role and RoleBinding `monitoring-read-<namespace>` | each namespace you scrape | `get`, `list` and `watch` on Services, EndpointSlices and Pods, for the `prometheus` ServiceAccount only; Prometheus's endpoint discovery reads Pods for the `pod` label, so the grant also shows the Pod specs of that namespace |

The images are pinned by digest. The pods run as non-root with a read-only root filesystem, no capabilities and the `RuntimeDefault` seccomp profile, so they are admitted under the `restricted` Pod Security level, which the namespace enforces. They tolerate no taint by default, so they never land on GPU nodes tainted `nvidia.com/gpu`; `--node-selector` pins them further, for example `--node-selector role=system`, and `--toleration KEY[=VALUE]:EFFECT` lets them onto tainted nodes on a cluster that has no other.

Nothing else is created: no CRD, no DaemonSet, no host network or host port, no admission webhook, no EKS add-on, and by default no cluster-scoped object other than the namespace. `--cluster-wide-discovery` replaces the per-namespace grants with one read-only ClusterRole and ClusterRoleBinding named `<namespace>-prometheus`, for clusters where granting each namespace is impractical; Prometheus still reads only the namespaces given with `--scrape-namespace` and `--scrape-service`. An `apply` without the flag removes that pair again.

## What is read

Prometheus scrapes itself, the Pushgateway, and two kinds of target you name:

- Services that opt in. A Service in a namespace given with `--scrape-namespace` is scraped when it carries the label `awsome-distributed-ai/scrape: <monitoring namespace>` and a port named `metrics`. Only ready endpoints are scraped. Every Service label `awsome-distributed-ai/target-<name>: <value>` becomes the series label `<name>="<value>"`, and the series also carry `namespace`, `service` and `pod`. The common `prometheus.io/scrape` annotation is deliberately not used: a Service that opts in here with the label alone is not picked up by another collector that reads that annotation, and a Service annotated only for another collector is not scraped here.
- Existing exporters, named with `--scrape-service NAMESPACE/SERVICE:PORT`. `PORT` is the Service's port name, or the container (target) port number behind it, not the Service's own port number.

Service label keys become Prometheus label names with every character other than letters, digits and `_` turned into `_`, so two keys that differ only in such characters map to the same label; keep target label names distinct after that conversion.

No exporter is installed. Without one there are no GPU, node or EFA metrics. To read GPU metrics from the GPU Operator's DCGM exporter, add `--scrape-service gpu-operator/nvidia-dcgm-exporter:9400`; the bundled dashboard "GPUs (DCGM exporter)" reads it. `check` waits up to a minute for every exporter named with `--scrape-service` to have a target that is up, and names one that has none. The repository's [DCGM dashboard](../1click-dashboards-deployment/dashboards/DCGM_exporter_dashboard.json) is an export with a `${DS_PROMETHEUS}` input, which Grafana's file provisioning does not fill, so this directory provisions its own that reads the same metrics. Installing a second DCGM exporter on GPUs that already have one makes the two conflict, which is why this directory does not ship one. For EFA counters see [`../../efa-node-exporter`](../../efa-node-exporter/).

## Install

Requirements: Python 3.9 or later and `kubectl`, with a kubeconfig whose identity may create the objects above. `install.py apply` checks the permissions with `kubectl auth can-i` before it writes anything.

Set `OWNER` to a Kubernetes label value that no one else on the cluster uses, then paste the rest:

```bash
export OWNER=
export NAMESPACE=in-cluster-monitoring
```

```bash
if [ -z "$OWNER" ]; then echo "set OWNER to a label value no one else on this cluster uses"; else
  [ -d awsome-distributed-ai ] || git clone https://github.com/awslabs/awsome-distributed-ai.git
  export MONITORING_DIR="$PWD/awsome-distributed-ai/observability/prometheus-grafana/eks-in-cluster"
  python3 "$MONITORING_DIR/install.py" apply --owner "$OWNER" --namespace "$NAMESPACE"
  python3 "$MONITORING_DIR/install.py" check --namespace "$NAMESPACE"
fi
```

`--owner` names who owns the install, as a Kubernetes label value. Every object carries it, and `apply` and `delete` refuse any object that exists without it or with another owner, so installs with different owners cannot overwrite or remove each other. On a cluster from [`architectures/amazon-eks`](../../../architectures/amazon-eks/), add `--node-selector role=system` to keep the pods on the system nodes. Re-running `apply` with the same owner updates the install in place with server-side apply.

Grafana:

```bash
python3 "$MONITORING_DIR/install.py" check --namespace "$NAMESPACE" --show-password | grep password
kubectl -n "$NAMESPACE" port-forward service/grafana 3000:80
```

Open `http://localhost:3000/` and sign in as `admin` with the password the first command prints. Anonymous viewing is off unless `--anonymous-viewer` is given. When Grafana is reached through a proxy at another address, for example a notebook's `https://<host>/proxy/absolute/3000/`, give that address with `--root-url`; one install serves one address, plus the origins given with `--extra-origin`.

`apply` takes the whole list each time: re-run it with `--scrape-namespace` for every namespace to read, including one created after the install, which is granted once it exists. A namespace left off the list loses this owner's grant there. The grant goes when that namespace is deleted.

## Push metrics

`push.py` sends [Prometheus text exposition](https://prometheus.io/docs/instrumenting/exposition_formats/) to a Pushgateway group, for results that are computed once rather than scraped. It uses only the Python standard library.

```bash
printf '# TYPE example_result gauge\nexample_result{step="1"} 0.75\n' |
  python3 "$MONITORING_DIR/push.py" put --job example --group run=first --via kubectl --namespace "$NAMESPACE"
python3 "$MONITORING_DIR/push.py" list --job example --via kubectl --namespace "$NAMESPACE"
python3 "$MONITORING_DIR/push.py" delete --job example --group run=first --via kubectl --namespace "$NAMESPACE"
```

`--via kubectl` sends through the Kubernetes API server's Service proxy, so it works from any host with a kubeconfig, with no port-forward. The identity needs `update` on `services/proxy` in the monitoring namespace for `put`, `delete` for `delete` and `get` for `list`; `push.py` checks this first. Granted without a name, that permission reaches every Service in the namespace, Grafana and Prometheus included; narrow it to the Pushgateway with the resource name `pushgateway:http`, for example `kubectl -n in-cluster-monitoring create role pusher --verb=update --resource=services/proxy --resource-name=pushgateway:http`. A Role created by hand in the monitoring namespace is not part of the install, so `install.py delete` refuses to remove the namespace until that Role and its RoleBinding are deleted. From a pod in the cluster, `--via url --url http://pushgateway.in-cluster-monitoring.svc:9091` sends directly and needs no Kubernetes permission.

`put` replaces the whole group, so send every series of a group in one call, and put each run in its own group, for example `--group run=<name>`. The Pushgateway never expires a group: `delete` the groups of a finished run. The "Monitoring health" dashboard lists every group with the time of its last push.

## Command reference

| Flag | Default | Effect |
|---|---|---|
| `--owner` | required for `apply` and `delete` | Label value naming the owner; objects with another owner are refused |
| `--namespace` | `in-cluster-monitoring` | Namespace of the install |
| `--scrape-namespace` | none | Namespace whose opted-in Services are scraped; repeatable; the whole list each time |
| `--scrape-service` | none | Existing exporter `NAMESPACE/SERVICE:PORT`; repeatable |
| `--cluster-wide-discovery` | off | One read-only ClusterRole and ClusterRoleBinding instead of per-namespace grants |
| `--node-selector` | none | `KEY=VALUE` the three pods require; repeatable |
| `--toleration` | none | `KEY[=VALUE]:EFFECT` the three pods tolerate; repeatable |
| `--root-url` | `http://localhost:3000/` | Address the browser reaches Grafana on; a path enables sub-path serving |
| `--extra-origin` | none | Another origin allowed for Grafana Live; repeatable |
| `--anonymous-viewer` | off | Let unauthenticated visitors view dashboards |
| `--dashboard` | none | Another dashboard JSON file to provision; repeatable |
| `--home` | `monitoring-health.json` | File name of the Home dashboard among the provisioned ones |
| `--retention-time`, `--retention-size` | `24h`, `1GB` | Prometheus retention |
| `--scrape-interval` | `15s` | Prometheus scrape and evaluation interval |
| `--prometheus-image`, `--pushgateway-image`, `--grafana-image` | digest-pinned | Image overrides |
| `--context`, `--timeout` | current context, `300` | kubectl context; seconds for rollouts and namespace deletion |
| `--show-password` | off | With `check`, print the Grafana admin password |

`install.py` exits 0 on success, 1 when a command fails or `check` finds a component that does not answer, 2 on a usage error, 3 when it refuses (another owner's objects, a missing permission, objects it did not create), and 4 when `check` finds the install working and only an exporter named with `--scrape-service` without a target that is up.

`push.py` takes `put`, `delete` or `list`, `--job`, `--group KEY=VALUE` (repeatable), `--via kubectl` with `--context`, `--namespace` and `--service` (default `pushgateway`), or `--via url --url URL`, and `--input FILE` (default stdin).

## Remove

```bash
python3 "$MONITORING_DIR/install.py" delete --owner "$OWNER" --namespace "$NAMESPACE"
```

`delete` removes the namespace, the grants this owner made in other namespaces, and the cluster-wide read if it was created. It asks for `delete` on the monitoring namespace by name, so a grant narrowed with `resourceNames` to that namespace is enough. It first lists every object in the namespace and refuses, removing nothing, if any of them was not created by this owner's install, its controllers or Kubernetes itself. It also refuses, removing nothing, when it cannot list a resource kind in the namespace or this owner's grants across the cluster, because it would otherwise leave objects it cannot see.

## Known limits

- Prometheus and the Pushgateway keep their data on an `emptyDir`, which is lost when the pod is replaced, for example on a node replacement or when `apply` changes the Prometheus configuration.
- One Prometheus replica; no alerting rules; no remote write.
- Anyone who may port-forward to, or proxy through, the monitoring namespace can reach Grafana, Prometheus and the Pushgateway.
- `delete` refuses a namespace that holds an object it did not create. On a cluster where a controller writes an object into every namespace, for example a service mesh's root certificate ConfigMap, it always refuses; delete that object, then run `delete` again.
- Prometheus and the Pushgateway have no authentication of their own. Anything that can reach their pod or Service addresses can read metrics and push groups: a pod in the cluster or, with the VPC CNI, a host in the VPC that the nodes' security groups admit. The Pushgateway is scraped with `honor_labels`, so a push can write series under any `job` label. Restrict that traffic with a NetworkPolicy where a policy engine enforces it.

## Files

| File | Purpose |
|---|---|
| [`install.py`](./install.py) | `apply`, `check`, `delete` and `render` (prints the objects without a cluster) |
| [`monitoring_render.py`](./monitoring_render.py) | The objects, rendered from the options; no cluster access |
| [`push.py`](./push.py) | `put`, `delete` and `list` of Pushgateway groups |
| [`dashboards/gen_dashboards.py`](./dashboards/gen_dashboards.py) | Writes the bundled dashboards; `--check` fails when they are out of date |
| [`test_eks_in_cluster_monitoring.py`](./test_eks_in_cluster_monitoring.py) | Unit tests, no cluster: `python3 -m pytest observability/prometheus-grafana/eks-in-cluster` |

`install.py`, `monitoring_render.py` and the dashboards' JSON are also the bundle that [`architectures/amazon-eks/assets/eks-monitoring.yaml`](../../../architectures/amazon-eks/assets/eks-monitoring.yaml) installs, pinned by the sha256 of its uncompressed tar. A change to any of them changes that digest: print the new one with `python3 .github/scripts/stage-template-publish.py --manifest .github/template-publish-manifest.yml --stage-dir /dev/null --cfn-list /dev/null --shell-list /dev/null --obsolete-list /dev/null --print-bundle-digests` from the repository root, set it as `Sha256` under `Mappings` in that template, and run `bash architectures/amazon-eks/tests/lint-templates.sh`, whose check 11 compares the two.
