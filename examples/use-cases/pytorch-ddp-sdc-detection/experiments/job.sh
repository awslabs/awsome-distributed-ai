#!/bin/bash
# Runs the experiments as a Kubeflow PyTorchJob in your current kubectl context.
#
#   ./job.sh preflight [nodes]      # check the context, the PyTorchJob CRD, and idle GPU nodes
#   ./job.sh push                   # create the namespace if needed, and the ConfigMap with the code
#   ./job.sh start [gen_job args]   # create a uniquely named job and print its name
#   ./job.sh wait  <job> [secs]     # fail on timeout (default 10800 s), a failed pod, or torchrun != 0
#   ./job.sh fetch <job>            # copy logs, manifest, image digests, and results to ./runs/<job>/
#   ./job.sh stop  <job>            # delete one job
#   ./job.sh clean                  # delete every job and ConfigMap this tool created in $NS
#
# EXPECTED_CONTEXT: refuse to run unless kubectl's current context has this name.
# SELECTOR: label that selects the GPU nodes (default node.kubernetes.io/instance-type=p5.48xlarge;
#   on SageMaker HyperPod, node.kubernetes.io/instance-type=ml.p5.48xlarge).
# TOLERATIONS: space-separated taints the pods tolerate, as KEY[=VALUE]:EFFECT.
# NS, CONFIGMAP: namespace and ConfigMap name.
set -euo pipefail
cd "$(dirname "$0")"
NS=${NS:-sdc-detection}
CONFIGMAP=${CONFIGMAP:-sdc-order-scripts}
SELECTOR=${SELECTOR:-node.kubernetes.io/instance-type=p5.48xlarge}
TOLERATIONS=${TOLERATIONS:-}
LABEL=app.kubernetes.io/part-of=sdc-detection

ctx=$(kubectl config current-context)
if [ -n "${EXPECTED_CONTEXT:-}" ] && [ "$ctx" != "$EXPECTED_CONTEXT" ]; then
  echo "kubectl context is '$ctx', not '$EXPECTED_CONTEXT'; refusing to run" >&2
  exit 1
fi

pods() {
  kubectl get pods -n "$NS" -l "training.kubeflow.org/job-name=$1" \
    -o jsonpath='{range .items[*]}{.metadata.name}={.status.phase}{" "}{end}'
}

# Prints the node's NoSchedule and NoExecute taints that TOLERATIONS doesn't cover.
untolerated() {
  local key value effect tol tkv tkey tvalue teffect ok
  while IFS='|' read -r key value effect; do
    [ -n "$key" ] || continue
    case "$effect" in NoSchedule|NoExecute) ;; *) continue ;; esac
    ok=no
    for tol in $TOLERATIONS; do
      tkv=${tol%:*}; teffect=${tol##*:}; tkey=${tkv%%=*}
      if [ "$tkv" = "$tkey" ]; then tvalue=; else tvalue=${tkv#*=}; fi
      if [ "$tkey" = "$key" ] && [ "$teffect" = "$effect" ] &&
         { [ "$tkv" = "$tkey" ] || [ "$tvalue" = "$value" ]; }; then ok=yes; break; fi
    done
    [ "$ok" = yes ] || printf '%s ' "$key${value:+=$value}:$effect"
  done < <(kubectl get node "$1" -o jsonpath='{range .spec.taints[*]}{.key}|{.value}|{.effect}{"\n"}{end}')
}

case "${1:-}" in
  preflight)
    want=${2:-4}
    echo "context: $ctx"
    kubectl get crd pytorchjobs.kubeflow.org >/dev/null
    idle=0
    for n in $(kubectl get nodes -l "$SELECTOR" -o jsonpath='{.items[*].metadata.name}'); do
      cordoned=$(kubectl get node "$n" -o jsonpath='{.spec.unschedulable}')
      used=$(kubectl describe node "$n" | awk '/Allocated resources/,0' | awk '$1=="nvidia.com/gpu"{print $2; exit}')
      taints=$(untolerated "$n")
      echo "$n gpus_requested=${used:-0} cordoned=${cordoned:-false} untolerated_taints=${taints:-none}"
      if [ "${used:-0}" = "0" ] && [ "${cordoned:-false}" != "true" ] && [ -z "$taints" ]; then
        idle=$((idle + 1))
      fi
    done
    echo "idle nodes: $idle (need $want)"
    [ "$idle" -ge "$want" ] ;;
  push)
    kubectl get namespace "$NS" >/dev/null 2>&1 || kubectl create namespace "$NS"
    kubectl create configmap "$CONFIGMAP" -n "$NS" \
      --from-file=../sdc_order.py --from-file=run_order.py --from-file=common.py \
      --from-file=model.py --from-file=faults.py --dry-run=client -o yaml |
      kubectl label --local -f - "$LABEL" -o yaml | kubectl apply -f - ;;
  start)
    shift
    tol_args=()
    for tol in $TOLERATIONS; do tol_args+=(--toleration "$tol"); done
    # Your own gen_job.py arguments come last, so they override SELECTOR and TOLERATIONS.
    python3 gen_job.py --namespace "$NS" --configmap "$CONFIGMAP" --node-selector "$SELECTOR" \
      ${tol_args[@]+"${tol_args[@]}"} "$@" > job.yaml
    job=$(awk '/^  name:/ {print $2; exit}' job.yaml)
    kubectl create -f job.yaml >/dev/null   # create, not apply: never overwrite an existing job
    echo "$job" ;;
  wait)
    job=$2; tmo=${3:-10800}; t=0
    while :; do
      if line=$(kubectl logs -n "$NS" "$job-master-0" -c pytorch 2>/dev/null | grep -m1 '^TORCHRUN_EXIT'); then
        echo "$line"
        [ "$(echo "$line" | awk '{print $2}')" = "0" ] || { echo "torchrun failed" >&2; exit 1; }
        case "$(pods "$job")" in *Failed*) echo "a pod failed: $(pods "$job")" >&2; exit 1 ;; esac
        exit 0
      fi
      case "$(pods "$job" 2>/dev/null || true)" in *Failed*) echo "a pod failed: $(pods "$job")" >&2; exit 1 ;; esac
      if [ "$t" -ge "$tmo" ]; then echo "timed out after ${t}s: $(pods "$job")" >&2; exit 1; fi
      sleep 30; t=$((t + 30))
    done ;;
  fetch)
    job=$2; out=runs/$job; mkdir -p "$out"
    for p in $(kubectl get pods -n "$NS" -l "training.kubeflow.org/job-name=$job" -o name); do
      kubectl logs -n "$NS" "${p#pod/}" -c pytorch > "$out/${p#pod/}.log"
    done
    kubectl get pytorchjob -n "$NS" "$job" -o yaml > "$out/job.yaml"
    kubectl get pods -n "$NS" -l "training.kubeflow.org/job-name=$job" \
      -o jsonpath='{range .items[*]}{.metadata.name} {.status.containerStatuses[0].imageID}{"\n"}{end}' > "$out/images.txt"
    kubectl cp -n "$NS" -c pytorch "$job-master-0:/results" "$out/files"
    [ -f "$out/files/DONE" ] || { echo "no DONE marker in the copied results" >&2; exit 1; }
    kubectl exec -n "$NS" -c pytorch "$job-master-0" -- touch /results/FETCHED   # lets the pod exit
    echo "fetched $out" ;;
  stop)
    kubectl delete pytorchjob -n "$NS" "$2" --wait=false ;;
  clean)
    kubectl delete pytorchjob -n "$NS" -l "$LABEL" --wait=false
    kubectl delete configmap -n "$NS" -l "$LABEL" ;;
  *)
    sed -n '2,16p' "$0"; exit 1 ;;
esac
