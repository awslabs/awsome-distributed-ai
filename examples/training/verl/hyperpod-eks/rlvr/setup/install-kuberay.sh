# Deploy KubeRay
NS_COUNT=$(kubectl get namespace kuberay | grep kuberay | wc -l)
if [ "$NS_COUNT" == "1" ]; then
    echo "Namespace kuberay already exists"
else
    kubectl create namespace kuberay
fi
# Deploy the KubeRay operator with the Helm chart repository
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
#Install both CRDs and Kuberay operator v1.2.0
helm install kuberay-operator kuberay/kuberay-operator --version 1.4.2 --namespace kuberay
# Kuberay operator pod will be deployed onto head pod
kubectl get pods --namespace kuberay