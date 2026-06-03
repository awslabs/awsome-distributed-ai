import boto3
import os
import subprocess
import cfnresponse
from botocore.exceptions import ClientError
import yaml

def lambda_handler(event, context):
    """
    Handle CloudFormation custom resource requests for managing Helm Charts
    """
    try: 
        request_type = event['RequestType']

        if request_type == 'Create':
            response_data = on_create()
        elif request_type == 'Update':
            response_data = on_update()
        elif request_type == 'Delete':
            response_data = on_delete()
        else:
            raise ValueError(f"Invalid request type: {request_type}")

        cfnresponse.send(
            event,
            context,
            cfnresponse.SUCCESS,
            response_data
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        cfnresponse.send(
            event,
            context,
            cfnresponse.FAILED,
            {
                "Status": "FAILED",
                "Reason": str(e)
            }
        )

def write_kubeconfig(cluster_name, region):
    """
    Generate kubeconfig using boto3
    """
    # Initialize EKS client
    eks = boto3.client('eks', region_name=region)
    
    try:
        # Get cluster info
        cluster = eks.describe_cluster(name=cluster_name)['cluster']
        
        # Generate kubeconfig content
        kubeconfig = {
            'apiVersion': 'v1',
            'kind': 'Config',
            'clusters': [{
                'cluster': {
                    'server': cluster['endpoint'],
                    'certificate-authority-data': cluster['certificateAuthority']['data']
                },
                'name': cluster_name
            }],
            'contexts': [{
                'context': {
                    'cluster': cluster_name,
                    'user': cluster_name
                },
                'name': cluster_name
            }],
            'current-context': cluster_name,
            'preferences': {},
            'users': [{
                'name': cluster_name,
                'user': {
                    'exec': {
                        'apiVersion': 'client.authentication.k8s.io/v1beta1',
                        'command': 'aws-iam-authenticator',
                        'args': [
                            'token',
                            '-i',
                            cluster_name
                        ]
                    }
                }
            }]
        }
        
        # Use /tmp instead of ~/.kube
        kubeconfig_dir = '/tmp/.kube'
        os.makedirs(kubeconfig_dir, exist_ok=True)
        kubeconfig_path = os.path.join(kubeconfig_dir, 'config')
        
        with open(kubeconfig_path, 'w') as f:
            yaml.dump(kubeconfig, f, default_flow_style=False)
        
        # Make sure kubectl can read it
        os.chmod(kubeconfig_path, 0o600)
        
        # Set KUBECONFIG environment variable
        os.environ['KUBECONFIG'] = kubeconfig_path
        
        return True
        
    except ClientError as e:
        print(f"Error getting cluster info: {str(e)}")
        raise


def on_create():
    """
    Handle Create request to clone and install a Helm chart
    """
    try:
        # Initialize response data
        response_data = {
            "Status": "SUCCESS",
            "Reason": "Helm chart installed successfully"
        }

        # Ensure required environment variables are set
        required_env_vars = [
            'GITHUB_REPO_URL',
            'CHART_PATH',
            'NAMESPACE',
            'RELEASE_NAME',
            'CLUSTER_NAME',
            'AWS_REGION'
        ]
        
        for var in required_env_vars:
            if var not in os.environ:
                raise ValueError(f"Missing required environment variable: {var}")
            
         # Set HELM_CACHE_HOME and HELM_CONFIG_HOME
        os.environ['HELM_CACHE_HOME'] = '/tmp/.helm/cache'
        os.environ['HELM_CONFIG_HOME'] = '/tmp/.helm/config'
        
        # Create directories
        os.makedirs('/tmp/.helm/cache', exist_ok=True)
        os.makedirs('/tmp/.helm/config', exist_ok=True)

        # Configure kubectl using boto3
        write_kubeconfig(os.environ['CLUSTER_NAME'], os.environ['AWS_REGION'])

        # Add required Helm repositories
        subprocess.run(['helm', 'repo', 'add', 'nvidia', 'https://nvidia.github.io/k8s-device-plugin'], check=True)
        subprocess.run(['helm', 'repo', 'add', 'eks', 'https://aws.github.io/eks-charts/'], check=True)
        subprocess.run(['helm', 'repo', 'update'], check=True)

        # Clone the GitHub repository
        branch = os.environ.get('GITHUB_REPO_BRANCH', 'main')
        clone_cmd = [
            'git', 'clone',
            '--branch', branch,
            os.environ['GITHUB_REPO_URL'],
            '/tmp/helm-charts'
        ]
        subprocess.run(clone_cmd, check=True)

        # DEBUG: Check what branch we're on and list chart directories
        print(f"Cloned branch: {branch}")
        subprocess.run(['git', '-C', '/tmp/helm-charts', 'branch'], check=True)
        subprocess.run(['ls', '-la', '/tmp/helm-charts/helm_chart/HyperPodHelmChart/charts/'], check=True)
        subprocess.run(['cat', '/tmp/helm-charts/helm_chart/HyperPodHelmChart/Chart.yaml'], check=True)

        # DEBUG: Show the Chart.yaml dependencies
        print("Chart.yaml contents:")
        with open('/tmp/helm-charts/helm_chart/HyperPodHelmChart/Chart.yaml', 'r') as f:
            print(f.read())

        print("Running helm dependency update...")
        # Update dependencies
        subprocess.run(['helm', 'dependency', 'update', f"/tmp/helm-charts/{os.environ['CHART_PATH']}"], check=True)

        # Install the Helm chart
        install_cmd = [
            'helm', 'install',
            os.environ['RELEASE_NAME'],
            f"/tmp/helm-charts/{os.environ['CHART_PATH']}",
            '--namespace', os.environ['NAMESPACE']
        ]
        subprocess.run(install_cmd, check=True)

        # Clean up cloned repository
        subprocess.run(['rm', '-rf', '/tmp/helm-charts'], check=True)

        return response_data

    except subprocess.CalledProcessError as e:
        raise Exception(f"Command failed: {e.cmd}. Return code: {e.returncode}")
    except Exception as e:
        raise Exception(f"Failed to install Helm chart: {str(e)}")


def on_update():
    """
    Handle Update request to upgrade an existing Helm release
    """
    try:
        response_data = {
            "Status": "SUCCESS",
            "Reason": "Helm chart updated successfully"
        }

        # Verify required environment variables
        required_env_vars = [
            'GITHUB_REPO_URL',
            'CHART_PATH',
            'RELEASE_NAME',
            'CLUSTER_NAME',
            'AWS_REGION'
        ]
        
        for var in required_env_vars:
            if var not in os.environ:
                raise ValueError(f"Missing required environment variable: {var}")
            
        # Set HELM_CACHE_HOME and HELM_CONFIG_HOME
        os.environ['HELM_CACHE_HOME'] = '/tmp/.helm/cache'
        os.environ['HELM_CONFIG_HOME'] = '/tmp/.helm/config'
        
        # Create directories
        os.makedirs('/tmp/.helm/cache', exist_ok=True)
        os.makedirs('/tmp/.helm/config', exist_ok=True)

        # Configure kubectl using boto3
        write_kubeconfig(os.environ['CLUSTER_NAME'], os.environ['AWS_REGION'])

        # Add required Helm repositories
        subprocess.run(['helm', 'repo', 'add', 'nvidia', 'https://nvidia.github.io/k8s-device-plugin'], check=True)
        subprocess.run(['helm', 'repo', 'add', 'eks', 'https://aws.github.io/eks-charts/'], check=True)
        subprocess.run(['helm', 'repo', 'update'], check=True)

        # Clone the updated chart
        clone_cmd = [
            'git', 'clone',
            os.environ['GITHUB_REPO_URL'],
            '/tmp/helm-charts'
        ]
        subprocess.run(clone_cmd, check=True)

        # Update dependencies if any
        subprocess.run(['helm', 'dependency', 'update', f"/tmp/helm-charts/{os.environ['CHART_PATH']}"], check=True)

        # Upgrade the release
        upgrade_cmd = [
            'helm', 'upgrade',
            os.environ['RELEASE_NAME'],
            f"/tmp/helm-charts/{os.environ['CHART_PATH']}"
        ]
        subprocess.run(upgrade_cmd, check=True)

        # Clean up
        subprocess.run(['rm', '-rf', '/tmp/helm-charts'], check=True)

        return response_data

    except subprocess.CalledProcessError as e:
        raise Exception(f"Command failed: {e.cmd}. Return code: {e.returncode}")
    except Exception as e:
        raise Exception(f"Failed to update Helm chart: {str(e)}")

def on_delete():
    """
    Handle Delete request to uninstall a Helm release
    """
    try:
        response_data = {
            "Status": "SUCCESS",
            "Reason": "Helm chart uninstalled successfully"
        }

        # Verify required environment variables
        required_env_vars = [
            'RELEASE_NAME',
            'CLUSTER_NAME',
            'AWS_REGION'
        ]
        
        for var in required_env_vars:
            if var not in os.environ:
                raise ValueError(f"Missing required environment variable: {var}")

        # Configure kubectl using boto3
        write_kubeconfig(os.environ['CLUSTER_NAME'], os.environ['AWS_REGION'])

        # Uninstall the release
        uninstall_cmd = [
            'helm', 'uninstall',
            os.environ['RELEASE_NAME']
        ]
        subprocess.run(uninstall_cmd, check=True)

        return response_data

    except Exception as e:
        print(f"Error during deletion: {str(e)}")
        # Return SUCCESS anyway to allow stack deletion to proceed
        return {
            "Status": "SUCCESS",
            "Reason": f"Proceeding with deletion despite error: {str(e)}"
        }
