# Pre-baking Enroot/Pyxis into a custom AMI

The all-in-one template installs Enroot/Pyxis at **first boot** via
`PostInstallScriptUrl`, which is fast to deploy and avoids an Image Builder step. For
**frequent scaling** in production, pre-baking Enroot/Pyxis into a custom AMI drops node
boot time from ~8–12 min to ~3 min and pins every node to a deterministic state.

This is a separate, standalone path: build the AMI once with
[`pcs-ready-dlami-with-enroot-pyxis.yaml`](../assets/pcs-ready-dlami-with-enroot-pyxis.yaml),
then pass the resulting `ami-xxx` as `AmiId` to the cluster.

## Step 1: Build the AMI (~30 min one-time, separate stack)

[![Launch](../images/launch-stack.svg)](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ready-dlami-with-enroot-pyxis.yaml&stackName=pcs-dlami)

```bash
aws cloudformation create-stack \
  --stack-name pcs-dlami \
  --template-url https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ready-dlami-with-enroot-pyxis.yaml \
  --parameters ParameterKey=SlurmVersion,ParameterValue=25.11 \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM
```

The AMI is **single-Slurm-version by design**: Pyxis is a SPANK plugin whose ABI is
locked to its compile-time Slurm version, so pass the same `SlurmVersion` you'll use on
the cluster.

## Step 2: Read the resulting AMI ID

From the stack output `DLAMIforPCSAmiId`:

```bash
AMI_ID=$(aws cloudformation describe-stacks \
  --stack-name pcs-dlami \
  --query 'Stacks[0].Outputs[?OutputKey==`DLAMIforPCSAmiId`].OutputValue' \
  --output text)
echo "$AMI_ID"   # ami-0xxxxxxxxxxxxxxxx
```

## Step 3: Pass it to the cluster as `AmiId`

Optionally skip the boot-time Enroot/Pyxis install (it's already baked in) by setting
`PostInstallScriptUrl` to a single space:

```bash
aws cloudformation create-stack \
  --stack-name pcs-ml-cluster \
  --template-url https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=us-east-1a \
    ParameterKey=AmiId,ParameterValue=$AMI_ID \
    ParameterKey=PostInstallScriptUrl,ParameterValue=' ' \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM
```

Leaving `PostInstallScriptUrl` at its default (empty → auto-install Enroot/Pyxis from the
templates bucket) also works on a pre-baked AMI: the installer detects Enroot/Pyxis is
already present and is a fast idempotent no-op. Passing a single space skips the
download+check entirely, shaving a few seconds off boot.

## Optional features of `pcs-ready-dlami-with-enroot-pyxis.yaml`

Defaults are off:
- `BuildSchedule=Weekly`/`Monthly` for scheduled rebuilds against a moving base AMI
- `EnableLifecyclePolicy=true` to deprecate older AMIs after `LifecycleDeprecateAfterWeeks`
- `PublishToSsm=true` to publish the latest AMI ID to an SSM parameter for downstream stacks

For production deploys that pin the AMI explicitly per cluster, none of these are needed.

> The build path is validated by [tests/infra-test.md Test 8](../tests/infra-test.md#test-8-pre-baked-ami-build-standalone-dlami-template).
