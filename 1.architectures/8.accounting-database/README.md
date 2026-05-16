# Deploy an accounting database for Slurm

This sample setup a Amazon RDS database for HPC and Machine Learning cluster.
You can use for Slurm accounting and generate report of your cluster usage.
For more information you can visit Slurm documentation on [accounting](https://slurm.schedmd.com/accounting.html).

You will need at least two private subnets in different avaibility zones to deploy the database.

## Deploy

Deploy the accounting database using the 1-click deploy:

[<kbd> <br> 1-Click Deploy 🚀 <br> </kbd>](https://console.aws.amazon.com/cloudformation/home?#/stacks/quickcreate?templateURL=https%3A%2F%2Fawsome-distributed-training.s3.amazonaws.com%2Ftemplates%2Fcf_database-accounting.yaml&stackName=slurm-accounting-database)

**Note** or you can deploy using AWS cli and CloudFormation:
  ```bash
  aws cloudformation deploy --stack-name slurm-accounting-database \
  --template-file cf_database-accounting.yaml \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND \
  --parameter-overrides VpcId=XXX SubnetIds=XXX,XXX
  ```

## Get database parameters
In this section, you will retrieve the database parameter that are used by Slurm to connect to the accounting database.

### Get the Database URI

Retrieve the `DatabaseHost` to connect to the LDAP User Interface.
```bash
DATABASE_URI=$(aws cloudformation describe-stacks \
 --stack-name slurm-accounting-database \
 --query 'Stacks[0].Outputs[?OutputKey==`DatabaseHost`].OutputValue' \
 --output text)
```
  Copy URL into a Web Browser.

### Get the Database Admin User
The database admin user is by default `custeradmin` if you didn't change it on creation.

Get the Database admin user name
```bash
 DATABASE_ADMIN=$(aws cloudformation describe-stacks \
 	--stack-name slurm-accounting-database \
  --query 'Stacks[0].Outputs[?OutputKey==`DatabaseAdminUser`].OutputValue' \
  --output text)
```

### Get the Database password
The password to access the database was generated randomly and stored in AWS Secret Manager under `AccountingClusterAdminSecre-XXX` output of the cloudformation stack.

Get the Secret ARN
```bash
DATABASE_SECRET_ARN=$(aws cloudformation describe-stacks \
  --stack-name slurm-accounting-database \
  --query 'Stacks[0].Outputs[?OutputKey==`DatabaseSecretArn`].OutputValue' \
  --output text)
```

## Configure AWS ParallelCluster
Starting with version 3.3.0, AWS ParallelCluster supports Slurm accounting with the cluster configuration parameter `SlurmSettings / Database`.

To use the database created previously for accounting, add the following in the `SlurmSettings` section of your cluster configuration file:

```yaml
    Database:
      Uri: ${DATABASE_URI}:3306
      UserName: ${DATABASE_ADMIN}
      PasswordSecretArn: ${DATABASE_SECRET_ARN}
    CustomSlurmSettings:
      # Enable accounting for GPU resources.
      # - AccountingStorageTRES: gres/gpu
      - AccountingStorageTRES: gres/gpu
```

## Amazon SageMaker HyperPod Orchestrated by Slurm
There are two steps to setup Slurm with the accounting database:
1. Add database configuration file
1. Configure Slurm accounting

### Add database configuration file
You need to execute the following commands on the controller node to configure the database connectivity for Slurm.

The Aurora cluster parameter group deployed by `cf_database-accounting.yaml` sets `require_secure_transport: ON`, so slurmdbd must connect over TLS and verify the RDS server certificate. The RDS public CA is not in the default OS trust store on Amazon Linux or Ubuntu, so first download the RDS global CA bundle and make it readable:

```bash
# Download the RDS public CA bundle so slurmdbd can verify the TLS server certificate
# required by the cluster parameter group (require_secure_transport: ON).
sudo curl -o /etc/ssl/certs/rds-global-bundle.pem \
  https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem
sudo chmod 644 /etc/ssl/certs/rds-global-bundle.pem
```

Then write the slurmdbd configuration. The `StorageParameters=SSL_CA=...` line points slurmdbd at the bundle you just downloaded; see the Slurm [`StorageParameters` documentation](https://slurm.schedmd.com/slurmdbd.conf.html#OPT_StorageParameters) for details.

```bash
cat > /opt/slurm/etc/slurmdbd.conf << EOF
AuthType=auth/munge
DbdHost=$(hostname) # Slurm controller ip address.
DbdPort=6819
SlurmUser=slurm
LogFile=/var/log/slurmdbd.log
StorageType=accounting_storage/mysql
StorageUser=${DATABASE_ADMIN}
StoragePass=$(aws secretsmanager get-secret-value --secret-id ${DATABASE_SECRET_ARN} --query SecretString --output text)
StorageHost=${DATABASE_URI}
StoragePort=3306
StorageParameters=SSL_CA=/etc/ssl/certs/rds-global-bundle.pem
EOF
```

### Configure Slurm accounting

```bash
cat >> /opt/slurm/etc/slurm.conf << EOF
# ACCOUNTING
JobAcctGatherType=jobacct_gather/linux
JobAcctGatherFrequency=60
AccountingStorageType=accounting_storage/slurmdbd
AccountingStorageHost=$(hostname)
AccountingStorageUser=${DATABASE_ADMIN}
AccountingStoragePort=6819
AccountingStorageTRES=gres/gpu
EOF
```

Restart the slurmctld to pickup the configuration change.
```bash
sudo systemctl restart slurmdctld
sudo scontrol reconfigure
```

For more info how to use Slurm accounting you can read some examples on the [HPC blog](https://aws.amazon.com/blogs/compute/enabling-job-accounting-for-hpc-with-aws-parallelcluster-and-amazon-rds/) 

## Delete the database
Once you delete your cluster no longer need to keep Slurm acocunting data, you can delete the database.
You can use the command below to delete the AWs CloudFormation stack of the database.
**ALL** accounting will be deleted.

```bash
aws cloudformation delete-stack --stack-name slurm-accounting-database
```
